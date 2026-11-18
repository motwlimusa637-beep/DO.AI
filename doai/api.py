"""HTTP API and dashboard. Start with: uvicorn doai.api:app --host 0.0.0.0 --port $PORT"""

import io
import os
from contextlib import asynccontextmanager
from datetime import UTC, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import __version__, agent, pipeline
from .config import claude_enabled, settings
from .db import Anomaly, AuditLog, Cell, KpiHourly, Recommendation, SessionLocal, audit, init_db, utcnow

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    if os.getenv("DOAI_SEED_ON_START", "").lower() in ("1", "true", "yes"):
        with SessionLocal() as s:
            if not s.scalar(select(func.count()).select_from(Cell)):
                pipeline.seed_demo(s)
                pipeline.run_detection(s)
                s.commit()
    yield


app = FastAPI(title="DO.AI", version=__version__, lifespan=lifespan)


def get_session():
    with SessionLocal() as s:
        yield s


def require_token(authorization: str | None = Header(default=None), x_doai_token: str | None = Header(default=None)):
    if not settings.api_token:
        return "anonymous"
    supplied = x_doai_token or (authorization or "").removeprefix("Bearer ").strip()
    if supplied != settings.api_token:
        raise HTTPException(401, "missing or wrong API token")
    return "token-user"


def _iso(v):
    """Always UTC in API output (Postgres may hand back the session's local zone)."""
    if v is None:
        return None
    return (v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)).isoformat()


# ---------- pages ----------
@app.get("/health")
def health():
    return {"ok": True, "version": __version__}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


# ---------- read ----------
@app.get("/api/status")
def status(s: Session = Depends(get_session), _=Depends(require_token)):
    latest = s.scalar(select(func.max(KpiHourly.ts)))
    return {
        "claude_enabled": claude_enabled(),
        "model": settings.reasoning_model,
        "cells": s.scalar(select(func.count()).select_from(Cell)),
        "kpi_rows": s.scalar(select(func.count()).select_from(KpiHourly)),
        "latest_hour": _iso(latest),
        "open_anomalies": s.scalar(select(func.count()).select_from(Anomaly).where(Anomaly.status == "open")),
        "pending_recommendations": s.scalar(
            select(func.count()).select_from(Recommendation).where(Recommendation.status == "pending")
        ),
    }


@app.get("/api/regions")
def regions(s: Session = Depends(get_session), _=Depends(require_token)):
    latest = s.scalar(select(func.max(KpiHourly.ts)))
    if latest is None:
        return []
    since = latest - timedelta(hours=24)
    q = (
        select(
            Cell.region,
            func.count(func.distinct(Cell.cell_id)).label("cells"),
            func.avg(KpiHourly.prb_util_pct).label("prb"),
            func.max(KpiHourly.prb_util_pct).label("prb_max"),
            func.avg(KpiHourly.dl_throughput_mbps).label("dl"),
            func.avg(KpiHourly.drop_rate_pct).label("drop"),
            func.avg(KpiHourly.availability_pct).label("avail"),
        )
        .join(KpiHourly, KpiHourly.cell_id == Cell.cell_id)
        .where(KpiHourly.ts > since)
        .group_by(Cell.region)
        .order_by(Cell.region)
    )
    return [
        {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r._mapping.items()} for r in s.execute(q)
    ]


@app.get("/api/cells/{cell_id}/kpis")
def cell_kpis(cell_id: str, hours: int = 72, s: Session = Depends(get_session), _=Depends(require_token)):
    latest = s.scalar(select(func.max(KpiHourly.ts)).where(KpiHourly.cell_id == cell_id))
    if latest is None:
        raise HTTPException(404, "no data for cell")
    rows = s.scalars(
        select(KpiHourly)
        .where(KpiHourly.cell_id == cell_id, KpiHourly.ts > latest - timedelta(hours=min(hours, 24 * 14)))
        .order_by(KpiHourly.ts)
    )
    return [
        {
            "ts": _iso(r.ts),
            "prb_util_pct": r.prb_util_pct,
            "dl_throughput_mbps": r.dl_throughput_mbps,
            "active_users": r.active_users,
            "drop_rate_pct": r.drop_rate_pct,
            "availability_pct": r.availability_pct,
        }
        for r in rows
    ]


@app.get("/api/anomalies")
def anomalies(status: str | None = None, limit: int = 100, s: Session = Depends(get_session), _=Depends(require_token)):
    q = select(Anomaly).order_by(Anomaly.ts.desc()).limit(min(limit, 500))
    if status:
        q = q.where(Anomaly.status == status)
    return [
        {
            "id": a.id,
            "cell_id": a.cell_id,
            "ts": _iso(a.ts),
            "kpi": a.kpi,
            "category": a.category,
            "value": round(a.value, 2),
            "baseline": round(a.baseline, 2),
            "zscore": round(a.zscore, 1),
            "status": a.status,
            "analysis": a.analysis,
        }
        for a in s.scalars(q)
    ]


def _rec(r: Recommendation) -> dict:
    return {
        "id": r.id,
        "anomaly_id": r.anomaly_id,
        "cell_id": r.cell_id,
        "action": r.action,
        "parameters": r.parameters,
        "rationale": r.rationale,
        "expected_impact": r.expected_impact,
        "risk": r.risk,
        "status": r.status,
        "reviewer": r.reviewer,
        "review_note": r.review_note,
        "created_at": _iso(r.created_at),
        "reviewed_at": _iso(r.reviewed_at),
    }


@app.get("/api/recommendations")
def recommendations(status: str | None = None, s: Session = Depends(get_session), _=Depends(require_token)):
    q = select(Recommendation).order_by(Recommendation.created_at.desc()).limit(200)
    if status:
        q = q.where(Recommendation.status == status)
    return [_rec(r) for r in s.scalars(q)]


@app.get("/api/recommendations/export.csv")
def export_approved(s: Session = Depends(get_session), _=Depends(require_token)):
    rows = [_rec(r) for r in s.scalars(select(Recommendation).where(Recommendation.status == "approved"))]
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=approved_change_requests.csv"},
    )


@app.get("/api/audit")
def audit_log(limit: int = 50, s: Session = Depends(get_session), _=Depends(require_token)):
    q = select(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 500))
    return [{"id": a.id, "at": _iso(a.at), "actor": a.actor, "event": a.event, "detail": a.detail} for a in s.scalars(q)]


# ---------- actions ----------
class Review(BaseModel):
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=2, max_length=128)
    note: str | None = None


@app.post("/api/recommendations/{rec_id}/review")
def review(rec_id: int, body: Review, s: Session = Depends(get_session), _=Depends(require_token)):
    r = s.get(Recommendation, rec_id)
    if r is None:
        raise HTTPException(404, "not found")
    if r.status != "pending":
        raise HTTPException(409, f"already {r.status}")
    r.status, r.reviewer, r.review_note, r.reviewed_at = body.decision, body.reviewer, body.note, utcnow()
    audit(s, body.reviewer, f"recommendation_{body.decision}", recommendation_id=rec_id, note=body.note)
    s.commit()
    return _rec(r)


@app.post("/api/detect")
def detect(s: Session = Depends(get_session), _=Depends(require_token)):
    ids = pipeline.run_detection(s)
    s.commit()
    return {"new_anomalies": ids}


@app.post("/api/seed")
def seed(days: int = 14, s: Session = Depends(get_session), _=Depends(require_token)):
    if os.getenv("DOAI_ALLOW_SEED", "true").lower() not in ("1", "true", "yes"):
        raise HTTPException(403, "seeding disabled")
    out = pipeline.seed_demo(s, days=min(days, 30))
    out["new_anomalies"] = pipeline.run_detection(s)
    s.commit()
    return out


@app.post("/api/ingest")
async def ingest(
    file: UploadFile = File(...),
    vendor: str = Form(...),
    s: Session = Depends(get_session),
    _=Depends(require_token),
):
    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(413, "file too large for the PoC (50 MB)")
    try:
        name = (file.filename or "").lower()
        df = pd.read_parquet(io.BytesIO(raw)) if name.endswith(".parquet") else pd.read_csv(io.BytesIO(raw))
        out = pipeline.ingest_file(s, df, vendor.lower(), actor="upload", filename=file.filename or "")
    except PermissionError as e:
        audit(s, "upload", "ingest_rejected", file=file.filename, reason=str(e))
        s.commit()
        raise HTTPException(422, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    s.commit()
    return out


def _need_claude():
    if not claude_enabled():
        raise HTTPException(503, "Claude is not configured: set ANTHROPIC_API_KEY")


@app.post("/api/anomalies/{anomaly_id}/investigate")
def investigate(anomaly_id: int, s: Session = Depends(get_session), _=Depends(require_token)):
    _need_claude()
    try:
        out = agent.investigate(s, anomaly_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    s.commit()
    return out


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.post("/api/ask")
def ask(body: Question, s: Session = Depends(get_session), _=Depends(require_token)):
    _need_claude()
    out = agent.ask(s, body.question)
    s.commit()
    return out
