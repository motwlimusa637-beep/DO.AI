"""Pipeline steps: seed demo data, ingest files, detect anomalies, investigate."""

from datetime import timedelta

import pandas as pd
from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session

from . import agent, privacy
from .config import claude_enabled, settings
from .db import Anomaly, Cell, KpiHourly, audit
from .detect import detect_anomalies
from .normalize import UNIFIED_KPIS, normalize
from .synthetic import make_cells, make_kpis

KPI_COLS = list(UNIFIED_KPIS)


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.astype(object).where(pd.notna(df), None)
    recs = df.to_dict("records")
    for r in recs:
        if isinstance(r.get("ts"), pd.Timestamp):
            r["ts"] = r["ts"].to_pydatetime()
    return recs


def store_kpis(session: Session, kpis: pd.DataFrame) -> int:
    """Replace any existing rows for the same cells and time range, then insert."""
    if kpis.empty:
        return 0
    kpis = kpis[["cell_id", "ts", *KPI_COLS]]
    lo, hi = kpis.ts.min().to_pydatetime(), kpis.ts.max().to_pydatetime()
    cells = kpis.cell_id.unique().tolist()
    for i in range(0, len(cells), 500):
        session.execute(
            delete(KpiHourly).where(
                KpiHourly.cell_id.in_(cells[i : i + 500]), KpiHourly.ts >= lo, KpiHourly.ts <= hi
            )
        )
    recs = _records(kpis)
    for i in range(0, len(recs), 5000):
        session.execute(insert(KpiHourly), recs[i : i + 5000])
    return len(recs)


def seed_demo(session: Session, days: int = 14, reset: bool = True) -> dict:
    """Load synthetic cells and KPIs (with injected faults) so the demo works with no real data."""
    cells = make_cells()
    if reset:
        session.execute(delete(Anomaly))
        session.execute(delete(KpiHourly))
        session.execute(delete(Cell))
    for r in cells.to_dict("records"):
        session.merge(Cell(**r))
    kpis, faults = make_kpis(cells, days=days)
    n = store_kpis(session, kpis)
    audit(session, "system", "seed_demo", cells=len(cells), rows=n, injected_faults=faults)
    return {"cells": len(cells), "kpi_rows": n, "injected_faults": faults}


def ingest_file(session: Session, df: pd.DataFrame, vendor: str, actor: str = "upload", filename: str = "") -> dict:
    privacy.enforce(df)  # raises PermissionError -> nothing stored
    result = normalize(df, vendor, session=session)
    frame = result.frame
    known = set(session.scalars(select(Cell.cell_id)))
    new_cells = sorted(set(frame.cell_id) - known)
    for cid in new_cells:
        site = cid.rsplit("-", 1)[0]
        session.add(Cell(cell_id=cid, site_id=site, region="unknown", vendor=vendor, neighbors=[]))
    n = store_kpis(session, frame)
    summary = {
        "file": filename,
        "vendor": vendor,
        "rows": n,
        "mapped": result.mapped,
        "learned_by_claude": result.learned,
        "ignored_columns": result.ignored,
        "new_cells": len(new_cells),
    }
    audit(session, actor, "ingest", **summary)
    return summary


def run_detection(session: Session, history_days: int = 8, lookback_hours: int = 24) -> list[int]:
    latest = session.scalar(select(func.max(KpiHourly.ts)))
    if latest is None:
        return []
    since = latest - timedelta(days=history_days)
    rows = session.execute(select(KpiHourly).where(KpiHourly.ts >= since)).scalars().all()
    df = pd.DataFrame([{c: getattr(r, c) for c in ["cell_id", "ts", *KPI_COLS]} for r in rows])
    regions = dict(session.execute(select(Cell.cell_id, Cell.region)).all())
    found = detect_anomalies(df, z_threshold=settings.anomaly_z, lookback_hours=lookback_hours, regions=regions)
    new_ids = []
    for r in found.to_dict("records"):
        ts = pd.Timestamp(r["ts"]).to_pydatetime()
        exists = session.scalar(
            select(Anomaly.id).where(Anomaly.cell_id == r["cell_id"], Anomaly.kpi == r["kpi"], Anomaly.ts == ts)
        )
        if exists:
            continue
        a = Anomaly(cell_id=r["cell_id"], ts=ts, kpi=r["kpi"], category=r["category"], value=r["value"], baseline=r["baseline"], zscore=r["zscore"])
        session.add(a)
        session.flush()
        new_ids.append(a.id)
    audit(session, "system", "detection", new_anomalies=len(new_ids))
    return new_ids


def investigate_open(session: Session, limit: int = 5, client=None) -> list[dict]:
    if not claude_enabled() and client is None:
        return []
    open_ids = session.scalars(
        select(Anomaly.id).where(Anomaly.status == "open").order_by(func.abs(Anomaly.zscore).desc()).limit(limit)
    ).all()
    out = []
    for aid in open_ids:
        res = agent.investigate(session, aid, client=client)
        session.commit()
        out.append({"anomaly_id": aid, "proposals": res["proposals"]})
    return out
