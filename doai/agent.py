"""Claude agent: investigates anomalies, proposes changes, answers questions.

Claude only ever:
  * reads data through guarded, read-only tools, and
  * writes recommendations to a pending queue that an engineer must approve.
It never changes the network. Every run is written to the audit log.
"""

import json
import os
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from . import db, sqlguard
from .config import settings
from .db import Anomaly, Cell, KpiHourly, Recommendation, audit
from .detect import forecast_load

ALLOWED_ACTIONS = [
    "adjust_antenna_tilt",
    "adjust_tx_power",
    "load_balancing_offset",
    "activate_capacity_carrier",
    "dispatch_field_check",
    "monitor_only",
]

SYSTEM_PROMPT = f"""You are DO.AI, a radio network optimization analyst supporting stc engineers in Saudi Arabia.

You work with cell-level 5G KPIs (no subscriber data). Tables you can query with SQL:
- cells(cell_id, site_id, region, vendor, technology, neighbors JSON)
- kpi_hourly(cell_id, ts UTC, prb_util_pct, dl_throughput_mbps, ul_throughput_mbps, active_users, drop_rate_pct, availability_pct)
- anomalies(id, cell_id, ts, kpi, category fault|congestion, value, baseline = peer median that hour, zscore, status, analysis)
- recommendations(id, anomaly_id, cell_id, action, parameters, rationale, expected_impact, risk, status)
Local time in Saudi Arabia is UTC+3.

How to work:
- Gather evidence with tools before concluding. Compare the cell to its own history and to its neighbors.
- Separate what the data shows from what you infer. State confidence (low/medium/high).
- Recommend changes only with propose_change, using one of: {", ".join(ALLOWED_ACTIONS)}.
  Prefer the least risky action that fits the evidence. Hardware or outage symptoms -> dispatch_field_check.
  Capacity exhaustion (PRB near saturation with falling throughput) -> load balancing or capacity carrier.
- Your proposals are reviewed by an engineer; never claim a change was applied.
- Answer in the language of the question (Arabic or English). Be concise and specific; cite numbers and times."""

TOOLS = [
    {
        "name": "query_kpis",
        "description": "Run one read-only SQL SELECT against the KPI database. Results are capped in rows. Use aggregates for large ranges.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement"}},
            "required": ["sql"],
        },
    },
    {
        "name": "get_cell_context",
        "description": "Cell inventory, its neighbors, and hourly KPIs for the cell and neighbors around a time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cell_id": {"type": "string"},
                "around": {"type": "string", "description": "ISO timestamp to center on; default latest"},
                "hours": {"type": "integer", "description": "Window size in hours (default 12, max 72)"},
            },
            "required": ["cell_id"],
        },
    },
    {
        "name": "forecast_load",
        "description": "Forecast a KPI for a cell for the next N hours (seasonal model).",
        "input_schema": {
            "type": "object",
            "properties": {
                "cell_id": {"type": "string"},
                "kpi": {"type": "string", "enum": ["prb_util_pct", "active_users", "dl_throughput_mbps"]},
                "hours": {"type": "integer", "description": "1-48, default 24"},
            },
            "required": ["cell_id"],
        },
    },
    {
        "name": "propose_change",
        "description": "Submit a recommended network change for engineer review. It is NOT applied.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cell_id": {"type": "string"},
                "action": {"type": "string", "enum": ALLOWED_ACTIONS},
                "parameters": {"type": "object", "description": "e.g. {\"tilt_delta_deg\": 1}"},
                "rationale": {"type": "string"},
                "expected_impact": {"type": "string"},
                "risk": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["cell_id", "action", "rationale", "risk"],
        },
    },
]


def _jsonable(v):
    if isinstance(v, (datetime, date, pd.Timestamp)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _rows(result) -> list[dict]:
    return [{k: _jsonable(v) for k, v in r._mapping.items()} for r in result]


_ro_engine = None


def _readonly_engine():
    global _ro_engine
    if _ro_engine is None:
        url = os.getenv("DATABASE_URL_READONLY")
        _ro_engine = db.make_engine(url) if url else db.engine
    return _ro_engine


class ToolBox:
    def __init__(self, session: Session, anomaly_id: int | None = None, engine=None):
        self.session = session
        self.anomaly_id = anomaly_id
        self.engine = engine or _readonly_engine()
        self.proposals: list[int] = []

    def run(self, name: str, args: dict):
        try:
            return getattr(self, f"t_{name}")(**args)
        except sqlguard.UnsafeQuery as e:
            return {"error": f"query refused: {e}"}
        except Exception as e:  # report tool errors back to Claude instead of crashing
            return {"error": f"{type(e).__name__}: {e}"}

    def t_query_kpis(self, sql: str):
        safe = sqlguard.check(sql, settings.max_query_rows)
        if self.engine.dialect.name == "sqlite":
            # Local dev only: SQLite allows one writer, so read through the session's own connection.
            return {"rows": (rows := _rows(self.session.execute(text(safe)))), "row_count": len(rows)}
        with self.engine.connect() as conn:
            if conn.dialect.name == "postgresql":
                conn.execute(text("SET TRANSACTION READ ONLY"))
            rows = _rows(conn.execute(text(safe)))
            conn.rollback()
        return {"row_count": len(rows), "rows": rows}

    def t_get_cell_context(self, cell_id: str, around: str | None = None, hours: int = 12):
        cell = self.session.get(Cell, cell_id)
        if cell is None:
            return {"error": f"unknown cell {cell_id}"}
        hours = max(1, min(int(hours), 72))
        center = (
            pd.Timestamp(around).tz_localize("UTC").to_pydatetime()
            if around and pd.Timestamp(around).tzinfo is None
            else pd.Timestamp(around).tz_convert("UTC").to_pydatetime()
            if around
            else self.session.scalar(select(KpiHourly.ts).where(KpiHourly.cell_id == cell_id).order_by(KpiHourly.ts.desc()).limit(1))
        )
        lo, hi = center - timedelta(hours=hours / 2), center + timedelta(hours=hours / 2)
        ids = [cell_id, *cell.neighbors]
        q = (
            select(KpiHourly)
            .where(KpiHourly.cell_id.in_(ids), KpiHourly.ts >= lo, KpiHourly.ts <= hi)
            .order_by(KpiHourly.cell_id, KpiHourly.ts)
        )
        series: dict[str, list] = {}
        for k in self.session.scalars(q):
            series.setdefault(k.cell_id, []).append(
                {
                    "ts": k.ts.isoformat(),
                    "prb": k.prb_util_pct,
                    "dl": k.dl_throughput_mbps,
                    "users": k.active_users,
                    "drop": k.drop_rate_pct,
                    "avail": k.availability_pct,
                }
            )
        return {
            "cell": {"cell_id": cell.cell_id, "site_id": cell.site_id, "region": cell.region, "vendor": cell.vendor},
            "neighbors": cell.neighbors,
            "kpis": series,
        }

    def t_forecast_load(self, cell_id: str, kpi: str = "prb_util_pct", hours: int = 24):
        hours = max(1, min(int(hours), 48))
        since = datetime.now(UTC) - timedelta(days=10)
        rows = self.session.scalars(
            select(KpiHourly).where(KpiHourly.cell_id == cell_id, KpiHourly.ts >= since)
        ).all()
        if not rows:
            rows = self.session.scalars(select(KpiHourly).where(KpiHourly.cell_id == cell_id)).all()
        df = pd.DataFrame([{"cell_id": r.cell_id, "ts": r.ts, kpi: getattr(r, kpi)} for r in rows])
        return {"cell_id": cell_id, "kpi": kpi, "forecast": forecast_load(df, cell_id, kpi, hours)}

    def t_propose_change(self, cell_id, action, rationale, risk, parameters=None, expected_impact=None):
        if action not in ALLOWED_ACTIONS:
            return {"error": f"action must be one of {ALLOWED_ACTIONS}"}
        if self.session.get(Cell, cell_id) is None:
            return {"error": f"unknown cell {cell_id}"}
        rec = Recommendation(
            anomaly_id=self.anomaly_id,
            cell_id=cell_id,
            action=action,
            parameters=parameters or {},
            rationale=rationale,
            expected_impact=expected_impact,
            risk=risk,
        )
        self.session.add(rec)
        self.session.flush()
        self.proposals.append(rec.id)
        return {"recommendation_id": rec.id, "status": "pending engineer approval"}


def _client():
    import anthropic

    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def run_agent(session: Session, prompt: str, anomaly_id: int | None = None, client=None, actor="agent") -> dict:
    """Tool-use loop. Returns final text, tool trace and proposal ids."""
    client = client or _client()
    tools = ToolBox(session, anomaly_id=anomaly_id)
    messages = [{"role": "user", "content": prompt}]
    trace = []
    final_text = ""
    usage = {"input_tokens": 0, "output_tokens": 0}
    for _ in range(settings.max_agent_steps):
        resp = client.messages.create(
            model=settings.reasoning_model,
            max_tokens=2000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            tools=TOOLS,
            messages=messages,
        )
        u = getattr(resp, "usage", None)
        if u is not None:
            usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
            usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        messages.append({"role": "assistant", "content": resp.content})
        texts = [b.text for b in resp.content if b.type == "text"]
        if texts:
            final_text = "\n".join(texts).strip()
        if resp.stop_reason != "tool_use":
            break
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            out = tools.run(block.name, dict(block.input))
            trace.append({"tool": block.name, "input": block.input, "error": out.get("error") if isinstance(out, dict) else None})
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out, default=_jsonable)[:20000],
                    "is_error": bool(isinstance(out, dict) and out.get("error")),
                }
            )
        messages.append({"role": "user", "content": results})
    else:
        final_text = final_text or "Stopped: step limit reached before a conclusion."

    audit(
        session,
        actor,
        "agent_run",
        prompt=prompt[:2000],
        anomaly_id=anomaly_id,
        tools=trace,
        proposals=tools.proposals,
        answer=final_text[:4000],
        model=settings.reasoning_model,
        usage=usage,
    )
    return {"answer": final_text, "trace": trace, "proposals": tools.proposals, "usage": usage}


def investigate(session: Session, anomaly_id: int, client=None) -> dict:
    a = session.get(Anomaly, anomaly_id)
    if a is None:
        raise ValueError(f"anomaly {anomaly_id} not found")
    prompt = (
        f"Investigate anomaly #{a.id} (detector label: {a.category}): cell {a.cell_id}, KPI {a.kpi} = {a.value:.2f} "
        f"at {a.ts.isoformat()} UTC; its regional peers' median that hour was {a.baseline:.2f} (z-score {a.zscore:.1f}).\n"
        "1) Gather evidence (cell and neighbors around that time, recent history).\n"
        "2) Give the most likely root cause, the evidence, and your confidence.\n"
        "3) If an action is justified, call propose_change once. Otherwise say why not.\n"
        "Finish with a short summary an on-shift engineer can read in 30 seconds."
    )
    out = run_agent(session, prompt, anomaly_id=a.id, client=client)
    a.status = "investigated"
    a.analysis = out["answer"]
    return out


def ask(session: Session, question: str, client=None, actor="user") -> dict:
    return run_agent(session, question, client=client, actor=actor)
