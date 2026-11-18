"""Normalize vendor-specific KPI exports into the unified kpi_hourly schema.

Known counter names are mapped from a built-in table plus mappings learned earlier
(stored in counter_mappings). Unknown columns are sent to Claude once - only the
column names and a few sample values, never whole files - and the answer is saved
for reuse. The built-in names below are illustrative; replace them with the exact
counter names in stc's PM exports.
"""

import json
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import claude_enabled, settings
from .db import CounterMapping

UNIFIED_KPIS = {
    "prb_util_pct": "Downlink PRB utilization, percent 0-100",
    "dl_throughput_mbps": "Average downlink user throughput, Mbps",
    "ul_throughput_mbps": "Average uplink user throughput, Mbps",
    "active_users": "Average connected/active users in the hour",
    "drop_rate_pct": "Call/session drop rate, percent 0-100",
    "availability_pct": "Cell availability, percent 0-100",
}

CELL_ALIASES = {"cell_id", "cell", "cell name", "cellname", "eutrancell", "nrcellcu", "cell_name"}
TIME_ALIASES = {"ts", "timestamp", "datetime", "start time", "starttime", "period_start", "date_time"}

# vendor -> {source column: (target kpi, scale)}
BUILTIN: dict[str, dict[str, tuple[str, float]]] = {
    "ericsson": {
        "DL_PRB_Utilization": ("prb_util_pct", 1.0),
        "DL_User_Throughput_Mbps": ("dl_throughput_mbps", 1.0),
        "UL_User_Throughput_Mbps": ("ul_throughput_mbps", 1.0),
        "RRC_Connected_Users_Avg": ("active_users", 1.0),
        "Drop_Rate_Pct": ("drop_rate_pct", 1.0),
        "Cell_Availability_Pct": ("availability_pct", 1.0),
    },
    "nokia": {
        "PRB_USAGE_DL_RATIO": ("prb_util_pct", 100.0),
        "DL_THP_KBPS": ("dl_throughput_mbps", 0.001),
        "UL_THP_KBPS": ("ul_throughput_mbps", 0.001),
        "AVG_ACTIVE_UE": ("active_users", 1.0),
        "E_RAB_DROP_RATIO": ("drop_rate_pct", 100.0),
        "CELL_AVAIL_RATIO": ("availability_pct", 100.0),
    },
    "huawei": {
        "DL.PRB.Usage(%)": ("prb_util_pct", 1.0),
        "User.DL.Avg.Thp(Mbps)": ("dl_throughput_mbps", 1.0),
        "User.UL.Avg.Thp(Mbps)": ("ul_throughput_mbps", 1.0),
        "User.Active.Avg": ("active_users", 1.0),
        "Call.Drop.Rate(%)": ("drop_rate_pct", 1.0),
        "Cell.Avail(%)": ("availability_pct", 1.0),
    },
}


@dataclass
class NormalizeResult:
    frame: pd.DataFrame
    mapped: dict[str, str] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)
    learned: dict[str, str] = field(default_factory=dict)


def _find(columns, aliases) -> str | None:
    for c in columns:
        if str(c).strip().lower() in aliases:
            return c
    return None


def _known_mappings(session: Session | None, vendor: str) -> dict[str, tuple[str | None, float]]:
    known: dict[str, tuple[str | None, float]] = dict(BUILTIN.get(vendor, {}))
    if session is not None:
        rows = session.scalars(select(CounterMapping).where(CounterMapping.vendor == vendor))
        for m in rows:
            known[m.source_name] = (m.target_kpi, m.scale)
    return known


MAPPING_PROMPT = """You map telecom vendor performance counters to a unified KPI schema.

Unified KPIs:
{kpis}

Vendor: {vendor}
Unmapped columns with sample values:
{columns}

For each column, return the unified KPI it measures and the multiplier that converts
its values to the unified unit, or null if it matches none. Respond with JSON only:
{{"mappings": [{{"source": "<column>", "target": "<kpi or null>", "scale": <number>, "reason": "<short>"}}]}}"""


def ask_claude_for_mappings(vendor: str, samples: dict[str, list], client=None) -> list[dict]:
    """Ask Claude (fast model) to map unknown counters. Sends names and a few values only."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt = MAPPING_PROMPT.format(
        kpis="\n".join(f"- {k}: {v}" for k, v in UNIFIED_KPIS.items()),
        vendor=vendor,
        columns="\n".join(f"- {c}: {vals}" for c, vals in samples.items()),
    )
    msg = client.messages.create(
        model=settings.fast_model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start : end + 1])
    out = []
    for m in data.get("mappings", []):
        target = m.get("target")
        if target is not None and target not in UNIFIED_KPIS:
            target = None
        out.append({"source": m["source"], "target": target, "scale": float(m.get("scale") or 1.0)})
    return out


def normalize(
    df: pd.DataFrame,
    vendor: str,
    session: Session | None = None,
    use_claude: bool | None = None,
    client=None,
) -> NormalizeResult:
    vendor = vendor.lower()
    cell_col = _find(df.columns, CELL_ALIASES)
    time_col = _find(df.columns, TIME_ALIASES)
    if cell_col is None or time_col is None:
        raise ValueError("File needs a cell column and a timestamp column")

    known = _known_mappings(session, vendor)
    value_cols = [c for c in df.columns if c not in (cell_col, time_col)]
    unknown = [c for c in value_cols if c not in known]

    result = NormalizeResult(frame=pd.DataFrame())
    if use_claude is None:
        use_claude = claude_enabled()
    if unknown and use_claude:
        samples = {c: df[c].dropna().head(3).tolist() for c in unknown}
        for m in ask_claude_for_mappings(vendor, samples, client=client):
            known[m["source"]] = (m["target"], m["scale"])
            if m["target"]:
                result.learned[m["source"]] = m["target"]
            if session is not None:
                session.add(
                    CounterMapping(
                        vendor=vendor,
                        source_name=m["source"],
                        target_kpi=m["target"],
                        scale=m["scale"],
                        origin="claude",
                    )
                )

    out = pd.DataFrame(
        {
            "cell_id": df[cell_col].astype(str).str.strip(),
            "ts": pd.to_datetime(df[time_col], utc=True),
        }
    )
    for col in value_cols:
        target, scale = known.get(col, (None, 1.0))
        if target is None:
            result.ignored.append(col)
            continue
        out[target] = pd.to_numeric(df[col], errors="coerce") * scale
        result.mapped[col] = target

    # Basic data-quality optimization: drop exact duplicates and impossible values.
    out = out.drop_duplicates(subset=["cell_id", "ts"], keep="last")
    for pct in ("prb_util_pct", "drop_rate_pct", "availability_pct"):
        if pct in out:
            out.loc[(out[pct] < 0) | (out[pct] > 100), pct] = None
    for k in UNIFIED_KPIS:
        if k not in out:
            out[k] = None
    result.frame = out.reset_index(drop=True)
    return result
