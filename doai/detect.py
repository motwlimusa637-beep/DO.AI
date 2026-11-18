"""Anomaly detection and a simple load forecast (no Claude involved here).

A cell is compared with its *peers* (the other cells in its region) at the same
hour, not only with its own past. A mass-gathering surge raises load on every cell
in Makkah or Mina at once, so peer-relative deviation stays normal and the surge is
not flagged as a fault; a single cell breaking while its peers are fine is.

Each event is labelled 'congestion' (the cell is out of capacity, e.g. a Hajj
surge hot spot: a capacity action is needed) or 'fault' (something broke).

An hour is flagged when both hold:
  * statistically unusual: robust z-score of the cell's peer-relative deviation,
    against that cell's own history of deviations, beyond the threshold; and
  * operationally material: a floor engineers would care about (below).
"""

import numpy as np
import pandas as pd

# KPI -> (bad direction, materiality test on value vs peer median)
WATCHED = {
    "dl_throughput_mbps": ("low", lambda v, peer: v < 0.5 * peer),
    "drop_rate_pct": ("high", lambda v, peer: v - peer >= 2.0),
    "availability_pct": ("low", lambda v, peer: v < 95.0),
}
# Minimum spread so very stable KPIs don't flag noise
SPREAD_FLOOR = {"dl_throughput_mbps": 2.0, "drop_rate_pct": 0.1, "availability_pct": 0.2}
MIN_HISTORY = 24

COLUMNS = ["cell_id", "ts", "kpi", "category", "value", "baseline", "zscore"]
# A throughput/drop problem is load-driven ("congestion") when the cell's PRB
# utilization is near saturation, or clearly above its peers' at that hour.
CONGESTION_PRB = 80.0
CONGESTION_PRB_ABOVE_PEERS = 15.0
# Load alone does not push drop rate this far above peers: treat as a fault even on a busy cell.
DROP_FAULT_ABOVE_PEERS = 3.0


def detect_anomalies(
    kpis: pd.DataFrame,
    z_threshold: float = 3.0,
    lookback_hours: int = 24,
    regions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return anomaly events in the last `lookback_hours` of data.

    `regions` maps cell_id -> region (peer group). Without it, all cells are peers.
    `baseline` in the result is the peer median at that hour.
    """
    if kpis.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = kpis.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["peer_group"] = df["cell_id"].map(regions or {}).fillna("all")
    if "prb_util_pct" in df:
        df["peer_prb"] = df.groupby(["peer_group", "ts"])["prb_util_pct"].transform("median")
    cutoff = df["ts"].max() - pd.Timedelta(hours=lookback_hours)

    found = []
    for kpi, (direction, material) in WATCHED.items():
        if kpi not in df or df[kpi].isna().all():
            continue
        peer = df.groupby(["peer_group", "ts"])[kpi].transform("median")
        dev = df[kpi] - peer
        hist = df["ts"] <= cutoff
        stats = dev[hist].groupby(df.loc[hist, "cell_id"]).agg(
            ["median", lambda s: float(np.median(np.abs(s - np.median(s)))), "count"]
        )
        stats.columns = ["med", "mad", "n"]
        cols = ["cell_id", "ts", kpi] + (["prb_util_pct", "peer_prb"] if "prb_util_pct" in df else [])
        recent = df.loc[~hist, cols].assign(dev=dev[~hist], peer=peer[~hist])
        recent = recent.join(stats, on="cell_id")
        recent = recent[recent["n"] >= MIN_HISTORY]
        scale = np.maximum(1.4826 * recent["mad"], SPREAD_FLOOR[kpi])
        z = (recent["dev"] - recent["med"]) / scale
        unusual = z <= -z_threshold if direction == "low" else z >= z_threshold
        hits = recent[unusual & material(recent[kpi], recent["peer"])]
        for idx, r in hits.iterrows():
            found.append(
                {
                    "cell_id": r.cell_id,
                    "ts": r.ts,
                    "kpi": kpi,
                    "category": _category(kpi, r),
                    "value": float(r[kpi]),
                    "baseline": float(r["peer"]),
                    "zscore": float(z.loc[idx]),
                }
            )
    return collapse_events(pd.DataFrame(found, columns=COLUMNS))


def _category(kpi: str, row) -> str:
    """'congestion' when a throughput/drop problem coincides with a saturated cell, else 'fault'."""
    prb, peer_prb = row.get("prb_util_pct"), row.get("peer_prb")
    if kpi not in ("dl_throughput_mbps", "drop_rate_pct") or prb is None or prb != prb:
        return "fault"
    if kpi == "drop_rate_pct" and row[kpi] - row["peer"] >= DROP_FAULT_ABOVE_PEERS:
        return "fault"
    if prb >= CONGESTION_PRB or (peer_prb == peer_prb and peer_prb is not None and prb - peer_prb >= CONGESTION_PRB_ABOVE_PEERS):
        return "congestion"
    return "fault"


def collapse_events(anoms: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive hourly hits for the same cell+KPI into one event (worst hour kept)."""
    if anoms.empty:
        return anoms
    anoms = anoms.sort_values(["cell_id", "kpi", "ts"]).reset_index(drop=True)
    gap = anoms.groupby(["cell_id", "kpi"])["ts"].diff() > pd.Timedelta(hours=1)
    anoms["event"] = gap.groupby([anoms.cell_id, anoms.kpi]).cumsum()
    worst_idx = anoms.groupby(["cell_id", "kpi", "event"])["zscore"].apply(lambda s: s.abs().idxmax())
    return anoms.loc[worst_idx.values].drop(columns="event").reset_index(drop=True)


def forecast_load(kpis: pd.DataFrame, cell_id: str, kpi: str = "prb_util_pct", hours: int = 24) -> list[dict]:
    """Seasonal forecast: same hour of day, averaged over the last 3 days, trended by recent change."""
    df = kpis[kpis.cell_id == cell_id].copy()
    if df.empty:
        return []
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").set_index("ts")[kpi].dropna()
    last = df.index.max()
    recent = df[df.index > last - pd.Timedelta(days=3)]
    prior = df[(df.index <= last - pd.Timedelta(days=3)) & (df.index > last - pd.Timedelta(days=6))]
    by_hour = recent.groupby(recent.index.hour).mean()
    trend = (recent.mean() / prior.mean()) if len(prior) and prior.mean() else 1.0
    trend = float(np.clip(trend, 0.7, 1.5))
    out = []
    for h in range(1, hours + 1):
        t = last + pd.Timedelta(hours=h)
        v = float(by_hour.get(t.hour, recent.mean())) * (trend ** 0.5)
        if kpi.endswith("_pct"):
            v = min(v, 100.0)
        out.append({"ts": t.isoformat(), kpi: round(v, 2)})
    return out
