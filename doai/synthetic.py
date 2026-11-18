"""Synthetic network data, so the PoC never needs real stc data.

Generates cells in Makkah, Mina, Arafat, Riyadh and Jeddah with daily traffic
patterns, a Hajj-style surge on Mashaer cells, and a few injected faults
(throughput collapse, drop-rate spike, outage) for the detector to find.
Exports in vendor-specific column formats to exercise normalization.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from .normalize import BUILTIN

REGIONS = {
    "Makkah": ("MKK", "ericsson", 8),
    "Mina": ("MNA", "ericsson", 6),
    "Arafat": ("ARF", "nokia", 6),
    "Riyadh": ("RUH", "huawei", 8),
    "Jeddah": ("JED", "nokia", 6),
}
SURGE_REGIONS = {"Mina", "Arafat", "Makkah"}


def make_cells() -> pd.DataFrame:
    rows = []
    for region, (code, vendor, n_sites) in REGIONS.items():
        for s in range(1, n_sites + 1):
            for sector in (1, 2, 3):
                rows.append(
                    {
                        "cell_id": f"{code}-{s:03d}-{sector}",
                        "site_id": f"{code}-{s:03d}",
                        "region": region,
                        "vendor": vendor,
                        "technology": "5G",
                    }
                )
    cells = pd.DataFrame(rows)
    # Neighbours: other sectors of the same site plus sector 1 of adjacent sites.
    neigh = {}
    for _, c in cells.iterrows():
        code, site, sector = c.cell_id.split("-")
        site_n = int(site)
        same = [f"{code}-{site}-{x}" for x in (1, 2, 3) if str(x) != sector]
        adj = [f"{code}-{n:03d}-1" for n in (site_n - 1, site_n + 1) if n >= 1 and f"{code}-{n:03d}" in set(cells.site_id)]
        neigh[c.cell_id] = same + adj
    cells["neighbors"] = cells.cell_id.map(neigh)
    return cells


def make_kpis(
    cells: pd.DataFrame,
    days: int = 14,
    end: datetime | None = None,
    seed: int = 7,
) -> tuple[pd.DataFrame, list[dict]]:
    """Return unified-schema KPIs and the list of injected faults (ground truth)."""
    rng = np.random.default_rng(seed)
    end = (end or datetime.now(UTC)).replace(minute=0, second=0, microsecond=0)
    hours = pd.date_range(end=end, periods=days * 24, freq="h", tz="UTC")
    hod = hours.hour.to_numpy()
    # Daily shape: low at 04:00, peaks around 21:00 local (UTC+3)
    local = (hod + 3) % 24
    daily = 0.35 + 0.65 * np.clip(np.sin((local - 6) / 24 * 2 * np.pi) * 0.5 + 0.5, 0, 1)
    surge_start = len(hours) - 72  # last 3 days look like a mass-gathering surge

    frames = []
    for _, c in cells.iterrows():
        base_users = rng.uniform(60, 220)
        load = daily * rng.uniform(0.9, 1.1, len(hours))
        if c.region in SURGE_REGIONS:
            load[surge_start:] *= rng.uniform(1.8, 2.4)
        users = base_users * load
        prb = np.clip(users / (base_users * 2.6) * 100 + rng.normal(0, 3, len(hours)), 2, 99)
        dl = np.clip(320 * (1 - prb / 115) + rng.normal(0, 12, len(hours)), 3, None)
        ul = np.clip(dl * rng.uniform(0.12, 0.18), 1, None)
        drop = np.clip(0.25 + (prb / 100) ** 4 * 1.2 + rng.normal(0, 0.05, len(hours)), 0.02, None)
        avail = np.clip(99.95 + rng.normal(0, 0.03, len(hours)), 0, 100)
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": c.cell_id,
                    "ts": hours,
                    "prb_util_pct": prb.round(2),
                    "dl_throughput_mbps": dl.round(2),
                    "ul_throughput_mbps": ul.round(2),
                    "active_users": users.round(1),
                    "drop_rate_pct": drop.round(3),
                    "availability_pct": avail.round(3),
                }
            )
        )
    kpis = pd.concat(frames, ignore_index=True)

    # Inject faults in the last day
    faults = []
    ids = cells.cell_id.tolist()
    picks = rng.choice(ids, size=3, replace=False)
    kinds = ["throughput_collapse", "drop_spike", "outage"]
    for cell_id, kind in zip(picks, kinds, strict=True):
        t0 = hours[-int(rng.integers(6, 20))]
        mask = (kpis.cell_id == cell_id) & (kpis.ts >= t0) & (kpis.ts < t0 + timedelta(hours=3))
        if kind == "throughput_collapse":
            kpis.loc[mask, "dl_throughput_mbps"] *= 0.15
        elif kind == "drop_spike":
            kpis.loc[mask, "drop_rate_pct"] += 4.5
        else:
            kpis.loc[mask, "availability_pct"] = 20.0
            kpis.loc[mask, "active_users"] *= 0.1
        faults.append({"cell_id": cell_id, "kind": kind, "start": t0.isoformat()})
    return kpis, faults


def to_vendor_format(kpis: pd.DataFrame, vendor: str) -> pd.DataFrame:
    """Rename unified columns to a vendor's export names (inverse of normalization)."""
    mapping = BUILTIN[vendor]
    out = pd.DataFrame(
        {
            "Cell Name": kpis.cell_id,
            "Start Time": kpis.ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    for source, (target, scale) in mapping.items():
        out[source] = (kpis[target] / scale).round(6)
    return out
