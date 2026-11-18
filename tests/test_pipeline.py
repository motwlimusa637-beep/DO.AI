import pandas as pd

from doai import pipeline
from doai.db import Anomaly, KpiHourly
from doai.normalize import normalize
from doai.synthetic import make_cells, make_kpis, to_vendor_format


def test_vendor_formats_round_trip():
    cells = make_cells()
    kpis, _ = make_kpis(cells, days=1)
    sample = kpis[kpis.cell_id == cells.cell_id[0]]
    for vendor in ("ericsson", "nokia", "huawei"):
        res = normalize(to_vendor_format(sample, vendor), vendor, use_claude=False)
        assert not res.ignored
        got = res.frame.sort_values("ts").reset_index(drop=True)
        exp = sample.sort_values("ts").reset_index(drop=True)
        assert (got.dl_throughput_mbps - exp.dl_throughput_mbps).abs().max() < 0.01


def test_unknown_columns_ignored_without_claude():
    df = pd.DataFrame({"Cell Name": ["A-001-1"], "Start Time": ["2026-11-18T10:00:00Z"], "MysteryCounter": [3]})
    res = normalize(df, "ericsson", use_claude=False)
    assert res.ignored == ["MysteryCounter"]


def test_claude_mapping_is_used_and_learned(session):
    class FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                class B:
                    type = "text"
                    text = '{"mappings":[{"source":"pmDlThpKbps","target":"dl_throughput_mbps","scale":0.001,"reason":"kbps"}]}'

                class M:
                    content = [B()]

                return M()

    df = pd.DataFrame({"Cell Name": ["A-001-1"], "Start Time": ["2026-11-18T10:00:00Z"], "pmDlThpKbps": [150000]})
    res = normalize(df, "ericsson", session=session, use_claude=True, client=FakeClient())
    assert res.learned == {"pmDlThpKbps": "dl_throughput_mbps"}
    assert abs(res.frame.dl_throughput_mbps[0] - 150.0) < 1e-9
    session.flush()
    # Second time it is known, no Claude call needed
    res2 = normalize(df, "ericsson", session=session, use_claude=False)
    assert res2.mapped == {"pmDlThpKbps": "dl_throughput_mbps"}


def test_seed_and_detect_finds_injected_faults(session):
    out = pipeline.seed_demo(session, days=10)
    assert session.query(KpiHourly).count() == out["kpi_rows"]
    ids = pipeline.run_detection(session)
    anoms = session.query(Anomaly).all()
    found = {a.cell_id for a in anoms}
    injected = {f["cell_id"] for f in out["injected_faults"]}
    assert ids, "detector should flag something"
    assert injected <= found, f"missed injected faults: {injected - found}"
    # The surge must not be reported as faults: anything else flagged is congestion
    extra = [a for a in anoms if a.cell_id not in injected]
    assert all(a.category == "congestion" for a in extra), [(a.cell_id, a.kpi, a.category) for a in extra]
    # Running again does not duplicate
    assert pipeline.run_detection(session) == []


def test_ingest_rejects_pii(session):
    df = pd.DataFrame({"Cell Name": ["A-001-1"], "Start Time": ["2026-11-18T10:00:00Z"], "MSISDN": ["966512345678"]})
    try:
        pipeline.ingest_file(session, df, "ericsson")
    except PermissionError:
        assert session.query(KpiHourly).count() == 0
    else:
        raise AssertionError("PII file was accepted")
