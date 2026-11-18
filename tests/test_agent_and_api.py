"""Agent loop and API, with a scripted fake Claude client (no network calls)."""

from types import SimpleNamespace as NS

from fastapi.testclient import TestClient

from doai import agent, pipeline
from doai.db import Anomaly, AuditLog, Recommendation


class ScriptedClient:
    """Plays back a fixed sequence of Claude responses and records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        return self.responses.pop(0)


def tool_use(id_, name, inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def text(t):
    return NS(type="text", text=t)


def test_investigation_reads_data_and_queues_proposal(session):
    pipeline.seed_demo(session, days=10)
    pipeline.run_detection(session)
    a = session.query(Anomaly).first()
    client = ScriptedClient(
        [
            NS(
                stop_reason="tool_use",
                content=[
                    tool_use("t1", "get_cell_context", {"cell_id": a.cell_id, "around": a.ts.isoformat(), "hours": 6}),
                    tool_use("t2", "query_kpis", {"sql": f"SELECT avg(prb_util_pct) AS p FROM kpi_hourly WHERE cell_id = '{a.cell_id}'"}),
                    tool_use("t3", "query_kpis", {"sql": "DELETE FROM kpi_hourly"}),
                ],
                usage=NS(input_tokens=100, output_tokens=20),
            ),
            NS(
                stop_reason="tool_use",
                content=[
                    tool_use(
                        "t4",
                        "propose_change",
                        {"cell_id": a.cell_id, "action": "dispatch_field_check", "rationale": "sudden break", "risk": "low"},
                    )
                ],
                usage=NS(input_tokens=100, output_tokens=20),
            ),
            NS(stop_reason="end_turn", content=[text("Likely hardware fault. Confidence: medium.")], usage=NS(input_tokens=50, output_tokens=10)),
        ]
    )
    out = agent.investigate(session, a.id, client=client)
    session.commit()

    assert out["answer"].startswith("Likely hardware")
    assert [t["tool"] for t in out["trace"]] == ["get_cell_context", "query_kpis", "query_kpis", "propose_change"]
    assert out["trace"][1]["error"] is None
    assert "refused" in out["trace"][2]["error"]  # the DELETE was blocked
    rec = session.get(Recommendation, out["proposals"][0])
    assert rec.status == "pending" and rec.anomaly_id == a.id
    assert session.get(Anomaly, a.id).status == "investigated"
    assert session.query(AuditLog).filter_by(event="agent_run").count() == 1
    # Tool results were sent back to Claude
    sent = client.requests[1]["messages"][2]["content"]
    assert {r["tool_use_id"] for r in sent} == {"t1", "t2", "t3"}


def test_api_flow(session):
    from doai.api import app

    with TestClient(app) as c:
        assert c.get("/health").json()["ok"]
        assert c.get("/").status_code == 200
        seeded = c.post("/api/seed?days=10").json()
        assert seeded["new_anomalies"]
        st = c.get("/api/status").json()
        assert st["cells"] > 0 and st["claude_enabled"] is False
        assert len(c.get("/api/regions").json()) == 5
        anoms = c.get("/api/anomalies").json()
        assert anoms
        assert c.get(f"/api/cells/{anoms[0]['cell_id']}/kpis?hours=24").json()
        # Claude endpoints say clearly that the key is missing
        assert c.post("/api/ask", json={"question": "hello there"}).status_code == 503

        # Review flow
        with session.begin():
            session.add(Recommendation(cell_id=anoms[0]["cell_id"], action="monitor_only", rationale="x", risk="low"))
        rid = c.get("/api/recommendations").json()[0]["id"]
        r = c.post(f"/api/recommendations/{rid}/review", json={"decision": "approved", "reviewer": "Eng. Sara"})
        assert r.json()["status"] == "approved"
        assert c.post(f"/api/recommendations/{rid}/review", json={"decision": "rejected", "reviewer": "x y"}).status_code == 409
        csv = c.get("/api/recommendations/export.csv").text
        assert "monitor_only" in csv

        # Upload: clean file accepted, PII file rejected
        clean = "Cell Name,Start Time,DL_PRB_Utilization\nMKK-001-1,2026-11-18T10:00:00Z,50\n"
        assert c.post("/api/ingest", files={"file": ("a.csv", clean)}, data={"vendor": "ericsson"}).status_code == 200
        bad = "Cell Name,Start Time,MSISDN\nMKK-001-1,2026-11-18T10:00:00Z,966512345678\n"
        assert c.post("/api/ingest", files={"file": ("b.csv", bad)}, data={"vendor": "ericsson"}).status_code == 422


def test_api_token_required_when_set(monkeypatch):
    import doai.api as api_mod

    object.__setattr__(api_mod.settings, "api_token", "s3cret")
    try:
        with TestClient(api_mod.app) as c:
            assert c.get("/api/status").status_code == 401
            assert c.get("/api/status", headers={"X-DOAI-Token": "s3cret"}).status_code == 200
            assert c.get("/health").status_code == 200
    finally:
        object.__setattr__(api_mod.settings, "api_token", None)
