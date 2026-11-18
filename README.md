# DO.AI — Claude-powered network data optimization (proof of concept)

DO.AI turns cell-level 5G network KPIs into explained, approvable optimization decisions:

- **Normalizes** vendor KPI exports (Ericsson, Nokia, Huawei) into one schema. Unknown counters are mapped by Claude once and remembered.
- **Detects** anomalies by comparing each cell with its regional peers, and labels them **fault** (something broke) or **congestion** (out of capacity, e.g. a Hajj surge hot spot).
- **Investigates** with a Claude agent that reads data through guarded, read-only tools and writes root-cause analyses.
- **Recommends** changes into an approval queue. Nothing is ever applied to the network; approved items export as change requests.
- **Answers questions** in Arabic or English over the KPI data.
- **Keeps personal data out:** a privacy gateway rejects any file with subscriber identifiers (MSISDN, IMSI, IMEI, national ID) before storage.

> **Data residency.** This PoC is meant to run on Railway with **synthetic or anonymized, aggregated network data only**. Railway has no Saudi Arabia region, and the standard Claude API processes data outside the Kingdom. Production on real stc data must run in-Kingdom (see "Path to production").

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # add ANTHROPIC_API_KEY to enable Claude
export $(grep -v '^#' .env | xargs)
uvicorn doai.api:app --reload   # http://localhost:8000
```

With `DOAI_SEED_ON_START=true` the app loads 14 days of synthetic data for 102 cells (Makkah, Mina, Arafat, Riyadh, Jeddah) with a Hajj-style surge and three injected faults.

Tests: `pytest -q` · Lint: `ruff check .` · PII check: `python scripts/pii_scan.py data tests`

## Deploy on Railway

1. **Database:** in your Railway project, **Create → Database → PostgreSQL**.
2. **Web service:** **Create → GitHub Repo →** select this repo. Railway reads `railway.json` and builds the `Dockerfile`.
3. **Variables** (web service → Variables):
   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` |
   | `ANTHROPIC_API_KEY` | your key from console.anthropic.com |
   | `DOAI_API_TOKEN` | a long random string (the dashboard asks for it) |
   | `DOAI_SEED_ON_START` | `true` |
4. **Domain:** web service → Settings → Networking → **Generate Domain**.
5. **Region:** web service → Settings → Region. Pick the one closest to Saudi Arabia (EU West, Amsterdam).
6. **Hourly pipeline (optional):** **Create → GitHub Repo →** the same repo again, rename it `pipeline`, then Settings → **Config-as-code path** = `/railway.worker.json`. Give it the same variables. It runs detection and Claude investigations at minute 5 of every hour.

Every push to `main` redeploys. Enable **PR environments** in project settings to get a preview per pull request.

## Project layout

```
doai/
  api.py         FastAPI app + dashboard (static/index.html)
  pipeline.py    seed, ingest, detect, investigate
  privacy.py     privacy gateway (rejects personal data)
  normalize.py   vendor counters -> unified KPIs (+ Claude mapping)
  detect.py      peer-relative anomaly detection, fault vs congestion, forecast
  agent.py       Claude agent: tools, guardrails, audit
  sqlguard.py    read-only SQL guard for agent queries
  synthetic.py   synthetic network data generator
  jobs.py        CLI / cron entrypoints
tests/           25 tests, no network calls (Claude is scripted)
scripts/pii_scan.py   CI check that no committed data contains personal data
railway.json / railway.worker.json   Railway config-as-code
```

## Claude agent guardrails

- Tools: `query_kpis` (single SELECT, allow-listed tables, row cap, read-only transaction on Postgres), `get_cell_context`, `forecast_load`, `propose_change` (pending queue only).
- Models: `claude-sonnet-5-5` for investigations, `claude-haiku-5-5` for counter mapping. Set via `DOAI_REASONING_MODEL` / `DOAI_FAST_MODEL`.
- Step limit per run (`DOAI_MAX_AGENT_STEPS`), prompt caching on the system prompt, and every prompt, tool call, answer and token count written to `audit_log`.
- For production, also set `DATABASE_URL_READONLY` to a Postgres role with SELECT-only rights.

## Path to production

1. Replace the illustrative counter names in `normalize.py` with stc's actual PM export names.
2. Agree data access and classification with stc; PoC data must contain no personal data.
3. Move the same containers to stc's in-Kingdom cloud (or a Saudi cloud region) and point the Claude client at an in-Kingdom endpoint once one is confirmed.
4. Replace the shared API token with stc SSO and role-based access.
5. Connect approved change requests to stc's SON/OSS change process.
