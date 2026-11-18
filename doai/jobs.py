"""Command-line jobs. Used by the Railway cron service and for local work.

  python -m doai.jobs init                 create tables
  python -m doai.jobs seed [--days 14]     load synthetic demo data
  python -m doai.jobs ingest FILE --vendor ericsson
  python -m doai.jobs detect               find anomalies in the last 24h
  python -m doai.jobs investigate [--limit 5]
  python -m doai.jobs pipeline             detect + investigate (cron entrypoint)
  python -m doai.jobs make-samples         write vendor-format sample files to data/samples
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from . import pipeline
from .db import SessionLocal, init_db


def main(argv=None):
    p = argparse.ArgumentParser(prog="doai.jobs")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    s = sub.add_parser("seed")
    s.add_argument("--days", type=int, default=14)
    i = sub.add_parser("ingest")
    i.add_argument("file")
    i.add_argument("--vendor", required=True, choices=["ericsson", "nokia", "huawei", "other"])
    sub.add_parser("detect")
    inv = sub.add_parser("investigate")
    inv.add_argument("--limit", type=int, default=5)
    pl = sub.add_parser("pipeline")
    pl.add_argument("--limit", type=int, default=5)
    sub.add_parser("make-samples")
    args = p.parse_args(argv)

    init_db()
    if args.cmd == "make-samples":
        from .synthetic import make_cells, make_kpis, to_vendor_format

        cells = make_cells()
        kpis, _ = make_kpis(cells, days=2, seed=11)
        out = Path("data/samples")
        out.mkdir(parents=True, exist_ok=True)
        for vendor in ("ericsson", "nokia", "huawei"):
            vcells = cells[cells.vendor == vendor].cell_id.head(3)
            sub_k = kpis[kpis.cell_id.isin(vcells)]
            to_vendor_format(sub_k, vendor).to_csv(out / f"sample_{vendor}.csv", index=False)
        print(f"wrote samples to {out}")
        return

    with SessionLocal() as session:
        if args.cmd == "init":
            result = "tables ready"
        elif args.cmd == "seed":
            result = pipeline.seed_demo(session, days=args.days)
        elif args.cmd == "ingest":
            df = pd.read_parquet(args.file) if args.file.endswith(".parquet") else pd.read_csv(args.file)
            result = pipeline.ingest_file(session, df, args.vendor, actor="cli", filename=args.file)
        elif args.cmd == "detect":
            result = {"new_anomalies": pipeline.run_detection(session)}
        elif args.cmd == "investigate":
            result = pipeline.investigate_open(session, limit=args.limit)
        elif args.cmd == "pipeline":
            new = pipeline.run_detection(session)
            session.commit()
            result = {"new_anomalies": new, "investigated": pipeline.investigate_open(session, limit=args.limit)}
        session.commit()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
