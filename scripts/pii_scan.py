"""CI guard: fail if any committed data file contains personal data patterns.

Usage: python scripts/pii_scan.py [paths...]   (default: data/ and tests/)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doai.privacy import scan_dataframe  # noqa: E402


def main(paths):
    files = []
    for p in map(Path, paths or ["data", "tests"]):
        files += [f for f in (p.rglob("*") if p.is_dir() else [p]) if f.suffix in (".csv", ".parquet")]
    failed = False
    for f in files:
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, dtype=str)
        report = scan_dataframe(df)
        if not report.ok:
            failed = True
            print(f"PII suspected in {f}: {'; '.join(report.findings)}")
    print(f"scanned {len(files)} data file(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
