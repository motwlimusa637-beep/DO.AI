"""Privacy gateway: keep subscriber-level personal data out of the platform.

Runs before anything is stored. A file is rejected if it contains a column whose
name suggests a personal identifier, or values that look like one. Rejecting is
the safe default for the PoC: the platform only accepts cell-level aggregates.
"""

import re
from dataclasses import dataclass, field

import pandas as pd

# Column names that indicate subscriber-level data.
_PII_COLUMN_HINTS = re.compile(
    r"(msisdn|imsi|imei|iccid|phone|mobile_?no|subscriber|customer|national_?id|"
    r"iqama|passport|email|first_?name|last_?name|full_?name|user_?ip|ue_?ip|"
    r"latitude|longitude|\blat\b|\blon\b|address)",
    re.IGNORECASE,
)

# Value patterns (checked on string forms of values).
_VALUE_PATTERNS = {
    # Saudi mobile numbers: 9665XXXXXXXX, +9665..., 05XXXXXXXX
    "msisdn": re.compile(r"^(\+?9665\d{8}|05\d{8})$"),
    # IMSI with Saudi MCC 420, 15 digits
    "imsi": re.compile(r"^420\d{12}$"),
    # Generic 15-digit identifiers (IMEI etc.)
    "imei": re.compile(r"^\d{15}$"),
    # Saudi national ID / Iqama: 10 digits starting with 1 or 2
    "national_id": re.compile(r"^[12]\d{9}$"),
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE),
}


@dataclass
class PrivacyReport:
    ok: bool
    findings: list[str] = field(default_factory=list)


def scan_dataframe(df: pd.DataFrame, sample_rows: int = 500) -> PrivacyReport:
    findings: list[str] = []
    for col in df.columns:
        if _PII_COLUMN_HINTS.search(str(col)):
            findings.append(f"column '{col}' looks like personal data")
    sample = df.head(sample_rows)
    for col in sample.columns:
        series = sample[col].dropna()
        if series.empty:
            continue
        # Only check text and integer-like columns; floats are KPI measurements.
        if pd.api.types.is_float_dtype(series):
            continue
        values = series.astype(str).str.strip()
        for kind, pattern in _VALUE_PATTERNS.items():
            hits = values.str.match(pattern).sum()
            if hits:
                findings.append(f"column '{col}': {hits} value(s) look like {kind}")
                break
    return PrivacyReport(ok=not findings, findings=findings)


def enforce(df: pd.DataFrame) -> None:
    report = scan_dataframe(df)
    if not report.ok:
        raise PermissionError(
            "Rejected by privacy gateway (cell-level aggregates only): "
            + "; ".join(report.findings)
        )
