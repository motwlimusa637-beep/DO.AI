import pandas as pd
import pytest

from doai import privacy


def test_clean_kpi_file_passes():
    df = pd.DataFrame({"Cell Name": ["MKK-001-1"], "Start Time": ["2026-11-18T10:00:00Z"], "DL_PRB_Utilization": [55.2]})
    assert privacy.scan_dataframe(df).ok


@pytest.mark.parametrize(
    "column,value",
    [
        ("MSISDN", "966512345678"),
        ("subscriber", "x"),
        ("IMSI", "420011234567890"),
        ("national_id", "1012345678"),
    ],
)
def test_pii_columns_rejected(column, value):
    df = pd.DataFrame({"cell_id": ["MKK-001-1"], column: [value]})
    with pytest.raises(PermissionError):
        privacy.enforce(df)


def test_pii_values_in_innocent_column_rejected():
    df = pd.DataFrame({"cell_id": ["A-1"], "note": ["0551234567"]})
    report = privacy.scan_dataframe(df)
    assert not report.ok and "msisdn" in report.findings[0]
