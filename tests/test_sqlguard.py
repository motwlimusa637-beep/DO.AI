import pytest

from doai.sqlguard import UnsafeQuery, check


def test_select_is_wrapped_with_limit():
    q = check("SELECT cell_id, avg(prb_util_pct) FROM kpi_hourly GROUP BY cell_id", max_rows=50)
    assert q.endswith("LIMIT 50") and "kpi_hourly" in q


def test_cte_allowed():
    check("WITH x AS (SELECT * FROM kpi_hourly) SELECT * FROM x JOIN cells c ON c.cell_id = x.cell_id")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM kpi_hourly",
        "SELECT 1; DROP TABLE cells",
        "SELECT * FROM audit_log",
        "SELECT * FROM counter_mappings",
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT * FROM cells -- sneaky",
        "UPDATE recommendations SET status='approved'",
        "",
    ],
)
def test_unsafe_rejected(sql):
    with pytest.raises(UnsafeQuery):
        check(sql)


def test_keywords_inside_strings_are_fine():
    check("SELECT * FROM recommendations WHERE rationale LIKE '%drop table%'")
