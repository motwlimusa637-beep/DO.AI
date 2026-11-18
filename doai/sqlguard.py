"""Guard for SQL written by the Claude agent: one read-only SELECT on allowed tables.

This is defence in depth. In production also give the agent a database role that
only has SELECT on these tables (set DATABASE_URL_READONLY).
"""

import re

from .db import READABLE_TABLES

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|merge|call|"
    r"execute|attach|detach|pragma|vacuum|set|reset|lock|listen|notify|load|"
    r"pg_\w+|information_schema|sqlite_master)\b",
    re.IGNORECASE,
)
_TABLE_REF = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", re.IGNORECASE)
_CTE_NAME = re.compile(r"(?:\bwith\b|,)\s*([a-zA-Z_]\w*)\s+as\s*\(", re.IGNORECASE)


class UnsafeQuery(ValueError):
    pass


def _strip_strings(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def check(sql: str, max_rows: int = 200) -> str:
    """Return a safe, row-limited query or raise UnsafeQuery."""
    q = sql.strip().rstrip(";").strip()
    if not q:
        raise UnsafeQuery("empty query")
    if "--" in q or "/*" in q:
        raise UnsafeQuery("comments are not allowed")
    bare = _strip_strings(q)
    if ";" in bare:
        raise UnsafeQuery("only one statement is allowed")
    if not re.match(r"^\s*(select|with)\b", bare, re.IGNORECASE):
        raise UnsafeQuery("only SELECT queries are allowed")
    if _FORBIDDEN.search(bare):
        raise UnsafeQuery("query uses a forbidden keyword or system table")
    ctes = {m.lower() for m in _CTE_NAME.findall(bare)}
    for table in _TABLE_REF.findall(bare):
        name = table.lower().split(".")[-1]
        if name not in READABLE_TABLES and name not in ctes:
            raise UnsafeQuery(f"table '{table}' is not readable")
    return f"SELECT * FROM ({q}) AS agent_q LIMIT {int(max_rows)}"
