"""Database models. PostgreSQL on Railway, SQLite locally and in tests."""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import pg_driver, settings


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Cell(Base):
    """Radio cell inventory (no subscriber data)."""

    __tablename__ = "cells"
    cell_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    site_id: Mapped[str] = mapped_column(String(64))
    region: Mapped[str] = mapped_column(String(64))
    vendor: Mapped[str] = mapped_column(String(32))
    technology: Mapped[str] = mapped_column(String(8), default="5G")
    neighbors: Mapped[list] = mapped_column(JSON, default=list)


class KpiHourly(Base):
    """Unified, vendor-neutral hourly KPIs per cell."""

    __tablename__ = "kpi_hourly"
    __table_args__ = (UniqueConstraint("cell_id", "ts", name="uq_kpi_cell_ts"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    prb_util_pct: Mapped[float | None] = mapped_column(Float)
    dl_throughput_mbps: Mapped[float | None] = mapped_column(Float)
    ul_throughput_mbps: Mapped[float | None] = mapped_column(Float)
    active_users: Mapped[float | None] = mapped_column(Float)
    drop_rate_pct: Mapped[float | None] = mapped_column(Float)
    availability_pct: Mapped[float | None] = mapped_column(Float)


class Anomaly(Base):
    __tablename__ = "anomalies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    kpi: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(16), default="fault")  # fault|congestion
    value: Mapped[float] = mapped_column(Float)
    baseline: Mapped[float] = mapped_column(Float)
    zscore: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|investigated
    analysis: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Recommendation(Base):
    """A proposed network change. Never applied automatically."""

    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    anomaly_id: Mapped[int | None] = mapped_column(ForeignKey("anomalies.id"))
    cell_id: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    rationale: Mapped[str] = mapped_column(Text)
    expected_impact: Mapped[str | None] = mapped_column(Text)
    risk: Mapped[str] = mapped_column(String(16), default="medium")
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|approved|rejected
    reviewer: Mapped[str | None] = mapped_column(String(128))
    review_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CounterMapping(Base):
    """Learned vendor counter -> unified KPI mappings (reused, so Claude is called once per new format)."""

    __tablename__ = "counter_mappings"
    __table_args__ = (UniqueConstraint("vendor", "source_name", name="uq_mapping"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vendor: Mapped[str] = mapped_column(String(32))
    source_name: Mapped[str] = mapped_column(String(128))
    target_kpi: Mapped[str | None] = mapped_column(String(64))
    scale: Mapped[float] = mapped_column(Float, default=1.0)
    origin: Mapped[str] = mapped_column(String(16), default="builtin")  # builtin|claude|human


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actor: Mapped[str] = mapped_column(String(128))
    event: Mapped[str] = mapped_column(String(64))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


# Tables the Claude agent is allowed to read with SQL.
READABLE_TABLES = {"cells", "kpi_hourly", "anomalies", "recommendations"}


def make_engine(url: str | None = None):
    url = pg_driver(url or settings.database_url)
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db(bind=None) -> None:
    Base.metadata.create_all(bind or engine)


def audit(session: Session, actor: str, event: str, **detail) -> None:
    session.add(AuditLog(actor=actor, event=event, detail=detail))
