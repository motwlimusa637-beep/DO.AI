"""Runtime settings, all from environment variables (Railway Variables in deployment)."""

import os
from dataclasses import dataclass


def _db_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./doai.db")
    return pg_driver(url)


def pg_driver(url: str) -> str:
    """Railway gives postgres:// or postgresql:// URLs; pin the psycopg (v3) driver."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


@dataclass(frozen=True)
class Settings:
    database_url: str = _db_url()
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    # Model used for investigations, recommendations and Q&A
    reasoning_model: str = os.getenv("DOAI_REASONING_MODEL", "claude-sonnet-5-5")
    # Cheaper model for high-volume work such as counter mapping
    fast_model: str = os.getenv("DOAI_FAST_MODEL", "claude-haiku-5-5")
    max_agent_steps: int = int(os.getenv("DOAI_MAX_AGENT_STEPS", "8"))
    max_query_rows: int = int(os.getenv("DOAI_MAX_QUERY_ROWS", "200"))
    # Simple shared token protecting the API in the PoC; replace with SSO for stc.
    api_token: str | None = os.getenv("DOAI_API_TOKEN")
    anomaly_z: float = float(os.getenv("DOAI_ANOMALY_Z", "3.0"))


settings = Settings()


def claude_enabled() -> bool:
    return bool(settings.anthropic_api_key)
