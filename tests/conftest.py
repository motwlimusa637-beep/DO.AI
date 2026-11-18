import os
import sys
from pathlib import Path

# Use a throwaway SQLite database and no Claude key for every test run.
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(__file__).parent / "_test.db")
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("DOAI_API_TOKEN", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from doai.db import Base, SessionLocal, engine  # noqa: E402


@pytest.fixture()
def session():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as s:
        yield s


def pytest_sessionfinish(session, exitstatus):
    engine.dispose()
    p = Path(__file__).parent / "_test.db"
    if p.exists():
        p.unlink()
