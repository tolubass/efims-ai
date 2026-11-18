"""Shared pytest fixtures for the EFIMS AI test suite.

EFIMS_API_KEY is set here to a made-up test value before the app is ever
imported — never a real key, and never read from the real environment.
Audit logging is redirected to a temporary directory for every test (see
``_isolate_audit_log`` below), so running the suite never writes to the
real ``logs/``.
"""

from __future__ import annotations

import os

# Set before importing src.api.main: src.api.security.verify_api_key reads
# this from the environment on every request, and src.api.lifespan's
# warn_if_api_key_missing reads it once at startup.
os.environ.setdefault("EFIMS_API_KEY", "pytest-fixture-key-not-real")

import pytest
from fastapi.testclient import TestClient

import src.governance.prediction_audit_logger as prediction_audit_logger
from src.api.main import app

TEST_API_KEY: str = os.environ["EFIMS_API_KEY"]
AUTH_HEADERS: dict[str, str] = {"X-API-Key": TEST_API_KEY}
WRONG_AUTH_HEADERS: dict[str, str] = {"X-API-Key": "wrong-key-for-tests"}


@pytest.fixture(autouse=True)
def _isolate_audit_log(tmp_path, monkeypatch):
    """Redirect prediction_audit_logger.LOG_DIR to a per-test temp directory.

    log_prediction() resolves LOG_DIR from the module namespace at call
    time, so patching the module attribute here is enough to redirect every
    audit write during this test — the real logs/ directory is never
    touched by the test suite.
    """
    monkeypatch.setattr(prediction_audit_logger, "LOG_DIR", tmp_path / "logs")


@pytest.fixture(scope="session")
def client():
    """A TestClient with the app's lifespan (model loading) run once per session.

    Runs real startup (model_registry/*/v1.0/model.pkl + category_mappings.json
    load) exactly once, since none of this is cheap to repeat per test.
    """
    with TestClient(app) as test_client:
        yield test_client
