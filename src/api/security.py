"""API-key authentication for the EFIMS AI Service.

Reads the expected key from the ``EFIMS_API_KEY`` environment variable and
compares it, constant-time, against the ``X-API-Key`` request header. If no
key is configured at all, protected routes fail closed with 503 by default
— set ``EFIMS_ALLOW_NO_KEY=true`` only for local development, never in a
shared environment. The key's value is never logged or returned, correct or
not.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from src.utils.logger import get_logger

logger = get_logger(__name__)

API_KEY_HEADER_NAME: str = "X-API-Key"
ENV_API_KEY: str = "EFIMS_API_KEY"
ENV_ALLOW_NO_KEY: str = "EFIMS_ALLOW_NO_KEY"

_WWW_AUTHENTICATE_HEADER: dict = {"WWW-Authenticate": "ApiKey"}

# auto_error=False so a missing header doesn't short-circuit with FastAPI's
# own generic 403 before verify_api_key can run — and so Swagger renders an
# "Authorize" button for this scheme on /docs.
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def _allow_no_key_configured() -> bool:
    """Return True if EFIMS_ALLOW_NO_KEY is explicitly set to "true"."""
    return os.environ.get(ENV_ALLOW_NO_KEY, "false").strip().lower() == "true"


def warn_if_api_key_missing() -> None:
    """Log a startup warning if EFIMS_API_KEY isn't set. Call once at startup."""
    if os.environ.get(ENV_API_KEY):
        return

    if _allow_no_key_configured():
        logger.warning(
            "%s is not set and %s=true — severity/anomaly/recovery endpoints are "
            "UNAUTHENTICATED. This must never be used outside local development.",
            ENV_API_KEY,
            ENV_ALLOW_NO_KEY,
        )
    else:
        logger.warning(
            "%s is not set — severity/anomaly/recovery endpoints will return 503 "
            "until it is configured (or %s=true for local development only).",
            ENV_API_KEY,
            ENV_ALLOW_NO_KEY,
        )


async def verify_api_key(provided_key: str = Security(_api_key_header)) -> None:
    """FastAPI dependency: require a valid X-API-Key header.

    Raises:
        HTTPException(503): EFIMS_API_KEY isn't configured on the server,
            and EFIMS_ALLOW_NO_KEY isn't set to "true".
        HTTPException(401): the header is missing or doesn't match, with a
            ``WWW-Authenticate: ApiKey`` header.
    """
    expected_key = os.environ.get(ENV_API_KEY)

    if not expected_key:
        if _allow_no_key_configured():
            return
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured on this server.",
        )

    if not provided_key or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers=_WWW_AUTHENTICATE_HEADER,
        )
