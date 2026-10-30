"""Date and time helpers for efims-ai, standardised on West Africa Time.

Every timestamp in EFIMS is expressed in West Africa Time (WAT, UTC+1).
Nigeria does not observe daylight saving time, so WAT is a fixed +01:00 offset
all year round.

Consistent WAT handling matters for fraud detection because:

* Transactions at unusual hours are more suspicious, so "business hours" must
  be judged in local time, not UTC or server time.
* Nigerian public holidays affect business-hour calculations.
* Regulatory deadlines and SLAs are calculated in WAT.

Naive datetimes (no ``tzinfo``) are always assumed to be UTC and converted to
WAT before use. Never compare naive and aware datetimes directly; pass them
through ``to_wat()`` first.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

WAT: timezone = timezone(timedelta(hours=1), name="WAT")

# Nigerian banking hours, in WAT: Monday-Friday, 08:00 (inclusive) to 17:00 (exclusive).
BUSINESS_START_HOUR: int = 8
BUSINESS_END_HOUR: int = 17

# Fixed-date Nigerian public holidays as (month, day). Eid and Easter are
# excluded because their dates vary each year.
PUBLIC_HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 1): "New Year's Day",
    (5, 1): "Workers' Day",
    (6, 12): "Democracy Day",
    (10, 1): "Independence Day",
    (12, 25): "Christmas Day",
    (12, 26): "Boxing Day",
}


def now_wat() -> datetime:
    """Return the current date and time in WAT (timezone-aware)."""
    return datetime.now(WAT)


def to_wat(dt: datetime) -> datetime:
    """Convert ``dt`` to WAT.

    A naive datetime (no ``tzinfo``) is assumed to be UTC.
    """
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(WAT)


def _resolve(dt: Optional[datetime]) -> datetime:
    """Return ``dt`` in WAT, or the current WAT time if ``dt`` is None."""
    return now_wat() if dt is None else to_wat(dt)


def is_business_hours(dt: Optional[datetime] = None) -> bool:
    """Return True if ``dt`` falls within Nigerian banking hours.

    Banking hours are Monday to Friday, 08:00 to 17:00 WAT (08:00:00 counts as
    inside, 17:00:00 as outside). Saturdays and Sundays always return False.
    Public holidays are not considered here; combine with ``is_public_holiday``
    if needed.

    Args:
        dt: Datetime to check. Naive values are treated as UTC. Defaults to now.
    """
    local = _resolve(dt)
    if local.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    return BUSINESS_START_HOUR <= local.hour < BUSINESS_END_HOUR


def is_public_holiday(dt: Optional[datetime] = None) -> bool:
    """Return True if ``dt`` (in WAT) is a fixed-date Nigerian public holiday.

    Args:
        dt: Datetime to check. Naive values are treated as UTC. Defaults to now.
    """
    local = _resolve(dt)
    return (local.month, local.day) in PUBLIC_HOLIDAYS


def format_wat_timestamp(dt: Optional[datetime] = None) -> str:
    """Format ``dt`` in WAT as ``"YYYY-MM-DD HH:MM:SS WAT"``.

    Example: ``"2026-09-23 14:52:15 WAT"``

    Args:
        dt: Datetime to format. Naive values are treated as UTC. Defaults to now.
    """
    return f"{_resolve(dt):%Y-%m-%d %H:%M:%S} WAT"


def time_since_seconds(past_dt: datetime) -> float:
    """Return the number of seconds elapsed since ``past_dt``.

    Used for event-to-detection time in the recovery propensity model.
    Returns a negative value (and logs a warning) if ``past_dt`` is in the future.
    """
    seconds = (now_wat() - to_wat(past_dt)).total_seconds()
    if seconds < 0:
        logger.warning("time_since_seconds received a future datetime: %s", past_dt)
    return seconds


def days_between(start_dt: datetime, end_dt: datetime) -> float:
    """Return the number of days (fractional) from ``start_dt`` to ``end_dt``.

    Used for case age and SLA tracking. The result is negative if ``end_dt``
    is before ``start_dt``.
    """
    return (to_wat(end_dt) - to_wat(start_dt)).total_seconds() / 86_400
