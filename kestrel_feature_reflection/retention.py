"""Age-based retention for the reflection feature's cognition tables (#1674 P4).

The unbounded-cognition-tables concern (#1674) covers ``reflection_sessions``
and ``reflection_insights`` (defined in core, written by this feature). Core
must not import features, so the feature prunes its own tables — and it does so
inside the nightly ``sleep`` cycle via the reflection hook (see ``hooks.py`` /
``feature.on_pre_sleep``), not on a separate cron. This keeps memory
maintenance unified under sleep (the #1674 P3 doctrine).

Policy (Option A — feature-only, no core schema change):

- ``reflection_sessions`` — pure telemetry; age-prune past ``sessions_days``.
- ``reflection_insights`` — split by ``actionable``:
    * non-actionable: age-prune past ``insights_days`` (observations that
      didn't propose an action are the cheapest to forget);
    * actionable: the agent's pending self-improvement TODOs — preserved far
      longer (``actionable_days``) but still bounded so the table can't grow
      without limit.

Opt-in/off by default (``enabled=false``), matching core ``[forgetting]`` —
the Sovereign turns retention on deliberately.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_ENABLED = False
DEFAULT_SESSIONS_DAYS = 90
DEFAULT_INSIGHTS_DAYS = 90
DEFAULT_ACTIONABLE_DAYS = 365
DEFAULT_MAX_ROWS = 5000  # per-table per-run cap; backlog drains over nights


def load_reflection_retention_config() -> Dict[str, Any]:
    """Resolve ``[reflection.retention]`` from kestrel.toml, fully defaulted.

    Always returns ``{enabled, sessions_days, insights_days, actionable_days,
    max_rows}``; malformed/non-positive values fall back to the compiled-in
    defaults with a warning rather than silently disabling the rail.
    """
    section: Dict[str, Any] = {}
    try:
        from kestrel_sovereign.config import load_section
        retention = load_section("reflection") or {}
        sub = retention.get("retention")
        section = sub if isinstance(sub, dict) else {}
    except Exception as e:  # noqa: BLE001
        logger.debug("[reflection.retention] config load failed (defaults): %s", e)

    enabled = section.get("enabled", DEFAULT_ENABLED)
    if not isinstance(enabled, bool):
        logger.warning("[reflection.retention] enabled is not a bool: %r", enabled)
        enabled = DEFAULT_ENABLED

    return {
        "enabled": enabled,
        "sessions_days": _coerce_days(
            section.get("sessions_days"), DEFAULT_SESSIONS_DAYS, "sessions_days"),
        "insights_days": _coerce_days(
            section.get("insights_days"), DEFAULT_INSIGHTS_DAYS, "insights_days"),
        "actionable_days": _coerce_days(
            section.get("actionable_days"), DEFAULT_ACTIONABLE_DAYS, "actionable_days"),
        "max_rows": _coerce_days(
            section.get("max_rows"), DEFAULT_MAX_ROWS, "max_rows"),
    }


def _coerce_days(value: Any, fallback: int, key: str) -> int:
    if value is None:
        return fallback
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        logger.warning("[reflection.retention] %s is not an int: %r", key, value)
        return fallback
    if coerced <= 0:
        logger.warning("[reflection.retention] %s must be > 0: %r", key, value)
        return fallback
    return coerced


def parse_ts_utc(value: Any) -> Optional[datetime]:
    """Parse a stored timestamp to naive-UTC for safe cross-format comparison.

    reflection_insights.created_at is written via ``datetime.isoformat()``
    (ISO/`T`, possibly tz-aware); reflection_sessions rows may use SQLite
    ``CURRENT_TIMESTAMP`` (space-separated). A raw SQL string ``<`` would
    mis-sort across those formats, so callers parse + compare in Python.
    Returns None when absent/unparseable.
    """
    if value is None:
        return None
    dt = value if isinstance(value, datetime) else None
    if dt is None:
        try:
            dt = datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def utc_cutoff(days: int) -> datetime:
    """Naive-UTC timestamp ``days`` in the past."""
    from datetime import timedelta
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
