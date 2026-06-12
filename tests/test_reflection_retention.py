"""Age-based retention for reflection_sessions + reflection_insights (#1674 P4).

Option A — feature-only, no core schema change: telemetry sessions and
non-actionable insights age out at a normal window; actionable insights (pending
self-improvement TODOs) are preserved far longer but still bounded.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_feature_reflection.db_helpers import ReflectionDatabaseHelper
from kestrel_feature_reflection import retention as R


pytestmark = pytest.mark.asyncio
AGENT = "did:test:reflection-retention"


@pytest.fixture
async def db(tmp_path):
    d = await AsyncDatabase.sqlite(str(tmp_path / "reflection.db"))
    yield d
    await d.close()


def _ago(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


async def _add_session(db, *, days_old: int, created_fmt: str = "iso", agent_id=AGENT):
    ts = _ago(days_old)
    created = ts.isoformat() if created_fmt == "iso" else ts.strftime("%Y-%m-%d %H:%M:%S")
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    await db.execute(
        """INSERT INTO reflection_sessions (id, agent_id, trigger, created_at)
           VALUES (?, ?, ?, ?)""",
        (sid, agent_id, "sleep", created),
    )
    return sid


async def _add_insight(db, *, days_old: int, actionable: bool, agent_id=AGENT):
    iid = f"ins-{uuid.uuid4().hex[:8]}"
    await db.execute(
        """INSERT INTO reflection_insights
           (id, agent_id, type, title, actionable, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (iid, agent_id, "pattern", "t", 1 if actionable else 0, _ago(days_old).isoformat()),
    )
    return iid


async def _ids(db, table, agent_id=AGENT):
    rows = await db.fetchall(f"SELECT id FROM {table} WHERE agent_id = ?", (agent_id,))
    return {r[0] for r in rows or []}


# --------------------------------------------------------------------------


async def test_prune_sessions_age_window(db):
    h = ReflectionDatabaseHelper(db, AGENT)
    old = await _add_session(db, days_old=100)
    recent = await _add_session(db, days_old=10)

    out = await h.prune_old_records(
        sessions_days=90, insights_days=90, actionable_days=365)

    assert out["sessions_deleted"] == 1
    assert await _ids(db, "reflection_sessions") == {recent}


async def test_prune_insights_actionable_preserved_longer(db):
    h = ReflectionDatabaseHelper(db, AGENT)
    stale_obs = await _add_insight(db, days_old=100, actionable=False)   # > 90 → gone
    fresh_obs = await _add_insight(db, days_old=10, actionable=False)    # kept
    pending = await _add_insight(db, days_old=100, actionable=True)      # < 365 → kept
    ancient_todo = await _add_insight(db, days_old=400, actionable=True)  # > 365 → gone

    out = await h.prune_old_records(
        sessions_days=90, insights_days=90, actionable_days=365)

    assert out["insights_deleted"] == 2
    assert await _ids(db, "reflection_insights") == {fresh_obs, pending}


async def test_prune_handles_space_separated_timestamps(db):
    """reflection_sessions written with SQLite CURRENT_TIMESTAMP (space format)
    must compare correctly against the cutoff (Python-side parsing)."""
    h = ReflectionDatabaseHelper(db, AGENT)
    old_space = await _add_session(db, days_old=200, created_fmt="space")

    out = await h.prune_old_records(
        sessions_days=90, insights_days=90, actionable_days=365)

    assert out["sessions_deleted"] == 1
    assert await _ids(db, "reflection_sessions") == set()


async def test_prune_scoped_to_agent(db):
    h = ReflectionDatabaseHelper(db, AGENT)
    mine = await _add_session(db, days_old=200)
    await _add_session(db, days_old=200, agent_id="someone-else")

    out = await h.prune_old_records(
        sessions_days=90, insights_days=90, actionable_days=365)

    assert out["sessions_deleted"] == 1
    assert await _ids(db, "reflection_sessions", "someone-else")  # other agent untouched


async def test_prune_max_rows_caps_and_drains(db):
    h = ReflectionDatabaseHelper(db, AGENT)
    for _ in range(5):
        await _add_session(db, days_old=200)

    first = await h.prune_old_records(
        sessions_days=90, insights_days=90, actionable_days=365, max_rows=2)
    assert first["sessions_deleted"] == 2
    assert len(await _ids(db, "reflection_sessions")) == 3

    second = await h.prune_old_records(
        sessions_days=90, insights_days=90, actionable_days=365, max_rows=100)
    assert second["sessions_deleted"] == 3
    assert await _ids(db, "reflection_sessions") == set()


# --------------------------------------------------------------------------
# config


async def test_config_default_is_opt_out_with_defaults(monkeypatch):
    import kestrel_sovereign.config as cfg
    monkeypatch.setattr(cfg, "load_section", lambda name: {})
    c = R.load_reflection_retention_config()
    assert c["enabled"] is False
    assert c["sessions_days"] == R.DEFAULT_SESSIONS_DAYS
    assert c["actionable_days"] == R.DEFAULT_ACTIONABLE_DAYS


async def test_config_reads_values(monkeypatch):
    import kestrel_sovereign.config as cfg
    monkeypatch.setattr(
        cfg, "load_section",
        lambda name: {"retention": {"enabled": True, "sessions_days": 30,
                                    "insights_days": 45, "actionable_days": 200}}
        if name == "reflection" else {},
    )
    c = R.load_reflection_retention_config()
    assert c == {"enabled": True, "sessions_days": 30, "insights_days": 45,
                 "actionable_days": 200, "max_rows": R.DEFAULT_MAX_ROWS}


async def test_config_non_bool_enabled_fails_safe(monkeypatch):
    import kestrel_sovereign.config as cfg
    monkeypatch.setattr(
        cfg, "load_section",
        lambda name: {"retention": {"enabled": "true"}} if name == "reflection" else {},
    )
    assert R.load_reflection_retention_config()["enabled"] is False


async def test_config_non_positive_window_falls_back(monkeypatch):
    import kestrel_sovereign.config as cfg
    monkeypatch.setattr(
        cfg, "load_section",
        lambda name: {"retention": {"sessions_days": 0, "insights_days": "soon"}}
        if name == "reflection" else {},
    )
    c = R.load_reflection_retention_config()
    assert c["sessions_days"] == R.DEFAULT_SESSIONS_DAYS
    assert c["insights_days"] == R.DEFAULT_INSIGHTS_DAYS
