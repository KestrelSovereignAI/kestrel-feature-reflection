"""Validation tests for reflect/get_insights/training_cycle params (#1946).

The kestrel-sovereign #1925 dogfooding sweep surfaced agent-facing tool bugs
where enum-like parameters silently fell through instead of failing loudly:

  - ``reflect(scope=...)`` was documented but IGNORED — the Mind layer always
    analyzed ``scope="today"``. Now scope is threaded to the analyzer AND an
    unknown scope returns a clear failure.
  - ``reflect(depth=...)`` / ``training_cycle(depth=...)`` silently mapped an
    unknown value to ``normal``. Now validated against shallow/normal/deep.
  - ``get_insights(type_filter=...)`` put an unvalidated string straight into a
    SQL WHERE — a typo returned 0 rows indistinguishable from "no insights".
    Now validated against the InsightType enum.

These mirror the existing ``propose_improvement.change_type`` pattern.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_feature_reflection.feature import ReflectionFeature


def _make_feature() -> ReflectionFeature:
    """A ReflectionFeature with the heavy init bypassed.

    We only exercise the @tool method bodies' validation + threading, so we
    attach just the collaborators those bodies touch and leave everything else
    at the validation guard's mercy.
    """
    agent = SimpleNamespace(did="did:test:validation")
    feat = ReflectionFeature(agent=agent)
    feat._db_helper = None
    feat._mind_checker = None
    feat._arms_checker = None
    feat._memory_checker = None
    feat._prioritizer = MagicMock()
    feat._prioritizer.prioritize.return_value = []
    feat._training_manager = MagicMock()
    feat._training_manager.run_training_cycle = AsyncMock(return_value={})
    return feat


# ---------------------------------------------------------------------------
# reflect(scope=...) / reflect(depth=...)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reflect_invalid_scope_fails_clearly():
    feat = _make_feature()
    res = await feat.reflect(scope="yesterday")
    assert res.status is ToolResultStatus.ERROR
    assert "Invalid scope" in (res.error or "")
    # lists the valid options so the agent can self-correct
    assert "today" in (res.error or "")


@pytest.mark.asyncio
async def test_reflect_invalid_depth_fails_clearly():
    feat = _make_feature()
    res = await feat.reflect(depth="medium")
    assert res.status is ToolResultStatus.ERROR
    assert "Invalid depth" in (res.error or "")
    assert "shallow" in (res.error or "")


@pytest.mark.asyncio
async def test_reflect_valid_scope_is_threaded_to_mind_checker():
    """A non-default scope must actually reach the Mind checker, proving scope
    is no longer a dead parameter."""
    feat = _make_feature()
    mind = MagicMock()
    mind.run_all = AsyncMock(return_value=[])
    feat._mind_checker = mind

    res = await feat.reflect(scope="week", depth="deep")
    assert res.status in (ToolResultStatus.OK, ToolResultStatus.PARTIAL)
    mind.run_all.assert_awaited_once()
    kwargs = mind.run_all.await_args.kwargs
    assert kwargs["scope"] == "week"
    assert kwargs["depth"] == "deep"


@pytest.mark.asyncio
async def test_reflect_normalizes_case():
    feat = _make_feature()
    mind = MagicMock()
    mind.run_all = AsyncMock(return_value=[])
    feat._mind_checker = mind

    res = await feat.reflect(scope="WEEK", depth="Deep")
    assert res.status in (ToolResultStatus.OK, ToolResultStatus.PARTIAL)
    kwargs = mind.run_all.await_args.kwargs
    assert kwargs["scope"] == "week"
    assert kwargs["depth"] == "deep"


# ---------------------------------------------------------------------------
# Mind checker threads scope to the analyzer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mind_checker_threads_scope_to_analyzer():
    from kestrel_feature_reflection.checks.mind import MindChecker

    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(return_value=[])
    checker = MindChecker(agent=SimpleNamespace(storage=None), analyzer=analyzer)

    await checker.run_all(scope="month", depth="shallow")

    analyzer.analyze.assert_awaited_once()
    kwargs = analyzer.analyze.await_args.kwargs
    assert kwargs["scope"] == "month"
    assert kwargs["depth"] == "shallow"


# ---------------------------------------------------------------------------
# get_insights(type_filter=...)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_insights_invalid_type_filter_fails_clearly():
    feat = _make_feature()
    feat._db_helper = MagicMock()
    feat._db_helper.get_insights = AsyncMock(return_value=[])

    res = await feat.get_insights(type_filter="successes")  # typo: not an enum
    assert res.status is ToolResultStatus.ERROR
    assert "Invalid type_filter" in (res.error or "")
    assert "pattern" in (res.error or "")
    # must NOT have hit the DB with the bad value
    feat._db_helper.get_insights.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_insights_valid_type_filter_normalized_and_passed():
    feat = _make_feature()
    feat._db_helper = MagicMock()
    feat._db_helper.get_insights = AsyncMock(return_value=[])

    res = await feat.get_insights(type_filter="FAILURE")
    assert res.status is ToolResultStatus.OK
    kwargs = feat._db_helper.get_insights.await_args.kwargs
    assert kwargs["type_filter"] == "failure"


@pytest.mark.asyncio
async def test_get_insights_no_filter_still_works():
    feat = _make_feature()
    feat._db_helper = MagicMock()
    feat._db_helper.get_insights = AsyncMock(return_value=[])

    res = await feat.get_insights()
    assert res.status is ToolResultStatus.OK
    kwargs = feat._db_helper.get_insights.await_args.kwargs
    assert kwargs["type_filter"] is None


# ---------------------------------------------------------------------------
# training_cycle(depth=...)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_training_cycle_invalid_depth_fails_before_iterating():
    feat = _make_feature()
    res = await feat.training_cycle(depth="quick")  # documented-away alias
    assert res.status is ToolResultStatus.ERROR
    assert "Invalid depth" in (res.error or "")
    # must not have started any training iteration
    feat._training_manager.run_training_cycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_training_cycle_valid_depth_runs():
    feat = _make_feature()
    res = await feat.training_cycle(depth="Deep", iterations=1)
    assert res.status in (ToolResultStatus.OK, ToolResultStatus.PARTIAL)
    kwargs = feat._training_manager.run_training_cycle.await_args.kwargs
    assert kwargs["depth"] == "deep"
