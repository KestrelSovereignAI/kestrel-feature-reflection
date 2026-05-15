"""Tests for the per-turn fact-capture STOP hook (kestrel-sovereign #1238).

Acceptance criterion from the parent ticket:
> in a test conversation where the agent learns three structural facts
> (a renaming, a package location, a bug observation), all three are
> persisted without the user explicitly asking.

The hook reads the turn context off the enriched STOP HookInput
(kestrel-sovereign #1269) — these tests construct that HookInput
directly and drive ``OnStopReflectionHook.execute`` with a MagicMock
agent, mirroring the kestrel-sovereign test pattern for the loop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookInput
from kestrel_feature_reflection.on_stop_hook import (
    RESERVED_FACT_TOOL_NAMES,
    OnStopReflectionHook,
    create_on_stop_reflection_hook,
    filter_fact_tools,
    format_turn_transcript,
    per_turn_reflection_disabled,
)


def _fact_tool_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"test {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _llm_response(tool_calls=None, content=""):
    r = MagicMock()
    r.content = content
    r.tool_calls = tool_calls
    return r


def _tc(call_id: str, tool_name: str, arguments: dict):
    """Build a tool-call mock. ``name`` is a reserved MagicMock ctor kwarg,
    so it must be assigned after construction."""
    m = MagicMock(id=call_id, arguments=arguments)
    m.name = tool_name
    return m


def _three_fact_tool_calls():
    """Three structural facts: a rename, a package path, a pattern."""
    return [
        _tc("r1", "save_fact", {"subject": "project", "predicate": "renamed_from",
                                "value": "CodeEditFeature -> CodeFeature", "confidence": 1.0}),
        _tc("r2", "save_fact", {"subject": "project", "predicate": "reflection_pkg_path",
                                "value": "kestrel_feature_reflection/", "confidence": 1.0}),
        _tc("r3", "strategy_add_pattern", {"pattern": "bare pip in CI breaks",
                                           "implication": "use uv pip"}),
    ]


def _make_agent(*, llm_response, fact_tools_loaded=True, disabled_llm=False):
    agent = MagicMock()
    agent.did = "did:test:reflection"
    agent.llm_service = MagicMock()
    agent.llm_service.disabled = disabled_llm
    agent.llm_service.generate_with_messages = AsyncMock(return_value=llm_response)

    tools = [
        _fact_tool_schema("save_fact"),
        _fact_tool_schema("strategy_add_pattern"),
        _fact_tool_schema("strategy_add_blocker"),
        _fact_tool_schema("github_issue_view"),  # non-fact, must be filtered
    ] if fact_tools_loaded else [_fact_tool_schema("github_issue_view")]
    agent._build_all_tools = MagicMock(return_value=tools)
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent._visible_known_tool_names = MagicMock(return_value=set())
    agent._build_tool_calls_msg = MagicMock(return_value=[{"id": "x"}])
    agent._execute_tool_batch = AsyncMock()
    agent.observability_store = MagicMock()
    agent.observability_store.log_llm_call = AsyncMock(return_value="evt")
    return agent


def _stop_input(**kw):
    base = dict(
        session_id="s-1",
        hook_event_name=HookEvent.STOP.value,
        user_message="rename CodeEditFeature to CodeFeature",
        response_text="Renamed. The package now lives at kestrel_feature_reflection/.",
    )
    base.update(kw)
    return HookInput(**base)


# --- acceptance ------------------------------------------------------------


async def test_three_structural_facts_persisted_unprompted(monkeypatch):
    monkeypatch.delenv("KESTREL_PER_TURN_REFLECTION_DISABLED", raising=False)
    agent = _make_agent(llm_response=_llm_response(_three_fact_tool_calls()))
    hook = OnStopReflectionHook(agent)

    out = await hook.execute(_stop_input())

    # STOP hooks never gate.
    assert out.permission_decision is None or out.permission_decision.name == "ALLOW"

    # LLM call used only the three fact tools, not the github tool.
    llm_kwargs = agent.llm_service.generate_with_messages.await_args.kwargs
    passed = {t["function"]["name"] for t in llm_kwargs["tools"]}
    assert passed == set(RESERVED_FACT_TOOL_NAMES)
    assert llm_kwargs["session_id"] == "s-1"

    # All three saves dispatched in one batch through the agent's
    # hook-enforced tool path.
    agent._execute_tool_batch.assert_awaited_once()
    dispatched = agent._execute_tool_batch.await_args.args[0]
    assert [tc.name for tc in dispatched] == [
        "save_fact", "save_fact", "strategy_add_pattern"
    ]

    # Observability recorded once with the per_turn phase.
    agent.observability_store.log_llm_call.assert_awaited_once()
    obs = agent.observability_store.log_llm_call.await_args.kwargs
    assert obs["provider"] == "reflection"
    assert obs["metadata"]["phase"] == "per_turn"
    assert obs["metadata"]["tool_calls_count"] == 3
    assert obs["agent_did"] == "did:test:reflection"


async def test_no_facts_learned_zero_saves(monkeypatch):
    monkeypatch.delenv("KESTREL_PER_TURN_REFLECTION_DISABLED", raising=False)
    agent = _make_agent(llm_response=_llm_response(tool_calls=None))
    hook = OnStopReflectionHook(agent)

    await hook.execute(_stop_input())

    agent._execute_tool_batch.assert_not_called()
    agent.observability_store.log_llm_call.assert_awaited_once()
    assert agent.observability_store.log_llm_call.await_args.kwargs[
        "metadata"]["tool_calls_count"] == 0


# --- isolation / opt-out ---------------------------------------------------


async def test_env_disabled_short_circuits(monkeypatch):
    monkeypatch.setenv("KESTREL_PER_TURN_REFLECTION_DISABLED", "1")
    agent = _make_agent(llm_response=_llm_response(_three_fact_tool_calls()))
    hook = OnStopReflectionHook(agent)

    out = await hook.execute(_stop_input())

    agent.llm_service.generate_with_messages.assert_not_called()
    agent._execute_tool_batch.assert_not_called()
    assert out is not None  # still returns allow()


async def test_disabled_llm_service_skips(monkeypatch):
    monkeypatch.delenv("KESTREL_PER_TURN_REFLECTION_DISABLED", raising=False)
    agent = _make_agent(
        llm_response=_llm_response(_three_fact_tool_calls()), disabled_llm=True
    )
    hook = OnStopReflectionHook(agent)
    await hook.execute(_stop_input())
    agent.llm_service.generate_with_messages.assert_not_called()


async def test_no_fact_tools_loaded_skips(monkeypatch):
    monkeypatch.delenv("KESTREL_PER_TURN_REFLECTION_DISABLED", raising=False)
    agent = _make_agent(
        llm_response=_llm_response(_three_fact_tool_calls()),
        fact_tools_loaded=False,
    )
    hook = OnStopReflectionHook(agent)
    await hook.execute(_stop_input())
    agent.llm_service.generate_with_messages.assert_not_called()


async def test_llm_failure_never_breaks_turn(monkeypatch):
    monkeypatch.delenv("KESTREL_PER_TURN_REFLECTION_DISABLED", raising=False)
    agent = _make_agent(llm_response=_llm_response())
    agent.llm_service.generate_with_messages = AsyncMock(
        side_effect=RuntimeError("LLM down")
    )
    hook = OnStopReflectionHook(agent)

    # Must NOT raise — STOP is post-yield, a failure here can't surface.
    out = await hook.execute(_stop_input())
    assert out is not None


async def test_tool_batch_failure_swallowed(monkeypatch):
    monkeypatch.delenv("KESTREL_PER_TURN_REFLECTION_DISABLED", raising=False)
    agent = _make_agent(llm_response=_llm_response(_three_fact_tool_calls()))
    agent._execute_tool_batch = AsyncMock(side_effect=RuntimeError("dispatch boom"))
    hook = OnStopReflectionHook(agent)
    out = await hook.execute(_stop_input())
    assert out is not None  # swallowed, turn unaffected


# --- helpers ---------------------------------------------------------------


def test_filter_fact_tools_keeps_only_three():
    schemas = [
        _fact_tool_schema("save_fact"),
        _fact_tool_schema("strategy_add_pattern"),
        _fact_tool_schema("strategy_add_blocker"),
        _fact_tool_schema("strategy_add_decision"),
        _fact_tool_schema("github_issue_view"),
    ]
    kept = {t["function"]["name"] for t in filter_fact_tools(schemas)}
    assert kept == set(RESERVED_FACT_TOOL_NAMES)


def test_transcript_includes_user_response_and_aligned_tool_calls():
    hi = _stop_input(
        tool_calls=[{"name": "github_view", "arguments": {"issue": 1238}}],
        tool_results=[{"tool_call_id": "tc", "name": "github_view",
                       "result": {"status": "ok", "data": {"title": "X"}}}],
    )
    t = format_turn_transcript(hi)
    assert "rename CodeEditFeature" in t
    assert "github_view" in t
    assert "status" in t  # tool-result envelope rendered
    assert "Renamed." in t


def test_transcript_handles_cancel_before_dispatch():
    """tool_calls present, tool_results empty (streaming cancel edge from
    kestrel-sovereign #1269) — must not raise, still renders the call."""
    hi = _stop_input(
        tool_calls=[{"name": "will_cancel", "arguments": {}}],
        tool_results=[],
    )
    t = format_turn_transcript(hi)
    assert "will_cancel" in t


def test_disabled_env_truthy_variants(monkeypatch):
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("KESTREL_PER_TURN_REFLECTION_DISABLED", truthy)
        assert per_turn_reflection_disabled() is True
    for falsy in ("0", "false", "no", "", "off"):
        monkeypatch.setenv("KESTREL_PER_TURN_REFLECTION_DISABLED", falsy)
        assert per_turn_reflection_disabled() is False


def test_factory_returns_none_without_llm_service():
    bare = MagicMock(spec=[])  # no llm_service attribute
    assert create_on_stop_reflection_hook(bare) is None
    assert create_on_stop_reflection_hook(None) is None


def test_factory_returns_hook_with_llm_service():
    agent = MagicMock()
    agent.llm_service = MagicMock()
    hook = create_on_stop_reflection_hook(agent)
    assert isinstance(hook, OnStopReflectionHook)
    assert HookEvent.STOP in hook.events
