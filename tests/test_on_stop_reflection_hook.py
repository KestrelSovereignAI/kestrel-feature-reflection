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
    format_turn_transcript,
    per_turn_reflection_disabled,
)


def _fact_tool_obj(name: str):
    """An AgentTool-shaped mock: .name + .schema.to_openai_format() + .execute."""
    t = MagicMock()
    t.name = name
    schema = MagicMock()
    schema.to_openai_format.return_value = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"test {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    t.schema = schema
    t.execute = AsyncMock(return_value={"success": True})
    return t


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

    # Fact tools are sourced by walking agent.features and calling
    # feature.get_tools() — exposure-state-independent (codex review).
    if fact_tools_loaded:
        mem = MagicMock()
        mem.get_tools = MagicMock(return_value=[_fact_tool_obj("save_fact")])
        strat = MagicMock()
        strat.get_tools = MagicMock(return_value=[
            _fact_tool_obj("strategy_add_pattern"),
            _fact_tool_obj("strategy_add_blocker"),
        ])
        other = MagicMock()
        other.get_tools = MagicMock(return_value=[_fact_tool_obj("github_issue_view")])
        agent.features = {
            "MemoryAgencyFeature": mem,
            "StrategicMemoryFeature": strat,
            "GitHubFeature": other,
        }
    else:
        other = MagicMock()
        other.get_tools = MagicMock(return_value=[_fact_tool_obj("github_issue_view")])
        agent.features = {"GitHubFeature": other}

    # Hook-enforced single-tool entry point: invoke the execute_fn.
    async def _exec_with_hooks(tool_name, feature_name, args, sid, execute_fn):
        return await execute_fn()

    agent._execute_tool_with_hooks = AsyncMock(side_effect=_exec_with_hooks)
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
    # Reflection LLM call is isolated from the user conversation cursor.
    assert llm_kwargs["session_id"] == "per-turn-reflection::s-1"

    # Each of the three saves dispatched through the hook-enforced
    # single-tool entry point (PRE/POST_TOOL_USE fire per call).
    assert agent._execute_tool_with_hooks.await_count == 3
    dispatched_names = [
        c.args[0] for c in agent._execute_tool_with_hooks.await_args_list
    ]
    assert dispatched_names == ["save_fact", "save_fact", "strategy_add_pattern"]
    # The owning feature name is passed for permission lookup.
    feature_names = {
        c.args[1] for c in agent._execute_tool_with_hooks.await_args_list
    }
    assert feature_names == {"MemoryAgencyFeature", "StrategicMemoryFeature"}

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

    agent._execute_tool_with_hooks.assert_not_called()
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
    agent._execute_tool_with_hooks.assert_not_called()
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
    agent._execute_tool_with_hooks = AsyncMock(side_effect=RuntimeError("dispatch boom"))
    hook = OnStopReflectionHook(agent)
    out = await hook.execute(_stop_input())
    assert out is not None  # swallowed, turn unaffected


# --- helpers ---------------------------------------------------------------


async def test_only_reserved_tools_offered_to_llm(monkeypatch):
    """The reflection LLM call is offered ONLY the three fact tools even
    though the agent has other features (github) loaded — sourced by
    name from agent.features, not the full tool catalog."""
    monkeypatch.delenv("KESTREL_PER_TURN_REFLECTION_DISABLED", raising=False)
    agent = _make_agent(llm_response=_llm_response(tool_calls=None))
    hook = OnStopReflectionHook(agent)
    await hook.execute(_stop_input())
    offered = {
        t["function"]["name"]
        for t in agent.llm_service.generate_with_messages.await_args.kwargs["tools"]
    }
    assert offered == set(RESERVED_FACT_TOOL_NAMES)


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


async def test_unenriched_stop_warns_once_and_skips(monkeypatch, caplog):
    """Against a sovereign runtime predating #1269, STOP fires with no
    turn context. The hook must NOT silently no-op forever — it warns
    once (actionable: upgrade sovereign) then stays quiet, and never
    issues an LLM call."""
    import logging as _logging
    import kestrel_feature_reflection.on_stop_hook as mod

    monkeypatch.delenv("KESTREL_PER_TURN_REFLECTION_DISABLED", raising=False)
    mod._warned_unenriched_stop = False  # reset latch for the test

    agent = _make_agent(llm_response=_llm_response(_three_fact_tool_calls()))
    hook = OnStopReflectionHook(agent)

    bare = HookInput(session_id="s", hook_event_name=HookEvent.STOP.value)

    with caplog.at_level(_logging.WARNING):
        await hook.execute(bare)
        await hook.execute(bare)  # second call must not warn again

    agent.llm_service.generate_with_messages.assert_not_called()
    warnings = [r for r in caplog.records if "predates the" in r.message]
    assert len(warnings) == 1, "must warn exactly once, not per turn"


def test_transcript_preserves_toolresult_envelope_payload():
    """When the tool-result payload lives in ToolResult fields
    (status/confirmation/data) rather than a ``result`` key — or the
    list item IS the envelope with no ``result`` wrapper — the transcript
    must still surface it, so the reflection LLM can capture facts a tool
    discovered (codex review)."""
    # Case A: result key present and meaningful — rendered as-is.
    a = _stop_input(
        tool_calls=[{"name": "find_path", "arguments": {}}],
        tool_results=[{"tool_call_id": "t", "name": "find_path",
                       "result": {"status": "ok", "data": {"path": "src/x.py"}}}],
    )
    ta = format_turn_transcript(a)
    assert "src/x.py" in ta and "status" in ta

    # Case B: no "result" wrapper — envelope IS the dict. Must not drop it.
    b = _stop_input(
        tool_calls=[{"name": "probe", "arguments": {}}],
        tool_results=[{"tool_call_id": "t", "name": "probe",
                       "status": "error", "error": "connection refused"}],
    )
    tb = format_turn_transcript(b)
    assert "connection refused" in tb

    # Case C: result present but empty/None — fall back to envelope sibs.
    c = _stop_input(
        tool_calls=[{"name": "check", "arguments": {}}],
        tool_results=[{"tool_call_id": "t", "name": "check",
                       "result": None, "confirmation": "saved fact #42"}],
    )
    tc_ = format_turn_transcript(c)
    assert "saved fact #42" in tc_


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


async def test_reflection_llm_call_uses_isolated_session_id(monkeypatch):
    """The reflection LLM call MUST NOT reuse the user turn's session_id —
    stateful/continuation providers anchor cursor state on it, so reusing
    it would corrupt the next user-facing turn. A distinct namespaced id
    is used instead (codex review P1)."""
    monkeypatch.delenv("KESTREL_PER_TURN_REFLECTION_DISABLED", raising=False)
    agent = _make_agent(llm_response=_llm_response(_three_fact_tool_calls()))
    hook = OnStopReflectionHook(agent)

    await hook.execute(_stop_input(session_id="user-conv-42"))

    sid = agent.llm_service.generate_with_messages.await_args.kwargs["session_id"]
    assert sid != "user-conv-42", "must not reuse the user conversation cursor"
    assert "user-conv-42" in sid and sid.startswith("per-turn-reflection")
    # Observability still attributes cost to the real conversation.
    obs_sid = agent.observability_store.log_llm_call.await_args.kwargs["session_id"]
    assert obs_sid == "user-conv-42"


def test_get_hooks_memoizes_same_instance():
    """The hooks manager unregisters by identity — get_hooks() called
    repeatedly must return the SAME hook object, or disable/re-enable
    leaves a stale STOP hook duplicating per-turn reflection (codex
    review P2)."""
    from kestrel_feature_reflection.feature import ReflectionFeature

    feat = ReflectionFeature.__new__(ReflectionFeature)  # skip heavy __init__
    feat.agent = MagicMock()
    feat.agent.llm_service = MagicMock()

    first = feat.get_hooks()
    second = feat.get_hooks()
    assert len(first) == 1 and len(second) == 1
    assert first[0] is second[0], "get_hooks must return the memoized instance"


def test_get_hooks_empty_without_llm_service():
    from kestrel_feature_reflection.feature import ReflectionFeature

    feat = ReflectionFeature.__new__(ReflectionFeature)
    feat.agent = MagicMock(spec=[])  # no llm_service
    assert feat.get_hooks() == []
