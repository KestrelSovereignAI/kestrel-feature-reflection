"""Real component writers use the original MemorySystem privacy lifetime.

These Agents own native SQLite and real privacy/storage services. This does not
claim READY publication or generated reflections; full native MCP diagnostics
are recorded separately.
"""

import asyncio
from uuid import uuid4

import pytest
from kestrel_sovereign.privacy import PrivacyMode

from kestrel_feature_reflection.models import (
    CheckStatus,
    HealthCheck,
    Insight,
    InsightType,
    LayerResult,
    ReflectionLayer,
    ReflectionResult,
)

pytestmark = pytest.mark.asyncio


def reflection_input(feature):
    insight = Insight(id=uuid4().hex, type=InsightType.PATTERN,
                      title="Owned derivative", description="Owned transcript derivative")
    check = HealthCheck(id=uuid4().hex, layer=ReflectionLayer.MIND,
                        name="Owned input", description="Writer input, not model output",
                        status=CheckStatus.PASS, details={"insights": [insight.to_dict()]})
    return ReflectionResult(id=uuid4().hex, trigger="owned-writer-input",
                            source_privacy_epoch=feature.agent._get_privacy_policy_epoch(),
                            mind=LayerResult(layer=ReflectionLayer.MIND, checks=[check]))


@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL,
                                  PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED])
async def test_reflection_derivatives_obey_the_native_privacy_gate(reflection_writer, mode):
    feature = reflection_writer
    async with feature.agent._get_privacy_transition_lock():
        feature.agent.storage.set_privacy_mode(mode)
    result = reflection_input(feature)
    await feature._persist_reflection(result)
    session = await feature._db.fetchone("SELECT id FROM reflection_sessions WHERE id = ?", (result.id,))
    insights = await feature._db.fetchall("SELECT id FROM reflection_insights WHERE session_id = ?", (result.id,))
    assert bool(session) is (mode == PrivacyMode.NORMAL), "Private reflection session was persisted"
    assert bool(insights) is (mode == PrivacyMode.NORMAL), "Private reflection insight was persisted"


@pytest.mark.parametrize("writer", ["store_session", "store_insight"])
async def test_privacy_transition_waits_for_the_original_reflection_write(
    reflection_writer, monkeypatch, writer,
):
    feature = reflection_writer
    entered, release, transition_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = getattr(feature._db_helper, writer)

    async def pause_original(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    async def change_privacy():
        transition_started.set()
        async with feature.agent._get_privacy_transition_lock():
            feature.agent.storage.set_privacy_mode(PrivacyMode.EPHEMERAL)

    monkeypatch.setattr(feature._db_helper, writer, pause_original)
    result = reflection_input(feature)
    writing = asyncio.create_task(feature._persist_reflection(result))
    transition = None
    try:
        async with asyncio.timeout(10):
            await entered.wait()
            transition = asyncio.create_task(change_privacy())
            await transition_started.wait()
            await asyncio.sleep(0)
            assert not transition.done(), "Privacy transition crossed an original reflection write"
            release.set()
            await writing
            await transition
        assert await feature._db.fetchone("SELECT id FROM reflection_sessions WHERE id = ?", (result.id,))
        assert await feature._db.fetchone("SELECT id FROM reflection_insights WHERE session_id = ?", (result.id,))
    finally:
        release.set()
        await asyncio.gather(writing, *([transition] if transition is not None else []), return_exceptions=True)


@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL,
                                  PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED])
async def test_reflection_uses_the_current_privacy_conversation(reflection_writer, mode):
    feature = reflection_writer
    await feature.agent.privacy_agent.add_conversation("user", "Earlier durable garden conversation")
    async with feature.agent._get_privacy_transition_lock():
        feature.agent.storage.set_privacy_mode(mode)
        feature.agent.privacy_agent.set_mode(mode)
    await feature.agent.privacy_agent.add_conversation("user", "Current garden conversation")
    expected = await feature.agent.privacy_agent.get_conversation_history(limit=50)
    interactions = await feature.analyzer._load_interactions("all", 50)
    assert [row["content"] for row in interactions] == [row["content"] for row in expected], (
        "Reflection bypassed the current privacy-aware conversation")
    prompt = feature.analyzer._build_prompt(interactions, [], "normal")
    assert "Current garden conversation" in prompt


@pytest.mark.parametrize("source", ["missing", "another_agent"])
@pytest.mark.parametrize("sink", ["persist", "deliver"])
async def test_reflection_rejects_unowned_source_policies(
    reflection_writer, native_tool_effect_agent, source, sink,
):
    feature = reflection_writer
    result = reflection_input(feature)
    other = await native_tool_effect_agent(did="did:test:other-reflection-source")
    result.source_privacy_epoch = None if source == "missing" else other._get_privacy_policy_epoch()
    if sink == "persist":
        await feature._persist_reflection(result)
        assert not await feature._db.fetchone(
            "SELECT id FROM reflection_sessions WHERE id = ?", (result.id,)
        ), "Reflection persisted a result without its original source policy"
    else:
        response = feature._reflect_to_tool_result(result)
        assert "Owned transcript derivative" not in repr(response.to_dict()), (
            "Reflection delivered data without its original source policy")


async def test_real_reflect_producer_records_its_original_policy(reflection_writer, monkeypatch):
    feature = reflection_writer
    original = feature._reflect_to_tool_result
    observed = []

    def observe_result(result):
        observed.append(result.source_privacy_epoch)
        return original(result)

    monkeypatch.setattr(feature, "_reflect_to_tool_result", observe_result)
    await feature.reflect()
    assert observed == [feature.agent._get_privacy_policy_epoch()], (
        "Reflect did not record its original source policy before the checks")


@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL,
                                  PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED])
async def test_coherence_check_uses_the_same_private_conversation(reflection_writer, mode):
    feature = reflection_writer
    await feature.agent.privacy_agent.add_conversation("assistant", "Earlier durable garden answer")
    async with feature.agent._get_privacy_transition_lock():
        feature.agent.storage.set_privacy_mode(mode)
        feature.agent.privacy_agent.set_mode(mode)
    await feature.agent.privacy_agent.add_conversation("user", "Current private garden question")
    check = await feature._mind_checker.check_response_coherence()
    assert check.status is (CheckStatus.PASS if mode == PrivacyMode.NORMAL else CheckStatus.SKIP), (
        "Coherence check bypassed the current privacy-aware conversation")


@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL,
                                  PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED])
async def test_decryption_check_uses_the_same_private_conversation(reflection_writer, mode):
    feature = reflection_writer
    await feature.agent.privacy_agent.add_conversation("user", "Earlier durable question")
    await feature.agent.privacy_agent.add_conversation("assistant", "Earlier durable answer")
    async with feature.agent._get_privacy_transition_lock():
        feature.agent.storage.set_privacy_mode(mode)
        feature.agent.privacy_agent.set_mode(mode)
    await feature.agent.privacy_agent.add_conversation("user", "Current private question")
    expected = await feature.agent.privacy_agent.get_conversation_history(limit=5)
    check = await feature._memory_checker.check_conversation_decryption()
    assert check.message == f"Can read {len(expected)} messages", (
        "Decryption check bypassed the current privacy-aware conversation")
