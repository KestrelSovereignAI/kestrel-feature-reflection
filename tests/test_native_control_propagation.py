"""Classified errors come from Core's actual native authority guard.

No model result, ownership receipt or exception classification is fabricated.
"""
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    is_execution_control_error,
)

from kestrel_feature_reflection.hooks import ReflectionSleepHook
from kestrel_feature_reflection.models import ReflectionResult

pytestmark = pytest.mark.asyncio


def original_authority_rejection(feature, carrier):
    try:
        feature.agent._get_privacy_transition_lock().require_native_scope(None)
    except ExecutionAuthorityError as error:
        result = error if carrier == "direct" else ExceptionGroup("original native control", [error])
        assert is_execution_control_error(result)
        return result
    pytest.fail("Core accepted an absent original native scope")


@pytest.mark.parametrize("carrier", ["direct", "group"])
@pytest.mark.parametrize("boundary", [
    "persist", "gate_exit", "reflect", "pre_sleep", "post_consolidation",
    "analyzer_reader", "coherence_reader", "memory_reader", "analyzer_generation", "arms_generation",
])
async def test_original_native_control_survives_every_reflection_boundary(
    reflection_writer, monkeypatch, carrier, boundary,
):
    feature = reflection_writer
    error = original_authority_rejection(feature, carrier)
    await feature.agent.privacy_agent.add_conversation("user", "Owned native control input")
    result = ReflectionResult(id=uuid4().hex, trigger="owned-control-input",
                              source_privacy_epoch=feature.agent._get_privacy_policy_epoch())
    if boundary == "persist":
        original = feature._db_helper.store_session
        async def original_write_then_fence(*args, **kwargs):
            await original(*args, **kwargs)
            raise error
        monkeypatch.setattr(feature._db_helper, "store_session", original_write_then_fence)
        operation = feature._persist_reflection(result)
    elif boundary == "gate_exit":
        original = feature.agent.memory_system.durable_writes
        @asynccontextmanager
        async def original_gate_then_fence():
            async with original() as allowed:
                yield allowed
            raise error
        monkeypatch.setattr(feature.agent.memory_system, "durable_writes", original_gate_then_fence)
        operation = feature._persist_reflection(result)
    elif boundary in {"reflect", "pre_sleep", "post_consolidation"}:
        original = feature._arms_checker.run_all
        async def original_checks_then_fence(*args, **kwargs):
            await original(*args, **kwargs)
            raise error
        monkeypatch.setattr(feature._arms_checker, "run_all", original_checks_then_fence)
        if boundary == "reflect":
            operation = feature.reflect()
        elif boundary == "pre_sleep":
            operation = ReflectionSleepHook(feature).on_pre_sleep(feature.agent)
        else:
            operation = ReflectionSleepHook(feature).on_post_consolidation(feature.agent, {"episodes_created": 1})
    elif boundary in {"analyzer_reader", "coherence_reader", "memory_reader"}:
        original = feature.agent._raw_storage.conversation.get_conversation_history
        async def original_read_then_fence(*args, **kwargs):
            await original(*args, **kwargs)
            raise error
        monkeypatch.setattr(feature.agent._raw_storage.conversation, "get_conversation_history", original_read_then_fence)
        if boundary == "analyzer_reader":
            operation = feature.analyzer._load_interactions("all", 50)
        elif boundary == "coherence_reader":
            operation = feature._mind_checker.check_response_coherence()
        else:
            operation = feature._memory_checker.check_conversation_decryption()
    else:
        async def original_authority_refusal(*args, **kwargs):
            raise error
        monkeypatch.setattr(feature.agent.llm_service, "generate", original_authority_refusal)
        operation = (feature.analyzer.analyze(scope="all") if boundary == "analyzer_generation"
                     else feature._arms_checker.check_llm_connectivity())
    with pytest.raises(type(error)) as captured:
        await operation
    assert captured.value is error, "Reflection replaced original native control evidence"
