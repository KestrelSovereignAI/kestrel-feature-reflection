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
        if carrier == "timeout":
            try:
                raise TimeoutError("Original authority carried by a timeout") from error
            except TimeoutError as timeout:
                result = timeout
        else:
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


@pytest.mark.parametrize("carrier", ["direct", "group"])
@pytest.mark.parametrize("boundary", ["registered_stop", "stop_live_policy", "ticket_insight", "ticket_proposal"])
async def test_registered_stop_and_ticket_approval_keep_native_control(
    reflection_writer, monkeypatch, carrier, boundary,
):
    from types import SimpleNamespace

    from kestrel_sdk.hooks.base import HookEvent, HookInput
    from kestrel_sovereign.features.security.approval_queue import ApprovalQueue

    from kestrel_feature_reflection.models import (
        ChangeType,
        ImprovementProposal,
        Insight,
        InsightType,
    )
    from kestrel_feature_reflection.on_stop_hook import (
        OnStopReflectionHook,
        live_force_local_only,
    )
    from kestrel_feature_reflection.ticket_creator import TicketCreator

    feature = reflection_writer
    error = original_authority_rejection(feature, carrier)
    if boundary == "registered_stop":
        hook = OnStopReflectionHook(feature.agent)
        original = hook._capture
        async def original_capture_then_fence(*args, **kwargs):
            await original(*args, **kwargs)
            raise error
        monkeypatch.setattr(hook, "_capture", original_capture_then_fence)
        operation = hook.execute(HookInput(hook_event_name=HookEvent.STOP.value, session_id="owned-stop"))
    elif boundary == "stop_live_policy":
        def original_authority_refusal():
            raise error
        monkeypatch.setattr(feature.agent.llm_service, "_current_force_local_only", original_authority_refusal)
        with pytest.raises(type(error)) as captured:
            live_force_local_only(feature.agent.llm_service)
        assert captured.value is error
        return
    else:
        queue = ApprovalQueue(agent=feature.agent)
        async def original_authority_refusal(**kwargs):
            raise error
        monkeypatch.setattr(queue, "request_approval", original_authority_refusal)
        # Exercise only the pre-write approval stage. No configured GitHub
        # client or token exists; construction/publication is not claimed.
        creator = object.__new__(TicketCreator)
        creator.github = None
        creator.repo = TicketCreator.DEFAULT_REPO
        security = SimpleNamespace(approval_queue=queue)
        if boundary == "ticket_insight":
            item = Insight(id=uuid4().hex, type=InsightType.PATTERN,
                           title="Owned approval input", description="Not generated output")
            operation = creator.create_ticket_from_insight(item, security)
        else:
            item = ImprovementProposal(id=uuid4().hex, insight_id=None, change_type=ChangeType.PROMPT,
                                       title="Owned approval input", description="Boundary input",
                                       proposed_change="Owned proposed guidance")
            operation = creator.create_ticket_from_proposal(item, security)
    with pytest.raises(type(error)) as captured:
        await operation
    assert captured.value is error, "Reflection erased native control at a registered boundary"


@pytest.mark.parametrize("carrier", ["direct", "group", "timeout"])
@pytest.mark.parametrize("boundary", [
    "guidance", "insight", "proposal", "actionable", "ticket_handler",
    "self_model_pointer", "self_model_handler", "self_modify_approval",
    "ticket_economics", "model_economics",
])
async def test_optional_consumers_preserve_original_native_control(
    reflection_writer, monkeypatch, carrier, boundary,
):
    from types import SimpleNamespace

    from kestrel_sovereign.features.security.approval_queue import ApprovalQueue

    from kestrel_feature_reflection.approval import ApprovalHandler
    from kestrel_feature_reflection.economics import EconomicGate
    from kestrel_feature_reflection.models import ChangeType, ImprovementProposal
    from kestrel_feature_reflection.self_model import SelfModelManager
    from kestrel_feature_reflection.self_model_handler import SelfModelHandler
    from kestrel_feature_reflection.ticket_creator import TicketCreator
    from kestrel_feature_reflection.ticket_handler import TicketHandler

    feature = reflection_writer
    error = original_authority_rejection(feature, carrier)
    async def original_authority_refusal(*args, **kwargs):
        raise error
    if boundary.endswith("economics"):
        # Protocol input refuses at the original Core authority guard. This
        # cannot grant economic eligibility and never creates a wallet result.
        def reject_economic_read():
            raise error
        gate = EconomicGate(SimpleNamespace(is_paid_tier=reject_economic_read))
        with pytest.raises(type(error)) as captured:
            (gate.can_create_tickets() if boundary == "ticket_economics"
             else gate.can_update_self_model())
        assert captured.value is error
        return
    if boundary == "self_modify_approval":
        queue = ApprovalQueue(agent=feature.agent)
        monkeypatch.setattr(queue, "request_approval", original_authority_refusal)
        handler = ApprovalHandler(feature.agent, feature._db_helper)
        monkeypatch.setattr(handler, "_get_security_feature", lambda: SimpleNamespace(approval_queue=queue))
        proposal = ImprovementProposal(id=uuid4().hex, insight_id=None,
                                       title="Owned approval input", description="Boundary input",
                                       change_type=ChangeType.PROMPT, proposed_change="Owned guidance")
        operation = handler.request_approval(proposal)
    else:
        # Fault admission to a real owned database; no SQL/model/provider
        # result is fabricated and no decentralized service is configured.
        monkeypatch.setattr(feature._db, "fetchall", original_authority_refusal)
        if boundary in {"guidance", "insight", "proposal", "actionable"}:
            methods = {"guidance": (feature._db_helper.get_active_guidance, ()),
                       "insight": (feature._db_helper.get_insight_by_id, ("owned-input",)),
                       "proposal": (feature._db_helper.get_proposal_by_id, ("owned-input",)),
                       "actionable": (feature._db_helper.get_actionable_insights, ())}
            method, args = methods[boundary]
            operation = method(*args)
        elif boundary == "ticket_handler":
            creator = object.__new__(TicketCreator)
            handler = TicketHandler(creator, None, feature._db_helper, feature.agent)
            operation = handler.create_improvement_ticket("owned-input")
        else:
            # Test the original pre-provider database stage only. Do not
            # bypass configuration to publish or claim provider acceptance.
            manager = object.__new__(SelfModelManager)
            manager._db, manager.agent_did = feature._db, feature.agent.did
            manager._current_model, manager._key_hash = None, None
            manager.storage = None
            if boundary == "self_model_pointer":
                operation = manager._get_latest_model_cid()
            else:
                handler = SelfModelHandler(manager, None, feature._db_helper, feature.agent)
                operation = handler.get_self_model()
    with pytest.raises(type(error)) as captured:
        await operation
    assert captured.value is error, "Optional consumer replaced original native control"
