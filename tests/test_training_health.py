"""Training consumes the real serialized health model, including absent layers."""

import pytest

from kestrel_feature_reflection.formatters import ReflectionResultFormatter
from kestrel_feature_reflection.models import (
    CheckStatus,
    HealthCheck,
    LayerResult,
    ReflectionLayer,
    ReflectionResult,
    Severity,
)
from kestrel_feature_reflection.training import TrainingManager


@pytest.mark.parametrize("status,severity,counts,healthy", [
    (CheckStatus.PASS, None, (1, 0, 0), True),
    (CheckStatus.SKIP, None, (0, 0, 0), False),
    (CheckStatus.WARN, Severity.LOW, (0, 1, 0), False),
    (CheckStatus.FAIL, Severity.HIGH, (0, 1, 0), False),
    (CheckStatus.FAIL, Severity.CRITICAL, (0, 0, 1), False),
])
def test_training_counts_serialized_checks(status, severity, counts, healthy):
    check = HealthCheck(id="training-health", layer=ReflectionLayer.ARMS,
                        name="Training health", description="Serialized health input",
                        status=status, severity=severity)
    result = ReflectionResult(id="training", trigger="test",
                              arms=LayerResult(layer=ReflectionLayer.ARMS, checks=[check]))
    data = ReflectionResultFormatter.format_reflection_result(result)
    health = TrainingManager(None)._analyze_iteration_health(data, 1)
    assert (health["passed"], health["warn"], health["critical"]) == counts
    assert health["healthy"] is healthy


@pytest.mark.asyncio
async def test_training_preserves_canonical_action_content_before_ticket_admission(reflection_writer):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.features.security.feature import SecurityFeature

    from kestrel_feature_reflection.models import ActionItem
    feature = reflection_writer
    try:
        feature.agent._get_privacy_transition_lock().require_native_scope(None)
    except ExecutionAuthorityError as error:
        original = error
    captured = []
    class AdmissionOnlyTicketPort:
        async def create_ticket_from_insight(self, *, insight, security_feature, timeout):
            captured.append(insight)
            # Stop before any GitHub client, token, approval or issue exists.
            raise original
    feature._ticket_creator = AdmissionOnlyTicketPort()
    feature.agent.features["SecurityFeature"] = SecurityFeature(feature.agent)
    action = ActionItem(id="owned-action", priority=Severity.HIGH,
                        title="Actual action title", description="Actual action problem",
                        fix_description="Actual proposed repair")
    with pytest.raises(ExecutionAuthorityError) as refusal:
        await TrainingManager(feature)._create_tickets_for_actions([action.to_dict()])
    assert refusal.value is original and len(captured) == 1
    insight, = captured
    assert (insight.title, insight.description, insight.suggested_action) == (
        action.title, action.description, action.fix_description), "Training discarded canonical ActionItem content"


@pytest.mark.parametrize("actionable", [False, True])
def test_canonical_action_serialization_preserves_ticket_eligibility(actionable):
    from kestrel_feature_reflection.models import ActionItem
    action = ActionItem(id="ticket-eligibility", priority=Severity.LOW,
                        title="Owned action", description="Owned input", actionable=actionable)
    assert action.to_dict().get("actionable") is actionable, "ActionItem serialization lost ticket eligibility"
