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
