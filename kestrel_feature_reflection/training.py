"""
Training cycle functionality for reflection feature.

Handles intensive reflection training sessions for rapid self-improvement.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List

from kestrel_sdk.tools import ToolResultStatus
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.kestrel_config.constants import APPROVAL_TIMEOUT_SHORT

from .formatters import ReflectionResultFormatter
from .models import CheckStatus, Insight, InsightType, Severity

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages training cycles for intensive reflection."""

    def __init__(self, reflection_feature):
        """
        Initialize the training manager.

        Args:
            reflection_feature: The parent reflection feature instance
        """
        self.feature = reflection_feature

    async def run_training_cycle(
        self,
        iterations: int = 3,
        depth: str = "normal",
        create_tickets: bool = False,
    ) -> Dict[str, Any]:
        """
        Run intensive training cycle for rapid self-improvement.

        Unlike the nightly sleep hook (long-term consolidation), this is for
        active, intensive improvement sessions - like meditation or training.

        Each iteration:
        1. Run layered reflection
        2. Optionally create GitHub tickets for action items
        3. Report health score trend

        Args:
            iterations: Number of reflection cycles (default: 3)
            depth: Analysis depth ('shallow', 'normal', 'deep')
            create_tickets: Whether to create GitHub issues for action items

        Returns:
            Training cycle results with health trend
        """
        results = []
        tickets_created = []

        for i in range(1, iterations + 1):
            logger.info(f"Training cycle iteration {i}/{iterations}")

            # Run reflection. ReflectionFeature.reflect() now returns a
            # ToolResult envelope (#1061 wave 9); the legacy
            # format_reflection_result dict that this manager consumes
            # via .get(...) lives under .data.
            reflection_envelope = await self.feature.reflect(scope="all", depth=depth)
            if reflection_envelope.status is ToolResultStatus.ERROR:
                return {
                    "success": False,
                    "error": reflection_envelope.error,
                    "iterations_completed": len(results),
                    "health_trend": results,
                    "tickets_created": tickets_created,
                }
            reflection_result = reflection_envelope.data
            if not isinstance(reflection_result, dict) or not reflection_result:
                raise TypeError("Reflection training requires a nonempty reflection result")

            # Extract health score
            iteration_result = self._analyze_iteration_health(reflection_result, i)
            results.append(iteration_result)

            # Create tickets if enabled
            if create_tickets and self.feature._ticket_creator:
                tickets = await self._create_tickets_for_actions(
                    reflection_result.get("actions", [])
                )
                tickets_created.extend(tickets)

            # Stop early if healthy
            if iteration_result["healthy"]:
                logger.info(f"Agent is HEALTHY after iteration {i}")
                break

            # Brief pause between iterations
            if i < iterations:
                await asyncio.sleep(1.0)

        # Build summary using formatter
        return ReflectionResultFormatter.format_training_result(
            iterations_completed=len(results),
            health_trend=results,
            tickets_created=tickets_created,
        )

    def _analyze_iteration_health(
        self,
        reflection_result: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Analyze the health status of a single iteration.

        Args:
            reflection_result: The reflection result to analyze
            iteration: The iteration number

        Returns:
            Health analysis for this iteration
        """
        critical = 0
        warn = 0
        passed = 0

        for layer_name in ["arms", "memory", "mind"]:
            layer = reflection_result.get(layer_name) or {}
            for check in layer.get("checks", []):
                status = check.get("status", "")
                severity = check.get("severity", "")
                if status == CheckStatus.FAIL.value and severity == Severity.CRITICAL.value:
                    critical += 1
                elif status in [CheckStatus.FAIL.value, CheckStatus.WARN.value]:
                    warn += 1
                elif status == CheckStatus.PASS.value:
                    passed += 1

        return {
            "iteration": iteration,
            "passed": passed,
            "warn": warn,
            "critical": critical,
            "healthy": passed > 0 and critical == 0 and warn == 0,
        }

    async def _create_tickets_for_actions(
        self,
        actions: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Create GitHub tickets for actionable items.

        Args:
            actions: List of action items from reflection

        Returns:
            List of created ticket URLs
        """
        tickets = []

        for action in actions:
            if not action.get("actionable", True):
                continue

            # Create insight from action for ticket creation
            insight = Insight(
                id=str(uuid.uuid4()),
                type=InsightType.IMPROVEMENT,
                title=action["title"][:100],
                description=action["description"],
                confidence=0.8,
                actionable=True,
                suggested_action=action["fix_description"],
            )

            try:
                # Get security feature
                security = self.feature._get_feature("security")

                if security:
                    url = await self.feature._ticket_creator.create_ticket_from_insight(
                        insight=insight,
                        security_feature=security,
                        timeout=APPROVAL_TIMEOUT_SHORT,
                    )
                    if url:
                        tickets.append(url)

            except Exception as e:
                if is_execution_control_error(e):
                    raise
                logger.warning(f"Failed to create ticket: {e}")

        return tickets