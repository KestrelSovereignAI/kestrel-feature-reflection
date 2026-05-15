"""
Per-turn fact-capture hook (kestrel-sovereign #1238, consumer half).

The sleep-cycle reflection (``ReflectionSleepHook``) runs every ~4 hours
and produces ``Insight`` objects through the gather → analyze → propose →
review pipeline. That's the right cadence for behavioral meta-reflection,
but it's too slow for *structural facts the agent learns mid-conversation*
— a package was renamed, a tool lives at a non-obvious path, a peer
agent uses a specific convention, a bug was observed. Those facts decay
out of working context before the cron fires.

This hook closes that gap with a tighter feedback loop: after every turn
(``HookEvent.STOP``), it issues one LLM call that asks the model what
structural facts it learned this turn and lets it persist them via the
three fact-save tools (``save_fact``, ``strategy_add_pattern``,
``strategy_add_blocker``). Zero saves is a valid outcome.

It reads the full turn context directly off ``HookInput`` —
``user_message``, ``response_text``, ``tool_calls``, ``tool_results`` —
populated by kestrel-sovereign #1269. No storage round-trip.

Design invariants:
- **Fail-isolated.** STOP is post-yield; a reflection failure must never
  surface to the user or break the turn. Every exit returns
  ``HookOutput.allow()``.
- **Single round.** One LLM call, one batch of tool calls, no follow-up
  LLM round. We capture facts, we don't start a sub-agent loop.
- **Opt-out.** ``KESTREL_PER_TURN_REFLECTION_DISABLED=1`` disables it.
  Default on.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput

logger = logging.getLogger(__name__)


# The only tools this hook exposes to the reflection LLM call. Filtering to
# this set keeps the model from firing unrelated subagent dispatches during
# what is supposed to be a cheap, bounded fact-capture step.
RESERVED_FACT_TOOL_NAMES: frozenset[str] = frozenset({
    "save_fact",
    "strategy_add_pattern",
    "strategy_add_blocker",
})


PER_TURN_REFLECTION_SYSTEM_PROMPT = """You just finished a turn. Before moving on, take one structured moment to capture what you learned.

Look at the transcript below and ask:
1. What structural facts did this turn surface that are worth persisting beyond this conversation? (A file/package was renamed; a tool lives at a non-obvious path; a previously-broken assumption was corrected; a peer agent uses a specific convention; a bug was observed.)
2. What patterns or failure modes did you observe? (A category of bug; a repeating user preference; a workflow that consistently fails.)

For each fact worth persisting, call `save_fact` (subject/predicate/value/confidence).
For each pattern worth recording, call `strategy_add_pattern`.
For each open blocker the user surfaced, call `strategy_add_blocker`.

Rules:
- One tool call per distinct fact/pattern/blocker. Only persist what would be useful to a future you, in a future conversation, with no access to this transcript.
- If nothing structural was learned, emit no tool calls and return an empty response. That is a valid, common outcome — do not invent facts.
- Do not narrate or address the user. Output text is discarded; only tool calls have effect.
- Confidence: 1.0 for things you directly verified, 0.7-0.9 for strong inference, lower for guesses.
"""


def per_turn_reflection_disabled() -> bool:
    """True when the per-turn fact-capture is globally disabled via env."""
    val = os.environ.get("KESTREL_PER_TURN_REFLECTION_DISABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _tool_schema_name(tool: Dict[str, Any]) -> Optional[str]:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(fn, dict):
        return fn.get("name")
    return tool.get("name") if isinstance(tool, dict) else None


def filter_fact_tools(all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the three fact-save tool schemas."""
    return [
        t for t in (all_tools or [])
        if _tool_schema_name(t) in RESERVED_FACT_TOOL_NAMES
    ]


def format_turn_transcript(
    hook_input: HookInput,
    *,
    max_chars: int = 12_000,
) -> str:
    """Build a compact transcript from the STOP HookInput.

    #1269 populates ``user_message``, ``response_text``, ``tool_calls``
    and ``tool_results`` so we never have to query storage here.
    ``tool_calls[i]`` and ``tool_results[i]`` are aligned by index in
    normal flows; on a streaming cancel before dispatch ``tool_results``
    can be empty while ``tool_calls`` still carries what the LLM emitted.
    """
    lines: List[str] = []

    if hook_input.user_message:
        lines.append(f"[user] {hook_input.user_message[:2000]}")

    tool_calls = hook_input.tool_calls or []
    tool_results = hook_input.tool_results or []
    for i, call in enumerate(tool_calls):
        name = call.get("name", "?") if isinstance(call, dict) else "?"
        args = call.get("arguments") if isinstance(call, dict) else None
        if isinstance(args, (dict, list)):
            arg_repr = json.dumps(args, default=str)[:300]
        else:
            arg_repr = str(args)[:300] if args is not None else ""
        lines.append(f"[tool-call] {name}({arg_repr})")
        if i < len(tool_results):
            res = tool_results[i]
            res_repr = (
                json.dumps(res.get("result"), default=str)[:1200]
                if isinstance(res, dict)
                else str(res)[:1200]
            )
            lines.append(f"[tool-result] {res_repr}")

    if hook_input.response_text:
        lines.append(f"[assistant final] {hook_input.response_text[:2500]}")

    transcript = "\n\n".join(lines)
    if len(transcript) > max_chars:
        transcript = "[transcript truncated]\n" + transcript[-max_chars:]
    return transcript


class OnStopReflectionHook(Hook):
    """SDK ``HookEvent.STOP`` hook that captures per-turn structural facts.

    Holds a reference to the agent (via the reflection feature) so it can
    reuse the orchestrator's tool-build + dispatch helpers — the same path
    a user-driven ``save_fact`` takes, so PRE/POST_TOOL_USE hooks and
    observability fire identically.
    """

    def __init__(self, agent):
        super().__init__(
            name="per_turn_reflection",
            events=[HookEvent.STOP],
            # Run late — after any audit/security STOP hooks. This is
            # bookkeeping, not a gate.
            priority=900,
            timeout=30.0,
        )
        self.agent = agent

    async def execute(self, input: HookInput) -> HookOutput:  # noqa: A002 (SDK contract)
        """Capture facts learned this turn. Always returns allow()."""
        try:
            await self._capture(input)
        except Exception as exc:  # never break the turn — STOP is post-yield
            logger.warning(
                f"[per-turn-reflection] swallowed error (turn unaffected): {exc}",
                exc_info=True,
            )
        return HookOutput.allow("per-turn reflection complete")

    async def _capture(self, input: HookInput) -> None:
        if per_turn_reflection_disabled():
            return

        agent = self.agent
        if agent is None:
            return

        llm_service = getattr(agent, "llm_service", None)
        if llm_service is None or getattr(llm_service, "disabled", False):
            # No LLM (or PayerKind.NONE) — nothing to reflect with.
            return

        build_all_tools = getattr(agent, "_build_all_tools", None)
        if not callable(build_all_tools):
            return
        fact_tools = filter_fact_tools(build_all_tools())
        if not fact_tools:
            # Agent doesn't have the memory/strategy features loaded.
            return

        transcript = format_turn_transcript(input)
        if not transcript.strip():
            return

        messages = [
            {"role": "system", "content": PER_TURN_REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]

        start = time.monotonic()
        response = await llm_service.generate_with_messages(
            messages=messages,
            tools=fact_tools,
            force_local_only=False,
            session_id=input.session_id or None,
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        tool_calls = getattr(response, "tool_calls", None)
        has_tool_calls = bool(tool_calls)

        if not has_tool_calls:
            await self._log(agent, duration_ms, 0, input.session_id, success=True)
            return

        # Single round of dispatch through the orchestrator's batch path so
        # PRE/POST_TOOL_USE hooks + observability fire exactly as they would
        # for a user-driven save. No follow-up LLM round.
        features_by_tool_name = self._safe_call(agent, "_visible_features_by_tool_name", {})
        known_tools = self._safe_call(agent, "_visible_known_tool_names", set())
        build_tool_calls_msg = getattr(agent, "_build_tool_calls_msg", None)
        execute_tool_batch = getattr(agent, "_execute_tool_batch", None)
        if not callable(execute_tool_batch) or not callable(build_tool_calls_msg):
            await self._log(agent, duration_ms, 0, input.session_id, success=False,
                            error="agent missing tool-dispatch helpers")
            return

        reflection_messages = list(messages)
        reflection_messages.append({
            "role": "assistant",
            "content": getattr(response, "content", "") or "",
            "tool_calls": build_tool_calls_msg(tool_calls),
        })
        await execute_tool_batch(
            tool_calls,
            features_by_tool_name,
            known_tools,
            reflection_messages,
            0,
            None,
            session_id=input.session_id or "per_turn_reflection",
        )
        await self._log(
            agent, duration_ms, len(tool_calls), input.session_id, success=True
        )

    @staticmethod
    def _safe_call(agent, attr: str, default):
        fn = getattr(agent, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return default
        return default

    @staticmethod
    async def _log(
        agent,
        duration_ms: int,
        tool_calls_count: int,
        session_id: Optional[str],
        *,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Best-effort observability — mirrors the #1239 instrumentation
        pattern so per-turn reflection cost/latency is measurable
        separately from normal turns."""
        store = getattr(agent, "observability_store", None)
        if store is None:
            return
        log_llm_call = getattr(store, "log_llm_call", None)
        if not callable(log_llm_call):
            return
        try:
            await log_llm_call(
                provider="reflection",
                model="per_turn_reflection",
                duration_ms=duration_ms,
                success=success,
                session_id=session_id,
                error_message=error,
                metadata={
                    "phase": "per_turn",
                    "tool_calls_count": tool_calls_count,
                },
                agent_did=getattr(agent, "did", None),
            )
        except Exception as exc:
            logger.debug(f"[per-turn-reflection] observability log failed: {exc}")


def create_on_stop_reflection_hook(agent) -> Optional[OnStopReflectionHook]:
    """Factory mirroring ``create_reflection_hook``. Returns None when the
    agent can't support the hook (no llm_service)."""
    if agent is None or not hasattr(agent, "llm_service"):
        return None
    return OnStopReflectionHook(agent)
