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

# One-time warning latch: if STOP fires with an unenriched HookInput we
# warn ONCE (not every turn) and then stay quiet. Distinguishes a
# misconfiguration ("running against a kestrel-sovereign that predates
# the #1269 STOP enrichment") from the normal "nothing to capture this
# turn" path.
_warned_unenriched_stop = False


# The only tools this hook exposes to the reflection LLM call. Filtering to
# this set keeps the model from firing unrelated subagent dispatches during
# what is supposed to be a cheap, bounded fact-capture step.
RESERVED_FACT_TOOL_NAMES: frozenset[str] = frozenset({
    "save_fact",
    "strategy_add_pattern",
    "strategy_add_blocker",
})


def _is_denied(result: Any) -> bool:
    """True when a tool result is a permission-denied / failed capture.

    Three shapes reach the counter and none of them wrote a fact, so none
    may be counted as an ``executed`` capture in observability:

    * the PRE_TOOL_USE blocking envelope ``{success: False, ...}`` that
      ``_execute_tool_with_hooks`` returns when a save is denied/queued;
    * a flat ToolResult dict with an error/denied ``status`` string;
    * a ``ToolResult`` OBJECT (e.g. ``save_fact`` returning
      ``ToolResult.failed(...)`` when the knowledge graph is unavailable),
      whose ``status`` is a ``ToolResultStatus`` enum, not a dict key.

    PARTIAL is treated as a capture — per the sovereign wrapper it still ran
    the action.
    """
    if isinstance(result, dict):
        if result.get("success") is False:
            return True
        status = result.get("status")
        return isinstance(status, str) and status.lower() in ("error", "denied")
    # ToolResult (or any status-bearing object): a failed/denied status means
    # nothing was written.
    status_obj = getattr(result, "status", None)
    if status_obj is None:
        return False
    status_name = (getattr(status_obj, "name", None) or str(status_obj)).upper()
    return status_name in ("ERROR", "DENIED", "FAILED")


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
            if isinstance(res, dict):
                # #1269 wraps each as {tool_call_id, name, arguments,
                # result}. ``result`` is the audit summary, itself often a
                # ToolResult.to_dict() envelope (status/confirmation/data/
                # error) — those carry the actual facts a tool surfaced
                # (a discovered path, an observed failure). Only rendering
                # res["result"] loses everything when the payload lives in
                # sibling keys or when the list item IS the envelope (no
                # "result" wrapper). Render res["result"] when present and
                # non-empty, else the whole envelope, so tool-learned facts
                # always reach the reflection LLM (codex review).
                payload = res.get("result")
                if payload in (None, {}, [], ""):
                    payload = {
                        k: v for k, v in res.items()
                        if k not in ("tool_call_id", "name", "arguments")
                    } or res
                res_repr = json.dumps(payload, default=str)[:1200]
            else:
                res_repr = str(res)[:1200]
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

        # Source the three fact tools straight from their owning feature
        # objects rather than ``_build_all_tools()``. On a fresh agent
        # ``_build_all_tools()`` only carries feature *dispatcher* schemas
        # plus already-promoted direct tools — ``save_fact`` /
        # ``strategy_add_*`` are sub-tools that may not appear until an
        # unrelated prior interaction happens to expose them, which would
        # make this hook silently inactive in the default session (codex
        # review). Walking ``agent.features`` is exposure-state-independent.
        fact_tool_objs: Dict[str, Any] = {}
        fact_tool_feature: Dict[str, str] = {}
        for feature_name, feature in (getattr(agent, "features", {}) or {}).items():
            get_tools = getattr(feature, "get_tools", None)
            if not callable(get_tools):
                continue
            try:
                tools = get_tools()
            except Exception:
                continue
            for t in tools or []:
                tname = getattr(t, "name", None)
                if tname in RESERVED_FACT_TOOL_NAMES and tname not in fact_tool_objs:
                    fact_tool_objs[tname] = t
                    fact_tool_feature[tname] = feature_name
        if not fact_tool_objs:
            # Agent doesn't have the memory/strategy features loaded.
            return

        try:
            fact_tools = [
                t.schema.to_openai_format() for t in fact_tool_objs.values()
            ]
        except Exception as exc:
            logger.debug(f"[per-turn-reflection] tool schema build failed: {exc}")
            return

        # In the enriched world (kestrel-sovereign #1269) a completed turn
        # ALWAYS carries at least user_message + response_text. If every
        # #1269 field is absent, this STOP came from a sovereign runtime
        # that predates the enrichment — the hook can't function there.
        # That's a deployment misconfiguration, not "nothing to capture",
        # so surface it loudly ONCE instead of silently no-opping forever.
        enriched = any((
            input.user_message,
            input.response_text,
            input.tool_calls,
            input.tool_results,
        ))
        if not enriched:
            global _warned_unenriched_stop
            if not _warned_unenriched_stop:
                _warned_unenriched_stop = True
                logger.warning(
                    "[per-turn-reflection] STOP HookInput has no turn context "
                    "(user_message/response_text/tool_calls/tool_results all "
                    "empty). This kestrel-sovereign runtime predates the "
                    "#1269 STOP enrichment, so per-turn fact-capture is "
                    "INACTIVE. Upgrade to a kestrel-sovereign build that "
                    "includes #1269 (post-0.11.0). This warning logs once "
                    "per process."
                )
            return

        transcript = format_turn_transcript(input)
        if not transcript.strip():
            return

        messages = [
            {"role": "system", "content": PER_TURN_REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]

        # CRITICAL: do NOT reuse the user turn's session_id. Stateful /
        # continuation-backed providers (e.g. CodexAdapter) anchor
        # continuation state (previous_response_id) on session_id — passing
        # the user's id here would overwrite the conversation's cursor with
        # the reflection prompt + fact tools, corrupting the next
        # user-facing turn. Namespace a distinct id so reflection calls are
        # isolated from the user conversation entirely.
        reflection_session_id = (
            f"per-turn-reflection::{input.session_id}"
            if input.session_id
            else "per-turn-reflection"
        )

        # ``_execute_tool_with_hooks`` is the single hook-enforced entry
        # point (PRE/POST_TOOL_USE: permissions, audit) and works for
        # sub-tools the orchestrator batch path can't see on a fresh agent.
        exec_with_hooks = getattr(agent, "_execute_tool_with_hooks", None)

        def _dispatch(tname: str, call_args: Dict[str, Any]):
            """Run one fact tool through the hook-enforced path.

            ``_execute_tool_with_hooks``'s ``execute_fn`` contract CHANGED
            across the supported sovereign range and the executor must work
            under both (this package supports ``kestrel-sovereign>=0.28.0``):

            * sovereign < #1866 (e.g. 0.28.0) calls ``execute_fn()`` with no
              arguments — PRE_TOOL_USE MODIFY rewrites were dropped there.
            * sovereign >= #1866 calls ``execute_fn(args)`` with the post-hook
              args as a single positional, so MODIFY rewrites are honored.

            A fixed-arity lambda breaks on one side or the other — the prior
            zero-arity form raised ``TypeError`` (swallowed) on new sovereign,
            silently disabling ALL per-turn fact capture (F365). Accept the
            optional positional: use the post-hook args when passed, else fall
            back to the pre-hook ``call_args`` (exactly the old behavior).
            """
            if callable(exec_with_hooks):
                return exec_with_hooks(
                    tname,
                    fact_tool_feature.get(tname, "ReflectionFeature"),
                    call_args,
                    reflection_session_id,
                    lambda *a, _t=fact_tool_objs[tname], _fallback=call_args: (
                        _t.execute(
                            **(a[0] if a and isinstance(a[0], dict) else _fallback)
                        )
                    ),
                )
            # Older sovereign without the helper — direct execute. Fact-save
            # tools are low-risk memory writes; degrade rather than skip.
            return fact_tool_objs[tname].execute(**call_args)

        # Stateful routes (openai:plan / codex app-server) REJECT a
        # ``generate_with_messages`` that advertises tools without a
        # ``tool_executor`` — the codex adapter raises
        # ``"openai:plan ... requires a tool_executor"``, which the hook was
        # swallowing, so plan-route agents captured ZERO facts indefinitely
        # (F047). Supply an executor restricted to the reserved fact tools:
        # on stateful routes codex runs it inline and returns the calls as
        # ``executed_tool_calls``; stateless routes ignore it and return
        # ``tool_calls`` for the post-hoc loop below.
        async def _fact_tool_executor(name: str, call_args: Any):
            if name not in fact_tool_objs:
                # The reflection turn must never execute arbitrary tools.
                return {
                    "success": False,
                    "error": f"tool '{name}' not permitted in reflection",
                }
            safe_args = call_args if isinstance(call_args, dict) else {}
            return await _dispatch(name, safe_args)

        start = time.monotonic()
        response = await llm_service.generate_with_messages(
            messages=messages,
            tools=fact_tools,
            force_local_only=False,
            session_id=reflection_session_id,
            tool_executor=_fact_tool_executor,
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        # Calls codex already ran inline on a stateful route. The adapter
        # attaches them as ``executed_tool_calls`` (id/name/arguments/result);
        # they must NOT be re-dispatched below or facts double-save (F047).
        inline_executed = getattr(response, "executed_tool_calls", None) or []
        inline_ids = {
            e.get("id") for e in inline_executed
            if isinstance(e, dict) and e.get("id")
        }
        executed = sum(
            1 for e in inline_executed
            if isinstance(e, dict) and not _is_denied(e.get("result"))
        )

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls and not inline_executed:
            await self._log(agent, duration_ms, 0, input.session_id, success=True)
            return

        # Post-hoc dispatch for stateless routes: a single round, no follow-up
        # LLM turn — we capture, we don't loop. Skip any call already executed
        # inline (belt-and-suspenders for a route that returns both).
        for tc in tool_calls:
            tname = getattr(tc, "name", None)
            if tname not in fact_tool_objs:
                continue  # model emitted something outside the allowed set
            tc_id = getattr(tc, "id", None)
            if tc_id and tc_id in inline_ids:
                continue  # already executed inline on the stateful route
            raw_args = getattr(tc, "arguments", {})
            args = raw_args if isinstance(raw_args, dict) else {}
            try:
                result = await _dispatch(tname, args)
                if not _is_denied(result):
                    executed += 1
            except Exception as exc:
                logger.warning(
                    f"[per-turn-reflection] fact tool '{tname}' failed: {exc}"
                )
        await self._log(
            agent, duration_ms, executed, input.session_id, success=True
        )

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
