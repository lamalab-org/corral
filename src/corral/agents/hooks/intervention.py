"""State-backed interventions for the current async agent-session API."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from corral.agents.hooks.core import CriticalHookError, HookCallback, HookContext
from corral.agents.utils import convert_outermost_triple_quotes
from corral.core.action import Action


def _rewrite_workspace_paths(
    steps: Sequence[Mapping[str, Any]],
    old_workspace: str | None,
    new_workspace: str | None,
) -> list[dict[str, Any]]:
    """Return detached trace steps whose paths target the current workspace."""
    rewritten = json.loads(json.dumps(list(steps)))
    if not old_workspace or not new_workspace or old_workspace == new_workspace:
        return rewritten

    logger.info(f"Rewriting trace paths: {old_workspace} -> {new_workspace}")
    for step in rewritten:
        content = step.get("content")
        if isinstance(content, str):
            step["content"] = content.replace(old_workspace, new_workspace)
        for tool_call in step.get("tool_calls", ()):
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                function["arguments"] = arguments.replace(old_workspace, new_workspace)
            elif isinstance(arguments, dict):
                encoded = json.dumps(arguments, ensure_ascii=False)
                function["arguments"] = json.loads(
                    encoded.replace(old_workspace, new_workspace)
                )
    return rewritten


def _detect_workspace_from_trace(messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Best-effort extraction of a Corral workspace path from a saved trace."""
    pattern = re.compile(r"(/results/[^/\s]+/[^/\s]+_exec_[^/\s]+)/")
    for message in messages:
        if message.get("role") != "assistant":
            continue
        values: list[str] = []
        content = message.get("content")
        if isinstance(content, str):
            values.append(content)
        for tool_call in message.get("tool_calls", ()):
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get("function")
            if not isinstance(function, Mapping):
                continue
            arguments = function.get("arguments")
            values.append(
                arguments
                if isinstance(arguments, str)
                else json.dumps(arguments, ensure_ascii=False)
            )
        for value in values:
            match = pattern.search(value)
            if match:
                return match.group(1)
    return None


def _load_trace_steps(
    trace_path: str,
    num_steps: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """Load the selected assistant decisions and their source workspace."""
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    raw_messages = trace.get("messages", ())
    messages = [item for item in raw_messages if isinstance(item, Mapping)]
    workspace = _detect_workspace_from_trace(messages)
    assistant = [dict(item) for item in messages if item.get("role") == "assistant"]
    if not assistant:
        logger.warning(f"No assistant messages found in trace: {trace_path}")
        return [], workspace
    if num_steps >= 0:
        return assistant[:num_steps], workspace
    if abs(num_steps) >= len(assistant):
        logger.warning(
            f"num_steps={num_steps} would exclude all {len(assistant)} messages"
        )
        return [], workspace
    return assistant[:num_steps], workspace


def _parse_react_actions(text: str) -> list[Action]:
    """Parse ReAct XML-like actions into current immutable ``Action`` values."""
    actions: list[Action] = []
    matches = re.finditer(
        r"<action>(.*?)</action>(?:.*?<action_input>(.*?)</action_input>)?",
        text,
        re.DOTALL,
    )
    for match in matches:
        raw_arguments = match.group(2)
        if raw_arguments is None:
            continue
        try:
            converted = convert_outermost_triple_quotes(raw_arguments.strip())
            converted = converted.replace("True", "true").replace("False", "false")
            arguments = json.loads(converted)
            if not isinstance(arguments, dict):
                raise ValueError("action_input must be a JSON object")
            actions.append(Action(name=match.group(1).strip(), arguments=arguments))
        except (json.JSONDecodeError, TypeError, ValueError, SyntaxError) as exc:
            logger.error(f"Failed to parse intervention action: {exc}")
    return actions


def _strip_react_actions(text: str) -> str:
    return re.sub(
        r"<action>.*?</action>(?:\s*<action_input>.*?</action_input>)?",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


async def _execute(
    context: HookContext,
    action: Action,
    *,
    content: str | None = None,
) -> None:
    current = Action(
        name=action.name,
        arguments=action.arguments,
        content=content,
        metadata={"source": "hook-intervention"},
    )
    try:
        response = await context.session.execute(current)
    except Exception as exc:
        raise CriticalHookError(
            f"Unexpected error executing intervention tool {action.name!r}: {exc}"
        ) from exc
    if not response.success:
        raise CriticalHookError(
            f"Tool {action.name!r} execution failed during intervention: "
            f"{response.error}"
        )


async def _inject_react_text(
    context: HookContext,
    text: str,
    *,
    execute_tools: bool,
    wrap_thought: bool,
) -> None:
    """Inject ReAct text, recording every effect in the session State."""
    if not execute_tools:
        content = _strip_react_actions(text)
        if wrap_thought and content and not content.startswith("<thought>"):
            content = f"<thought>{content}</thought>"
        if content:
            await context.session.record_message(
                {"role": "assistant", "content": content}
            )
        return

    segments = [
        item.strip() for item in re.split(r"(?=<thought>)", text) if item.strip()
    ]
    for segment in segments:
        actions = _parse_react_actions(segment)
        if not actions and "<action>" in segment:
            raise CriticalHookError(
                "Failed to parse intervention actions despite execute_tools=True: "
                f"{segment[:200]}"
            )
        if not actions:
            await context.session.record_message(
                {"role": "assistant", "content": segment}
            )
            continue
        for index, action in enumerate(actions):
            await _execute(
                context,
                action,
                content=segment if index == 0 else None,
            )


async def _inject_tool_calls(
    context: HookContext,
    step: Mapping[str, Any],
    *,
    execute_tools: bool,
) -> None:
    """Inject one provider-native trace step through canonical Actions."""
    content = step.get("content")
    rendered_content = content if isinstance(content, str) else None
    raw_calls = step.get("tool_calls")
    calls = raw_calls if isinstance(raw_calls, list | tuple) else ()
    if not execute_tools:
        if rendered_content:
            await context.session.record_message(
                {"role": "assistant", "content": rendered_content}
            )
        return

    for index, raw_call in enumerate(calls):
        if not isinstance(raw_call, Mapping):
            raise CriticalHookError("trace intervention contains an invalid tool call")
        try:
            action = Action.from_tool_call(raw_call)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CriticalHookError(
                f"Failed to parse trace intervention tool call: {exc}"
            ) from exc
        await _execute(
            context,
            action,
            content=rendered_content if index == 0 else None,
        )


def create_intervention_hook(
    intervention_map: Mapping[str, str],
    execute_tools: bool = False,
) -> HookCallback:
    """Create a BEFORE_TASK-compatible State-backed intervention hook."""

    async def intervention_hook(context: HookContext) -> None:
        intervention = intervention_map.get(context.task_id)
        if intervention is None:
            return
        logger.info(f"Injecting intervention for task {context.task_id}")
        context.metadata["intervention_applied"] = True
        context.metadata["intervention_thought"] = intervention
        is_react = type(context.agent).__name__ == "ReActAgent"
        has_actions = "<action>" in intervention
        if is_react or has_actions:
            await _inject_react_text(
                context,
                intervention,
                execute_tools=execute_tools,
                wrap_thought=is_react,
            )
            return
        await context.session.record_message(
            {"role": "assistant", "content": intervention}
        )
        if execute_tools:
            logger.warning(
                "execute_tools=True had no structured action to execute in the "
                f"intervention for {type(context.agent).__name__}"
            )

    return intervention_hook


def create_trace_intervention_hook(
    trace_pool: Mapping[str, str | Sequence[str]],
    num_steps: int,
    execute_tools: bool = False,
) -> HookCallback:
    """Create an intervention that replays selected trace decisions via State."""

    def eligible_traces(pool: str | Sequence[str]) -> list[str]:
        paths = [pool] if isinstance(pool, str) else list(pool)
        eligible: list[str] = []
        for path in paths:
            trace = json.loads(Path(path).read_text(encoding="utf-8"))
            count = sum(
                1
                for message in trace.get("messages", ())
                if isinstance(message, Mapping) and message.get("role") == "assistant"
            )
            if (num_steps >= 0 and count >= num_steps) or (
                num_steps < 0 and count > abs(num_steps)
            ):
                eligible.append(path)
        return eligible

    async def trace_intervention_hook(context: HookContext) -> None:
        pool = trace_pool.get(context.task_id)
        if pool is None:
            return
        eligible = eligible_traces(pool)
        if not eligible:
            pool_size = 1 if isinstance(pool, str) else len(pool)
            logger.warning(
                f"No traces with enough steps for task {context.task_id} "
                f"(num_steps={num_steps}, pool_size={pool_size})"
            )
            return

        trace_path = random.choice(eligible)
        steps, trace_workspace = _load_trace_steps(trace_path, num_steps)
        if not steps:
            return
        steps = _rewrite_workspace_paths(
            steps,
            trace_workspace,
            context.session.workspace,
        )
        context.metadata["intervention_applied"] = True
        context.metadata["intervention_num_steps"] = len(steps)
        context.metadata["intervention_trace_path"] = trace_path

        for index, step in enumerate(steps, start=1):
            logger.debug(f"Injecting trace step {index}/{len(steps)}")
            if step.get("tool_calls"):
                await _inject_tool_calls(
                    context,
                    step,
                    execute_tools=execute_tools,
                )
            else:
                content = step.get("content")
                await _inject_react_text(
                    context,
                    content if isinstance(content, str) else "",
                    execute_tools=execute_tools,
                    wrap_thought=False,
                )

    return trace_intervention_hook


__all__ = [
    "create_intervention_hook",
    "create_trace_intervention_hook",
]
