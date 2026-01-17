"""Intervention hook for injecting thoughts and actions into agent execution.

This module provides a hook that can inject pre-defined thoughts at the start of a task.
(For example take reasoning from Claude and inject it into GPT).

This is useful for context engineering scenarios such as the ones below or running ablations:
- Guiding agents based on successful/failed trajectories
- Providing hints or constraints
"""

import json
import re

from loguru import logger

from corral.agents.hooks.adapter import HookAdapterRegistry
from corral.agents.hooks.core import CriticalHookError, HookCallback, HookContext
from corral.agents.schema import Action
from corral.agents.utils import LiteLLMMessage, convert_outermost_triple_quotes


def _default_intervention(context: HookContext, intervention: str, execute_tools: bool):
    """Generic intervention - just add as assistant message."""
    context.messages.append(LiteLLMMessage(role="assistant", content=intervention))
    if execute_tools:
        logger.warning(
            f"execute_tools not supported for {context.agent.__class__.__name__}"
        )


def _react_intervention(context: HookContext, intervention: str, execute_tools: bool):
    """ReAct-specific intervention with optional tool execution.

    When execute_tools=True, executes in interleaved mode: each thought-action pair
    is added as an assistant message, the tool is executed, and the observation is
    appended before moving to the next segment.

    Raises:
        CriticalHookError: If execute_tools=True and tool execution fails
    """
    # Strip action tags if not executing
    if not execute_tools:
        intervention = re.sub(
            r"<action>.*?</action>(?:\s*<action_input>.*?</action_input>)?",
            "",
            intervention,
            flags=re.DOTALL,
        ).strip()

        # Wrap in thought tags
        if not intervention.startswith("<thought>"):
            intervention = f"<thought>{intervention}</thought>"

        context.messages.append(LiteLLMMessage(role="assistant", content=intervention))
        return

    # Interleaved execution: split intervention into thought-action segments
    # Split on <thought> tags to get individual reasoning steps
    segments = re.split(r"(?=<thought>)", intervention)

    for _segment in segments:
        segment = _segment.strip()
        if not segment:
            continue

        # Append this thought-action segment as assistant message
        context.messages.append(LiteLLMMessage(role="assistant", content=segment))

        # Execute any tools in this segment
        actions = _parse_react_actions(segment)

        # Check if actions were intended but couldn't be parsed
        if not actions and "<action>" in segment:
            raise CriticalHookError(
                "Failed to parse intervention actions despite execute_tools=True. "
                f"Segment contains <action> tags but parsing failed: {segment[:200]}"
            )

        for action in actions:
            try:
                tool_response = context.interface.execute_tool(
                    context.task_id, action.tool_name, action.arguments
                )

                # Check if tool execution succeeded
                if not tool_response.success:
                    raise CriticalHookError(
                        f"Tool '{action.tool_name}' execution failed during intervention: "
                        f"{tool_response.error}"
                    )

                observation = f"Observation: {tool_response.result}"
                context.messages.append(
                    LiteLLMMessage(
                        role="user", content=observation, name=action.tool_name
                    )
                )
            except CriticalHookError:
                # Re-raise critical errors
                raise
            except Exception as e:
                # Unexpected errors during tool execution are critical
                raise CriticalHookError(
                    f"Unexpected error executing intervention tool '{action.tool_name}': {e}"
                ) from e


def _toolcalling_intervention(
    context: HookContext, intervention: str, execute_tools: bool
):
    """ToolCalling-specific intervention with thought injection.

    ToolCallingAgent uses OpenAI function calling format, so we inject the
    intervention thought as an assistant message. Tool execution during
    intervention is not supported since ToolCallingAgent expects tool calls
    in the structured API format (tool_calls field), not as text.

    Args:
        context: Hook context with agent state
        intervention: The thought/content to inject
        execute_tools: If True, warns that tool execution is not supported
    """
    # Add the intervention thought as assistant message
    context.messages.append(LiteLLMMessage(role="assistant", content=intervention))

    if execute_tools:
        logger.warning(
            "execute_tools not supported for ToolCallingAgent. "
            "ToolCallingAgent uses structured function calling API format, not text-based tool calls. "
            "Only the intervention thought has been injected."
        )


def _parse_react_actions(text: str) -> list[Action]:
    """Parse ReAct-style actions from text.

    Args:
        text: Text containing ReAct-style action tags

    Returns:
        List of parsed actions
    """

    action_matches = re.finditer(
        r"<action>(.*?)</action>(?:.*?<action_input>(.*?)</action_input>)?",
        text,
        re.DOTALL,
    )

    actions = []
    for action_match in action_matches:
        tool_name = action_match.group(1).strip()
        try:
            action_input = action_match.group(2)
            if action_input is None:
                continue

            action_input = action_input.strip()
            converted_input = convert_outermost_triple_quotes(action_input)
            converted_input = converted_input.replace("True", "true").replace(
                "False", "false"
            )

            arguments = json.loads(converted_input)
            actions.append(Action(tool_name=tool_name, arguments=arguments))
        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            logger.error(f"Failed to parse intervention action: {e}")

    return actions


_intervention_registry = HookAdapterRegistry(default_adapter=_default_intervention)
_intervention_registry.register("ReActAgent", _react_intervention)
_intervention_registry.register("ToolCallingAgent", _toolcalling_intervention)


def create_intervention_hook(
    intervention_map: dict[str, str], execute_tools: bool = False
) -> HookCallback:
    """Factory function to create an intervention hook."""

    def intervention_hook(context: HookContext) -> None:
        if context.task_id not in intervention_map:
            return

        intervention_thought = intervention_map[context.task_id]
        logger.info(f"Injecting intervention for task {context.task_id}")

        context.metadata["intervention_applied"] = True
        context.metadata["intervention_thought"] = intervention_thought

        # Delegate to agent-specific adapter
        _intervention_registry.execute(context, intervention_thought, execute_tools)

    return intervention_hook
