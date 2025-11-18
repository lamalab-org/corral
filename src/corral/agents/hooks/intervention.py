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

from corral.agents.base_agent import Action
from corral.agents.hooks.adapter import HookAdapterRegistry
from corral.agents.hooks.core import HookCallback, HookContext
from corral.agents.utils import LiteLLMMessage, convert_outermost_triple_quotes


def _default_intervention(context: HookContext, intervention: str, execute_tools: bool):
    """Generic intervention - just add as assistant message."""
    context.messages.append(LiteLLMMessage(role="assistant", content=intervention))
    if execute_tools:
        logger.warning(
            f"execute_tools not supported for {context.agent.__class__.__name__}"
        )


def _react_intervention(context: HookContext, intervention: str, execute_tools: bool):
    """ReAct-specific intervention with optional tool execution."""
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

    # Execute tools if requested
    if execute_tools:
        actions = _parse_react_actions(intervention)
        for action in actions:
            try:
                tool_response = context.interface.execute_tool(
                    context.task_id, action.tool_name, action.arguments
                )
                observation = (
                    f"Observation: {tool_response.result}"
                    if tool_response.success
                    else f"Error: {tool_response.error}"
                )
                context.messages.append(
                    LiteLLMMessage(
                        role="user", content=observation, name=action.tool_name
                    )
                )
            except Exception as e:
                logger.error(f"Error executing tool {action.tool_name}: {e}")


def _toolcalling_intervention(
    context: HookContext, intervention: str, execute_tools: bool
):
    """ToolCalling-specific intervention."""
    context.messages.append(LiteLLMMessage(role="assistant", content=intervention))
    if execute_tools:
        logger.warning("execute_tools not supported for ToolCallingAgent")


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
