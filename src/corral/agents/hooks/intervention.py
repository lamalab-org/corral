"""Intervention hook for injecting thoughts and actions into agent execution.

This module provides a hook that can inject pre-defined thoughts at the start of a task. (For example take reasoning from Claude and inject it into GPT).
This is useful for context engineerting scenarios such as the ones below or running ablations
- Guiding agents based on successful/failed trajectories
- Providing hints or constraints

"""

import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from corral.agents.hooks.core import HookCallback, HookContext
from corral.agents.utils import LiteLLMMessage, convert_outermost_triple_quotes


@dataclass
class Action:
    """Represents an action parsed from intervention text."""

    tool_name: str
    arguments: dict[str, Any]


def create_intervention_hook(
    intervention_map: dict[str, str], execute_tools: bool = False
) -> HookCallback:
    """Factory function to create an intervention hook.

    The intervention hook injects pre-defined thoughts at the start of a task.
    Optionally, it can also parse and execute tools found in the intervention text.

    Args:
        intervention_map: Map of task_id -> intervention thought text
        execute_tools: Whether to execute tools found in intervention text.
                      Only supported for ReActAgent. Default False.

    Returns:
        A hook callback that can be registered at BEFORE_TASK hook point

    Example:
        >>> from corral.agents.hooks import AgentHooks, HookPoint
        >>> hooks = AgentHooks()
        >>> intervention = create_intervention_hook(
        ...     intervention_map={"task1": "<thought>Consider X...</thought>"},
        ...     execute_tools=False
        ... )
        >>> hooks.register(HookPoint.BEFORE_TASK, intervention)
    """

    def intervention_hook(context: HookContext) -> None:
        """Inject intervention thought at the start of a task."""

        # Only intervene if we have an intervention for this task
        if context.task_id not in intervention_map:
            return

        intervention_thought = intervention_map[context.task_id]
        logger.info(f"Injecting intervention for task {context.task_id}")

        # Store in metadata for tracking
        context.metadata["intervention_applied"] = True
        context.metadata["intervention_thought"] = intervention_thought

        if execute_tools:
            # Add full intervention (with potential tool calls)
            _inject_with_tool_execution(context, intervention_thought)
        else:
            # Strip tool calls and add only thoughts
            _inject_thought_only(context, intervention_thought)

    return intervention_hook


def _inject_thought_only(context: HookContext, intervention_thought: str) -> None:
    """Inject intervention thought without executing tools.

    Args:
        context: The hook context
        intervention_thought: The intervention text to inject
    """

    # Detect agent type and format accordingly
    agent_type = context.agent.__class__.__name__

    if agent_type == "ReActAgent":
        # Strip action tags for ReAct
        thought_only = re.sub(
            r"<action>.*?</action>(?:\s*<action_input>.*?</action_input>)?",
            "",
            intervention_thought,
            flags=re.DOTALL,
        ).strip()

        # Wrap in thought tags if not already
        if not thought_only.startswith("<thought>"):
            thought_only = f"<thought>{thought_only}</thought>"

        context.messages.append(LiteLLMMessage(role="assistant", content=thought_only))
        logger.info(
            f"Injected thought-only intervention for ReActAgent (task {context.task_id})"
        )

    elif agent_type == "ToolCallingAgent":
        # For tool calling, just add as plain text
        context.messages.append(
            LiteLLMMessage(role="assistant", content=intervention_thought)
        )
        logger.info(
            f"Injected intervention for ToolCallingAgent (task {context.task_id})"
        )

    else:
        # Generic fallback
        context.messages.append(
            LiteLLMMessage(role="assistant", content=intervention_thought)
        )
        logger.warning(f"Unknown agent type {agent_type}, using generic intervention")


def _inject_with_tool_execution(
    context: HookContext, intervention_thought: str
) -> None:
    """Inject intervention and execute any tools found in it.

    Args:
        context: The hook context
        intervention_thought: The intervention text containing potential tool calls
    """

    agent_type = context.agent.__class__.__name__

    if agent_type == "ToolCallingAgent":
        # Tool calling agents don't support executing tools from text
        logger.warning(
            "execute_intervention_tools=True ignored for ToolCallingAgent, using thought-only"
        )
        _inject_thought_only(context, intervention_thought)
        return

    # Format intervention for ReActAgent
    if not intervention_thought.strip().startswith("<thought>"):
        intervention_message = f"<thought>{intervention_thought}</thought>"
    else:
        intervention_message = intervention_thought

    context.messages.append(
        LiteLLMMessage(role="assistant", content=intervention_message)
    )

    # Parse and execute actions (ReAct specific)
    if agent_type == "ReActAgent":
        actions = _parse_react_actions(intervention_message)

        if actions:
            logger.info(
                f"Executing {len(actions)} tool(s) from intervention for task {context.task_id}"
            )

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
                            role="user",
                            content=observation,
                            name=action.tool_name,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error executing intervention tool {action.tool_name}: {e}"
                    )
                    context.messages.append(
                        LiteLLMMessage(
                            role="user",
                            content=f"Error: Failed to execute {action.tool_name}: {e}",
                            name=action.tool_name,
                        )
                    )
        else:
            logger.info(f"No tools found in intervention for task {context.task_id}")


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
