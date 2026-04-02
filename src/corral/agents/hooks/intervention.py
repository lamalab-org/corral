"""Intervention hook for injecting thoughts and actions into agent execution.

This module provides a hook that can inject pre-defined thoughts at the start of a task.
(For example take reasoning from Claude and inject it into GPT).

This is useful for context engineering scenarios such as the ones below or running ablations:
- Guiding agents based on successful/failed trajectories
- Providing hints or constraints
"""

import json
import re
from pathlib import Path

from loguru import logger

from corral.agents.hooks.adapter import HookAdapterRegistry
from corral.agents.hooks.core import CriticalHookError, HookCallback, HookContext
from corral.agents.schema import Action
from corral.agents.utils import LiteLLMMessage, convert_outermost_triple_quotes


def _load_trace_steps(trace_path: str, num_steps: int) -> list[dict]:
    """Load trace file and extract assistant messages.

    Args:
        trace_path: Path to trace JSON file
        num_steps: Number of steps to extract
            - Positive: first N assistant messages
            - -1: all except last assistant message
            - -2: all except last 2 assistant messages
            - etc.

    Returns:
        List of assistant message dicts (full message including tool_calls if present)
    """
    with Path(trace_path).open() as f:
        trace = json.load(f)

    messages = trace.get("messages", [])

    # Extract all assistant messages (full dict, not just content)
    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]

    if not assistant_messages:
        logger.warning(f"No assistant messages found in trace: {trace_path}")
        return []

    # Handle num_steps
    if num_steps >= 0:
        return assistant_messages[:num_steps]
    else:
        # -1 means all except last 1, -2 means all except last 2, etc.
        # Python slicing: lst[:-1] gives all except last, lst[:-2] all except last 2
        if abs(num_steps) >= len(assistant_messages):
            logger.warning(
                f"num_steps={num_steps} would exclude all {len(assistant_messages)} messages"
            )
            return []
        return assistant_messages[:num_steps]


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


def _inject_react_step(
    context: HookContext, step_msg: dict, execute_tools: bool
) -> None:
    """Inject a ReActAgent-style step (text-based actions).

    Args:
        context: Hook context
        step_msg: Full assistant message dict from trace
        execute_tools: Whether to execute tools
    """
    content = step_msg.get("content", "")

    if execute_tools:
        # Inject assistant message exactly as-is
        context.messages.append(LiteLLMMessage(role="assistant", content=content))

        # Parse and execute any tools in this step
        actions = _parse_react_actions(content)

        for action in actions:
            try:
                tool_response = context.interface.execute_tool(
                    context.task_id, action.tool_name, action.arguments
                )

                if not tool_response.success:
                    raise CriticalHookError(
                        f"Tool '{action.tool_name}' execution failed during "
                        f"intervention: {tool_response.error}"
                    )

                observation = f"Observation: {tool_response.result}"
                context.messages.append(
                    LiteLLMMessage(
                        role="user", content=observation, name=action.tool_name
                    )
                )
            except CriticalHookError:
                raise
            except Exception as e:
                raise CriticalHookError(
                    f"Unexpected error executing intervention tool "
                    f"'{action.tool_name}': {e}"
                ) from e
    else:
        # Strip action tags but do NOT wrap with thought tags
        stripped_content = re.sub(
            r"<action>.*?</action>(?:\s*<action_input>.*?</action_input>)?",
            "",
            content,
            flags=re.DOTALL,
        ).strip()

        context.messages.append(
            LiteLLMMessage(role="assistant", content=stripped_content)
        )


def _inject_toolcalling_step(
    context: HookContext, step_msg: dict, execute_tools: bool
) -> None:
    """Inject a ToolCallingAgent-style step (structured tool_calls).

    Args:
        context: Hook context
        step_msg: Full assistant message dict from trace (with tool_calls)
        execute_tools: Whether to execute tools
    """
    content = step_msg.get("content", "")
    tool_calls = step_msg.get("tool_calls", [])

    if execute_tools and tool_calls:
        # Format tool_calls for OpenAI-compatible API (ensure type field is present)
        # Traces may store tool_calls without the required 'type' field
        formatted_tool_calls = []
        for tool_call in tool_calls:
            formatted_call = {
                "id": tool_call.get("id"),
                "type": "function",  # Required by OpenAI API
                "function": tool_call.get("function", {}),
            }
            formatted_tool_calls.append(formatted_call)

        # Build the assistant message with properly formatted tool_calls
        assistant_msg = {
            "role": "assistant",
            "content": content,
            "tool_calls": formatted_tool_calls,
        }
        context.messages.append(assistant_msg)

        # Execute each tool call and append tool responses
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            function_info = tool_call.get("function", {})
            function_name = function_info.get("name", "")
            arguments_str = function_info.get("arguments", "{}")

            try:
                # Parse arguments (stored as JSON string in trace)
                arguments = json.loads(arguments_str)

                # Execute tool
                tool_response = context.interface.execute_tool(
                    context.task_id, function_name, arguments
                )

                if not tool_response.success:
                    raise CriticalHookError(
                        f"Tool '{function_name}' execution failed during "
                        f"intervention: {tool_response.error}"
                    )

                # Append tool response in OpenAI format
                context.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": str(tool_response.result),
                        "name": function_name,
                    }
                )
            except json.JSONDecodeError as e:
                raise CriticalHookError(
                    f"Failed to parse tool arguments for '{function_name}': {e}"
                ) from e
            except CriticalHookError:
                raise
            except Exception as e:
                raise CriticalHookError(
                    f"Unexpected error executing intervention tool "
                    f"'{function_name}': {e}"
                ) from e
    else:
        # Just inject the content (thought) without tool_calls
        context.messages.append(LiteLLMMessage(role="assistant", content=content))


def create_trace_intervention_hook(
    trace_pool: dict[str, str] | dict[str, list[str]],
    num_steps: int,
    execute_tools: bool = False,
) -> HookCallback:
    """Factory function to create an intervention hook from trace files.

    This loads assistant messages from recorded trace files and injects them
    as interventions. Messages are injected exactly as they appear in the trace
    (no wrapping with thought tags).

    Supports both:
    - ReActAgent traces: text-based <action>/<action_input> format
    - ToolCallingAgent traces: structured tool_calls format

    Args:
        trace_pool: Mapping of task_id to either:
            - a single trace file path (str) for backward compatibility
            - a list of trace file paths to sample from per trial
        num_steps: Number of assistant message steps to inject
            - Positive: first N steps (e.g., 1, 2, 3)
            - -1: all steps except the last one
            - -2: all steps except the last two
        execute_tools: Whether to execute tools found in the intervention
            - True: inject message as-is and execute any tools, observations added automatically
            - False: strip action/action_input tags (ReAct) or tool_calls (ToolCalling), inject thought only

    Returns:
        HookCallback to be registered with AgentHooks

    Example:
        >>> trace_pool = {
        ...     "task_1": ["/path/to/trace_a.json", "/path/to/trace_b.json"],
        ...     "task_2": ["/path/to/another_trace.json"],
        ... }
        >>> hook = create_trace_intervention_hook(trace_pool, num_steps=2, execute_tools=True)
        >>> hooks.register(HookPoint.BEFORE_TASK, hook)
    """
    import random

    def _eligible_traces(pool, ns):
        """Filter pool to traces with enough assistant steps for num_steps."""
        if isinstance(pool, str):
            return [pool]
        eligible = []
        for tp in pool:
            n_assistant = sum(
                1
                for m in json.loads(Path(tp).read_text()).get("messages", [])
                if m.get("role") == "assistant"
            )
            if ns >= 0 and n_assistant >= ns or ns < 0 and n_assistant > abs(ns):
                eligible.append(tp)
        return eligible

    def trace_intervention_hook(context: HookContext) -> None:
        if context.task_id not in trace_pool:
            return

        pool = trace_pool[context.task_id]
        eligible = _eligible_traces(pool, num_steps)

        if not eligible:
            logger.warning(
                f"No traces with enough steps for task {context.task_id} "
                f"(num_steps={num_steps}, pool_size={len(pool) if isinstance(pool, list) else 1})"
            )
            return

        trace_path = random.choice(eligible)

        steps = _load_trace_steps(trace_path, num_steps)

        if not steps:
            logger.warning(
                f"No steps to inject for task {context.task_id} from {trace_path}"
            )
            return

        logger.info(
            f"Injecting {len(steps)} intervention step(s) for task {context.task_id} "
            f"from {Path(trace_path).name}"
        )

        context.metadata["intervention_applied"] = True
        context.metadata["intervention_num_steps"] = len(steps)
        context.metadata["intervention_trace_path"] = trace_path

        for step_idx, step_msg in enumerate(steps):
            logger.debug(f"Injecting step {step_idx + 1}/{len(steps)}")

            # Detect format: ToolCallingAgent has tool_calls field
            if step_msg.get("tool_calls"):
                _inject_toolcalling_step(context, step_msg, execute_tools)
            else:
                _inject_react_step(context, step_msg, execute_tools)

    return trace_intervention_hook
