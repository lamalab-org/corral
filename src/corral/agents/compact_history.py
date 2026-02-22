import ast
import json
import re
from typing import Any

import yaml
from loguru import logger

from corral.agents.hooks import HookPoint
from corral.agents.prompt_utils import create_prompt
from corral.agents.schema import Action
from corral.agents.tool_calling import ToolCallingAgent
from corral.agents.utils import LiteLLMMessage, convert_to_openai_tool_format
from corral.router.routes import CorralRouter


class CompactHistoryAgent(ToolCallingAgent):
    """
    Agent derived from ToolCallingAgent that stores past tool exchanges as clean
    YAML instead of structured JSON tool-call / tool-result message pairs.

    After each tool call completes the exchange is stored as a single assistant
    message rendered with ``yaml.dump``, e.g.:

        - tool: write_file
          arguments:
            path: results.json
            content:
              - node_a: A
                node_b: B
                resistance: 66.364
          result: Successfully wrote to results.json

    Argument values and result strings that are themselves JSON or Python
    dict/list literals are recursively parsed before rendering, so no escaped
    brackets or ``\\n`` escape sequences survive into the history.

    All constructor arguments are identical to :class:`ToolCallingAgent`.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse(value: Any) -> Any:
        """Try to parse a string as JSON, then as a Python literal.
        Returns the parsed object if successful, otherwise the original value.
        """
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        # JSON path (objects, arrays, quoted strings, numbers, booleans)
        if stripped and stripped[0] in ("{", "[", '"'):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        # Python repr path (single-quoted strings, None, True/False, dicts, lists)
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, dict | list | tuple):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return value

    @classmethod
    def _deep_parse(cls, obj: Any) -> Any:
        """Recursively parse any string leaves that look like JSON / Python literals."""
        if isinstance(obj, dict):
            return {k: cls._deep_parse(cls._try_parse(v)) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [cls._deep_parse(cls._try_parse(item)) for item in obj]
        if isinstance(obj, str):
            parsed = cls._try_parse(obj)
            if parsed is not obj:
                return cls._deep_parse(parsed)
        return obj

    @classmethod
    def _extract_result(cls, raw_result: str) -> Any:
        """Parse the raw result string.

        Tool responses are often stringified dicts like::

            {'tool_name': 'write_file', 'result': 'OK', 'status': 'success', ...}

        When the parsed object is a dict that has a ``'result'`` key we return
        just that value (the actual answer).  Everything else (tool_name,
        arguments, duration, timestamp) is noise we discard.
        If parsing fails, or there is no ``'result'`` key, the raw string is
        returned as-is.
        """
        parsed = cls._try_parse(raw_result)
        if isinstance(parsed, dict) and "result" in parsed:
            return cls._deep_parse(parsed["result"])
        return cls._deep_parse(parsed)

    @classmethod
    def _format_tool_block(cls, name: str, arguments_json: str, result: str) -> str:
        """Render one tool call + result as a YAML list item."""
        try:
            args: dict = json.loads(arguments_json)
        except (json.JSONDecodeError, AttributeError):
            args = {"_raw": arguments_json}

        args_parsed = cls._deep_parse(args)
        result_parsed = cls._extract_result(result)

        block = {"tool": name, "arguments": args_parsed, "result": result_parsed}
        return yaml.dump(
            [block],
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ).rstrip()

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
        enable_surrender: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> str:
        tools = convert_to_openai_tool_format(
            interface.get_available_tools_for_task(task_id)
        )
        self._available_tools = tools

        if task_prompt is None:
            task_guide = interface.get_task_prompt(task_id)
        else:
            task_guide = task_prompt

        self.messages = create_prompt(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            task_guide=task_guide,
            history=history,
            examples=examples,
            surrender_prompt=self.surrender_prompt,
            enable_surrender=enable_surrender,
        )

        self._execute_hooks(HookPoint.BEFORE_TASK, interface, task_id)

        for _i in range(self.max_iterations):
            self._current_iteration = _i
            self._execute_hooks(HookPoint.BEFORE_ITERATION, interface, task_id)

            try:
                full_llm_response = self.get_llm_response(tools)
                llm_response = full_llm_response

                content = llm_response.content
                if content:
                    if enable_surrender and re.search(
                        r"(?:Final Answer:\s*)?SURRENDER", content, re.IGNORECASE
                    ):
                        logger.info(f"Agent retiring from task {task_id}")
                        self.messages.append(
                            LiteLLMMessage(role="assistant", content=content)
                        )
                        return "GIVE UP"

                    final_answer_match = re.search(
                        r"Final Answer: (.*)", content, re.DOTALL | re.IGNORECASE
                    )
                    self._execute_hooks(
                        HookPoint.AFTER_ITERATION,
                        interface,
                        task_id,
                        llm_response=full_llm_response,
                    )
                    if final_answer_match:
                        self.messages.append(
                            LiteLLMMessage(role="assistant", content=content)
                        )
                        return final_answer_match.group(1).strip()

                tool_calls = llm_response.tool_calls
                if tool_calls:
                    # Collect all tool results for this turn
                    call_lines: list[str] = []

                    # Preserve any free-form content alongside the tool calls
                    if content:
                        call_lines.append(content)

                    for called_tool in tool_calls:
                        function_name = str(called_tool.function.name)
                        result: str | None = None

                        try:
                            raw_arguments = json.loads(called_tool.function.arguments)
                            action = Action(
                                tool_name=called_tool.function.name,
                                arguments=raw_arguments,
                            )
                            function_call = interface.execute_tool(
                                task_id, action.tool_name, action.arguments
                            )
                            result = str(function_call.result)
                            if result is None:
                                result = str(function_call.error)
                        except json.JSONDecodeError as e:
                            result = f"Error parsing tool arguments: {e!s}"
                            logger.error(
                                f"JSON parsing error for tool {function_name}: {e}"
                            )
                        except Exception as e:
                            result = f"Error executing tool: {e!s}"
                            logger.error(
                                f"Tool execution error for {function_name}: {e}"
                            )

                        call_lines.append(
                            self._format_tool_block(
                                function_name,
                                called_tool.function.arguments,
                                result or "Unknown error occurred",
                            )
                        )

                    # Store the whole exchange as one plain-text assistant message
                    self.messages.append(
                        LiteLLMMessage(
                            role="assistant",
                            content="\n".join(call_lines),
                        )
                    )
                else:
                    self.messages.append(
                        LiteLLMMessage(
                            role="assistant",
                            content=llm_response.content,
                        )
                    )

                self._execute_hooks(
                    HookPoint.AFTER_ITERATION,
                    interface,
                    task_id,
                    llm_response=full_llm_response,
                )

            except Exception as e:
                logger.error(f"Error during agent iteration: {e}")
                self.messages.append(
                    LiteLLMMessage(
                        role="system",
                        content=f"Error during agent iteration: {e!s}",
                    )
                )

        self.messages.append(
            LiteLLMMessage(
                role="assistant",
                content="Error: Maximum iterations reached without finding a final answer.",
            )
        )
        return "Error solving the task. Maximum iterations reached."
