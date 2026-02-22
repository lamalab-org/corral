import re

import yaml
from loguru import logger

# Re-use the parsing/formatting helpers from CompactHistoryAgent without
# inheriting its run() method.
from corral.agents.compact_history import CompactHistoryAgent
from corral.agents.hooks import HookPoint
from corral.agents.prompt_utils import create_prompt
from corral.agents.react import ReActAgent
from corral.agents.utils import LiteLLMMessage
from corral.router.routes import CorralRouter


class CompactReActAgent(ReActAgent):
    """
    ReAct agent that stores tool observations as clean YAML instead of raw
    ``Observation: <stringified result>`` user messages.

    The thought / action / action_input XML that the LLM produces is kept
    verbatim in the assistant messages (the LLM needs to see its own reasoning
    chain).  Only the *observation* returned by the tool is reformatted:

        - tool: simulate_circuit_resistance
          arguments:
            topology: ...
            terminal_nodes:
              - A
              - B
          result: 4.7

    Any tool result that is itself a stringified JSON / Python dict (e.g. a tool
    response envelope with ``status``, ``duration``, ``timestamp`` fields) is
    recursively parsed and the inner ``result`` value is extracted and rendered.

    All constructor arguments are identical to :class:`ReActAgent`.
    """

    # Borrow helpers from CompactHistoryAgent as unbound class methods
    _try_parse = staticmethod(CompactHistoryAgent._try_parse)
    _deep_parse = classmethod(lambda _, obj: CompactHistoryAgent._deep_parse(obj))
    _extract_result = classmethod(
        lambda _, raw: CompactHistoryAgent._extract_result(raw)
    )

    @classmethod
    def _format_observation(
        cls, tool_name: str, arguments: dict, raw_result: str
    ) -> str:
        """Render a tool result as a YAML block for use as the observation."""
        args_parsed = CompactHistoryAgent._deep_parse(arguments)
        result_parsed = CompactHistoryAgent._extract_result(raw_result)

        block = {"tool": tool_name, "arguments": args_parsed, "result": result_parsed}
        return yaml.dump(
            [block],
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ).rstrip()

    @classmethod
    def _clean_assistant_message(cls, text: str) -> str:
        """Replace every <action_input>…</action_input> block in *text* with
        a clean YAML representation of its JSON content, so the stored
        assistant message contains no escaped strings or brackets."""

        def _replace(match: re.Match) -> str:
            raw = match.group(1).strip()
            parsed = CompactHistoryAgent._deep_parse(
                CompactHistoryAgent._try_parse(raw)
            )
            clean = yaml.dump(
                parsed,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ).rstrip()
            return f"<action_input>\n{clean}\n</action_input>"

        return re.sub(
            r"<action_input>(.*?)</action_input>",
            _replace,
            text,
            flags=re.DOTALL,
        )

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
        if task_prompt is None:
            task_guide = interface.get_task_guide(task_id)
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

        for _iteration in range(self.max_iterations):
            self._current_iteration = _iteration
            self._execute_hooks(HookPoint.BEFORE_ITERATION, interface, task_id)

            full_llm_response = self.get_llm_response()
            llm_response = full_llm_response.content

            # Clean the action_input blocks before storing so future context
            # sees YAML instead of escaped JSON strings.
            clean_llm_response = self._clean_assistant_message(llm_response)

            self.messages.append(
                LiteLLMMessage(
                    role="assistant",
                    content=clean_llm_response,
                    id=full_llm_response.id,
                )
            )

            thoughts, actions = self.parse_llm_response(llm_response)

            if enable_surrender and (
                surrender_match := re.search(
                    r"<surrender>(.*?)</surrender>",
                    llm_response,
                    re.DOTALL | re.IGNORECASE,
                )
            ):
                logger.info(
                    f"Agent surrendering from task {task_id}. Reason: {surrender_match[1].strip()}"
                )
                return "SURRENDER"

            final_answer_match = re.search(
                r"<final_answer>(.*?)</final_answer>", llm_response, re.DOTALL
            ) or re.search(r"Final Answer: (.*)", llm_response, re.DOTALL)

            if actions:
                for action in actions:
                    tool_response = interface.execute_tool(
                        task_id, action.tool_name, action.arguments
                    )

                    if tool_response.success:
                        observation = self._format_observation(
                            action.tool_name,
                            action.arguments,
                            str(tool_response.result),
                        )
                    else:
                        observation = self._format_observation(
                            action.tool_name,
                            action.arguments,
                            f"Error: {tool_response.error}",
                        )

                    self.messages.append(
                        LiteLLMMessage(
                            role="user",
                            content=observation,
                            name=action.tool_name,
                        )
                    )

            self._execute_hooks(
                HookPoint.AFTER_ITERATION,
                interface,
                task_id,
                llm_response=full_llm_response,
            )

            if final_answer_match:
                return final_answer_match.group(1).strip()

            if not actions and final_answer_match is None:
                self.messages.append(
                    LiteLLMMessage(
                        role="user",
                        content=(
                            "No actions to execute. This is due to parsing error or missing "
                            "action in the response. Please follow the format "
                            "<thought>[your reasoning]</thought>\n"
                            "<action>[tool name]</action>\n"
                            "<action_input>[tool arguments as JSON]</action_input>.\n\n"
                            "If you have the final answer, respond with:\n"
                            "<thought>[your reasoning]</thought>\n"
                            "<final_answer>[answer]</final_answer>. "
                            "For tool calls without arguments, use `<action_input>{}</action_input>`. "
                            "Remember the closing tags. Try again."
                        ),
                    )
                )

        self.messages.append(
            LiteLLMMessage(
                role="assistant",
                content="Error: Maximum iterations reached without finding a final answer.",
                name="compact-react-error",
            )
        )
        return "Error solving the task: unable to complete it in the iteration limit"
