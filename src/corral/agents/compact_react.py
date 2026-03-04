import json
import re
from typing import Any

import yaml
from loguru import logger

# Re-use the parsing/formatting helpers from CompactHistoryAgent without
# inheriting its run() method.
from corral.agents.compact_tool import CompactToolCallingAgent
from corral.agents.hooks import HookPoint
from corral.agents.prompt_utils import create_prompt
from corral.agents.react import (
    Action,
    ReActAgent,
    Thought,
    convert_outermost_triple_quotes,
)
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

    # Borrow helpers from CompactToolCallingAgent as unbound class methods
    _try_parse = staticmethod(CompactToolCallingAgent._try_parse)
    _deep_parse = classmethod(lambda _, obj: CompactToolCallingAgent._deep_parse(obj))
    _extract_result = classmethod(
        lambda _, raw: CompactToolCallingAgent._extract_result(raw)
    )

    @classmethod
    def _format_observation(
        cls, tool_name: str, arguments: dict, raw_result: str
    ) -> str:
        """Render a tool result as a YAML block for use as the observation."""
        args_parsed = CompactToolCallingAgent._deep_parse(arguments)
        result_parsed = CompactToolCallingAgent._extract_result(raw_result)

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
            parsed = CompactToolCallingAgent._deep_parse(
                CompactToolCallingAgent._try_parse(raw)
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

    def _maybe_unescape_llm_string(self, s: str) -> str:
        """
        If `s` looks like a quoted/escaped string containing \\n etc, unescape it.
        Handles cases like: "topology: |\\n  {...}\\nmeasurements: |\\n  {...}"
        """
        s = s.strip()

        # Heuristic: starts/ends with a quote AND contains escaped newlines
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"' and "\\n" in s:
            # 1) Try JSON string decoding (most correct for backslash escapes)
            try:
                return json.loads(s)
            except Exception:
                pass

            # 2) Fallback: YAML double-quoted string decoding handles \n, \", etc.
            try:
                result = yaml.safe_load(s)
                if isinstance(result, str):
                    return result
            except Exception:
                pass

        return s

    def _coerce_yaml_args(self, obj: Any) -> dict[str, Any]:
        """Ensure arguments are a dict, raising if not."""
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        raise ValueError(
            f"Expected YAML mapping/dict for action_input, got {type(obj).__name__}"
        )

    def _parse_embedded_json_fields(self, d: dict[str, Any]) -> dict[str, Any]:
        """
        Optional: if some top-level YAML fields are strings that contain JSON (common with `|` blocks),
        parse them into dicts/lists.
        """
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, str):
                sv = v.strip()
                if (sv.startswith("{") and sv.endswith("}")) or (
                    sv.startswith("[") and sv.endswith("]")
                ):
                    try:
                        out[k] = json.loads(sv)
                        continue
                    except json.JSONDecodeError:
                        pass
            out[k] = v
        return out

    def _safe_load_yaml(self, text: str) -> Any:
        """
        Load a YAML value from *text*, with two levels of fallback:

        1. Normal ``yaml.safe_load`` — covers the common case where the LLM
           returned a well-formed YAML mapping.
        2. If ``yaml.safe_load`` raises because it encounters multiple adjacent
           top-level scalars without a ``---`` separator (error "expected
           '<document start>'..."), scan the text for individual single-quoted
           YAML scalars with a regex, parse each one independently, and merge
           all resulting dicts.  This handles the pattern where the LLM
           accidentally produces two adjacent single-quoted YAML scalars, e.g.::

               'topology: |\\n  ...\\n''measurements: |\\n  ...'
        """
        # --- fast path ---
        try:
            loaded = yaml.safe_load(text)
            # If YAML returned a bare string it may be a nested YAML document
            # (the LLM wrapped a YAML doc in a YAML double-quoted/single-quoted string).
            if isinstance(loaded, str):
                # First attempt: re-parse as-is (covers JSON-escaped `\"` → real YAML)
                try:
                    result = yaml.safe_load(loaded)
                    if not isinstance(result, str):
                        return result
                    loaded = result
                except yaml.YAMLError:
                    pass
                # Second attempt: the string may contain literal \n (backslash-n) instead
                # of real newlines (common when the LLM produces '''…''' with \n inside).
                # Replace them with real newlines and re-parse.
                if "\\n" in loaded:
                    try:
                        result = yaml.safe_load(loaded.replace("\\n", "\n"))
                        if not isinstance(result, str):
                            return result
                    except yaml.YAMLError:
                        pass
            return loaded
        except yaml.YAMLError:
            pass

        # --- adjacent-scalar fallback ---
        # PyYAML cannot stream two bare scalars without a --- separator.
        # Extract every single-quoted scalar with a regex, decode '' → ',
        # parse each as YAML, and merge any dicts that result.
        scalar_contents = re.findall(r"'((?:[^']|'')*)'", text)
        if len(scalar_contents) > 1:
            merged: dict[str, Any] = {}
            for content in scalar_contents:
                cleaned_content = content.replace("''", "'")
                try:
                    parsed = yaml.safe_load(cleaned_content)
                    if isinstance(parsed, dict):
                        merged.update(parsed)
                except yaml.YAMLError:
                    pass
            if merged:
                return merged

        # If all else fails, re-raise the original error so the caller can log it.
        return yaml.safe_load(text)

    def parse_llm_response_yaml(
        self,
        response: str,
    ) -> tuple[list["Thought"] | None, list["Action"] | None]:
        """Parse LLM response into Thoughts and Actions, with YAML action inputs."""
        thought_matches = re.finditer(r"<thought>(.*?)</thought>", response, re.DOTALL)
        action_matches = re.finditer(
            r"<action>(.*?)</action>(?:.*?<action_input>(.*?)</action_input>)?",
            response,
            re.DOTALL,
        )

        thoughts = [Thought(m.group(1).strip()) for m in thought_matches]

        actions: list[Action] = []
        for m in action_matches:
            tool_name = m.group(1).strip()
            try:
                action_input = m.group(2)

                # Malformed tags (opening present but closing missing) -> group(2) becomes None
                if action_input is None:
                    return (thoughts if thoughts else None), None

                action_input = action_input.strip()
                converted_input = convert_outermost_triple_quotes(action_input)

                # NEW: unescape if the model returned a quoted/escaped string
                converted_input = self._maybe_unescape_llm_string(converted_input)

                loaded = self._safe_load_yaml(converted_input)

                arguments = self._coerce_yaml_args(loaded)

                # OPTIONAL: parse JSON inside YAML block scalars (your exact case)
                arguments = self._parse_embedded_json_fields(arguments)

                actions.append(Action(tool_name=tool_name, arguments=arguments))

            except (yaml.YAMLError, ValueError, SyntaxError) as e:
                logger.error(f"Parsing error: {e}")

        return (thoughts if thoughts else None), (actions if actions else None)

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
            logger.info(repr(clean_llm_response))

            self.messages.append(
                LiteLLMMessage(
                    role="assistant",
                    content=clean_llm_response,
                    id=full_llm_response.id,
                )
            )

            thoughts, actions = self.parse_llm_response_yaml(llm_response)

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
                raw_answer = final_answer_match.group(1).strip()
                try:
                    parsed = yaml.safe_load(raw_answer)
                    if parsed is not None and not isinstance(parsed, str):
                        return json.dumps(parsed)
                except Exception:
                    pass
                return raw_answer

            if not actions and final_answer_match is None:
                self.messages.append(
                    LiteLLMMessage(
                        role="user",
                        content=(
                            "No actions to execute. This is due to parsing error or missing "
                            "action in the response. Please follow the format "
                            "<thought>[your reasoning]</thought>\n"
                            "<action>[tool name]</action>\n"
                            "<action_input>[tool arguments as YAML]</action_input>.\n\n"
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
