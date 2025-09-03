import json
import re
from dataclasses import dataclass
from typing import Any

from promptstore import PromptStore

from corral.agents.base_agent import BaseAgent
from corral.agents.prompt_utils import create_prompt
from corral.agents.utils import LiteLLMMessage
from corral.evaluate import BenchmarkInterface


@dataclass
class Thought:
    """Represents agent's reasoning step"""

    content: str


@dataclass
class Action:
    """Represents an action to be taken"""

    tool_name: str
    arguments: dict[str, Any]


class ReActAgent(BaseAgent):
    """
    Agent that uses the ReAct framework to solve tasks.
    Based on https://arxiv.org/abs/2210.03629

    The ReAct agent follows a "Thought-Action-Observation" loop, where it reasons about
    the task, executes tools, and observes results before continuing to the next step.

    ## Required Prompt Fields

    The user prompt for ReActAgent must contain the following Jinja template fields:
    - **{{task_guide}}**: The main task instructions and description
    - **{{examples}}**: Few-shot examples formatted as a string (optional, can be empty)

    ### Default User Prompt Template:
    The default prompt (ID: "d880c4d3-fe60-4cf4-813b-2008076cd595") expects:

    ### Custom Prompt Requirements:
    If providing a custom user_prompt, it must:
    1. Include {{task_guide}} placeholder for task instructions
    2. Include {{examples}} placeholder for few-shot examples
    3. Instruct the agent to use "Thought:", "Action:", "Action Input:" format
    4. Specify "Final Answer:" format for completion
    5. Support Jinja templating with .fill() method

    Args:
        model (str): The model to use for running the agent. Defaults to "openai/gpt-4o".
        max_iterations (int, optional): The maximum number of iterations to run. Defaults to 10.
        api_endpoint (str, optional): The API endpoint URL for the LLM provider (e.g., OpenAI, VLLM, or self-hosted models) to handle tool/function calling requests. Defaults to None.
        system_prompt (str | Any, optional): The system prompt to use. Can be a string, PromptStore ID, or prompt object
            that implements .fill() method. Defaults to "You are a helpful AI assistant that solves tasks step by step."
        user_prompt (str | Any, optional): The user prompt template. Must contain {{task_guide}} and {{examples}} fields.
            Can be a string, PromptStore ID, or prompt object that implements .fill() method.
            Defaults to ReAct-formatted prompt with Thought-Action-Observation structure.
        extractor_prompt (str | Any, optional): The prompt to use for extracting final answers. Can be a string,
            PromptStore ID, or prompt object that implements .fill() method. Defaults to None.
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.7.
        prompt_store (PromptStore, optional): The prompt store to use. Defaults to None.
        system_prompt_id (str, optional): The ID of the system prompt to use. Defaults to "400fcecf-f5f2-464b-aff5-8a4377c9685c".
        user_prompt_id (str, optional): The ID of the user prompt to use. Defaults to "d880c4d3-fe60-4cf4-813b-2008076cd595".
        extractor_prompt_id (str, optional): The ID of the extractor prompt to use. Defaults to "9d37e4a0-26c5-438a-ba1b-a273388fcded".
        **kwargs: Additional keyword arguments to pass to the LiteLLM API
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_iterations: int = 10,
        api_endpoint: str | None = None,
        system_prompt: str | Any | None = None,
        user_prompt: str | Any | None = None,
        extractor_prompt: str | Any | None = None,
        temperature: float = 0.7,
        prompt_store: PromptStore | None = None,
        system_prompt_id: str = "400fcecf-f5f2-464b-aff5-8a4377c9685c",
        user_prompt_id: str | None = "d880c4d3-fe60-4cf4-813b-2008076cd595",
        extractor_prompt_id: str | None = "9d37e4a0-26c5-438a-ba1b-a273388fcded",
        **kwargs,
    ):
        """Initialize the agent"""
        super().__init__(
            model=model,
            max_iterations=max_iterations,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extractor_prompt=extractor_prompt,
            temperature=temperature,
            prompt_store=prompt_store,
            system_prompt_id=system_prompt_id,
            user_prompt_id=user_prompt_id,
            extractor_prompt_id=extractor_prompt_id,
            **kwargs,
        )

    def parse_llm_response(
        self, response: str
    ) -> tuple[Thought | None, list[Action] | None, bool, str | None]:
        """
        Parse LLM response into Thought, Actions, final flag, and parsing error.

        This parser:
        1. Returns a boolean flag for final answer detection (eliminates duplicate parsing)
        2. Handles empty/whitespace-only thoughts consistently (returns None)
        3. Uses robust line-by-line parsing for better accuracy
        4. Properly handles multiline thoughts and multiple actions
        5. Captures JSON parsing errors for better error feedback

        Returns:
            tuple: (thought, actions, is_final, parsing_error)
            - thought: Thought object or None if no meaningful thought found
            - actions: List of Action objects or None if no actions found
            - is_final: True if response contains "Final Answer:", False otherwise
            - parsing_error: String describing parsing error or None if no error
        """
        lines = response.split("\n")

        # Check for final answer
        is_final = any(line.startswith("Final Answer:") for line in lines)

        # Parse thought
        thought = None
        thought_lines = []
        in_thought = False

        # Parse actions
        actions = []
        current_action_name = None
        current_action_input_lines = []
        in_action_input = False
        parsing_error = None

        for line in lines:
            if line.startswith("Thought:"):
                # Finish any pending action first
                if (
                    in_action_input
                    and current_action_name
                    and current_action_input_lines
                ):
                    try:
                        action_input_str = "\n".join(current_action_input_lines)
                        arguments = json.loads(action_input_str)
                        actions.append(
                            Action(tool_name=current_action_name, arguments=arguments)
                        )
                    except json.JSONDecodeError as e:
                        if parsing_error is None:  # Only capture first parsing error
                            parsing_error = f"Invalid JSON in Action Input for '{current_action_name}': {e!s}"
                    current_action_name = None
                    current_action_input_lines = []

                # Start thought parsing
                thought_content = line[8:].strip()  # Remove "Thought:" prefix
                thought_lines = [thought_content] if thought_content else []
                in_thought = True
                in_action_input = False

            elif line.startswith("Action:"):
                # Finish previous thought if any
                if in_thought and thought_lines:
                    content = "\n".join(thought_lines).strip()
                    if content:
                        thought = Thought(content=content)
                in_thought = False

                # Finish any pending action first
                if (
                    in_action_input
                    and current_action_name
                    and current_action_input_lines
                ):
                    try:
                        action_input_str = "\n".join(current_action_input_lines)
                        arguments = json.loads(action_input_str)
                        actions.append(
                            Action(tool_name=current_action_name, arguments=arguments)
                        )
                    except json.JSONDecodeError as e:
                        if parsing_error is None:  # Only capture first parsing error
                            parsing_error = f"Invalid JSON in Action Input for '{current_action_name}': {e!s}"

                # Start new action parsing
                current_action_name = line[7:].strip()  # Remove "Action:" prefix
                current_action_input_lines = []
                in_action_input = False

            elif line.startswith("Action Input:"):
                # Start action input parsing
                action_input_content = line[
                    13:
                ].strip()  # Remove "Action Input:" prefix
                current_action_input_lines = (
                    [action_input_content] if action_input_content else []
                )
                in_action_input = True
                in_thought = False

            elif line.startswith("Final Answer:"):
                # Finish any pending parsing
                if in_thought and thought_lines:
                    content = "\n".join(thought_lines).strip()
                    if content:
                        thought = Thought(content=content)
                if (
                    in_action_input
                    and current_action_name
                    and current_action_input_lines
                ):
                    try:
                        action_input_str = "\n".join(current_action_input_lines)
                        arguments = json.loads(action_input_str)
                        actions.append(
                            Action(tool_name=current_action_name, arguments=arguments)
                        )
                    except json.JSONDecodeError as e:
                        if parsing_error is None:  # Only capture first parsing error
                            parsing_error = f"Invalid JSON in Action Input for '{current_action_name}': {e!s}"
                break

            else:
                # Continue current context
                if in_thought:
                    thought_lines.append(line)
                elif in_action_input:
                    current_action_input_lines.append(line)

        # Handle end of response (no Final Answer found)
        if not is_final:
            if in_thought and thought_lines:
                content = "\n".join(thought_lines).strip()
                if content:
                    thought = Thought(content=content)
            if in_action_input and current_action_name and current_action_input_lines:
                try:
                    action_input_str = "\n".join(current_action_input_lines)
                    arguments = json.loads(action_input_str)
                    actions.append(
                        Action(tool_name=current_action_name, arguments=arguments)
                    )
                except json.JSONDecodeError as e:
                    if parsing_error is None:  # Only capture first parsing error
                        parsing_error = f"Invalid JSON in Action Input for '{current_action_name}': {e!s}"

        return thought, actions if actions else None, is_final, parsing_error

    def run(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> str:
        """Main ReAct loop implementation

        Args:
            interface (BenchmarkInterface): The interface to use
            task_id (str): The task ID to solve
            history (List[Dict[str, Any]], optional): The history items to include. Defaults to None.
            task_prompt (str, optional): The task prompt to use. `task_prompt` is intended to be a plan or description about the task, that should always be provided when this agent is called as a subagent of a main orchestrator. Defaults to None.
            examples (List[str], optional): List with the few-shot examples to use. Defaults to None.

        Returns:
            str: The final answer to the task
        """
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
        )

        for _iteration in range(self.max_iterations):
            # Create prompt and get LLM response
            llm_response = self.get_llm_response().content

            # Parse response using improved parser
            thought, actions, is_final, parsing_error = self.parse_llm_response(
                llm_response
            )
            thought_prefix = f"Thought: {thought.content}\n" if thought else ""

            # Check for final answer using the parser's flag
            if is_final:
                final_answer_match = re.search(r"Final Answer: (.*)", llm_response)
                if final_answer_match:
                    self.messages.append(
                        LiteLLMMessage(
                            role="assistant",
                            content=f"{thought_prefix}Final Answer: {final_answer_match.group(1)}",
                        )
                    )
                    return final_answer_match.group(1).strip()

            # Provide feedback for parsing errors
            if parsing_error:
                self.messages.append(
                    LiteLLMMessage(
                        role="assistant", content=f"{thought_prefix}Action: (attempted)"
                    )
                )
                self.messages.append(
                    LiteLLMMessage(
                        role="user",
                        content=f"Error: {parsing_error}. Please ensure your Action Input is valid JSON format.",
                        name="parsing-error",
                    )
                )
                continue  # Skip to next iteration to let agent try again

            # Execute tools if actions exist
            if actions:
                for action in actions:
                    action_content = f"{thought_prefix}Action: {action.tool_name}\nAction Input: {json.dumps(action.arguments)}"
                    self.messages.append(
                        LiteLLMMessage(role="assistant", content=action_content)
                    )

                    # Execute tool and get response
                    tool_response = interface.execute_tool(
                        task_id, action.tool_name, action.arguments
                    )

                    observation = (
                        f"Observation: {tool_response.result}"
                        if tool_response.success
                        else f"Error: {tool_response.error}"
                    )

                    self.messages.append(
                        LiteLLMMessage(
                            role="user",
                            content=observation,
                            name=action.tool_name,
                        )
                    )
            else:
                self.messages.append(
                    LiteLLMMessage(
                        role="user",
                        content=str(llm_response),
                    )
                )

        self.messages.append(
            LiteLLMMessage(
                role="assistant",
                content="Error: Maximum iterations reached without finding a final answer.",
                name="react-error",
            )
        )

        return "Error solving the task: unable to complete it in the iteration limit"
