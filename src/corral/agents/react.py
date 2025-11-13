import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger
from promptstore import PromptStore

from corral.agents.base_agent import BaseAgent
from corral.agents.prompt_utils import create_prompt
from corral.agents.utils import LiteLLMMessage, convert_outermost_triple_quotes
from corral.router.routes import CorralRouter


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
        surrender_prompt (str | Any, optional):Instructions for how the agent can surrender from unsolvable tasks. This prompt is only included when enable_surrender=True in the run() method and will be added to the task prompt. Can be a string,
            PromptStore ID, or prompt object that implements .fill() method. Defaults to None.
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.7.
        prompt_store (PromptStore, optional): The prompt store to use. Defaults to None.
        system_prompt_id (str, optional): The ID of the system prompt to use. Defaults to "400fcecf-f5f2-464b-aff5-8a4377c9685c".
        user_prompt_id (str, optional): The ID of the user prompt to use. Defaults to "d880c4d3-fe60-4cf4-813b-2008076cd595".
        extractor_prompt_id (str, optional): The ID of the extractor prompt to use. Defaults to "9d37e4a0-26c5-438a-ba1b-a273388fcded".
        surrender_prompt_id (str, optional): The ID of the surrender prompt to use. Defaults to "1d9059d5-763e-4efd-93b6-308977635ef3".
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
        surrender_prompt: str | Any | None = None,
        temperature: float = 0.7,
        prompt_store: PromptStore | None = None,
        system_prompt_id: str = "400fcecf-f5f2-464b-aff5-8a4377c9685c",
        user_prompt_id: str | None = "d880c4d3-fe60-4cf4-813b-2008076cd595",
        extractor_prompt_id: str | None = "9d37e4a0-26c5-438a-ba1b-a273388fcded",
        surrender_prompt_id: str | None = "1d9059d5-763e-4efd-93b6-308977635ef3",
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
            surrender_prompt=surrender_prompt,
            temperature=temperature,
            prompt_store=prompt_store,
            system_prompt_id=system_prompt_id,
            user_prompt_id=user_prompt_id,
            extractor_prompt_id=extractor_prompt_id,
            surrender_prompt_id=surrender_prompt_id,
            **kwargs,
        )

    def parse_llm_response(
        self, response: str
    ) -> tuple[list[Thought] | None, list[Action] | None]:
        """Parse LLM response into Thoughts and Actions"""
        thought_matches = re.finditer(r"<thought>(.*?)</thought>", response, re.DOTALL)
        action_matches = re.finditer(
            r"<action>(.*?)</action>(?:.*?<action_input>(.*?)</action_input>)?",
            response,
            re.DOTALL,
        )

        thoughts = [Thought(match.group(1).strip()) for match in thought_matches]

        actions = []
        for action_match in action_matches:
            tool_name = action_match.group(1).strip()
            try:
                action_input = action_match.group(2)
                # Check if action_input tags are malformed (opening tag present but closing tag missing)
                if action_input is None:
                    return thoughts, None
                else:
                    action_input = action_match.group(2).strip()

                converted_input = convert_outermost_triple_quotes(action_input)

                # Handle Python boolean values
                converted_input = converted_input.replace("True", "true").replace(
                    "False", "false"
                )

                # Parse as JSON
                arguments = json.loads(converted_input)
                actions.append(Action(tool_name=tool_name, arguments=arguments))
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                logger.error(f"Parsing error: {e}")

        return thoughts if thoughts else None, actions if actions else None

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
        enable_surrender: bool = False,
        intervention_thought: str | None = None,
        execute_intervention_tools: bool = False,
    ) -> str:
        """Main ReAct loop implementation

        Args:
            interface (CorralRouter): The interface to use
            task_id (str): The task ID to solve
            history (List[Dict[str, Any]], optional): The history items to include. Defaults to None.
            task_prompt (str, optional): The task prompt to use. `task_prompt` is intended to be a plan or description about the task, that should always be provided when this agent is called as a subagent of a main orchestrator. Defaults to None.
            examples (List[str], optional): List with the few-shot examples to use. Defaults to None.
            enable_surrender (bool, optional): Whether to enable the surrender option, which allows the agent to give up solving a task. Defaults to False.
            intervention_thought (str, optional): An intervention thought to inject at the start of the task. Defaults to None.
            execute_intervention_tools (bool, optional): Whether to execute tools found in the intervention thought. Defaults to False and if tool execution is present it will be stripped out.

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
            surrender_prompt=self.surrender_prompt,
            enable_surrender=enable_surrender,
        )

        # Inject intervention thought if provided
        if intervention_thought:
            self._inject_intervention(
                intervention_thought=intervention_thought,
                interface=interface,
                task_id=task_id,
                execute_tools=execute_intervention_tools,
            )

        for _iteration in range(self.max_iterations):
            # Create prompt and get LLM response
            llm_response = self.get_llm_response().content

            self.messages.append(LiteLLMMessage(role="assistant", content=llm_response))

            # Parse response
            thoughts, actions = self.parse_llm_response(llm_response)

            # Check for surrender (XML format) if enabled
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

            # Check for final answer (XML format)
            final_answer_match = re.search(
                r"<final_answer>(.*?)</final_answer>", llm_response, re.DOTALL
            ) or re.search(r"Final Answer: (.*)", llm_response, re.DOTALL)

            # Execute tools if actions exist
            if actions:
                # Check for final answer
                for action in actions:
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
            if final_answer_match:
                return final_answer_match.group(1).strip()

            if not actions and final_answer_match is None:
                self.messages.append(
                    LiteLLMMessage(
                        role="user",
                        content="No actions to execute. This is due to parsing error or missing action in the response. Please follow the format <thought>[your reasoning]</thought>\n<action>[tool name]</action>\n<action_input>[tool arguments as JSON]</action_input>.\n\nIf you have the final answer, respond with:\n<thought>[your reasoning]</thought>\n<final_answer>[answer]</final_answer>. For tool calls without arguments, use `<action_input>{}</action_input>`. Remember the closing tags. Try again.",
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

    def _inject_intervention(
        self,
        intervention_thought: str,
        interface: CorralRouter,
        task_id: str,
        execute_tools: bool = False,
    ) -> None:
        """Inject intervention thought and optionally execute tools

        Args:
            intervention_thought (str): The intervention thought to inject from successful/failed trajectories.
            interface (CorralRouter): The interface to use for tool execution
            task_id (str): The task ID
            execute_tools (bool): Whether to execute tools found in the intervention
        """
        if not execute_tools:
            # Just add the thought and strip tool calls
            thought_only = re.sub(
                r"<action>.*?</action>(?:\s*<action_input>.*?</action_input>)?",
                "",
                intervention_thought,
                flags=re.DOTALL,
            ).strip()
            # Format as ReAct-style thought
            intervention_message = f"<thought>{thought_only}</thought>"
            self.messages.append(
                LiteLLMMessage(role="assistant", content=intervention_message)
            )
            logger.info(
                f"Injected intervention thought (tools stripped) for task {task_id}"
            )
        else:
            # Execute tools and add observations
            # Format the full intervention with thought tags if not already present
            if not intervention_thought.strip().startswith("<thought>"):
                intervention_message = f"<thought>{intervention_thought}</thought>"
            else:
                intervention_message = intervention_thought

            self.messages.append(
                LiteLLMMessage(role="assistant", content=intervention_message)
            )

            # Parse and execute any actions
            _, actions = self.parse_llm_response(intervention_message)
            if actions:
                logger.info(
                    f"Executing {len(actions)} tool(s) from intervention for task {task_id}"
                )
                for action in actions:
                    try:
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
                    except Exception as e:
                        logger.error(
                            f"Error executing intervention tool {action.tool_name}: {e}"
                        )
                        self.messages.append(
                            LiteLLMMessage(
                                role="user",
                                content=f"Error: Failed to execute {action.tool_name}: {e}",
                                name=action.tool_name,
                            )
                        )
            else:
                logger.info(
                    f"Injected intervention thought (no tools found) for task {task_id}"
                )
