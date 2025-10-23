import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger
from promptstore import PromptStore

from corral.agents.base_agent import BaseAgent
from corral.agents.prompt_utils import create_prompt
from corral.agents.utils import LiteLLMMessage
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
    ) -> tuple[Thought | None, list[Action] | None]:
        """Parse LLM response into Thought and Actions"""
        thought_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL)
        action_matches = re.finditer(
            r"<action>(.*?)</action>.*?<action_input>(.*?)</action_input>",
            response,
            re.DOTALL,
        )

        thought = Thought(thought_match.group(1).strip()) if thought_match else None

        actions = []
        for action_match in action_matches:
            tool_name = action_match.group(1).strip()
            try:
                action_input = action_match.group(2).strip()

                # Convert Python triple-quoted strings to JSON-escaped strings
                action_input = re.sub(
                    r'"""(.*?)"""',
                    lambda m: json.dumps(m.group(1)),
                    action_input,
                    flags=re.DOTALL,
                )

                # Handle Python boolean values
                action_input = action_input.replace("True", "true").replace(
                    "False", "false"
                )

                arguments = json.loads(action_input)
                actions.append(Action(tool_name=tool_name, arguments=arguments))
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")

        return thought, actions if actions else None

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> str:
        """Main ReAct loop implementation

        Args:
            interface (CorralRouter): The interface to use
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

            self.messages.append(LiteLLMMessage(role="assistant", content=llm_response))

            # Parse response
            thought, actions = self.parse_llm_response(llm_response)

            # Check for final answer (XML format)

            final_answer_match = re.search(
                r"<final_answer>(.*?)</final_answer>", llm_response, re.DOTALL
            )
            if not final_answer_match:
                final_answer_match = re.search(
                    r"Final Answer: (.*)", llm_response, re.DOTALL
                )

            if final_answer_match:
                return final_answer_match.group(1).strip()

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
            else:
                self.messages.append(
                    LiteLLMMessage(
                        role="user",
                        content="No actions to execute. This is due to parsing error or missing action in the response. Please use the tags <action> and <action_input> to specify your action, or <final_answer> to provide your final answer.",
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
