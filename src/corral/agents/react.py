import importlib.resources
import json
import re
from dataclasses import dataclass
from typing import Any

from promptstore import PromptStore

from corral.agents.prompt_utils import get_prompt
from corral.agents.utils import (
    LiteLLMMessage,
    TrackedAgentMixin,
    _build_user_content,
    llm_call,
)
from corral.evaluate import BenchmarkInterface
from corral.graph import GraphTrackerFactory


@dataclass
class Thought:
    """Represents agent's reasoning step"""

    content: str


@dataclass
class Action:
    """Represents an action to be taken"""

    tool_name: str
    arguments: dict[str, Any]


class ReActAgent(TrackedAgentMixin):
    """
    Agent that uses the ReAct framework to solve tasks
    Based on https://arxiv.org/abs/2210.03629
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_iterations: int = 10,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.7,
        prompt_store: PromptStore | None = None,
        graph_factory: GraphTrackerFactory | None = None,
        server_url: str = "http://localhost:8000",
        **kwargs,
    ):
        """Initialize the agent"""
        # Initialize the TrackedAgentMixin with server URL
        TrackedAgentMixin.__init__(
            self, graph_factory=graph_factory, server_url=server_url
        )

        self.model = model
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        if prompt_store:
            self.store = prompt_store
        else:
            with importlib.resources.path("corral.agents", "") as style_path:
                self.store = PromptStore(f"{style_path}/prompts")
        self.temperature = temperature
        self.kwargs = kwargs
        self.system_prompt = (
            get_prompt(
                self.store, system_prompt, "400fcecf-f5f2-464b-aff5-8a4377c9685c"
            ).fill({})
            if system_prompt is None
            else system_prompt
        )

        self.user_prompt = get_prompt(
            self.store,
            user_prompt,
            "d880c4d3-fe60-4cf4-813b-2008076cd595",
        )

    def get_llm_response(self, messages: list[LiteLLMMessage]) -> str:
        """Get response from the LLM using LiteLLM"""
        # Track the LLM prompt
        prompt_node_id = self.track_llm_prompt(messages)

        # Get response from LLM
        response = llm_call(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            api_endpoint=self.api_endpoint,
            **self.kwargs,
        ).content

        # Track the LLM response
        self.track_llm_response(response, prompt_node_id)

        return response

    def parse_llm_response(
        self, response: str
    ) -> tuple[Thought | None, list[Action] | None]:
        """Parse LLM response into Thought and Actions"""
        thought_match = re.search(
            r"Thought: (.*?)(?=\nAction:|Final Answer:|$)", response, re.DOTALL
        )
        action_matches = re.finditer(
            r"Action: (\w+)\nAction Input: ({.*?}(?=\nAction:|\nThought:|\nFinal Answer:|$))",
            response,
            re.DOTALL,
        )

        thought = Thought(thought_match.group(1).strip()) if thought_match else None

        actions = []
        for action_match in action_matches:
            tool_name = action_match.group(1).strip()
            try:
                action_input = action_match.group(2).strip()
                arguments = json.loads(action_input)
                actions.append(Action(tool_name=tool_name, arguments=arguments))
            except json.JSONDecodeError:
                pass

        # Track thought if present
        if thought:
            self.track_thought(thought.content)

        # Track actions if present
        if actions:
            for action in actions:
                self.track_tool_call(action.tool_name, action.arguments)

        return thought, actions if actions else None

    def create_prompt(
        self,
        task_guide: str | list,
        history: list[LiteLLMMessage],
        examples: list[str] | None,
        tools: str,
    ) -> list[LiteLLMMessage]:
        """Create prompt for LLM including context and history"""
        messages: list[LiteLLMMessage] = []
        messages.append(
            LiteLLMMessage(
                role="system",
                content=self.system_prompt,
            )
        )

        user_content = _build_user_content(
            agent="react",
            user_prompt=self.user_prompt,
            task_guide=task_guide,
            history=history,
            tools=tools,
            examples=examples,
        )

        messages.append(
            LiteLLMMessage(
                role="user",
                content=user_content,
            )
        )

        return messages

    def run_agent(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> tuple[str, list[LiteLLMMessage]]:
        """Main ReAct loop implementation"""
        if history is None:
            history = []
        if task_prompt is None:
            task_guide = interface.get_task_prompt(task_id)
            tools = interface.get_tools_guide(task_id)
        else:
            task_guide = task_prompt
            tools = interface.get_tools_guide(task_id)

        # Start tracking
        self.start_tracking(task_id=task_id, agent_type="ReActAgent")

        messages = self.create_prompt(task_guide, history, examples, tools)

        for _iteration in range(self.max_iterations):
            # Create prompt and get LLM response
            llm_response = self.get_llm_response(messages)

            # Parse response
            thought, actions = self.parse_llm_response(llm_response)
            thought_prefix = f"Thought: {thought.content}\n" if thought else ""

            # Check for final answer
            final_answer_match = re.search(r"Final Answer: (.*)", llm_response)

            if final_answer_match:
                answer = final_answer_match.group(1).strip()
                messages.append(
                    LiteLLMMessage(
                        role="assistant",
                        content=f"{thought_prefix}Final Answer: {answer}",
                    )
                )

                # Track final submission
                self.track_final_submission(answer)

                # Stop tracking
                self.stop_tracking(visualize=True)

                return answer, messages

            # Execute tools if actions exist
            if actions:
                for action in actions:
                    action_content = f"{thought_prefix}Action: {action.tool_name}\nAction Input: {json.dumps(action.arguments)}"
                    messages.append(
                        LiteLLMMessage(role="assistant", content=action_content)
                    )

                    # Execute tool and get response
                    tool_response = interface.execute_tool(
                        task_id, action.tool_name, action.arguments
                    )

                    # Track tool response
                    result = (
                        tool_response.result
                        if tool_response.success
                        else tool_response.error
                    )
                    self.track_tool_response(
                        result=result,
                        success=tool_response.success,
                        tool_name=action.tool_name,
                    )

                    observation = (
                        f"Observation: {tool_response.result}"
                        if tool_response.success
                        else f"Error: {tool_response.error}"
                    )

                    messages.append(
                        LiteLLMMessage(
                            role="user",
                            content=observation,
                            name=action.tool_name,
                        )
                    )

            # If no action or thought was parsed, break the loop
            if not thought and actions is None:
                break

        error_message = (
            "Error solving the task: unable to complete it in the iteration limit"
        )

        # Track error as final submission
        self.track_final_submission(error_message)

        # Stop tracking
        self.stop_tracking(visualize=True)

        return error_message, messages
