from promptstore import PromptStore

from corral.agents.base_agent import BaseAgent
from corral.agents.react import ReActAgent
from corral.agents.tool_calling import ToolCallingAgent
from corral.agents.utils import LiteLLMMessage
from corral.evaluate import BenchmarkInterface


class LLMPlanner(BaseAgent):
    """Agent that uses the LLM planner to generate plans.
    Then the low-level planner is called to execute the plan.
    Based on https://arxiv.org/abs/2212.04088

    Args:
        model (str): The model to use for running the agent
        max_iterations (int, optional): The maximum number of iterations to plan. Defaults to 10.
        api_endpoint (str, optional): The API endpoint URL for the LLM provider (e.g., OpenAI, VLLM, or self-hosted models) to handle tool/function calling requests. Defaults to None.
        system_prompt (str, optional): The system prompt to use.
            Defaults to "You are a helpful AI assistant that solves tasks step by step."
        user_prompt (str, optional): The user prompt to use. Defaults to a simple prompt with `task_guide`, `tools`, `iterations` and `examples`. `examples` is thought to include few-shot guide.
        extractor_prompt (str, optional): The prompt to use for the low-level planner. Defaults to None.
        temperature (float, optional): The temperature to use for sampling.
                Defaults to 0.7.
        prompt_store (PromptStore, optional): The prompt store to use. Defaults to None.
        system_prompt_id (str, optional): The ID of the system prompt to use. Defaults to "400fcecf-f5f2-464b-aff5-8a4377c9685c".
        user_prompt_id (str, optional): The ID of the user prompt to use. Defaults to "1c7f064f-9a3b-40f5-a555-94e551722d50".
        extractor_prompt_id (str, optional): The ID of the extractor prompt to use. Defaults to "9d37e4a0-26c5-438a-ba1b-a273388fcded".
        kwargs: Additional keyword arguments to pass to the LiteLLM API for all LLM calls
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_iterations: int = 10,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        extractor_prompt: str | None = None,
        temperature: float = 0.7,
        prompt_store: PromptStore | None = None,
        system_prompt_id: str = "400fcecf-f5f2-464b-aff5-8a4377c9685c",
        user_prompt_id: str = "1c7f064f-9a3b-40f5-a555-94e551722d50",
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

    def run(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage],
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> str:
        """Run the LLM planner agent

        Args:
            interface (BenchmarkInterface): The benchmark interface to use
            task_id (str): The task ID to solve
            history (List[LiteLLMMessage], optional): The history items to include. Defaults to None.
            task_prompt (str, optional): The task prompt to use. Defaults to None.
            examples (List[str], optional): List with the few-shot examples to use. Defaults to None.

        Returns:
            str: The final answer to the task
        """
        tools = interface.get_available_tools_for_task(task_id)

        if task_prompt is None:
            task_guide = interface.get_task_prompt(task_id)
        else:
            task_guide = task_prompt

        tool_usage = (
            False  # Default to False, could be passed as a parameter in the future
        )

        self.messages = self.create_prompt(
            task_guide=task_guide,
            history=history,
            examples=examples,
            agent_type="llm_planner",
            tools=tools,
        )

        if tool_usage:
            agent = ToolCallingAgent(
                model=self.model,
                max_iterations=self.max_iterations,
                api_endpoint=self.api_endpoint,
                temperature=self.temperature,
                prompt_store=self.store,
                **self.kwargs,
            )
        else:
            agent = ReActAgent(
                model=self.model,
                max_iterations=self.max_iterations,
                api_endpoint=self.api_endpoint,
                temperature=self.temperature,
                prompt_store=self.store,
                **self.kwargs,
            )

        for _i in range(self.max_iterations):
            plan = self.get_llm_response().content

            self.messages.append(
                LiteLLMMessage(
                    role="assistant", content=plan, name="high-level-planner"
                )
            )
            if plan is None:
                continue

            if "Final Answer:" in plan:
                return plan.split("Final Answer:")[1].strip()

            final_answer, low_level_planner_messages = agent.run_agent(
                interface=interface,
                task_id=task_id,
                task_prompt=plan,
            )

            self.messages.append(
                LiteLLMMessage(
                    role="assistant",
                    content=f"Answer submitted by the executor: {final_answer}.\nMessages by the executor:",
                    name="low-level-planner",
                )
            )
            self.messages.extend(low_level_planner_messages)

        return "Error: Maximum iterations reached"
