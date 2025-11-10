import json
import re
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv
from loguru import logger
from promptstore import PromptStore

from corral import CorralRouter, CorralRunner
from corral.agents import BaseAgent
from corral.agents.utils import LiteLLMMessage

subtask_path = "gpt-4o/retrosynthesis/level_2/subtasks/agent_logs-ToolCallingAgent-gpt-4o-2024-08-06-comprehensive"

root_path = Path(__file__).parent


def load_files() -> dict:
    data = {}
    file_path = root_path / subtask_path
    for file in file_path.glob("*.json"):
        with file.open("r") as f:
            r = json.load(f)
        name = file.name
        data[name] = r["messages"]

    return data


class ToolCallingAgent(BaseAgent):
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
        self.files = load_files()

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
        file_id = ""
        for file in self.files:
            if task_id in file:
                file_id = file

        if not file_id:
            raise Exception(f"Could not find file for the task: {task_id}")

        data = self.files.get(file_id, None)
        if data is None:
            raise ValueError("not file found")

        del self.files[file_id]

        self.messages = data
        self.token_usage = {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }

        llm_response = data[-1]["content"]

        if (
            llm_response
            == "Error: Maximum iterations reached without finding a final answer."
        ):
            return llm_response

        final_answer_match = re.search(
            r"<final_answer>(.*?)</final_answer>", llm_response, re.DOTALL
        ) or re.search(r"Final Answer: (.*)", llm_response, re.DOTALL)

        if final_answer_match:
            return final_answer_match.group(1).strip()

        else:
            json_schema = re.search(r"{(.*?)}", llm_response, re.DOTALL)
            if json_schema:
                return json_schema.group(0).strip().replace("```", "")

        return llm_response


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "claude-3-5-sonnet-20241022",
    task_ids: list | None = None,
    temperature: float = 0.0,
    run_name: str = "corral_benchmark_tests",
    verbose: str = "brief",
):
    """Run the benchmark with specified model and tasks"""

    interface = CorralRouter()

    agent = ToolCallingAgent(model=model, max_iterations=20, temperature=temperature)
    runner = CorralRunner(interface, agent)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        task_ids,
        trials_per_task=5,
        k_values=[1, 2, 3, 4, 5],
        verbose=True,
        tool_verbosity=verbose,
    )
    result.generate_report(f"{run_name}.json")
    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    verboses = [
        # "brief",
        # "workflow",
        "comprehensive",
    ]
    for verbose in verboses:
        logger.info(f"Running benchmark with verbosity: {verbose}")
        try:
            model = "gpt-4o-2024-08-06"
            run_name = f"gpt_4o-tool_calling-retro_lvl2_env-{verbose}_verbosity"
            run_benchmark(model=model, run_name=run_name, verbose=verbose)

        except Exception as e:
            logger.error(f"Benchmark failed: {e!s}")
            raise
