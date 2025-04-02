import json

import uvicorn
from loguru import logger
from tools import UnitConverterTool, app, calculator, number_converter

from corral.base import Environment, TaskDefinition, TaskGroup, Tool
from corral.server import create_benchmark_server


class TaskEnvironment(Environment):
    """Environment that works with a task group"""

    def __init__(
        self, task_id: str, task_group: TaskGroup, available_tools: dict[str, Tool]
    ):
        self.task_group = task_group
        self.task_id = task_id  # Individual task ID within the group
        self.available_tools = available_tools

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        # Initialize tools and environment
        self.tools = {}
        super().__init__(f"{task_group.group_id}_{task_id}")

        # Add required tools
        for tool_name in self.current_task.tools:
            if tool_name in available_tools:
                self.add_tool(available_tools[tool_name])

    def get_task_prompt(self) -> str:
        _input_data = self.task_group.get_task_input(self.task_id)

        prompt = f"""Task: {self.current_task.name}
Description: {self.current_task.description}

Required submission format:
"""
        for key, desc in self.current_task.submission_format.items():
            prompt += f"- {key}: {desc}\n"

        prompt += "\nAvailable input data:\n"

        if (
            self.current_task.input_from_task
            and self.current_task.input_from_task in self.task_group.results
        ):
            # If this task depends on a previous task, show its result
            previous_result = self.task_group.results[self.current_task.input_from_task]
            if isinstance(previous_result, dict) and "answer" in previous_result:
                prompt += f"Previous task result: {previous_result['answer']}\n"
            else:
                prompt += f"Previous task result: {previous_result}\n"

        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                prompt += f"- {key}: {value}\n"

        if self.current_task.input_from_task:
            status = (
                "available"
                if self.current_task.input_from_task in self.task_group.results
                else "not yet available"
            )
            prompt += f"\nThis task uses output from task: {self.current_task.input_from_task} ({status})"

        return prompt

    def score(self) -> float:
        """Score the submitted answer"""
        if not self.state.submitted_answer:
            return 0.0

        try:
            # Clean the submission - take only the numerical answer part
            submission_str = self.state.submitted_answer.strip()
            logger.info(f"Raw submission: {submission_str}")  # Debug logger.info

            # Try to parse as JSON first
            try:
                submission = json.loads(submission_str)
            except json.JSONDecodeError:
                # If not valid JSON, try to create a simple answer dict
                submission = {"answer": submission_str}

            # Store result in task group
            logger.info(f"Parsed submission: {submission}")  # Debug logger.info
            score = self.current_task.scoring_fn(submission)
            self.task_group.store_result(self.task_id, submission, score)

            return score
        except Exception as e:
            logger.info(f"Error scoring submission for task {self.task_id}: {e!s}")
            logger.info(f"Submission was: {self.state.submitted_answer}")
            return 0.0


def create_catalysis_environments() -> dict[str, Environment]:
    """Create environments for catalysis tasks"""

    def score_addition(result: dict) -> float:
        score = 0.0
        if "answer" in result:
            try:
                _answer = float(result["answer"])
                score = 1.0
            except ValueError:
                pass
        return score

    # Create task group
    task_group = TaskGroup(
        group_id="simple_math",
        tasks={
            "task1": TaskDefinition(
                name="First Addition",
                description="What is 3 + 5? Return your answer as a number.",
                tools=["calculator"],
                scoring_fn=score_addition,
                submission_format={"answer": "numerical result (example: 8)"},
                initial_input={"x": 3, "y": 5},
            ),
            "task2": TaskDefinition(
                name="Second Addition",
                description="Take the result from the previous task and add 4 to it. Return your answer as a number.",
                tools=["calculator"],
                scoring_fn=score_addition,
                submission_format={"answer": "numerical result (example: 12)"},
                input_from_task="task1",
            ),
        },
    )

    # logger.info task dependencies for reference
    logger.info("\nTask Dependencies:")
    for task_id, deps in task_group.get_task_dependencies().items():
        logger.info(f"- {task_id}: depends on {deps}")

    # Create environments for all tasks
    available_tools = {
        "calculator": calculator,
        "unit_converter": UnitConverterTool(),
        "number_converter": number_converter,
    }
    environments = {}

    for task_id in task_group.tasks:
        # All environments share the same task group instance
        environments[task_id] = TaskEnvironment(
            task_id=task_id, task_group=task_group, available_tools=available_tools
        )

    return environments


if __name__ == "__main__":
    # Create all environments
    environments = create_catalysis_environments()

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_task:
            logger.info(f"  Depends on: {env.current_task.input_from_task}")

    # Create and run server
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
