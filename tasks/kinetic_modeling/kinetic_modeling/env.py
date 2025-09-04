import os

from kinetic_modeling.tasks import return_tasks
from kinetic_modeling.tools import create_tools
from loguru import logger

from corral.backend.environment import Environment
from corral.backend.server import run_server
from corral.utils.task_group import TaskGroup

CORRAL_WORK_DIR = os.environ["CORRAL_WORK_DIR"]
BASE_WORK_DIR = os.environ.get(
    "CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/spectra_elucidation"
)


class KineticEnvironment(Environment):
    """Environment that works with a task group - simple composition approach"""

    def __init__(
        self,
        task_id: str,
        task_group: TaskGroup,
        work_dir: str,
    ):
        self.task_id = task_id
        self.task_group = task_group
        self.available_tools = create_tools()
        self.work_dir = work_dir

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        # Initialize environment
        super().__init__(f"{task_group.group_id}_{task_id}", base_work_dir=work_dir)

        self._add_task_tools()

    def _add_task_tools(self):
        """Add tools required for the current task to the environment"""
        for tool_name in self.current_task.tools:
            if tool_name in self.available_tools:
                self.add_tool(self.available_tools[tool_name])
            else:
                logger.warning(
                    f"Tool {tool_name} not found in available tools for task {self.task_id}"
                )

    def get_task_prompt(self) -> str:
        """Generate the task prompt for the current task"""

        prompt = f"""\nTask: {self.current_task.name}
Description: {self.current_task.description}

**Your Goal:**
Develop a kinetic model that accurately describes the experimental observations. Your final model should include:
1. A reaction mechanism (if mechanism discovery is required)
2. Rate laws for each reaction
3. Fitted kinetic parameters
4. Statistical validation of the model fit

**Success Criteria:**
- Model achieves R² > 0.95 on experimental data
- Physical reasonableness of parameters
- Statistical validation passes standard tests

Required submission format:
{self.current_task.submission_format}

"""

        prompt += "\nAvailable input data:\n"

        # Display input data from dependencies
        for dep_task_id in self.current_task.input_from_tasks:
            if dep_task_id in self.task_group.results:
                dep_result = self.task_group.results[dep_task_id]
                if isinstance(dep_result, dict) and "answer" in dep_result:
                    prompt += f"- Input from {dep_task_id}: {dep_result['answer']}\n"
                else:
                    prompt += f"- Input from {dep_task_id}: {dep_result}\n"

        # Display initial input data
        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                if key != "work_dir":
                    prompt += f"- {key}: {value}\n"

        # Add note about dependencies
        if self.current_task.input_from_tasks:
            status = []
            for dep_id in self.current_task.input_from_tasks:
                status_text = (
                    "available"
                    if dep_id in self.task_group.results
                    else "not yet available"
                )
                status.append(f"{dep_id} ({status_text})")

            prompt += f"\n\nThis task uses output from tasks: {', '.join(status)}"

        logger.info(f"PROMPT : {prompt}")

        return prompt

    def score(self) -> float:
        """Score the submitted answer"""
        if not self.state.submitted_answer:
            return 0.0

        try:
            # Clean the submission
            submission_str = self.state.submitted_answer.strip()
            logger.info(f"Raw submission: {submission_str}")
            score = self.current_task.scoring_fn(
                prediction=submission_str, ground_truth=self.current_task.scoring_inputs
            )
            self.task_group.store_result(
                self.task_id, {"answer": submission_str}, score
            )
            logger.info(f"Score for task {self.task_id}: {score}")
            return score

        except Exception as e:
            logger.error(f"Error scoring submission for task {self.task_id}: {e!s}")
            logger.error(f"Submission was: {self.state.submitted_answer}")
            return 0.0


def create_task_environments(subtask_level: bool, env_level: int, rag: bool):
    """Create all task environments for the kinetic modeling benchmark."""

    tasks = return_tasks(subtask_level, env_level, rag)

    group_id = f"kinetic_modeling_level_{env_level}"

    logger.info(f"Creating task group {group_id} with tasks: {list(tasks.keys())}")
    task_group = TaskGroup(group_id=group_id, tasks=tasks)

    logger.info("\nTask Dependencies:")
    for task_id, deps in task_group.get_task_dependencies().items():
        logger.info(f"- {task_id}: depends on {deps}")

    # Print ordering of tasks
    ordered_tasks = task_group.get_ordered_tasks()
    logger.info("\nTask Execution Order:")
    for i, task_id in enumerate(ordered_tasks):
        logger.info(f"{i+1}. {task_id}")

    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = KineticEnvironment(
            task_id=task_id,
            task_group=task_group,
            work_dir=BASE_WORK_DIR,
        )

    return environments


# Server setup
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kinetic Modeling Server")
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("CORRAL_HOST", "0.0.0.0"),
        help="Host to run the server on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CORRAL_PORT", "8000")),
        help="Port to run the server on",
    )
    parser.add_argument(
        "--subtask_level",
        type=bool,
        default=False,
        help="Whether to use subtask level",
    )
    parser.add_argument(
        "--env_level",
        type=int,
        help="Environment level to use (1-3)",
    )
    parser.add_argument(
        "--rag",
        type=bool,
        default=False,
        help="Whether to use RAG tools",
    )
    args = parser.parse_args()
    environments = create_task_environments(
        subtask_level=args.subtask_level, env_level=args.env_level, rag=args.rag
    )
    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"  Depends on: {env.current_task.input_from_tasks}")

    run_server(
        environments=environments,
        host=args.host,
        port=args.port,
    )
