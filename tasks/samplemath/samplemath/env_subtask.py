import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from samplemath.tools import calculator, percentage_calculator

from corral.backend.env import Environment, Toolset, build_environments
from corral.backend.server import run_server
from corral.backend.task import InputRef, TaskDefinition
from corral.backend.tool import Tool

# Base working directory
if "CORRAL_WORK_DIR" not in os.environ:
    raise OSError("Environment variable 'CORRAL_WORK_DIR' is not set.")
BASE_WORK_DIR = os.environ["CORRAL_WORK_DIR"]


def addition_score(expected_answer: float | None = None):
    """Factory function that returns a scoring function for addition tasks"""

    def score_fn(result: str) -> float:
        """Score an addition task with expected answer validation"""
        try:
            # Handle string submissions
            if isinstance(result, str):
                result = result.strip()
                # Try to parse as JSON first
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict) and "answer" in parsed_result:
                        answer = float(parsed_result["answer"])
                    else:
                        # If not a dict with answer, treat the whole thing as the answer
                        answer = float(result)
                except json.JSONDecodeError:
                    # If not JSON, treat as direct numerical answer
                    answer = float(result)
            else:
                # If already parsed
                if isinstance(result, dict) and "answer" in result:
                    answer = float(result["answer"])
                else:
                    answer = float(result)

            if expected_answer is not None:
                # Check if answer matches expected value (within tolerance)
                return 1.0 if abs(answer - expected_answer) < 0.001 else 0.0
            else:
                # Just check if it's a valid number
                return 1.0

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Error parsing result for addition_score: {e}, result was: {result}"
            )
            return 0.0

    return score_fn


def multiplication_score(expected_answer: float | None = None):
    """Factory function that returns a scoring function for multiplication tasks"""

    def score_fn(result: str) -> float:
        """Score a multiplication task with expected answer validation"""
        try:
            # Handle string submissions
            if isinstance(result, str):
                result = result.strip()
                # Try to parse as JSON first
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict) and "answer" in parsed_result:
                        answer = float(parsed_result["answer"])
                    else:
                        # If not a dict with answer, treat the whole thing as the answer
                        answer = float(result)
                except json.JSONDecodeError:
                    # If not JSON, treat as direct numerical answer
                    answer = float(result)
            else:
                # If already parsed
                if isinstance(result, dict) and "answer" in result:
                    answer = float(result["answer"])
                else:
                    answer = float(result)

            if expected_answer is not None:
                # Check if answer matches expected value (within tolerance)
                return 1.0 if abs(answer - expected_answer) < 0.001 else 0.0
            else:
                # Just check if it's a valid number
                return 1.0

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Error parsing result for multiplication_score: {e}, result was: {result}"
            )
            return 0.0

    return score_fn


# Registry of scoring functions
SCORING_FUNCTIONS = {
    "addition_score": addition_score,
    "multiplication_score": multiplication_score,
}


def get_scoring_function(name: str, params: dict | None = None) -> Callable:
    """Get a scoring function by name from the registry, with optional parameters"""
    fn = SCORING_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Scoring function '{name}' not found in the registry")

    # If it's a factory function (i.e., takes arguments), call with params
    if params:
        try:
            return fn(**params)
        except Exception as e:
            raise ValueError(
                f"Error initializing scoring function '{name}' with params {params}: {e}"
            ) from e
    else:
        return fn


def load_tasks_from_json(
    json_path: str | Path, work_dir: str
) -> dict[str, TaskDefinition]:
    """Load task definitions from a JSON file.

    Args:
        json_path: Path to the JSON file containing task definitions
        work_dir: Working directory to use for task execution

    Returns:
        dictionary of task definitions keyed by task ID
    """
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Task definition file not found: {json_path}")

    with Path(json_path).open() as f:
        task_data = json.load(f)

    tasks = {}
    for entry in task_data:
        task_id = entry["id"]
        scoring_fn_name = entry.get("scoring_function")
        if scoring_fn_name is None:
            raise ValueError(
                f"Task '{task_id}' is missing a 'scoring_function'. "
                f"Available scoring functions: {sorted(SCORING_FUNCTIONS)}"
            )
        scoring_params = entry.get("scoring_params", {})
        scoring_fn = get_scoring_function(scoring_fn_name, scoring_params)

        initial_input = entry.get("initial_input", {}).copy()
        if "work_dir" not in initial_input:
            initial_input["work_dir"] = work_dir

        tasks[task_id] = TaskDefinition(
            name=entry["name"],
            description=entry["description"],
            tools=entry.get("tools", []),
            scoring_fn=scoring_fn,
            submission_format=entry.get("submission_format", ""),
            input_map={
                dep: InputRef(dep) for dep in entry.get("input_from_tasks", [])
            },
            initial_input=initial_input,
            prompt_fn=_samplemath_prompt,
            resolve_answer=False,
        )

    return tasks


def _samplemath_prompt(env: Environment) -> str:
    """Task prompt rendering the resolved dependency outputs.

    Inputs are resolved strictly: by the time the prompt is requested every
    dependency has run (the runner enforces topological order), so there is no
    "not yet available" placeholder.
    """
    task = env.current_task
    prompt = f"""Task: {task.name}
    Description: {task.description}

    Required submission format:
    {task.submission_format}

    """

    # Strict resolution: raises if a dependency has not produced an output.
    resolved = env.state.resolve_inputs(task)

    prompt += "\nAvailable input data:\n"

    # Display resolved inputs from dependencies
    for input_name, ref in task.input_map.items():
        prompt += f"- {input_name} (from {ref.task_id}): {resolved[input_name]}\n"

    # Display initial input data
    for key, value in task.initial_input.items():
        if key != "work_dir":
            prompt += f"- {key}: {value}\n"

    # Add workspace info
    if env.state.workspace:
        prompt += "\nIMPORTANT: You have access to filesystem tools. All files will be saved in your isolated workspace.\n"

    logger.info(f"DEBUG: Generated prompt for {env.task_id}:\n{prompt}")
    return prompt


def create_environments(
    task_json_path: str | Path,
    taskgroup_common_tools: dict[str, Tool] | None = None,
    work_dir: str = BASE_WORK_DIR,
) -> dict[str, Environment]:
    """Create environments for tasks defined in a JSON file

    Args:
        task_json_path: Path to the JSON file with task definitions
        taskgroup_common_tools: dictionary of Tools which are common for subtasks, for example file system tools
        work_dir: Working directory for task execution

    Returns:
        dictionary of environments keyed by task ID
    """

    logger.info(f"Creating environments from {task_json_path} with work_dir {work_dir}")

    # Load tasks from JSON
    tasks = load_tasks_from_json(task_json_path, work_dir)

    name = Path(task_json_path).stem  # Use filename (without extension) as label
    logger.info(f"Creating linked task environments for group: {name}")

    # Create environments for all tasks; grouping is derived from the graph
    return build_environments(
        tasks,
        base_work_dir=work_dir,
        name=name,
        toolset=Toolset(
            pool={
                "calculator": calculator,
                "percentage_calculator": percentage_calculator,
            },
            common=taskgroup_common_tools or {},
        ),
    )


if __name__ == "__main__":
    # Determine tasks file path
    if len(sys.argv) > 1:
        tasks_json_path = sys.argv[1]
    else:
        tasks_json_path = os.environ.get(
            "CORRAL_TASKS_PATH",
            Path(__file__).parent / "tasks" / "catalysis_tasks.json",
        )

    # Get server settings from environment if provided
    host = os.environ.get("CORRAL_HOST", "0.0.0.0")
    port = int(os.environ.get("CORRAL_PORT", "8000"))
    work_dir = os.environ.get("CORRAL_WORK_DIR", BASE_WORK_DIR)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    # Create environments
    environments = create_environments(
        task_json_path=tasks_json_path,
        work_dir=work_dir,
    )

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_map:
            logger.info(f"  Depends on: {sorted(env.current_task.dependencies())}")

    # Run server
    run_server(environments, host, port)
