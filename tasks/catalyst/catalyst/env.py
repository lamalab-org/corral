import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from loguru import logger
from score import (
    check_adsorption_sites,
    check_adsorption_structure,
    check_mp_structure,
    check_slab_structure,
    check_slabs_json,
)
from tools import create_tools

from corral.base import Environment, Tool
from corral.io import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from corral.server import create_benchmark_server

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/temp")


@dataclass
class TaskDefinition:
    """Definition of a task with its requirements and scoring"""

    name: str
    description: str
    tools: list[str]
    scoring_fn: Callable[[dict | str], float]
    submission_format: dict[str, str]
    # Either use output from another task or custom input
    input_from_tasks: list[str] = field(
        default_factory=list
    )  # input required from some of the previous tasks in the group
    initial_input: dict[str, Any] = field(
        default_factory=dict
    )  # initial input for the task, if required

    # Helper method to check if task has dependencies
    def has_dependencies(self) -> bool:
        return len(self.input_from_tasks) > 0


@dataclass
class TaskGroup:
    """Container for related tasks"""

    group_id: str
    tasks: dict[str, TaskDefinition]
    results: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def get_task_input(self, task_id: str) -> dict[str, Any]:
        """Get input for a task either from other tasks or initial input"""
        task = self.tasks.get(task_id)
        if not task:
            return {}

        # Start with the initial input
        combined_input = task.initial_input.copy() if task.initial_input else {}

        # Add inputs from dependent tasks
        for dep_task_id in task.input_from_tasks:
            if dep_task_id in self.results:
                # Add the dependent task's result to the input
                dep_result = self.results[dep_task_id]
                if isinstance(dep_result, dict) and "answer" in dep_result:
                    # Extract the relevant part of the result
                    combined_input[f"result_from_{dep_task_id}"] = dep_result["answer"]
                else:
                    combined_input[f"result_from_{dep_task_id}"] = dep_result

        return combined_input

    def store_result(self, task_id: str, result: dict[str, Any], score: float) -> None:
        """Store task result and score"""
        self.results[task_id] = result
        self.scores[task_id] = score

    def get_task_dependencies(self) -> dict[str, list[str]]:
        """Get dictionary of task dependencies"""
        return {task_id: task.input_from_tasks for task_id, task in self.tasks.items()}

    def get_ordered_tasks(self) -> list[str]:
        """Return tasks in dependency order"""
        # Simple topological sort
        dependencies = self.get_task_dependencies()
        visited = set()
        ordered = []

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            for dep in dependencies.get(task_id, []):
                visit(dep)
            ordered.append(task_id)

        for task_id in self.tasks:
            visit(task_id)

        return ordered

    def check_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        return all(
            dep_task_id in self.results and self.results[dep_task_id] is not None
            for dep_task_id in task.input_from_tasks
        )


class TaskEnvironment(Environment):
    """Environment that works with a task group"""

    def __init__(
        self,
        task_id: str,
        task_group: TaskGroup,
        available_tools: dict[str, Tool],
        fs_tools: dict[str, Tool] | None = None,
    ):
        self.task_group = task_group
        self.task_id = task_id
        self.available_tools = available_tools
        self.fs_tools = fs_tools or {}
        self._trail_name = None

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        # Initialize tools and environment
        self.tools = {}
        super().__init__(f"{task_group.group_id}_{task_id}")

        # Add required tools for the task
        for tool_name in self.current_task.tools:
            if tool_name in available_tools:
                self.add_tool(available_tools[tool_name])

        # Add all file system tools
        for tool in self.fs_tools.values():
            self.add_tool(tool)

    def get_task_prompt(self) -> str:
        _combined_input = self.task_group.get_task_input(self.task_id)

        prompt = f"""Task: {self.current_task.name}
Description: {self.current_task.description}

Required submission format:
"""
        for key, desc in self.current_task.submission_format.items():
            prompt += f"- {key}: {desc}\n"

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
                prompt += f"- {key}: {value}\n"

        # Add IO tools description for saving results
        prompt += "\nIMPORTANT: You have access to filesystem tools which allow you to read and write files. Also you can retry many times to get the correct answer. "
        prompt += "Since some task results will be used in subsequent tasks, make sure to save your results using appropriate filenames. "
        prompt += (
            "This will help you reference and retrieve these files in later tasks."
        )

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

        logger.info(f"Task prompt for {self.task_id}:\n{prompt}")
        return prompt

    def score(self) -> float:
        """Score the submitted answer"""
        if not self.state.submitted_answer:
            return 0.0

        try:
            submission_str = self.state.submitted_answer.strip()
            logger.info(f"Raw submission: {submission_str}")

            # Try to parse as JSON first
            try:
                submission = json.loads(submission_str)

                # If submission is a dict, look for answer field or alternatives
                if isinstance(submission, dict):
                    # Check for alternative keys if "answer" not present
                    if "answer" not in submission:
                        # Check for other common keys
                        possible_keys = ["ans", "answers"]
                        for key in possible_keys:
                            if key in submission:
                                # Map alternative key to "answer"
                                submission["answer"] = submission[key]
                                logger.info(
                                    f"Found alternative key '{key}', mapped to 'answer'"
                                )
                                break

                    # Now extract answer for scoring, default to full submission if still no answer key
                    answer_for_scoring = submission.get("answer", submission_str)
                else:
                    # If submission parsed as JSON but is not a dict (e.g., a list or primitive),
                    # use it directly and wrap in a dict for storage
                    answer_for_scoring = submission
                    submission = {"answer": submission}

            except json.JSONDecodeError:
                # If not valid JSON, use the string directly for scoring
                # and create a dict for storage
                answer_for_scoring = submission_str
                submission = {"answer": submission_str}

            # Store full result in task group
            logger.info(f"Parsed submission: {submission}")
            logger.info(f"Using for scoring: {answer_for_scoring}")

            # Call the scoring function with the extracted answer
            score = self.current_task.scoring_fn(answer_for_scoring)

            self.task_group.store_result(self.task_id, submission, score)

            return score
        except Exception as e:
            logger.error(f"Error scoring submission for task {self.task_id}: {e!s}")
            logger.error(f"Submission was: {self.state.submitted_answer}")
            return 0.0


def create_catalysis_environments(
    fs_tools: dict[str, Tool] | None = None, work_dir: str = BASE_WORK_DIR
) -> dict[str, Environment]:
    """Create environments for catalysis tasks"""

    # Ensure work directory exists
    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)

    # Create task group with improved dependency management
    task_group = TaskGroup(
        group_id="catalyst",
        tasks={
            "retrieve_structure": TaskDefinition(
                name="Retrieve Bulk Structure",
                description="Retrieve structure of Si from Materials Project using the MP-149 ID and save it as a CIF file. Submit the path to the CIF file.",
                tools=["get_structure_from_mp_text"],
                scoring_fn=check_mp_structure,
                submission_format={"answer": "/path/to/bulk_structure.cif"},
                initial_input={
                    "mp_id": "mp-149",  # Silicon
                    "work_dir": str(work_dir_path),
                },
            ),
            "enumerate_slabs": TaskDefinition(
                name="Enumerate Possible Slabs",
                description="Enumerate possible slabs from the bulk Si structure with Miller index (1,1,1) and save the result as a JSON file. Submit the path to the JSON file.",
                tools=["enumerate_slabs_text"],
                scoring_fn=check_slabs_json,
                submission_format={"answer": "/path/to/slabs.json"},
                input_from_tasks=["retrieve_structure"],
                initial_input={
                    "miller_index": (1, 1, 1),
                    "min_slab_size": 12,
                    "min_vacuum_size": 5,
                    "work_dir": str(work_dir_path),
                },
            ),
            "choose_slab": TaskDefinition(
                name="Choose Slab",
                description="Choose one slab from the enumerated slabs (by index) and save it as a CIF file. Submit the path to the CIF file.",
                tools=["choose_slab_text"],
                scoring_fn=check_slab_structure,
                submission_format={"answer": "/path/to/chosen_slab.cif"},
                input_from_tasks=["enumerate_slabs"],
                initial_input={
                    "index": 0,  # Default to first slab
                    "work_dir": str(work_dir_path),
                },
            ),
            "create_molecule": TaskDefinition(
                name="Create CO2 Molecule",
                description="Retrieve CO2 molecule structure using MP - ID save it as a CIF file. Submit the path to the CIF file.",
                tools=["get_structure_from_mp_text"],
                scoring_fn=check_mp_structure,
                submission_format={"answer": "/path/to/co2.cif"},
                initial_input={"mp_id": "mp-20066", "work_dir": str(work_dir_path)},
            ),
            "get_adsorption_sites": TaskDefinition(
                name="Identify Adsorption Sites",
                description="Determine possible adsorption sites on the chosen slab and save the results as a JSON file. Submit the path to the JSON file.",
                tools=["get_adsorption_sites_text"],
                scoring_fn=check_adsorption_sites,
                submission_format={"answer": "/path/to/adsorption_sites.json"},
                input_from_tasks=["choose_slab"],
                initial_input={
                    "work_dir": str(work_dir_path),
                },
            ),
            "choose_adsorption_site": TaskDefinition(
                name="Choose Adsorption Site",
                description="Choose one adsorption site (preferably a top site) from the identified sites and save the coordinates to a file. Submit the path to the file.",
                tools=["choose_adsorption_site_text"],
                scoring_fn=lambda path: 1.0 if Path(path).exists() else 0.0,
                submission_format={"answer": "/path/to/chosen_site.json"},
                input_from_tasks=["get_adsorption_sites"],
                initial_input={
                    "site_type": "top",  # Default to top site
                    "index": 0,  # Default to first site of the type
                    "work_dir": str(work_dir_path),
                },
            ),
            "add_adsorbate": TaskDefinition(
                name="Add CO2 to Silicon Slab",
                description="Place the CO2 molecule on the chosen slab at the specified adsorption site with a height of approximately 2.0 Å and save the combined structure as a CIF file. Submit the path to the CIF file.",
                tools=["add_adsorbate_to_slab_text"],
                scoring_fn=check_adsorption_structure,
                submission_format={"answer": "/path/to/slab_with_co2.cif"},
                input_from_tasks=[
                    "choose_slab",
                    "create_molecule",
                    "choose_adsorption_site",
                ],
                initial_input={
                    "height": 2.0,  # Å above the surface
                    "work_dir": str(work_dir_path),
                },
            ),
        },
    )

    # Print task dependencies for reference
    logger.info("\nTask Dependencies:")
    for task_id, deps in task_group.get_task_dependencies().items():
        logger.info(f"- {task_id}: depends on {deps}")

    # Print ordering of tasks
    ordered_tasks = task_group.get_ordered_tasks()
    logger.info("\nTask Execution Order:")
    for i, task_id in enumerate(ordered_tasks):
        logger.info(f"{i+1}. {task_id}")

    # Create all available tools
    # This assumes you have a function to create the tools
    available_tools = create_tools()

    # Create environments for all tasks
    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = TaskEnvironment(
            task_id=task_id,
            task_group=task_group,
            available_tools=available_tools,
            fs_tools=fs_tools,
        )

    return environments


if __name__ == "__main__":
    # Create file system manager and tools
    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)
    fs_manager = FSManager("file", base_path=BASE_WORK_DIR)

    fs_tools = {
        "list_files": ListFilesTool(fs_manager),
        "read_file": ReadFileTool(fs_manager),
        "write_file": WriteFileTool(fs_manager),
        "file_info": FileInfoTool(fs_manager),
        "cat_files": CatFilesTool(fs_manager),
        "copy_file": CopyFileTool(fs_manager),
    }

    # Create all environments with file system tools
    environments = create_catalysis_environments(fs_tools=fs_tools)

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"  Depends on: {env.current_task.input_from_tasks}")

    # Create and run server
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
