"""
WetLab (Qualitative Inorganic Analysis) Benchmark Server

Command-line arguments:
    --host: Host address to run the server (default: value of CORRAL_HOST env var or '0.0.0.0').
    --port: Port to run the server (default: value of CORRAL_PORT env var or 8000).
    --subtask_level: Whether to use subtask-level tasks (default: False).
"""

import argparse
import json
import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict

from wetlab.score import score_ion_list
from wetlab.tools import create_tools
from wetlab.engine import set_chemical_system, Solution, StockSolution

from corral.backend.env import Environment
from corral.backend.server import run_server
from corral.backend.task import TaskDefinition, TaskGroup

# BASE_WORK_DIR = os.environ.get(
#     "CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/WetLab"
# )

SCORING_FUNCTIONS = {
    "score_ion_list": score_ion_list,
}

REAGENTS_1 = {
    "Water": StockSolution(composition={}, description="Distilled water"),
    "HCl(0.02M)": StockSolution(composition={"H+": 0.02, "Cl-": 0.02}, description="HCl 0.02 M, in water"),
    "HCl(1M)": StockSolution(composition={"H+": 1, "Cl-":1}, description="HCl 1.0 M, in water"),
    "HCl(6M)": StockSolution(composition={"H+": 6, "Cl-": 6}, description="HCl 6.0 M, in water"),
    "HNO3": StockSolution(composition={"H+": 6, "NO3-": 6}, description="HNO3 6.0 M, in water"),
    "KOH(6M)": StockSolution(composition={"K+": 6, "OH-": 6}, description="KOH 6.0 M, in water"),
    "KOH(0.02M)": StockSolution(composition={"K+": 0.02, "OH-": 0.02}, description="KOH 0.02 M, in water"),
    "H2SO4": StockSolution(composition={"H+": 1, "HSO4-": 1}, description="H2SO4 1.0 M, in water"),
    "NH3": StockSolution(composition={"NH3": 5}, description="NH3 5.0 M, in water"),
    "NH4Cl": StockSolution(composition={"NH4+": 1, "Cl-": 1}, description="NH4Cl 1.0 M, in water"),
    "BUFFER_9": StockSolution(composition={"NH3": 0.36, "NH4+": 0.64, "Cl-": 1}, description="1.0 M ammonia buffer pH=9 (NH3 0.36 M, NH4Cl 0.64 M), in water"),
    "NH4I": StockSolution(composition={"NH4+": 1, "I-": 1}, description="NH4I 1.0 M, in water"),
    "H2S(acidic)": StockSolution(composition={"H2S": 0.1, "H+": 0.01, "Cl-": 0.01}, description="H2S 0.1 M, HCl 0.01 M, in water"),
    "(NH4)2S": StockSolution(composition={"NH4+": 0.2, "S-2": 0.1}, description="(NH4)2S 0.1 M, in water"),
    "(NH4)2CO3": StockSolution(composition={"NH4+": 0.4, "CO3-2": 0.2}, description="(NH4)2CO3 0.2 M, in water"),
    "(NH4)2HPO4": StockSolution(composition={"NH4+": 0.4, "HPO4-2": 0.2}, description="(NH4)2HPO4 0.2 M, in water"),
    "(NH4)2SO4": StockSolution(composition={"NH4+": 1, "SO4-2": 0.5}, description="(NH4)2SO4 0.5 M, in water"),
    "K2CrO4": StockSolution(composition={"K+": 0.4, "CrO4-2": 0.2}, description="K2CrO4 0.2 M, in water"),
    "NH4SCN": StockSolution(composition={"NH4+": 0.1, "SCN-": 0.1}, description="NH4SCN 0.1 M, in water"),
    "DMG": StockSolution(composition={"K+": 0.01, "Hdmg-": 0.01}, description="dimethylglyoxime potassium salt 0.01 M, in water"),
}

SYS_1 = "Cl N S I C K P Cr dmg"

REAGENTS_2 = {
    "AgNO3": StockSolution(composition={"Ag+": 0.1, "NO3-": 0.1}, description="AgNO3 0.1 M, in water"),
    "BaCl2": StockSolution(composition={"Ba+2": 0.1, "Cl-":0.2}, description="BaCl2 0.1 M, in water"),
    "CuSO4": StockSolution(composition={"Cu+2": 0.1, "SO4-2": 0.1}, description="CuSO4 0.1 M, in water"),
    "FeCl3": StockSolution(composition={"Fe+3": 0.05, "Cl-": 0.15}, description="FeCl3 0.05 M, in water"),
    "Hg(NO3)2": StockSolution(composition={"Hg+2": 0.05, "NO3-": 0.1}, description="Hg(NO3)2 0.05 M, in water"),
    "MnSO4": StockSolution(composition={"Mn+2": 0.1, "SO4-2": 0.1}, description="MnSO4 0.1 M, in water"),
    "Pb(NO3)2": StockSolution(composition={"Pb+2": 0.1, "NO3-": 0.2}, description="Pb(NO3)2 0. M, in water"),
}

SYS_2 = "Ag Ba Cu Fe(+3) Hg(+2) Mn Pb N(+5) Cl S(+6)"

@dataclass
class QualitativeAnalysisTask(TaskDefinition):
    task_sys: str = None
    sample_list: Dict[str, Solution] = None
    reagent_list: Dict[str, StockSolution] | str = None

    def __post_init__(self): # setting up the engine

        if self.reagent_list == "def_1":
            reagent_solutions = REAGENTS_1
            sys = ' '.join([SYS_1, self.task_sys])
        elif self.reagent_list == "def_2":
            reagent_solutions = REAGENTS_2
            sys = ' '.join([SYS_2, self.task_sys])
        elif self.reagent_list == "def_12":
            reagent_solutions = REAGENTS_1 | REAGENTS_2
            sys = ' '.join([SYS_1, SYS_2, self.task_sys])
        else:
            reagent_solutions = {reagent["label"]: StockSolution(composition=reagent["composition"], description=reagent["description"]) for reagent in self.reagent_list}
            sys = self.task_sys
        
        chemical_system = set_chemical_system(sys)
        original_samples = {sample["label"]: Solution(composition=sample["composition"], volume=sample["vol"], description="unknown") for sample in self.sample_list}
        # equilibrating the samples
        sample_solutions = {}
        for label, solution in original_samples.items():
            solution.equilibrate()
            if solution.has_precipitate:
                raise RuntimeError(f"Sample with label {label} resulted in a precipitate after equilibration.")
            else:
                sample_solutions[label] = solution

        object.__setattr__(self, "samples", sample_solutions)
        object.__setattr__(self, "reagents", reagent_solutions)
        object.__setattr__(self, "chemical_system", chemical_system)
        object.__setattr__(self, "sys", sys)

def load_tasks_from_json(
    json_path: Path, #work_dir: str = BASE_WORK_DIR
) -> dict[str, TaskDefinition]:
    tasks = {}
    for task_file in json_path.glob("*.json"):
        if not Path(task_file).is_file():
            raise ValueError(f"Task file {task_file} is not a valid file.")

        with task_file.open() as f:
            task_data = json.load(f)
        for data in task_data:
            task_id = data["id"]
            initial_input = data.get("initial_input", {}) #, {"work_dir": work_dir})
            input_from_tasks = data.get("input", {}).get("input_from_task", [])
            if not isinstance(input_from_tasks, list):
                input_from_tasks = []

            tasks[task_id] = QualitativeAnalysisTask(
                name=data["name"],
                description=data["input"]["prompt"],
                tools=data.get("tools", []),
                scoring_fn=SCORING_FUNCTIONS[data["scoring_fn"]],
                scoring_inputs=data["output"][0]["target"],
                submission_format=data.get("submission_format", ""),
                input_from_tasks=input_from_tasks,
                initial_input=initial_input,
                task_sys=data["input"]["sys"],
                sample_list=data["input"]["samples"],
                reagent_list=data["input"]["reagents"],
            )
    return tasks


class TaskEnvironment(Environment):
    """Environment that works with a task group

    Args:
        task_id (str): ID of the task to work on
        task_group (TaskGroup): Task group containing all the subtasks
        available_tools (dict[str, Tool]): All tools available in the environment

    Raises:
        ValueError: If task ID is not found in the task group
    """

    def __init__(
        self,
        task_id: str,
        task_group: TaskGroup,
        work_dir: str = "",
    ):
        self.task_id = task_id
        self.task_group = task_group
        self.available_tools = create_tools()
        self.work_dir = work_dir

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        # Initialize environment
        super().__init__(f"{task_group.group_id}_{task_id}" , base_work_dir=work_dir)

        compositions = self.current_task.samples | self.current_task.reagents
        self.hidden_args = {"compositions": compositions}

        logger.info(f"Initializing environment for task {self.task_id}")
        logger.info(f"Task name: {self.current_task}")
        self._add_task_tools()

    def configure_additional_apps(self):
        """Setting the Reaktoro chemical system"""
        set_chemical_system(self.current_task.chemical_system)
        logger.info(f"Reaktoro chemical system is set to '{self.current_task.sys}' for {self.task_id}.")

    def _add_task_tools(self):
        """Add tools required for the current task to the environment"""
        if len(self.current_task.tools) > 0:
            for tool_name in self.current_task.tools:
                if tool_name in self.available_tools:
                    self.add_tool(self.available_tools[tool_name])
                else:
                    logger.warning(
                        f"Tool {tool_name} not found in available tools for task {self.task_id}"
                    )
        else:
            # Adding all tools if no tools are specified
            for tool_name in self.available_tools:
                self.add_tool(self.available_tools[tool_name])

    def get_task_prompt(self) -> str:
        prompt = (
            f"Task {self.current_task.name}:\n"
            f"{self.current_task.description}\n\n"
            "Required submission format:\n"
            f"{self.current_task.submission_format}\n\n"
        )

        if len(self.current_task.input_from_tasks)>0:
            prompt += "\nAvailable data from previous subtasks:\n"

        # Display input data from dependencies
        for dep_task_id in self.current_task.input_from_tasks:
            dep_key = f"{self.task_group.group_id}_{dep_task_id}"
            if dep_key in self.task_group.results:
                dep_result = self.task_group.results[dep_key]
                task_prompt = self.task_group.tasks[dep_task_id].description
                if isinstance(dep_result, dict) and "answer" in dep_result:
                    prompt += f"- Input from '{dep_task_id}' with question: '{task_prompt}' and answer: '{dep_result['answer']}'\n"
                else:
                    prompt += f"- Input from '{dep_task_id}' with description: '{task_prompt}' and answer: '{dep_result}'\n"

        # Display initial input data
        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                if key != "work_dir":
                    prompt += f"- {key}: {value}\n"

        logger.info(f"Task prompt for {self.task_id}:\n{prompt}")
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


def create_qualysis_environments(
    #work_dir: str = BASE_WORK_DIR,
    subtask_level: bool = False,
) -> dict[str, Environment]:
    """Create environments for the WetLab (Qualitative Inorganic Analysis) benchmark tasks."""
    logger.info("Creating environments for Qualitative Inorganic Analysis tasks...")
    if subtask_level:
        json_path = Path(__file__).parent / "subtasks_json"
    else:
        json_path = Path(__file__).parent / "tasks_json"
    if not json_path.exists():
        raise ValueError(f"The path {json_path!r} does not exist.")

    logger.info(f"Loading tasks from {json_path!r}")

    tasks = load_tasks_from_json(json_path) #, work_dir=work_dir)

    group_id = "qualitative_inorganic_analysis"
    logger.info(f"Creating task group {group_id} with {len(tasks)} tasks")
    task_group = TaskGroup(
        group_id=group_id,
        tasks=tasks,
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

    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = TaskEnvironment(
            task_id=task_id,
            task_group=task_group,
            #work_dir=work_dir,
        )

    return environments


if __name__ == "__main__":

    load_dotenv()

    parser = argparse.ArgumentParser(description="Qualitative Inorganic Analysis Benchmark Server")
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
    args = parser.parse_args()

    #Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)

    # Create all environments with file system tools
    environments = create_qualysis_environments(
        #work_dir=BASE_WORK_DIR,
        subtask_level=args.subtask_level
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
