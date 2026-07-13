"""
WetLab (Qualitative Inorganic Analysis) Benchmark Server

Command-line arguments:
    --host: Host address to run the server (default: value of CORRAL_HOST env var or '0.0.0.0').
    --port: Port to run the server (default: value of CORRAL_PORT env var or 8000).
    --level: The level of tasks to run the benchmark on (default: 2)
    --subtask: Whether to use subtask-level tasks (default: False).
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict

from wetlab.score import score_ion_list, score_salt, none_checker
from wetlab.tools import create_tools
from wetlab.engine import set_chemical_system, Solution, StockSolution, Precipitate

from corral.backend.env import Environment, Toolset, build_environments
from corral.backend.server import run_server
from corral.backend.task import InputRef, TaskDefinition

SCORING_FUNCTIONS = {
    "none_checker": none_checker,
    "score_ion_list": score_ion_list,
    "score_salt": score_salt
}

BASIC_REAGENTS = {
    "Water": StockSolution(composition={}, description="Distilled water"),
    "H2SO4": StockSolution(composition={"H+": 1, "HSO4-": 1}, description="H2SO4 1.0 M"),
    "KOH": StockSolution(composition={"K+": 1, "OH-": 1}, description="KOH 1.0 M"),
}
BASIC_SYS = "K S(+6)"

REAGENTS_1 = {
    "Water": StockSolution(composition={}, description="distilled water"),
    "HCl(0.02M)": StockSolution(composition={"H+": 0.02, "Cl-": 0.02}, description="HCl 0.02 M"),
    "HCl(1M)": StockSolution(composition={"H+": 1, "Cl-":1}, description="HCl 1.0 M"),
    "HCl(6M)": StockSolution(composition={"H+": 6, "Cl-": 6}, description="HCl 6.0 M"),
    "HNO3(6M)": StockSolution(composition={"H+": 6, "NO3-": 6}, description="HNO3 6.0 M"),
    "HNO3(0.1M)": StockSolution(composition={"H+": 0.1, "NO3-": 0.1}, description="HNO3 0.1 M"),
    "KOH(6M)": StockSolution(composition={"K+": 6, "OH-": 6}, description="KOH 6.0 M"),
    "KOH(1M)": StockSolution(composition={"K+": 1, "OH-": 1}, description="KOH 1.0 M"),
    "KOH(0.05M)": StockSolution(composition={"K+": 0.05, "OH-": 0.05}, description="KOH 0.05 M"),
    "H2SO4(1M)": StockSolution(composition={"H+": 1, "HSO4-": 1}, description="H2SO4 1.0 M"),
    "NH3(1M)": StockSolution(composition={"NH3": 1}, description="NH3 1.0 M"),
    "NH3(5M)": StockSolution(composition={"NH3": 5}, description="NH3 5.0 M"),
    "NH4Cl": StockSolution(composition={"NH4+": 1, "Cl-": 1}, description="NH4Cl 1.0 M"),
    "BUFFER_9": StockSolution(composition={"NH3": 0.72, "NH4+": 1.28, "Cl-": 1.28}, description="2.0 M NH3/NH4Cl buffer pH=9.0"),
    "NH4I": StockSolution(composition={"NH4+": 1, "I-": 1}, description="NH4I 1.0 M"),
    "H2S(acidic)": StockSolution(composition={"H2S": 0.1, "H+": 0.05, "Cl-": 0.05}, description="H2S 0.1 M + HCl 0.05 M"),
    "(NH4)2S": StockSolution(composition={"NH4+": 1.4, "S-2": 0.5, "NO3-": 0.4}, description="(NH4)2S 0.5 M + NH4NO3 0.4 M, (pH=9.0)"),
    "(NH4)2CO3": StockSolution(composition={"NH4+": 0.4, "CO3-2": 0.2}, description="(NH4)2CO3 0.2 M"),
    "(NH4)2HPO4": StockSolution(composition={"NH4+": 0.4, "HPO4-2": 0.2}, description="(NH4)2HPO4 0.2 M"),
    "(NH4)2SO4": StockSolution(composition={"NH4+": 1, "SO4-2": 0.5}, description="(NH4)2SO4 0.5 M"),
    "K2CrO4": StockSolution(composition={"K+": 0.4, "CrO4-2": 0.2}, description="K2CrO4 0.2 M"),
    "NH4SCN": StockSolution(composition={"NH4+": 0.1, "SCN-": 0.1}, description="NH4SCN 0.1 M"),
    "DMG": StockSolution(composition={"K+": 0.01, "Hdmg-": 0.01}, description="dimethylglyoxime potassium salt 0.01 M"),
}
SYS_1 = "Cl N S I C K P Cr dmg"

REAGENTS_2 = {
    "Water": StockSolution(composition={}, description="distilled water"),
    "AgNO3": StockSolution(composition={"Ag+": 0.1, "NO3-": 0.1}, description="AgNO3 0.1 M"),
    "Ba(NO3)2": StockSolution(composition={"Ba+2": 0.1, "NO3-":0.2}, description="Ba(NO3)2 0.1 M"),
    "Pb(NO3)2": StockSolution(composition={"Pb+2": 0.1, "NO3-": 0.2}, description="Pb(NO3)2 0.1 M"),
    "HNO3(6M)": StockSolution(composition={"H+": 6, "NO3-": 6}, description="HNO3 6.0 M"),
    "HNO3(0.1M)": StockSolution(composition={"H+": 0.1, "NO3-": 0.1}, description="HNO3 0.1 M"),
    "HCl(1M)": StockSolution(composition={"H+": 1, "Cl-": 1}, description="HCl 1.0 M"),
    "KOH(6M)": StockSolution(composition={"K+": 6, "OH-": 6}, description="KOH 6.0 M"),
    "KOH(0.05M)": StockSolution(composition={"K+": 0.05, "OH-": 0.05}, description="KOH 0.05 M"),
    "H2SO4(1M)": StockSolution(composition={"H+": 1, "HSO4-": 1}, description="H2SO4 1.0 M"),
    "NH3(1M)": StockSolution(composition={"NH3": 1}, description="NH3 1.0 M"),
    "NH3(5M)": StockSolution(composition={"NH3": 5}, description="NH3 5.0 M"),
    "NH4Cl": StockSolution(composition={"NH4+": 1, "Cl-": 1}, description="NH4Cl 1.0 M"),
    "BUFFER_9": StockSolution(composition={"NH3": 0.72, "NH4+": 1.28, "Cl-": 1.28}, description="2.0 M NH3/NH4Cl buffer, (pH=9.0)"),
    "NH4I": StockSolution(composition={"NH4+": 1, "I-": 1}, description="NH4I 1.0 M"),
    "H2S(acidic)": StockSolution(composition={"H2S": 0.1, "H+": 0.05, "Cl-": 0.05}, description="H2S 0.1 M + HCl 0.05 M"),
    "(NH4)2S": StockSolution(composition={"NH4+": 1.4, "S-2": 0.5, "NO3-": 0.4}, description="(NH4)2S 0.5 M + NH4NO3 0.4 M, (pH=9.0)"),
    "(NH4)2CO3": StockSolution(composition={"NH4+": 0.4, "CO3-2": 0.2}, description="(NH4)2CO3 0.2 M"),
    "(NH4)2HPO4": StockSolution(composition={"NH4+": 0.4, "HPO4-2": 0.2}, description="(NH4)2HPO4 0.2 M"),
    "K2CrO4": StockSolution(composition={"K+": 0.4, "CrO4-2": 0.2}, description="K2CrO4 0.2 M"),
}
SYS_2 = "Ag Ba Pb N K S(+6) S(-2) Cl I C(+4) P Cr"


def _is_charge_neutral(solution: StockSolution | Solution, threshold: float=1e-10) -> bool:
    if type(solution) == StockSolution:
        solution = 1 * solution
    if abs(solution.state.charge()) > threshold:
        return False
    else:
        return True

@dataclass(frozen=True)
class QualitativeAnalysisTask(TaskDefinition):
    task_sys: str = None
    sample_list: Dict[str, Solution] = None
    reagent_set: str = None
    additional_reagents: Dict[str, StockSolution] = None
    initial: bool = True

    def __post_init__(self): # setting up the engine

        if self.reagent_set == "Basic":
            reagent_solutions = BASIC_REAGENTS.copy()
            sys = ' '.join([BASIC_SYS, self.task_sys])
        elif self.reagent_set == "R1":
            reagent_solutions = REAGENTS_1.copy()
            sys = ' '.join([SYS_1, self.task_sys])
        elif self.reagent_set == "R2":
            reagent_solutions = REAGENTS_2.copy()
            sys = ' '.join([SYS_2, self.task_sys])
        else:
            raise ValueError(f"Invalid reagent_set: {self.reagent_set}")

        if len(self.additional_reagents) > 0:
            additional_reagents = {reagent["label"]: StockSolution(composition=reagent["composition"], description=reagent["description"]) for reagent in self.additional_reagents}
            reagent_solutions |= additional_reagents
        
        chemical_system = set_chemical_system(sys)
        original_samples = {sample["label"]: Solution(composition=sample["composition"], volume=sample["vol"], description="unknown") for sample in self.sample_list}
        
        # checking charge neutrality    
        for label, solution in (original_samples | reagent_solutions).items() : 
            if not _is_charge_neutral(solution):
                solution_type = "Sample solution" if type(solution) == Solution else "Reagent solution"
                raise RuntimeError(f"{solution_type} with label {label!r} is not charge neutral!")
        
        # equilibrating the samples and checking for precipitates
        sample_solutions = {}
        for label, solution in original_samples.items():
            solution.equilibrate()
            if solution.has_precipitate:
                raise RuntimeError(f"Sample with label {label!r} resulted in a precipitate after equilibration.")
            else:
                sample_solutions[label] = solution

        object.__setattr__(self, "samples", sample_solutions)
        object.__setattr__(self, "reagents", reagent_solutions)
        object.__setattr__(self, "chemical_system", chemical_system)
        object.__setattr__(self, "sys", sys)


def load_tasks_from_json(
    json_path: Path,
) -> Dict[str, QualitativeAnalysisTask]:
    tasks = {}
    for task_file in json_path.glob("*.json"):
        if not Path(task_file).is_file():
            raise ValueError(f"Task file {task_file} is not a valid file.")

        with task_file.open() as f:
            task_data = json.load(f)
        for data in task_data:
            task_id = data["id"]
            initial_input = data.get("initial_input", {})
            input_from_tasks = data.get("input", {}).get("input_from_task", [])
            if not isinstance(input_from_tasks, list):
                input_from_tasks = []

            tasks[task_id] = QualitativeAnalysisTask(
                name=data["name"],
                description=data["input"]["prompt"],
                tools=data.get("tools", []),
                excluded_tools=tuple(data.get("excluded_tools", [])),
                scoring_fn=SCORING_FUNCTIONS[data["scoring_fn"]],
                scoring_inputs=data["output"][0]["target"],
                submission_format=data.get("submission_format", ""),
                input_map={dep: InputRef(dep) for dep in input_from_tasks},
                initial_input=initial_input,
                task_sys=data["input"]["sys"],
                sample_list=data["input"]["samples"],
                reagent_set=data["input"]["reagent_set"],
                additional_reagents=data["input"].get("additional_reagents",[]),
                initial=data["input"].get("initial", True)
            )
    return tasks


class QualitativeAnalysisEnvironment(Environment):
    """Environment for one QualitativeAnalysisTask.

    Linked tasks are normal tasks: they share output/score stores through
    their `CorralState` when created together.

    Args:
        task_id (str): ID of the task to work on
        task (TaskDefinition): Definition of the task
        base_work_dir (str): Base working directory
        group_tasks (dict[str, TaskDefinition]): All linked task definitions
        shared_task_runs (dict): Run store shared between linked environments
    """

    def configure_additional_apps(self):
        """Setting the Reaktoro chemical system and the Inventory"""
        set_chemical_system(self.current_task.chemical_system)
        logger.info(f"Reaktoro chemical system is set to '{self.current_task.sys}' for {self.task_id}.")

        if self.current_task.initial:
            logger.info("Resetting the Inventory...")
            new_samples = {label: sample.clone() for label, sample in self.current_task.samples.items()} # cloning the original task samples to make trials independent
            compositions = new_samples | self.current_task.reagents
            self.hidden_args = {"compositions": compositions}
            try:
                comp_file = Path(self.base_work_dir) / "compositions.pkl"
                os.remove(comp_file)
            except FileNotFoundError:
                logger.info("No leftover 'compositions.pkl' file to remove.")
        else:
            self.load_inventory()

    def get_task_prompt(self) -> str:
        prompt = (
            f"Task {self.current_task.name}:\n"
            f"{self.current_task.description}\n\n"
            "Required submission format:\n"
            f"{self.current_task.submission_format}\n\n"
        )

        if self.current_task.input_map:
            prompt += "\nAvailable data from previous subtasks:\n"

        # Display resolved inputs from dependencies
        for ref in self.current_task.input_map.values():
            if self.state.is_completed(ref.task_id):
                value = self.state.get_output(ref.task_id, ref.key)
                task_prompt = self.group_tasks[ref.task_id].description
                prompt += f"- Input from '{ref.task_id}' with question: '{task_prompt}' and answer: '{value}'\n"

        # Display initial input data
        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                if key != "work_dir":
                    prompt += f"- {key}: {value}\n"

        logger.info(f"Task prompt for {self.task_id}:\n{prompt}")
        return prompt

    def save_inventory(self):
        comp = {name: obj.to_dict() for name,obj in self.hidden_args["compositions"].items()}
        comp_file = Path(self.base_work_dir) / f"compositions.pkl"
        logger.info(f"Saving inventory to file: {str(comp_file)}")
        with open(comp_file, 'wb') as f:
            pickle.dump(comp, f)

    def load_inventory(self):
        comp_file = Path(self.base_work_dir) / f"compositions.pkl"
        logger.info(f"Loading inventory from file: {str(comp_file)}")
        with open(comp_file, 'rb') as f:
            data = pickle.load(f)
    
        comp = dict()
        for name, obj in data.items():
            if obj["type"] == "Precipitate":
                comp[name] = Precipitate.from_dict(obj)
            elif obj["type"] == "StockSolution":
                comp[name] = StockSolution.from_dict(obj)
            elif obj["type"] == "Solution":
                comp[name] = Solution.from_dict(obj)
            else:
                raise TypeError("Unknown object type when reading inventory from file!")
             
        self.hidden_args = {"compositions": comp}
        return

    def score(self) -> float:
        """Score the submitted answer"""
        # writing the final inventory to file
        self.save_inventory()
        
        if not self.state.submitted_answer:
            return 0.0

        try:
            # Clean the submission
            submission_str = self.state.submitted_answer.strip()
            logger.info(f"Raw submission: {submission_str!r}")
            score = self.current_task.scoring_fn(
                prediction=submission_str, ground_truth=self.current_task.scoring_inputs
            )
            self.state.store_task_output(self.task_id, submission_str, score)
            logger.info(f"Score for task {self.task_id}: {score}")
            return score

        except Exception as e:
            logger.error(f"Error scoring submission for task {self.task_id}: {e!s}")
            logger.error(f"Submission was: {self.state.submitted_answer!r}")
            return 0.0


def create_qualysis_environments(
    level: int = 2,
    subtask: bool = False,
) -> dict[str, Environment]:
    """Create environments for the WetLab (Qualitative Inorganic Analysis) benchmark tasks."""
    logger.info("Creating environments for Qualitative Inorganic Analysis tasks...")
    if subtask:
        json_path = Path(__file__).parent / "subtasks_json" / f"level_{level}"
    else:
        json_path = Path(__file__).parent / "tasks_json" / f"level_{level}"
    if not json_path.exists():
        raise ValueError(f"The path {json_path} does not exist.")

    logger.info(f"Loading tasks from {json_path}")

    tasks = load_tasks_from_json(json_path)

    logger.info(f"Creating linked task environments with {len(tasks)} tasks")

    # Wetlab keeps a subclass for its pickled inventory; grouping is derived.
    # An empty `tools` list means "the whole pool" and `excluded_tools` (a base
    # TaskDefinition field) is honoured generically by Toolset.resolve, so the
    # old `_add_task_tools` override is gone.
    return build_environments(
        tasks,
        name="wetlab",
        toolset=Toolset(
            pool=create_tools(),
            workspace_factory=None,
            select_all_when_unspecified=True,
        ),
        env_cls=QualitativeAnalysisEnvironment,
    )


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
        "--level",
        type=int,
        default=2,
        help="The level of tasks"
    )
    parser.add_argument(
        "--subtask",
        type=bool,
        default=False,
        help="Whether to use subtasks",
    )
    args = parser.parse_args()

    # Create all environments
    environments = create_qualysis_environments(
        level=args.level,
        subtask=args.subtask
    )

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_map:
            logger.info(f"  Depends on: {sorted(env.current_task.dependencies())}")

    run_server(
        environments=environments,
        host=args.host,
        port=args.port,
    )
