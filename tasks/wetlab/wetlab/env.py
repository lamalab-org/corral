"""WetLab (qualitative inorganic analysis) environment definitions."""

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from wetlab.engine import (
    ChemicalSystemSpec,
    Solution,
    StockSolution,
    WetlabEngine,
    WetlabState,
)
from wetlab.score import none_checker, score_ion_list, score_salt
from wetlab.tools import create_tools

from corral.core import ToolExecutionResult
from corral.core.environment import Environment, Toolset, build_environments
from corral.core.events import TaskConfigured, WorkspaceDelta
from corral.core.state import ExecutionState, TaskOutput
from corral.core.task import InputRef, TaskDefinition, with_fixed_inputs
from corral.core.tool import Tool
from corral.core.transition import environment_operations
from corral.report.logging import event, exception_fields

SCORING_FUNCTIONS = {
    "none_checker": none_checker,
    "score_ion_list": score_ion_list,
    "score_salt": score_salt,
}

BASIC_REAGENTS = {
    "Water": StockSolution(composition={}, description="Distilled water"),
    "H2SO4": StockSolution(
        composition={"H+": 1, "HSO4-": 1}, description="H2SO4 1.0 M"
    ),
    "KOH": StockSolution(composition={"K+": 1, "OH-": 1}, description="KOH 1.0 M"),
}
BASIC_SYS = "K S(+6)"

REAGENTS_1 = {
    "Water": StockSolution(composition={}, description="distilled water"),
    "HCl(0.02M)": StockSolution(
        composition={"H+": 0.02, "Cl-": 0.02}, description="HCl 0.02 M"
    ),
    "HCl(1M)": StockSolution(composition={"H+": 1, "Cl-": 1}, description="HCl 1.0 M"),
    "HCl(6M)": StockSolution(composition={"H+": 6, "Cl-": 6}, description="HCl 6.0 M"),
    "HNO3(6M)": StockSolution(
        composition={"H+": 6, "NO3-": 6}, description="HNO3 6.0 M"
    ),
    "HNO3(0.1M)": StockSolution(
        composition={"H+": 0.1, "NO3-": 0.1}, description="HNO3 0.1 M"
    ),
    "KOH(6M)": StockSolution(composition={"K+": 6, "OH-": 6}, description="KOH 6.0 M"),
    "KOH(1M)": StockSolution(composition={"K+": 1, "OH-": 1}, description="KOH 1.0 M"),
    "KOH(0.05M)": StockSolution(
        composition={"K+": 0.05, "OH-": 0.05}, description="KOH 0.05 M"
    ),
    "H2SO4(1M)": StockSolution(
        composition={"H+": 1, "HSO4-": 1}, description="H2SO4 1.0 M"
    ),
    "NH3(1M)": StockSolution(composition={"NH3": 1}, description="NH3 1.0 M"),
    "NH3(5M)": StockSolution(composition={"NH3": 5}, description="NH3 5.0 M"),
    "NH4Cl": StockSolution(
        composition={"NH4+": 1, "Cl-": 1}, description="NH4Cl 1.0 M"
    ),
    "BUFFER_9": StockSolution(
        composition={"NH3": 0.72, "NH4+": 1.28, "Cl-": 1.28},
        description="2.0 M NH3/NH4Cl buffer pH=9.0",
    ),
    "NH4I": StockSolution(composition={"NH4+": 1, "I-": 1}, description="NH4I 1.0 M"),
    "H2S(acidic)": StockSolution(
        composition={"H2S": 0.1, "H+": 0.05, "Cl-": 0.05},
        description="H2S 0.1 M + HCl 0.05 M",
    ),
    "(NH4)2S": StockSolution(
        composition={"NH4+": 1.4, "S-2": 0.5, "NO3-": 0.4},
        description="(NH4)2S 0.5 M + NH4NO3 0.4 M, (pH=9.0)",
    ),
    "(NH4)2CO3": StockSolution(
        composition={"NH4+": 0.4, "CO3-2": 0.2}, description="(NH4)2CO3 0.2 M"
    ),
    "(NH4)2HPO4": StockSolution(
        composition={"NH4+": 0.4, "HPO4-2": 0.2}, description="(NH4)2HPO4 0.2 M"
    ),
    "(NH4)2SO4": StockSolution(
        composition={"NH4+": 1, "SO4-2": 0.5}, description="(NH4)2SO4 0.5 M"
    ),
    "K2CrO4": StockSolution(
        composition={"K+": 0.4, "CrO4-2": 0.2}, description="K2CrO4 0.2 M"
    ),
    "NH4SCN": StockSolution(
        composition={"NH4+": 0.1, "SCN-": 0.1}, description="NH4SCN 0.1 M"
    ),
    "DMG": StockSolution(
        composition={"K+": 0.01, "Hdmg-": 0.01},
        description="dimethylglyoxime potassium salt 0.01 M",
    ),
}
SYS_1 = "Cl N S I C K P Cr dmg"

REAGENTS_2 = {
    "Water": StockSolution(composition={}, description="distilled water"),
    "AgNO3": StockSolution(
        composition={"Ag+": 0.1, "NO3-": 0.1}, description="AgNO3 0.1 M"
    ),
    "Ba(NO3)2": StockSolution(
        composition={"Ba+2": 0.1, "NO3-": 0.2}, description="Ba(NO3)2 0.1 M"
    ),
    "Pb(NO3)2": StockSolution(
        composition={"Pb+2": 0.1, "NO3-": 0.2}, description="Pb(NO3)2 0.1 M"
    ),
    "HNO3(6M)": StockSolution(
        composition={"H+": 6, "NO3-": 6}, description="HNO3 6.0 M"
    ),
    "HNO3(0.1M)": StockSolution(
        composition={"H+": 0.1, "NO3-": 0.1}, description="HNO3 0.1 M"
    ),
    "HCl(1M)": StockSolution(composition={"H+": 1, "Cl-": 1}, description="HCl 1.0 M"),
    "KOH(6M)": StockSolution(composition={"K+": 6, "OH-": 6}, description="KOH 6.0 M"),
    "KOH(0.05M)": StockSolution(
        composition={"K+": 0.05, "OH-": 0.05}, description="KOH 0.05 M"
    ),
    "H2SO4(1M)": StockSolution(
        composition={"H+": 1, "HSO4-": 1}, description="H2SO4 1.0 M"
    ),
    "NH3(1M)": StockSolution(composition={"NH3": 1}, description="NH3 1.0 M"),
    "NH3(5M)": StockSolution(composition={"NH3": 5}, description="NH3 5.0 M"),
    "NH4Cl": StockSolution(
        composition={"NH4+": 1, "Cl-": 1}, description="NH4Cl 1.0 M"
    ),
    "BUFFER_9": StockSolution(
        composition={"NH3": 0.72, "NH4+": 1.28, "Cl-": 1.28},
        description="2.0 M NH3/NH4Cl buffer, (pH=9.0)",
    ),
    "NH4I": StockSolution(composition={"NH4+": 1, "I-": 1}, description="NH4I 1.0 M"),
    "H2S(acidic)": StockSolution(
        composition={"H2S": 0.1, "H+": 0.05, "Cl-": 0.05},
        description="H2S 0.1 M + HCl 0.05 M",
    ),
    "(NH4)2S": StockSolution(
        composition={"NH4+": 1.4, "S-2": 0.5, "NO3-": 0.4},
        description="(NH4)2S 0.5 M + NH4NO3 0.4 M, (pH=9.0)",
    ),
    "(NH4)2CO3": StockSolution(
        composition={"NH4+": 0.4, "CO3-2": 0.2}, description="(NH4)2CO3 0.2 M"
    ),
    "(NH4)2HPO4": StockSolution(
        composition={"NH4+": 0.4, "HPO4-2": 0.2}, description="(NH4)2HPO4 0.2 M"
    ),
    "K2CrO4": StockSolution(
        composition={"K+": 0.4, "CrO4-2": 0.2}, description="K2CrO4 0.2 M"
    ),
}
SYS_2 = "Ag Ba Pb N K S(+6) S(-2) Cl I C(+4) P Cr"


def _is_charge_neutral(
    solution: StockSolution | Solution, threshold: float = 1e-10
) -> bool:
    if type(solution) is StockSolution:
        solution = 1 * solution
    return abs(solution.state.charge()) <= threshold


@dataclass(frozen=True)
class QualitativeAnalysisTask(TaskDefinition):
    task_sys: str = None
    sample_list: list[dict[str, Any]] = None
    reagent_set: str = None
    additional_reagents: list[dict[str, Any]] = None
    initial: bool = True
    chemical_system_spec: ChemicalSystemSpec | None = None
    initial_wetlab_state: Mapping[str, Any] | None = None
    sys: str | None = None

    def __post_init__(self):  # setting up the engine
        if self.reagent_set == "Basic":
            reagent_templates = BASIC_REAGENTS.copy()
            sys = f"{BASIC_SYS} {self.task_sys}"
        elif self.reagent_set == "R1":
            reagent_templates = REAGENTS_1.copy()
            sys = f"{SYS_1} {self.task_sys}"
        elif self.reagent_set == "R2":
            reagent_templates = REAGENTS_2.copy()
            sys = f"{SYS_2} {self.task_sys}"
        else:
            raise ValueError(f"Invalid reagent_set: {self.reagent_set}")

        spec = ChemicalSystemSpec(elements=sys)
        engine = WetlabEngine(spec)
        reagent_solutions = {
            label: reagent.clone(engine=engine)
            for label, reagent in reagent_templates.items()
        }

        if self.additional_reagents:
            additional_reagents = {
                reagent["label"]: engine.stock_solution(
                    composition=reagent["composition"],
                    description=reagent["description"],
                )
                for reagent in self.additional_reagents
            }
            reagent_solutions |= additional_reagents

        original_samples = {
            sample["label"]: engine.solution(
                composition=sample["composition"],
                volume=sample["vol"],
                description="unknown",
            )
            for sample in self.sample_list
        }

        # checking charge neutrality
        for label, solution in (original_samples | reagent_solutions).items():
            if not _is_charge_neutral(solution):
                solution_type = (
                    "Sample solution"
                    if type(solution) is Solution
                    else "Reagent solution"
                )
                raise RuntimeError(
                    f"{solution_type} with label {label!r} is not charge neutral!"
                )

        # equilibrating the samples and checking for precipitates
        sample_solutions = {}
        for label, solution in original_samples.items():
            solution.equilibrate()
            if solution.has_precipitate:
                raise RuntimeError(
                    f"Sample with label {label!r} resulted in a precipitate after equilibration."
                )
            sample_solutions[label] = solution

        initial_state = engine.snapshot(sample_solutions | reagent_solutions)
        object.__setattr__(self, "chemical_system_spec", spec)
        object.__setattr__(self, "initial_wetlab_state", initial_state.to_dict())
        object.__setattr__(self, "sys", sys)


def load_tasks_from_json(
    json_path: Path,
) -> dict[str, QualitativeAnalysisTask]:
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
                scoring_fn=with_fixed_inputs(
                    SCORING_FUNCTIONS[data["scoring_fn"]],
                    ground_truth=data["output"][0]["target"],
                ),
                scoring_inputs=data["output"][0]["target"],
                submission_format=data.get("submission_format", ""),
                resolve_answer=False,
                input_map={dep: InputRef(dep) for dep in input_from_tasks},
                initial_input=initial_input,
                task_sys=data["input"]["sys"],
                sample_list=data["input"]["samples"],
                reagent_set=data["input"]["reagent_set"],
                additional_reagents=data["input"].get("additional_reagents", []),
                initial=data["input"].get("initial", True),
            )
    return tasks


class QualitativeAnalysisEnvironment(Environment):
    """Environment for one QualitativeAnalysisTask.

    The active inventory is serialized in `State.environment`. Reaktoro
    objects are materialized only for the duration of a tool call and are never
    retained by the Environment.

    Args:
        task_id (str): ID of the task to work on
        task (TaskDefinition): Definition of the task
        base_work_dir (str): Base working directory
        group_tasks (dict[str, TaskDefinition]): All linked task definitions
    """

    def _state_for_configuration(self, state: ExecutionState) -> WetlabState:
        """Resolve the initial or latest upstream inventory from durable State."""
        task_spec = self.current_task.chemical_system_spec
        if task_spec is None:
            raise RuntimeError("Wetlab task has no ChemicalSystemSpec")

        if self.current_task.initial:
            raw_wetlab = self.current_task.initial_wetlab_state
            if not isinstance(raw_wetlab, Mapping):
                raise RuntimeError("Wetlab task has no initial inventory")
        else:
            raw_wetlab = None
            # JSON task definitions list cumulative dependencies in execution
            # order. The last one is the most recent inventory checkpoint.
            for ref in self.current_task.input_map.values():
                dependency = state.dependency_outputs.get(ref.task_id)
                if dependency is None:
                    continue
                candidate = dependency.metadata.get("wetlab")
                if isinstance(candidate, Mapping):
                    raw_wetlab = candidate
            if raw_wetlab is None:
                raise RuntimeError(
                    f"Task {self.task_id!r} requires an upstream WetlabState"
                )

        wetlab_state = WetlabState.from_dict(raw_wetlab)
        if wetlab_state.chemical_system != task_spec:
            raise RuntimeError(
                "Upstream WetlabState uses a different chemical-system configuration"
            )
        return wetlab_state

    def configure(self, state: ExecutionState) -> TaskConfigured:
        """Commit the task's structured chemistry snapshot to runtime State."""
        wetlab_state = self._state_for_configuration(state)
        event(
            "DEBUG",
            "environment.configuration_completed",
            subsystem="runtime",
            benchmark="wetlab",
            task_id=self.task_id,
            chemical_system=self.current_task.sys,
        )
        hidden = dict(state.environment.values.get("hidden_arguments", {}))
        hidden["wetlab"] = wetlab_state.to_dict()
        environment = {
            **dict(state.environment.values),
            "hidden_arguments": hidden,
        }
        next_environment = self.capture_environment(environment)
        operations = environment_operations(state.environment.values, next_environment)
        workspace = self.capture_workspace(state.workspace)
        workspace_delta = (
            None
            if workspace.files == state.workspace.files
            and workspace.artifacts == state.workspace.artifacts
            else WorkspaceDelta.from_workspace(workspace)
        )
        return TaskConfigured(
            status="Wetlab chemical system and structured inventory configured.",
            environment_operations=operations,
            workspace_delta=workspace_delta,
            expected_environment_revision=state.environment.revision,
            expected_workspace_revision=(
                state.workspace.revision if workspace_delta is not None else None
            ),
        )

    def get_task_prompt(self, state: ExecutionState) -> str:
        prompt = (
            f"Task {self.current_task.name}:\n"
            f"{self.current_task.description}\n\n"
            "Required submission format:\n"
            f"{self.current_task.submission_format}\n\n"
        )

        if self.current_task.input_map:
            prompt += "\nAvailable data from previous subtasks:\n"

        # Display resolved inputs from dependencies
        resolved = self.resolve_inputs(state)
        for input_name, ref in self.current_task.input_map.items():
            value = resolved[input_name]
            task_prompt = self.group_tasks[ref.task_id].description
            prompt += f"- Input from '{ref.task_id}' with question: '{task_prompt}' and answer: '{value}'\n"

        # Display initial input data
        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                if key != "work_dir":
                    prompt += f"- {key}: {value}\n"

        event(
            "DEBUG",
            "environment.prompt_generated",
            subsystem="runtime",
            benchmark="wetlab",
            task_id=self.task_id,
            prompt=prompt,
        )
        return prompt

    def execute_tool(
        self,
        state: ExecutionState,
        tool: Tool,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute against a disposable engine restored from `WetlabState`."""
        if "wetlab" not in tool.hidden_args:
            return super().execute_tool(state, tool, arguments)

        serialized = arguments.get("wetlab")
        if not isinstance(serialized, Mapping):
            raise TypeError("State hidden argument 'wetlab' must be an object")
        wetlab_state = WetlabState.from_dict(serialized)
        engine = WetlabEngine(wetlab_state.chemical_system)
        inventory = engine.restore(wetlab_state)
        call_arguments = {**arguments, "wetlab": inventory}
        content = tool.execute(**call_arguments)
        hidden = dict(state.environment.values.get("hidden_arguments", {}))
        hidden["wetlab"] = engine.snapshot(inventory).to_dict()
        return ToolExecutionResult(
            content=content,
            environment={
                **dict(state.environment.values),
                "hidden_arguments": hidden,
            },
        )

    @staticmethod
    def _wetlab_from_state(state: ExecutionState) -> WetlabState:
        hidden = state.environment.values.get("hidden_arguments", {})
        if not isinstance(hidden, Mapping):
            raise TypeError("State hidden_arguments must be an object")
        raw_wetlab = hidden.get("wetlab")
        if not isinstance(raw_wetlab, Mapping):
            raise TypeError("WetlabState is not configured")
        return WetlabState.from_dict(raw_wetlab)

    def get_task_output(self, state: ExecutionState) -> TaskOutput | None:
        """Publish the answer and current inventory for chained task execution."""
        output = super().get_task_output(state)
        if output is None:
            return None
        wetlab_state = self._wetlab_from_state(state)
        return TaskOutput(
            output=output.output,
            metadata={**dict(output.metadata), "wetlab": wetlab_state.to_dict()},
        )


def create_qualysis_environments(
    level: int = 2,
    subtask: bool = False,
) -> dict[str, Environment]:
    """Create environments for the WetLab (Qualitative Inorganic Analysis) benchmark tasks."""
    started = perf_counter()
    name = "wetlab"
    event(
        "INFO",
        "environment.started",
        subsystem="runtime",
        benchmark=name,
        operation="create",
    )
    if subtask:
        json_path = Path(__file__).parent / "subtasks_json" / f"level_{level}"
    else:
        json_path = Path(__file__).parent / "tasks_json" / f"level_{level}"
    event(
        "DEBUG",
        "environment.tasks_loading",
        subsystem="runtime",
        benchmark=name,
        task_source=str(json_path),
    )
    try:
        if not json_path.exists():
            raise ValueError(f"The path {json_path} does not exist.")
        tasks = load_tasks_from_json(json_path)
        environments = build_environments(
            tasks,
            name=name,
            toolset=Toolset(
                pool=create_tools(),
                workspace_factory=None,
                select_all_when_unspecified=True,
            ),
            env_cls=QualitativeAnalysisEnvironment,
        )
    except Exception as exc:
        event(
            "ERROR",
            "environment.failed",
            subsystem="runtime",
            benchmark=name,
            operation="create",
            status="failed",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            **exception_fields(exc),
        )
        raise

    event(
        "INFO",
        "environment.completed",
        subsystem="runtime",
        benchmark=name,
        operation="create",
        status="completed",
        duration_ms=round((perf_counter() - started) * 1000, 3),
        environment_count=len(environments),
    )
    return environments


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Inspect qualitative-analysis environments"
    )
    parser.add_argument("--level", type=int, default=2, help="The level of tasks")
    parser.add_argument(
        "--subtask",
        type=bool,
        default=False,
        help="Whether to use subtasks",
    )
    args = parser.parse_args()

    # Create all environments
    environments = create_qualysis_environments(level=args.level, subtask=args.subtask)

    for env_id, env in environments.items():
        event(
            "DEBUG",
            "environment.created",
            subsystem="runtime",
            benchmark="wetlab",
            task_id=env_id,
            task_name=env.current_task.name,
            dependencies=sorted(env.current_task.dependencies()),
        )
