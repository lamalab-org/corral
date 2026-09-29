"""Corral environments for the Stargazer radial-velocity benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from corral.core.environment import (
    Environment,
    EnvironmentSetup,
    Toolset,
    build_environments,
)
from corral.core.task import TaskDefinition
from corral.core.transition import ToolExecutionResult
from corral.runtime import permissions
from corral.tools.python_repl import PythonREPLTool
from stargazer.fit import validate_fit
from stargazer.models import load_task
from stargazer.protocol import (
    ASSISTED_PROTOCOL,
    execution_fingerprint,
    protocol_version,
    public_resources,
)
from stargazer.score import LEGACY_CRITERIA, EvaluationCriteria, make_stargazer_scorer
from stargazer.tools import create_tools

if TYPE_CHECKING:
    from corral.core.state import ExecutionState

TASK_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = TASK_ROOT / "data"
DEFAULT_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", str(TASK_ROOT / "CORRAL_WORK_DIR"))

PROTOCOL_VERSION = ASSISTED_PROTOCOL
SUBMISSION_FORMAT = (
    "Commit your final candidate JSON using submit_answer(answer=...). "
    'The JSON must contain "planets" and may contain "noise_jitter_ms". '
    "This ends the episode irrevocably, including malformed or incorrect candidates."
)


def _read_selectors(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(f"No Stargazer selector JSON found at {path}")
    selectors: list[dict[str, Any]] = []
    for file in files:
        with file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload if isinstance(payload, list) else [payload]
        if not all(isinstance(entry, dict) for entry in entries):
            raise ValueError(f"Selectors in {file} must be JSON objects")
        selectors.extend(entries)
    return selectors


def _task_matches_selector(task_file: Path, selector: dict[str, Any]) -> bool:
    with task_file.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    task_ids = selector.get("task_ids")
    if task_ids is not None:
        if not isinstance(task_ids, list) or not all(
            isinstance(task_id, str) for task_id in task_ids
        ):
            raise ValueError("selector task_ids must be a list of strings")
        if raw.get("task_id") not in task_ids:
            return False
    difficulty = int(raw.get("truth_difficulty", 10))
    minimum = int(selector.get("difficulty_min", difficulty))
    maximum = int(selector.get("difficulty_max", difficulty))
    return minimum <= difficulty <= maximum


def _task_prompt(env: Environment, _state: ExecutionState) -> str:
    context = env.current_task.scoring_inputs["public_context"]
    obs = context.observations
    assisted = env.current_task.scoring_inputs.get("analysis_assistance", True)
    assistance = (
        "\nUse stargazer_predict(planets) for the evaluator's RV signal and "
        "stargazer_diagnostics(planets, noise_jitter_ms=0.1) for fitted offsets, "
        "model_ms and residuals_ms arrays plus observation-only scalar diagnostics. "
        "Custom prediction times retain t_ref_days. "
        "STARGAZER_PUBLIC_RESOURCES['public_rv.py'] contains the shared source; "
        "read it in slices of at most 4,000 characters per tool call.\n"
        if assisted
        else ""
    )
    return f"""You are an RV data analyst detecting exoplanets from radial-velocity observations.
Interaction protocol: {protocol_version(assisted)}.

Dataset: {len(obs.times_days)} observations over {np.ptp(obs.times_days):.3f} days.
Stellar mass: {context.star_mass_sun} solar masses. Reference epoch: {context.t_ref_days} days.

Use PythonREPL to compute Lomb-Scargle periodograms, compare planet counts,
fit Keplerian models with multiple initializations, and inspect residuals.
Preloaded: np, times_days, rvs_ms, sigmas_ms, instruments, star_mass_sun,
t_ref_days, baselines, history, stargazer_planet_from_fit, STARGAZER_SUBMISSION_GUIDE.
Read STARGAZER_SUBMISSION_GUIDE before fitting. Use print() for results. No plotting.
{assistance}

Fit period P, amplitude K, eccentricity e, argument of periapsis omega,
mean anomaly M0 at t_ref, and offsets for each instrument. The helper
stargazer_planet_from_fit(P, K, e, omega, M0) converts to native fields:
P_days, m_sin_i_mjup, e, omega_rad, l_rad.
l_rad = (Omega_rad + omega_rad + M0) modulo 2*pi at times_days[0].
Use periods > 0.5 days, eccentricities 0 to 0.8, and at most {context.max_planets} planets.
Masses are clipped to [0.001, 30] Jupiter masses; jitter is at least 0.1 m/s.
Explicit noise_jitter_ms is recommended. If omitted, jitter is estimated using
legacy REBOUND residuals (axis={context.los_axis}, integrator={context.integrator_preference}).
Fit diagnostics use analytic RV-only Keplerians with fitted instrument offsets;
BIC counts five parameters per planet and one offset per instrument.
The null model uses zero jitter. RMS guidance is {context.maximum_rms_factor} times median sigma.

validate_fit(planets=[...], noise_jitter_ms=...) can be called repeatedly.
It returns format errors, the effective candidate, likelihood, BIC and residuals.
It contains no reference-based correctness feedback. history contains only these validations.
Choose planet count and parameters using observations. A low RMS alone does not establish planet recovery.
Tool use is bounded by Corral's configured agent iteration limit. Reserve time to commit an answer.

{SUBMISSION_FORMAT}
Example final JSON: {{"planets": [{{"P_days": 10.0, "m_sin_i_mjup": 0.1,
"e": 0.1, "omega_rad": 0.2, "l_rad": 1.0}}], "noise_jitter_ms": 0.5}}
Only the final candidate is graded. No final answer is an unfinished attempt.
"""


def _require_worker(env: Environment) -> None:
    if (
        not env.current_task.scoring_inputs.get("development_mode", False)
        and not permissions.enabled()
    ):
        raise RuntimeError(
            "Stargazer blind evaluations require restricted Docker workers; use development_mode=True only for local development"
        )


def _configure_trial(env: Environment, _state: ExecutionState) -> EnvironmentSetup:
    _require_worker(env)
    return EnvironmentSetup(
        hidden_arguments={
            "protocol_version": protocol_version(
                env.current_task.scoring_inputs.get("analysis_assistance", True)
            ),
            "analysis": {"history": [], "revision": 0},
            "analysis_session": None,
            "analysis_history_revision": 0,
        },
        status="Blind RV analysis configured.",
    )


class StargazerEnvironment(Environment):
    """Public analysis and irreversible, separately evaluated final submission."""

    def initial_event(self, **kwargs):
        _require_worker(self)
        # An independent trial must start with a fresh workspace. Same-execution
        # resume restores a committed snapshot instead of calling initial_event.
        if not self.current_task.allow_previous_attempt_context and self.workspace_path:
            path = Path(self.workspace_path)
            if path.exists() and any(path.iterdir()):
                raise ValueError("Blind trials require an empty initial workspace")
        event = super().initial_event(**kwargs)
        scorer = self.current_task.scoring_fn
        data = event.model_dump(mode="json")
        data["task"]["stargazer"] = {
            "protocol": protocol_version(
                self.current_task.scoring_inputs.get("analysis_assistance", True)
            ),
            "analysis_assistance": self.current_task.scoring_inputs.get(
                "analysis_assistance", True
            ),
            "provenance": self.current_task.scoring_inputs.get("provenance", {}),
            "bank_hash": scorer.bank_hash,
            "criteria": asdict(scorer.criteria),
            "development_mode": self.current_task.scoring_inputs.get(
                "development_mode", False
            ),
        }
        return type(event).model_validate(data)

    def validate_state_tool_catalog(self, state):
        _require_worker(self)
        catalog = super().validate_state_tool_catalog(state)
        hidden = state.environment.values.get("hidden_arguments", {})
        expected_protocol = protocol_version(
            self.current_task.scoring_inputs.get("analysis_assistance", True)
        )
        if hidden and hidden.get("protocol_version") != expected_protocol:
            raise ValueError(
                "Incompatible Stargazer checkpoint; start a fresh execution"
            )
        return catalog

    def execute_controller_tool(self, state: ExecutionState, prepared) -> Any:
        self.validate_state_tool_catalog(state)
        tool, arguments = prepared.tool, prepared.arguments
        if tool.name not in {"PythonREPL", "validate_fit"}:
            return super().execute_controller_tool(state, prepared)
        context = self.current_task.scoring_inputs["public_context"]
        values = dict(state.environment.values)
        hidden = json.loads(json.dumps(values["hidden_arguments"]))
        analysis = hidden["analysis"]
        if tool.name == "validate_fit":
            content = validate_fit(context, arguments)
            # Store effective candidates only, with controlled public errors.
            analysis["history"].append(content)
            analysis["revision"] += 1
            content = json.dumps(content, allow_nan=False)
        else:
            if not isinstance(tool, PythonREPLTool):
                raise TypeError("PythonREPL must use Corral's PythonREPLTool")
            public_data = {
                **asdict(context.observations),
                "star_mass_sun": context.star_mass_sun,
                "analysis_assistance": self.current_task.scoring_inputs.get(
                    "analysis_assistance", True
                ),
            }
            if public_data["analysis_assistance"]:
                public_data.update(
                    maximum_rms_factor=context.maximum_rms_factor,
                    los_axis=context.los_axis,
                    integrator_preference=context.integrator_preference,
                    public_resources=public_resources(),
                )
            history = (
                analysis["history"]
                if (
                    hidden["analysis_history_revision"] != analysis["revision"]
                    or hidden["analysis_session"] is None
                )
                else None
            )
            updates = {"history": history} if history is not None else None
            if permissions.enabled():
                result = tool.execute_repl(
                    code=arguments["input_code"],
                    initial_data=public_data,
                    checkpoint=hidden["analysis_session"],
                    workspace=self.workspace_path,
                    namespace_updates=updates,
                )
                content, hidden["analysis_session"] = result.output, result.checkpoint
            else:
                session = tool.create_session(public_data)
                try:
                    session.restore(hidden["analysis_session"])
                    content = session.execute(arguments["input_code"], updates)
                    hidden["analysis_session"] = session.snapshot()
                finally:
                    session.close()
            hidden["analysis_history_revision"] = analysis["revision"]
        return ToolExecutionResult(
            content=content, environment={**values, "hidden_arguments": hidden}
        )


def load_tasks_from_json(
    selector_path: str | Path,
    *,
    data_root: str | Path = DATA_ROOT,
    development_mode: bool = False,
    scorer: str = "complete",
    minimum_planet_score: float | None = None,
    analysis_assistance: bool = True,
) -> dict[str, TaskDefinition]:
    """Expand task-bank selectors into Corral task definitions."""
    selectors = _read_selectors(Path(selector_path))
    protocol = protocol_version(analysis_assistance)
    if scorer not in {"legacy", "complete"}:
        raise ValueError("scorer must be legacy or complete")
    criteria = (
        LEGACY_CRITERIA
        if scorer == "legacy"
        else EvaluationCriteria(minimum_planet_score=minimum_planet_score)
    )
    if scorer == "legacy" and minimum_planet_score is not None:
        raise ValueError("Per-planet thresholds require the complete scorer")
    root = Path(data_root)
    bank_files = sorted(root.glob("synthetic/*.json"))
    bank_hash = hashlib.sha256(
        b"".join(path.name.encode() + path.read_bytes() for path in bank_files)
    ).hexdigest()
    if (
        root.resolve() != DATA_ROOT.resolve()
        and not (root / "private-manifest.json").is_file()
    ):
        raise ValueError("Private evaluation requires a frozen bank manifest")
    if (root / "private-manifest.json").exists():
        from stargazer.bank import verify_bank

        bank_hash = verify_bank(root, purpose="evaluation")["bank_hash"]
    from stargazer.experiment import provenance

    code_provenance = provenance()
    version = execution_fingerprint(
        protocol=protocol,
        bank=bank_hash,
        criteria=asdict(criteria),
        development_mode=development_mode,
        source_hash=code_provenance["source_hash"],
    )

    tasks: dict[str, TaskDefinition] = {}
    for selector in selectors:
        source = str(selector.get("source", "synthetic"))
        if source not in {"synthetic", "real"}:
            raise ValueError(f"Unknown Stargazer task source: {source}")
        source_dir = root / source

        for task_file in sorted(source_dir.glob("*.json")):
            if not _task_matches_selector(task_file, selector):
                continue
            benchmark_task = load_task(task_file, source=source)
            task_id = benchmark_task.task_id
            if task_id in tasks:
                raise ValueError(f"Duplicate Stargazer task id: {task_id}")
            summary = benchmark_task.public_summary()
            tasks[task_id] = TaskDefinition(
                name=f"Stargazer {task_id}",
                description=(
                    "Recover an unseen planetary system from radial-velocity "
                    "measurements using statistical and physical model fitting."
                ),
                tools=["PythonREPL", "validate_fit"],
                scoring_fn=make_stargazer_scorer(
                    benchmark_task, criteria, bank_hash=bank_hash
                ),
                scorer_version=version,
                execution_version=version,
                allow_previous_attempt_context=False,
                submission_format=SUBMISSION_FORMAT,
                scoring_inputs={
                    "benchmark_task": benchmark_task,
                    "public_context": benchmark_task.public_fit_context(
                        maximum_rms_factor=criteria.maximum_rms_factor
                    ),
                    "development_mode": development_mode,
                    "analysis_assistance": analysis_assistance,
                    "provenance": code_provenance,
                },
                initial_input={
                    "task_id": task_id,
                    "source": source,
                    "num_observations": summary["num_observations"],
                },
                prompt_fn=_task_prompt,
                setup_fn=_configure_trial,
                resolve_answer=False,
            )
    if not tasks:
        raise ValueError(f"No Stargazer tasks matched selectors in {selector_path}")
    return tasks


def create_environments(
    *,
    level: int | str = 1,
    selector_path: str | Path | None = None,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    data_root: str | Path = DATA_ROOT,
    development_mode: bool = False,
    scorer: str = "complete",
    minimum_planet_score: float | None = None,
    analysis_assistance: bool = True,
) -> dict[str, Environment]:
    """Create one of the two scored Stargazer benchmark levels."""
    if isinstance(level, str) and level.isdigit():
        level = int(level)
    if level not in {1, 2}:
        raise ValueError("Stargazer level must be 1 or 2")
    split_name = f"level_{level}"
    path = (
        Path(selector_path)
        if selector_path
        else (TASK_ROOT / "environments" / split_name / "tasks_json")
    )
    if selector_path is None and Path(data_root) != DATA_ROOT:
        path = Path(data_root) / "selectors" / f"level_{level}.json"
    task_definitions = load_tasks_from_json(
        path,
        data_root=data_root,
        development_mode=development_mode,
        scorer=scorer,
        minimum_planet_score=minimum_planet_score,
        analysis_assistance=analysis_assistance,
    )
    logger.info(f"Creating {len(task_definitions)} Stargazer {split_name} environments")
    return build_environments(
        task_definitions,
        base_work_dir=str(work_dir),
        name=f"stargazer_{split_name}",
        toolset=Toolset(
            pool=create_tools(analysis_assistance=analysis_assistance),
            workspace_factory=None,
        ),
        env_cls=StargazerEnvironment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Stargazer environments")
    parser.add_argument(
        "selector_path",
        nargs="?",
        default=None,
        help="Optional task-bank selector JSON file or directory",
    )
    parser.add_argument("--level", choices=("1", "2"), default="1")
    args = parser.parse_args()

    Path(DEFAULT_WORK_DIR).mkdir(parents=True, exist_ok=True)
    environments = create_environments(
        level=args.level,
        selector_path=args.selector_path,
        work_dir=DEFAULT_WORK_DIR,
    )
    for task_id, environment in environments.items():
        logger.info("{}: {}", task_id, environment.current_task.name)


if __name__ == "__main__":
    main()
