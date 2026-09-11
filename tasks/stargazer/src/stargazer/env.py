"""Corral server for the Stargazer radial-velocity benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from corral.backend.env import Environment, Toolset, build_environments
from corral.backend.server import run_server
from corral.backend.task import TaskDefinition
from stargazer.evaluator import make_stargazer_scorer
from stargazer.models import load_task
from stargazer.tools import create_analysis_session, create_tools

TASK_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = TASK_ROOT / "data"
DEFAULT_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", str(TASK_ROOT / "CORRAL_WORK_DIR"))

SUBMISSION_FORMAT = """A JSON object with a `planets` list and top-level jitter:
{
  "planets": [
    {
      "P_days": <period in days, > 0.5>,
      "m_sin_i_mjup": <minimum mass in Jupiter masses>,
      "e": <eccentricity from 0 to 0.8>,
      "omega_rad": <argument of periapsis in radians>,
      "l_rad": <mean longitude at t_ref in radians>
    }
  ],
  "noise_jitter_ms": <optional non-negative jitter in m/s>
}
The optional compatibility fields `inc_rad` and `Omega_rad` may be included
on a planet but are not needed for the radial-velocity model."""


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


def _task_prompt(env: Environment) -> str:
    task = env.current_task
    benchmark_task = task.scoring_inputs["benchmark_task"]
    summary = benchmark_task.public_summary()
    maximum = int(task.scoring_inputs["max_evaluations"])
    return f"""Task: {task.name}

Infer the Keplerian planetary system that produced the observed stellar
radial-velocity time series.

Public dataset metadata:
- Task ID: {summary["task_id"]}
- Published difficulty: {summary["level_difficulty"]}
- Source: {summary["source"]}
- Observations: {summary["num_observations"]}
- Time span: {summary["observation_span_days"]:.6g} days
- Median uncertainty: {summary["median_uncertainty_ms"]:.6g} m/s
- Stellar mass: {summary["star_mass_solar"]:.6g} solar masses
- Instruments: {", ".join(summary["instrument_labels"])}
- Reference epoch t_ref: {summary["reference_epoch_days"]:.12g} days

The persistent `python_repl` preloads `times_days` (days), `rvs_ms` (m/s),
`sigmas_ms` (m/s), `instruments` (labels), `star_mass_sun` (solar masses), and
`t_ref_days`. Orbital angles are in radians. `l_rad` is mean longitude at
`t_ref_days`.

At most {maximum} valid diagnostic evaluations are available through
`evaluate_candidate`. Invalid candidates do not consume this allowance. The
tool only returns redacted feedback; it does not submit or score an answer.
Only the final Corral answer is scored, and no diagnostic candidate is used as
a fallback.

Return the final answer as canonical JSON in this schema:

{SUBMISSION_FORMAT}
"""


def _configure_trial(env: Environment) -> str:
    benchmark_task = env.current_task.scoring_inputs["benchmark_task"]
    maximum = int(env.current_task.scoring_inputs["max_evaluations"])
    evaluations: list[dict[str, Any]] = []
    evaluation_session = {
        "evaluations": evaluations,
        "max_evaluations": maximum,
        "locked": False,
    }
    observations = benchmark_task.observations
    env.hidden_args = {
        "benchmark_task": benchmark_task,
        "evaluation_session": evaluation_session,
        "star_mass_sun": benchmark_task.star_mass_sun,
        "analysis_session": create_analysis_session(
            times_days=observations.times_days,
            rvs_ms=observations.rvs_ms,
            sigmas_ms=observations.sigmas_ms,
            instruments=observations.instruments,
            star_mass_sun=benchmark_task.star_mass_sun,
        ),
    }
    return "Persistent RV analysis and evaluator sessions configured."


def load_tasks_from_json(
    selector_path: str | Path,
) -> dict[str, TaskDefinition]:
    """Expand task-bank selectors into Corral task definitions."""
    selectors = _read_selectors(Path(selector_path))
    tasks: dict[str, TaskDefinition] = {}
    for selector in selectors:
        source = str(selector.get("source", "synthetic"))
        if source not in {"synthetic", "real"}:
            raise ValueError(f"Unknown Stargazer task source: {source}")
        source_dir = DATA_ROOT / source
        maximum = int(selector.get("max_evaluations", 2))
        if maximum <= 0:
            raise ValueError("max_evaluations must be positive")

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
                tools=["python_repl", "planet_from_fit", "evaluate_candidate"],
                scoring_fn=make_stargazer_scorer(benchmark_task),
                submission_format=SUBMISSION_FORMAT,
                scoring_inputs={
                    "benchmark_task": benchmark_task,
                    "max_evaluations": maximum,
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
) -> dict[str, Environment]:
    """Create a scored Stargazer level or the separate real-data challenge."""
    if isinstance(level, str) and level.isdigit():
        level = int(level)
    if level not in {1, 2, 3, "real"}:
        raise ValueError("Stargazer level must be 1, 2, 3, or 'real'")
    split_name = "real" if level == "real" else f"level_{level}"
    path = (
        Path(selector_path)
        if selector_path
        else (TASK_ROOT / "environments" / split_name / "tasks_json")
    )
    task_definitions = load_tasks_from_json(path)
    logger.info(f"Creating {len(task_definitions)} Stargazer {split_name} environments")
    return build_environments(
        task_definitions,
        base_work_dir=str(work_dir),
        name=f"stargazer_{split_name}",
        toolset=Toolset(pool=create_tools(), workspace_factory=None),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stargazer benchmark server")
    parser.add_argument(
        "selector_path",
        nargs="?",
        default=None,
        help="Optional task-bank selector JSON file or directory",
    )
    parser.add_argument("--level", choices=("1", "2", "3", "real"), default="1")
    parser.add_argument("--host", default=os.environ.get("CORRAL_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("CORRAL_PORT", "8000"))
    )
    args = parser.parse_args()

    Path(DEFAULT_WORK_DIR).mkdir(parents=True, exist_ok=True)
    environments = create_environments(
        level=args.level,
        selector_path=args.selector_path,
        work_dir=DEFAULT_WORK_DIR,
    )
    run_server(environments, args.host, args.port)


if __name__ == "__main__":
    main()
