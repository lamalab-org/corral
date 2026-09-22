"""Corral environments for the Stargazer radial-velocity benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from corral.core.environment import Environment, Toolset, build_environments
from corral.core.task import EnvironmentSetup, TaskDefinition
from corral.core.transition import ToolExecutionResult
from corral.runtime import permissions
from corral.tools.python_repl import PythonREPLTool
from stargazer.models import load_task
from stargazer.score import make_stargazer_scorer, score_execution, submit_candidate
from stargazer.tools import create_tools

if TYPE_CHECKING:
    from corral.core.state import ExecutionState

TASK_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = TASK_ROOT / "data"
DEFAULT_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", str(TASK_ROOT / "CORRAL_WORK_DIR"))

SUBMISSION_FORMAT = (
    "Use submit_action to submit planet hypotheses to Stargazer. "
    "When finished, use Corral's final-answer tool to provide a brief summary. "
    "The final text does not submit another planet candidate or change the result."
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
    task = env.current_task
    benchmark_task = task.scoring_inputs["benchmark_task"]
    """Build the system prompt with task context."""
    obs = benchmark_task.public_observation()
    times = np.asarray(obs["times_days"], dtype=float)
    rvs = np.asarray(obs["rvs_ms"], dtype=float)
    sigmas = np.asarray(obs["sigmas_ms"], dtype=float)
    span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    median_sigma = float(np.median(sigmas))

    # Check for task-specific description and hints
    meta = obs.get("meta", {})
    task_description = meta.get("task_description", "")
    hints = meta.get("hints", {})
    reference = meta.get("reference", "")

    # Build task context section if available
    task_context_section = ""
    if task_description:
        task_context_section = f"\n{task_description}\n"
    if hints:
        hints_text = "\n".join(f"- {k}: {v}" for k, v in hints.items())
        task_context_section += f"\n### Hints\n{hints_text}\n"
    if reference:
        task_context_section += f"\n### Reference\n{reference}\n"

    prompt = textwrap.dedent(
        f"""
        You are an expert RV data analyst tasked with detecting exoplanets from radial velocity measurements.
        {task_context_section}
        ### Dataset Overview
        - Number of observations: {len(times)}
        - Time span: {span:.1f} days
        - RV range: {rvs.min():+.2f} to {rvs.max():+.2f} m/s
        - Median uncertainty: {median_sigma:.2f} m/s

        ### Your Task
        Analyze the RV data to identify planetary signals. Use the PythonREPL tool to:
        - Compute periodograms (Lomb-Scargle or other methods)
        - Test baseline models (available via `baselines` module)
        - Fit Keplerian orbital models (NOT simple sinusoids)
        - Analyze residuals and trigger optimization when needed

        ### CRITICAL: Keplerian Model Parameters
        When fitting Keplerian orbits, you MUST fit ALL of these parameters:
        - **P**: Period (days)
        - **K**: RV semi-amplitude (m/s)
        - **e**: Eccentricity (0 to 0.8)
        - **omega**: Argument of periastron (radians, 0 to 2π) - CRITICAL FOR ECCENTRIC ORBITS!
        - **M0**: Mean anomaly at reference time (radians)
        - **gamma**: Systemic velocity offset (m/s)

        For eccentric orbits (e > 0.1), the omega parameter significantly affects the RV curve shape.
        Always include omega in your fit AND in your submission!

        ### Submission Format
        Use Stargazer-native planet fields for highest reliability:
        - `P_days`, `m_sin_i_mjup`, `e`, `omega_rad`, `l_rad`
        - `l_rad` is mean longitude at reference epoch `t_ref = times_days[0]`
        - If your fit gives mean anomaly `M0` at `t_ref`, convert with: `l_rad = (Omega_rad + omega_rad + M0) % (2π)`

        When ready to submit, call submit_action with your BEST fitted parameters:
        """
        # Keep this example outside f-strings so blacken-docs can format its braces.
        """
        ```python
        {
            "planets": [
                {
                    "P_days": P,
                    "m_sin_i_mjup": m_sin_i,
                    "e": e,
                    "omega_rad": omega,
                    "l_rad": l_rad,
                    "inc_rad": inc,  # Optional, REBOUND geometry
                    "Omega_rad": Omega,  # Optional, REBOUND geometry
                }
            ],
            "rv_offset_ms": gamma,  # Systemic velocity
            "noise_jitter_ms": 0.5,  # Optional jitter term
        }
        ```
        """
        f"""
        Use helper function `stargazer_planet_from_fit(...)` in PythonREPL to convert
        `(P, K, e, omega, M0)` into a correct Stargazer planet dict.

        ### Response format for every turn
        1) Findings: concise hypothesis plus key numbers (candidate periods/powers/RMS).
        2) Plan/Next: 1\u20133 short bullets of what you'll do next.
        3) Code: one fenced code block with what you will run now (only if calling PythonREPL).
        4) Results: printed outputs interpreted; if ready, include submit_action parameters.
        - Keep prose and code separate; do not mix explanations inside code blocks.
        - Always print key metrics from code; avoid silent computations.

        ### Available Tools
        1. **PythonREPL**: Execute Python code for analysis
           - Pre-loaded variables (DO NOT import, just use directly):
             `times_days`, `rvs_ms`, `sigmas_ms`, `np`, `baselines`, `history`,
             `star_mass_sun`, `t_ref_days`, `stargazer_planet_from_fit`, `STARGAZER_SUBMISSION_GUIDE`
           - Example: `print(times_days.max() - times_days.min())`  # Correct
           - WRONG: `from times_days import times_days`  # Do NOT do this!
           - Always use print() to see outputs
           - No plotting allowed

        2. **submit_action**: Submit planet hypotheses
           - Max {benchmark_task.max_planets} planets
           - Period must be > 0.5 days
           - Eccentricity: 0 to 0.8
           - Submission mode: params_and_model

        ### Budget Constraints
        - Tool use is bounded by Corral's configured agent iteration limit.
        - There is no separate limit on the number of submit_action calls.
        - Keep refining and submitting candidates until one succeeds or your iterations run out.

        ### Mandatory Step 0: Read Protocol Guide First
        Before any fitting/submission, read `STARGAZER_SUBMISSION_GUIDE` in PythonREPL and set:
        `_protocol_guide_ack = True`.
        You are NOT allowed to call `submit_action` until this is done.

        ### Strategy (FOLLOW THIS ORDER)

        **Step 1: Periodogram Analysis**
        - Compute Lomb-Scargle periodogram
        - Identify strongest peak(s) and their periods

        **Step 2: Linear Sine Baseline (MANDATORY MODEL GATING)**
        - Before any Keplerian optimization, you MUST run a linear/sinusoidal baseline first.
        - Use `baselines.baseline_one_sine(observation)` (or equivalent linear sine fit).
        - Print at least: candidate period, baseline RMS, and RMS/median_sigma.

        **Step 3: Model Gating Decision (MANDATORY)**
        - Decide whether Kepler is needed based on baseline diagnostics.
        - If baseline RMS is already close to noise (RMS/median_sigma <= 1.5), prefer direct submission/refinement.
        - If baseline RMS is not close to noise, escalate to full Keplerian fitting.
        - Explicitly state: `Gate decision: Kepler=YES/NO` before running Kepler code.

        **Step 4: Keplerian Fitting (ONLY IF GATE=YES)**
        - Fit a FULL 6-parameter Keplerian: P, K, e, omega, M0, gamma
        - Use scipy.optimize.least_squares with bounds
        - For high eccentricity (e > 0.3), try multiple omega starting values
        - Use multi-start optimization to avoid local minima

        **Step 5: Check Fit Quality**
        - Compute residual RMS after fitting
        - Good fit: RMS ≈ {median_sigma:.2f} m/s (close to measurement uncertainty)
        - Bad fit: RMS >> {median_sigma:.2f} m/s → keep optimizing

        **Step 6: Submit ONLY After Convergence**
        - DO NOT submit until RMS is close to noise level
        - Include ALL fitted parameters in submission, especially omega_rad!
        - Double-check: did you include omega_rad in your submission?

        ### Common Mistakes to AVOID
        1. Jumping to Kepler before LS + linear-sine gating
        2. Submitting early with poor fit (high RMS)
        3. Forgetting omega_rad in submission (it will default to 0!)
        4. Using wrong phase convention (`l_rad` is mean longitude, not raw phase offset)
        5. Not doing multi-start optimization for eccentric orbits
        6. Reusing function names as variables in Python (e.g., `residuals = ...` after `def residuals(...)`).
           - If you define a function `residuals`, keep it callable.
           - Use names like `residual_vec`, `fit_residuals`, `model_rv_arr` for arrays.

        A successful fit should achieve residual RMS ≈ {median_sigma:.2f} m/s.
        If your RMS is much larger, your fit has NOT converged - keep optimizing!

        ### Mentor Guidance
        You may receive `[Mentor guidance]` messages from an expert reviewer.
        Treat this advice as high-priority — follow it before continuing your analysis.
        """
    ).strip()
    return prompt + (
        "\n\nCorral completion: after Stargazer reports done=true, finish with "
        "Corral's final-answer tool (`submit_answer`). The scientific result is "
        "determined by your submit_action calls; the final text only closes the run."
    )


def _update_submit_gate_from_text(text: str) -> bool:
    """Gate rule: if Kepler=YES and Best_RMS_over_med_sigma < 1.1, force next action to submit."""
    if not text:
        return False
    if re.search(r"Kepler\s*=\s*YES", text, flags=re.IGNORECASE) is None:
        return False
    m = re.search(
        r"Best_RMS_over_med_sigma\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        text,
    )
    if m is None:
        return False
    try:
        ratio = float(m.group(1))
    except ValueError:
        return False
    return ratio < 1.1


def _configure_trial(env: Environment, _state: ExecutionState) -> EnvironmentSetup:
    benchmark_task = env.current_task.scoring_inputs["benchmark_task"]
    return EnvironmentSetup(
        hidden_arguments={
            "benchmark_task": benchmark_task.task_id,
            "submission_session": {
                "history": [],
                "steps": 0,
                "done": False,
                "protocol_ack": False,
                "force_submit": False,
            },
            "analysis_session": None,
            "analysis_history_revision": 0,
        },
        status="RV analysis and submission state configured.",
    )


class StargazerEnvironment(Environment):
    """Run Stargazer tools with committed state and a public-data-only REPL worker."""

    def execute_controller_tool(self, state: ExecutionState, prepared) -> Any:
        tool = prepared.tool
        arguments = prepared.arguments
        if tool.name not in {"PythonREPL", "submit_action"}:
            return super().execute_controller_tool(state, prepared)
        benchmark_task = self.current_task.scoring_inputs["benchmark_task"]
        environment_values = dict(state.environment.values)
        hidden = json.loads(json.dumps(environment_values["hidden_arguments"]))
        if hidden["benchmark_task"] != benchmark_task.task_id:
            raise ValueError("Stargazer state belongs to a different task")
        submission = hidden["submission_session"]
        if submission["done"]:
            return "Stargazer is done. Finish with Corral's final-answer tool."
        # Forcing a submission the protocol gate would refuse locks both
        # tools: neither early return can clear either flag.
        if (
            tool.name == "PythonREPL"
            and submission["force_submit"]
            and submission["protocol_ack"]
        ):
            return (
                "Policy gate active: Best_RMS_over_med_sigma < 1.1 with Kepler=YES. "
                "Your next step MUST be submit_action now; skip summaries and extra analysis."
            )
        if tool.name == "PythonREPL":
            if not isinstance(tool, PythonREPLTool):
                raise TypeError("PythonREPL must use Corral's PythonREPLTool")
            observations = benchmark_task.observations
            public_data = {
                "times_days": observations.times_days,
                "rvs_ms": observations.rvs_ms,
                "sigmas_ms": observations.sigmas_ms,
                "instruments": observations.instruments,
                "star_mass_sun": benchmark_task.star_mass_sun,
            }
            # Upstream resets the REPL history binding after each submit call,
            # while ordinary REPL calls retain mutations to the current list.
            history_revision = len(submission["history"])
            history = (
                submission["history"]
                if (
                    hidden["analysis_history_revision"] != history_revision
                    or hidden["analysis_session"] is None
                )
                else None
            )
            namespace_updates = {"history": history} if history is not None else None
            if permissions.enabled():
                result = tool.execute_repl(
                    code=arguments["input_code"],
                    initial_data=public_data,
                    checkpoint=hidden["analysis_session"],
                    workspace=self.workspace_path,
                    namespace_updates=namespace_updates,
                )
                content = result.output
                hidden["analysis_session"] = result.checkpoint
                submission["protocol_ack"] = bool(
                    result.exports.get("_protocol_guide_ack", False)
                )
            else:
                session = tool.create_session(public_data)
                try:
                    session.restore(hidden["analysis_session"])
                    try:
                        content = session.execute(
                            arguments["input_code"], namespace_updates
                        )
                    except TimeoutError as exc:
                        content = f"AnalysisTimeoutError: {exc}"
                    submission["protocol_ack"] = bool(
                        session.exports().get("_protocol_guide_ack", False)
                    )
                    hidden["analysis_session"] = session.snapshot()
                finally:
                    session.close()
            hidden["analysis_history_revision"] = history_revision
            if _update_submit_gate_from_text(str(content)):
                submission["force_submit"] = True
        else:
            if not submission["protocol_ack"]:
                return (
                    "Submission blocked: protocol guide not acknowledged yet. "
                    "Run PythonREPL to read `STARGAZER_SUBMISSION_GUIDE`, then set "
                    "`_protocol_guide_ack = True` before calling submit_action."
                )
            content = submit_candidate(benchmark_task, arguments, submission)
        return ToolExecutionResult(
            content=content,
            environment={**environment_values, "hidden_arguments": hidden},
        )


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
                tools=["PythonREPL", "submit_action"],
                scoring_fn=make_stargazer_scorer(benchmark_task),
                state_scoring_fn=score_execution,
                submission_format=SUBMISSION_FORMAT,
                scoring_inputs={"benchmark_task": benchmark_task},
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
    task_definitions = load_tasks_from_json(path)
    logger.info(f"Creating {len(task_definitions)} Stargazer {split_name} environments")
    return build_environments(
        task_definitions,
        base_work_dir=str(work_dir),
        name=f"stargazer_{split_name}",
        toolset=Toolset(pool=create_tools(), workspace_factory=None),
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
