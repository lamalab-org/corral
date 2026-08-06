#!/usr/bin/env python
"""Serve the spectra_elucidation corral-mini task subset.

`spectra_elucidation.env` (the package's own server entrypoint) hardcodes its
task directory to ``tasks/spectra_elucidation/environments/level_<n>/tasks_json``
-- the *full* task set. corral-mini only wants the ~20-task subset selected by
``sampler/`` (see ``sampler/data/corral_mini_manifest_budget20.json``), which
lives alongside it at
``tasks/spectra_elucidation/environment-mini/environments/level_<n>/tasks_json``
as a directory of symlinks back into the full set.

Rather than editing the task package to special-case corral-mini, this script
reuses its task-loading/tool/server plumbing directly and just points
``json_path`` at the environment-mini directory instead. `glob()` + `open()`
follow symlinks fine, so this serves exactly the corral-mini tasks with the
exact same task content as the full run.

Prerequisites (see tasks/spectra_elucidation/README.md):
    cd tasks/spectra_elucidation
    uv venv --python 3.12 && uv sync
    # Node.js isotopic-distribution predictor for mass_spectrometry_spectra:
    mkdir -p CORRAL_WORK_DIR/js && cd CORRAL_WORK_DIR/js
    npm init -y && npm pkg set type=module && npm i --omit=dev isotopic-distribution

Usage (from anywhere, using the tasks/spectra_elucidation venv's Python)::

    cd tasks/spectra_elucidation
    export CORRAL_WORK_DIR="$PWD/CORRAL_WORK_DIR"
    export CORRAL_SPECTRA_JS_DIR="$PWD/CORRAL_WORK_DIR/js"
    .venv/bin/python /path/to/reports/claude-code/spectra_elucidation/serve_mini_env.py --level 1

Serves on http://localhost:8000 by default (override with --host/--port).
"""

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

# tasks/spectra_elucidation must be importable as `spectra_elucidation`. Run
# this with the tasks/spectra_elucidation venv's Python (see docstring) so
# corral + the task's own deps (rdkit, etc.) are already on sys.path; this
# just adds the package's own source root for good measure when invoked from
# elsewhere.
_TASK_ROOT = Path(__file__).resolve().parents[3] / "tasks" / "spectra_elucidation"
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from spectra_elucidation.env import load_tasks_from_json  # noqa: E402
from spectra_elucidation.tools import create_tools  # noqa: E402

from corral.backend.env import Toolset, build_environments  # noqa: E402
from corral.backend.server import run_server  # noqa: E402

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", str(_TASK_ROOT / "CORRAL_WORK_DIR"))
MINI_ENVIRONMENTS_ROOT = _TASK_ROOT / "environment-mini" / "environments"


def create_mini_environments(work_dir: str, level: int) -> dict:
    """Build environments from the corral-mini task subset for one level."""
    json_path = MINI_ENVIRONMENTS_ROOT / f"level_{level}" / "tasks_json"
    if not json_path.exists():
        raise ValueError(f"corral-mini task dir {json_path} does not exist.")

    logger.info(f"Loading corral-mini tasks from {json_path}")
    tasks = load_tasks_from_json(json_path, work_dir=work_dir)
    logger.info(f"Serving {len(tasks)} corral-mini task(s): {sorted(tasks)}")

    return build_environments(
        tasks,
        base_work_dir=work_dir,
        name="spectra_elucidation",
        toolset=Toolset(pool=create_tools(), workspace_factory=None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve the spectra_elucidation corral-mini task subset."
    )
    parser.add_argument(
        "--host", default=os.environ.get("CORRAL_HOST", "0.0.0.0"), help="Bind host."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CORRAL_PORT", "8000")),
        help="Bind port.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[1, 2],
        help="corral-mini level to serve.",
    )
    args = parser.parse_args()

    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)

    environments = create_mini_environments(work_dir=BASE_WORK_DIR, level=args.level)

    logger.info("Created corral-mini environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}: {env.current_task.name}")

    run_server(environments=environments, host=args.host, port=args.port)
