"""Serve the retrosynthesis corral-mini task subset.

`retrosynthesis.env` (the package's own server entrypoint) hardcodes its task
directory to ``tasks/retrosynthesis/environments/level_<n>/tasks_json`` -- the
*full* task set. corral-mini only wants a small subset per level, which lives
alongside it at
``tasks/retrosynthesis/environment-mini/environments/level_<n>/tasks_json`` as
a directory of symlinks back into the full set (``make_1.json`` for level 1,
``make_6.json``/``make_8.json`` for level 2, ``make_7.json``/``make_8.json``
for level 3).

Rather than editing the task package to special-case corral-mini, this script
reuses its task-loading/tool/server plumbing directly (same as
spectra_elucidation's ``serve_mini_env.py``) and just points ``json_path`` at
the environment-mini directory instead. `glob()` + `open()` follow symlinks
fine, so this serves exactly the corral-mini tasks with the exact same task
content as the full run.

Like the full server, this performs the same database connectivity/schema
check (`retrosynthesis.checks.check_database`) before serving -- the
retrosynthesis environment requires a PostgreSQL database with RDKit
extensions (`reactions_raw_db` + `reactions_production_db`; see
tasks/retrosynthesis/README.md and docker/setup_retro_db.sh). Database
credentials are read from RETRO_DB_HOST/PORT/NAME/USER/PASSWORD (or a
tasks/retrosynthesis/.env), defaulting to localhost:5432 -- override
RETRO_DB_PORT if your local Postgres/container publishes on a different port
(e.g. to avoid clashing with another Postgres already bound to 5432).

Prerequisites (see tasks/retrosynthesis/README.md):
    cd tasks/retrosynthesis/docker && bash setup_retro_db.sh   # once
    cd tasks/retrosynthesis
    uv venv --python 3.11.0 && uv sync

Usage (from anywhere, using the tasks/retrosynthesis venv's Python)::

    cd tasks/retrosynthesis
    .venv/bin/python /path/to/reports/claude-code/retrosynthesis/serve_mini_env.py --level 1

Serves on http://localhost:8000 by default (override with --host/--port).
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# tasks/retrosynthesis must be importable as `retrosynthesis`. Run this with
# the tasks/retrosynthesis venv's Python (see docstring) so corral + the
# task's own deps (rdkit, rxnmapper, psycopg2, etc.) are already on
# sys.path; this just adds the package's own source root for good measure
# when invoked from elsewhere.
_TASK_ROOT = Path(__file__).resolve().parents[3] / "tasks" / "retrosynthesis"
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

load_dotenv(_TASK_ROOT / ".env")

from retrosynthesis.checks import check_database  # noqa: E402
from retrosynthesis.env import load_tasks_from_json  # noqa: E402
from retrosynthesis.tools import create_tools  # noqa: E402

from corral.backend.env import Toolset, build_environments  # noqa: E402
from corral.backend.server import run_server  # noqa: E402

BASE_WORK_DIR = os.environ.get(
    "CORRAL_WORK_DIR", str(_TASK_ROOT / "CORRAL_WORK_DIR" / "rethrosynthesis")
)
MINI_ENVIRONMENTS_ROOT = _TASK_ROOT / "environment-mini" / "environments"


def create_mini_environments(work_dir: str, level: int) -> dict:
    """Build environments from the corral-mini task subset for one level."""
    json_path = MINI_ENVIRONMENTS_ROOT / f"level_{level}" / "tasks_json"
    if not json_path.exists():
        raise ValueError(f"corral-mini task dir {json_path} does not exist.")

    logger.info(f"Loading corral-mini tasks from {json_path}")
    tasks = load_tasks_from_json(json_path, work_dir=work_dir)
    logger.info(f"Serving {len(tasks)} corral-mini task(s): {sorted(tasks)}")

    tool_pool = create_tools()
    return build_environments(
        tasks,
        base_work_dir=work_dir,
        name="rethrosynthesis",
        toolset=Toolset(pool=tool_pool, common=tool_pool, workspace_factory=None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve the retrosynthesis corral-mini task subset."
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
        choices=[1, 2, 3],
        help="corral-mini level to serve.",
    )
    args = parser.parse_args()

    logger.info("Performing database checks before starting the environment...")
    try:
        check_database()
    except (ConnectionError, ValueError) as e:
        logger.error(f"Database check failed: {e}")
        logger.error(
            "Please ensure the retrosynthesis database is running and reachable "
            "(see tasks/retrosynthesis/README.md / docker/setup_retro_db.sh). "
            "Override RETRO_DB_HOST/RETRO_DB_PORT/etc. if it's not on the "
            "default localhost:5432."
        )
        raise SystemExit(1) from e

    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)

    environments = create_mini_environments(work_dir=BASE_WORK_DIR, level=args.level)

    logger.info("Created corral-mini environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}: {env.current_task.name}")

    run_server(environments=environments, host=args.host, port=args.port)
