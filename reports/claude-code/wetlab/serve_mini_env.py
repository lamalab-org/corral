#!/usr/bin/env python
"""Serve the wetlab corral-mini task subset.

`wetlab.env`'s own server entrypoint loads every task for a level from
``tasks/wetlab/wetlab/tasks_json/level_<n>`` (the package-local, "live" task
data `env.py` actually reads). corral-mini only wants a 2-task subset per
level (see ``sampler/data/corral_mini_manifest_budget20.json``).

Unlike spectra_elucidation, wetlab's ``environment-mini/`` symlinks do **not**
work here: they point into ``tasks/wetlab/environments/level_<n>/tasks_json``,
a stale top-level copy left over from before the task JSON schema changed
(``scoring_function`` there vs. the ``scoring_fn`` key `wetlab.env`'s loader
now requires) -- pointing at it raises ``KeyError: 'scoring_fn'``. This script
therefore loads from the package's own live task directory (guaranteed
schema-compatible, since it's what `wetlab.env` itself reads) and filters
down to the corral-mini task IDs there, rather than reusing the broken
symlinks.

Prerequisites (see tasks/wetlab/README.md):
    cd tasks/wetlab
    conda env create -f environment.yml
    # (installs reaktoro, matplotlib, and editable installs of corral + wetlab;
    # reaktoro is conda-forge-only, not on PyPI, so this cannot use uv/pip)

Usage (with the wetlab conda env's Python)::

    cd tasks/wetlab
    /opt/homebrew/Caskroom/miniconda/base/envs/wetlab/bin/python \\
        /path/to/reports/claude-code/wetlab/serve_mini_env.py --level 1

Serves on http://localhost:8000 by default (override with --host/--port).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

# tasks/wetlab must be importable as `wetlab`. Run this with the wetlab conda
# env's Python (see docstring) so corral + wetlab's own deps (reaktoro, etc.)
# are already on sys.path; this just adds the package's own source root for
# good measure when invoked from elsewhere.
_TASK_ROOT = Path(__file__).resolve().parents[3] / "tasks" / "wetlab"
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from wetlab.env import (  # noqa: E402
    QualitativeAnalysisEnvironment,
    load_tasks_from_json,
)
from wetlab.tools import create_tools  # noqa: E402

from corral.backend.env import Toolset, build_environments  # noqa: E402
from corral.backend.server import run_server  # noqa: E402

REPO_ROOT = _TASK_ROOT.parents[1]
MANIFEST_PATH = REPO_ROOT / "sampler" / "data" / "corral_mini_manifest_budget20.json"


def _corral_mini_task_ids(level: int) -> set[str]:
    """The corral-mini-selected task IDs for wetlab at this level."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    return set(manifest["tasks_by_environment"]["wetlab"][f"level_{level}"])


def create_mini_environments(level: int) -> dict:
    """Build environments from the corral-mini task subset for one level."""
    json_path = _TASK_ROOT / "wetlab" / "tasks_json" / f"level_{level}"
    if not json_path.exists():
        raise ValueError(f"Task dir {json_path} does not exist.")

    logger.info(f"Loading tasks from {json_path}")
    all_tasks = load_tasks_from_json(json_path)

    selected_ids = _corral_mini_task_ids(level)
    missing = selected_ids - set(all_tasks)
    if missing:
        raise ValueError(
            f"corral-mini selected task IDs not found in {json_path}: {missing}"
        )
    tasks = {tid: t for tid, t in all_tasks.items() if tid in selected_ids}
    logger.info(f"Serving {len(tasks)} corral-mini task(s): {sorted(tasks)}")

    # Must use QualitativeAnalysisEnvironment (not the generic Environment
    # class): it declares DEFAULT_CONCURRENCY = "process" (wetlab keeps its
    # active reaktoro chemical system in module globals, shared -- and
    # corruptible -- across trials) and owns configure_additional_apps/scoring.
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
    parser = argparse.ArgumentParser(
        description="Serve the wetlab corral-mini task subset."
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

    environments = create_mini_environments(level=args.level)

    logger.info("Created corral-mini environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}: {env.current_task.name}")

    run_server(environments=environments, host=args.host, port=args.port)
