import os
import sys
from pathlib import Path

from loguru import logger
from ml.env import (
    TaskGroupEnvironment,
    load_tasks_from_json,
)
from ml.score import BASE_WORK_DIR
from ml.tools import create_ml_tools

from corral.backend.server import run_server
from corral.backend.task import TaskGroup
from corral.backend.tool import Tool
from corral.sandbox import Sandbox, SandboxConfig, create_sandbox_tools

# Tools to keep from ml.tools (MP API retrieval only)
KEEP_TOOLS = {
    "get_structure_from_mp_text",
    "batch_retrieve_polymorphs",
    "get_bulk_polymorphs_data",
    "get_bulk_polymorphs_data_to_file",
}


def _get_retrieval_tools() -> dict[str, Tool]:
    """Cherry-pick only MP API retrieval tools from the ml tool set."""
    all_tools = create_ml_tools()
    return {name: tool for name, tool in all_tools.items() if name in KEEP_TOOLS}


def _create_sandbox_config() -> SandboxConfig:
    """Create sandbox configuration with scientific packages pre-installed."""
    return SandboxConfig(
        backend="subprocess",
        python_packages=[
            "numpy",
            "pandas",
            "scikit-learn>=1.6.1",
            "xgboost>=3.0.2",
            "scipy",
            "joblib",
        ],
    )


class SandboxAwareEnvironment(TaskGroupEnvironment):
    """TaskGroupEnvironment that keeps a sandbox in sync with trial directories.

    On construction, after each ``reset_state()``, and before **every** tool
    call the sandbox's working directory is updated to match the environment's
    current trial directory.  The per-call sync is required because multiple
    environments share a single sandbox instance — without it the last
    environment to be constructed (or reset) would "own" the sandbox's cwd
    even while a different environment's tools are being invoked.
    """

    def __init__(self, sandbox: Sandbox, **kwargs):
        self._sandbox = sandbox
        super().__init__(**kwargs)
        self._sync_sandbox()

    def _sync_sandbox(self):
        if self._sandbox and self.current_work_dir:
            self._sandbox.set_work_dir(self.current_work_dir)

    def reset_state(self) -> str:
        trial_id = super().reset_state()
        self._sync_sandbox()
        return trial_id

    def call_tool(self, tool_name, arguments):
        """Sync the sandbox to this environment's trial dir before every call."""
        self._sync_sandbox()
        return super().call_tool(tool_name, arguments)


def create_environments(
    task_json_path: str | Path,
    work_dir: str | Path = BASE_WORK_DIR,
) -> dict[str, SandboxAwareEnvironment]:
    """Create environments for code-execution ML tasks.

    Sets up environments with only MP API retrieval tools as subtask-specific
    tools, and sandbox tools (execute_python_code, run_in_terminal) as common
    tools available to all subtasks.

    Parameters
    ----------
    task_json_path
        Path to the JSON file with task definitions.
    work_dir
        Working directory for task execution.

    Returns
    -------
    dict[str, SandboxAwareEnvironment]
        Dictionary of environments keyed by task ID.
    """
    logger.info(
        f"[ml_code] Creating environments from {task_json_path} "
        f"with work_dir {work_dir}"
    )

    # Load tasks from JSON (reuses ml.env.load_tasks_from_json)
    tasks = load_tasks_from_json(task_json_path, work_dir)

    # Create task group
    group_id = Path(task_json_path).stem
    logger.info(f"[ml_code] Creating task group: {group_id}")
    task_group = TaskGroup(group_id=group_id, tasks=tasks)

    # Log task ordering
    ordered_tasks = task_group.get_ordered_tasks()
    logger.info(f"[ml_code] Task execution order: {ordered_tasks}")

    # Cherry-pick retrieval-only tools
    retrieval_tools = _get_retrieval_tools()
    logger.info(f"[ml_code] Retrieval tools: {list(retrieval_tools.keys())}")

    # Create sandbox + its tools
    sandbox_config = _create_sandbox_config()
    sandbox, sandbox_tools = create_sandbox_tools(sandbox_config)
    sandbox.start()
    logger.info(f"[ml_code] Sandbox tools: {list(sandbox_tools.keys())}")

    # Build environments: retrieval tools are subtask-specific (selected per
    # task via the JSON "tools" list), sandbox tools are common to all tasks.
    # SandboxAwareEnvironment keeps the sandbox cwd in sync with trial dirs.
    environments: dict[str, SandboxAwareEnvironment] = {}
    for task_id in task_group.tasks:
        environments[task_id] = SandboxAwareEnvironment(
            sandbox=sandbox,
            task_id=task_id,
            task_group=task_group,
            subtask_specific_tools=retrieval_tools,
            taskgroup_common_tools=sandbox_tools,
            base_work_dir=work_dir,
        )

    return environments


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python -m ml_code.env <tasks_json_path> [port]")
        sys.exit(1)

    tasks_json_path = sys.argv[1]

    if len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            logger.error(f"Invalid port: {sys.argv[2]}. Using default.")
            port = int(os.environ.get("CORRAL_PORT", "8000"))
    else:
        port = int(os.environ.get("CORRAL_PORT", "8000"))

    host = os.environ.get("CORRAL_HOST", "0.0.0.0")
    work_dir = os.environ.get("CORRAL_WORK_DIR", BASE_WORK_DIR)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    environments = create_environments(
        task_json_path=tasks_json_path,
        work_dir=work_dir,
    )

    logger.info("[ml_code] Created environments:")
    for env_id, env in environments.items():
        logger.info(f"  - {env_id}: {env.current_task.name}")
        tools_list = list(env.tools.keys())
        logger.info(f"    Tools: {tools_list}")

    logger.info(f"[ml_code] Running server on {host}:{port}")
    run_server(environments, host, port)
