import os
import shutil
from pathlib import Path

import litellm
from dotenv import load_dotenv
from loguru import logger

from corral import CorralRouter, CorralRunner
from corral.agents import ReActAgent
from corral.agents.hooks import AgentHooks, HookPoint, create_trace_intervention_hook
from corral.report import CorralWandbLogger

# Map task_id -> path to trace JSON file
# Map task_id -> path to trace JSON file
success_trace_map = {
    "task_4": "/Users/n0w0f/git/n0w0f_2026/corral_fixes/corral/intervention/claude_45/resistor/step_1/none/agent_logs-ReActAgent-claude-sonnet-4-5-20250929-workflow/task_4_20260121_182338.json",
}

failed_trace_map = {
    "task_4": "/Users/n0w0f/git/n0w0f_2026/corral_fixes/corral/intervention/claude_45/resistor/step_1/none/agent_logs-ReActAgent-claude-sonnet-4-5-20250929-workflow/task_4_20260121_192349.json",
}

successful_hook = create_trace_intervention_hook(
    success_trace_map, num_steps=1, execute_tools=True
)

failed_hook = create_trace_intervention_hook(
    failed_trace_map, num_steps=1, execute_tools=True
)


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def move_work_dir_files(work_dir: str, target_subdir: str):
    """
    Moves all contents of work_dir into a new subdirectory within work_dir,
    and creates the target subdirectory if it doesn't exist.
    """
    if not work_dir:
        logger.warning("CORRAL_WORK_DIR is not set. Skipping file movement.")
        return

    target_path = Path(work_dir) / target_subdir
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Moving work directory contents to: {target_path}")

    # Move all files and folders *except* the target subdirectory itself
    for item_path in Path(work_dir).iterdir():
        if item_path.name != target_subdir:
            try:
                shutil.move(item_path, target_path)
            except Exception as e:
                logger.error(f"Failed to move {item_path}: {e!s}")

    logger.success("Work directory files moved successfully.")


def run_benchmark(
    model: str = "claude-sonnet-4-5-20250929",
    _task_ids: list | None = None,
    temperature: float = 0.7,
    run_name: str = "corral_successrate_intervention",
    verbose: str = "brief",
    hooks=None,
):
    """Run the benchmark with specified model and tasks"""

    interface = CorralRouter(base_url="http://localhost:8001")
    wandblogger = CorralWandbLogger(
        project="corral_resistor_oss",
        group="gpt_oss",
        name=run_name,
    )
    agent = ReActAgent(
        model=model,
        max_iterations=20,
        temperature=temperature,
    )
    runner = CorralRunner(
        interface,
        agent,
        logger=wandblogger,
    )

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        task_ids=[
            "task_4",
        ],
        trials_per_task=15,
        k_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        verbose=True,
        tool_verbosity=verbose,
        hooks=hooks,
    )
    result.generate_report(f"{run_name}_try.json")
    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")

    verbose = "workflow"
    conditions = ["successful"]
    for condition in conditions:
        hooks = AgentHooks()  # Fresh hooks per condition
        if condition == "successful":
            hooks.register(HookPoint.BEFORE_TASK, successful_hook)
        elif condition == "failed":
            hooks.register(HookPoint.BEFORE_TASK, failed_hook)

        run_name = f"claude_45_react-resistor_{condition}-{verbose}_verbosity-single"
        model = "claude-sonnet-4-5-20250929"
        run_benchmark(model=model, run_name=run_name, verbose=verbose, hooks=hooks)
