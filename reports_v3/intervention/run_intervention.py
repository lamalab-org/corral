"""Main parametric runner for intervention experiments.

Runs a single condition (env, agent, intervention_type, num_steps).
Should be launched from its own run directory so traces/checkpoints are stored there.

Usage:
    # Baseline (no intervention) — uses task_selection.json for task IDs:
    python run_intervention.py \
        --env spectra --agent react \
        --intervention none \
        --task-selection /path/to/task_selection.json

    # Intervention — uses trace_registry.json for task IDs + trace paths:
    python run_intervention.py \
        --env spectra --agent react \
        --intervention success --num-steps 2 \
        --trace-registry /path/to/trace_registry.json
"""

import argparse
import json
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    AWS_PROFILE,
    AWS_REGION,
    ENVIRONMENTS,
    K_VALUES,
    MODEL,
    TEMPERATURE,
    TOOL_VERBOSITY,
    TRIALS_PER_CONDITION,
    WANDB_GROUP,
    WANDB_PROJECT,
)

from corral import CorralRouter, CorralRunner
from corral.agents import ReActAgent, ToolCallingAgent
from corral.agents.hooks import AgentHooks, HookPoint, create_trace_intervention_hook
from corral.report import CorralWandbLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Run intervention experiment")
    parser.add_argument(
        "--env",
        required=True,
        choices=["spectra", "wetlab", "resistor"],
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=["react", "toolcalling"],
    )
    parser.add_argument(
        "--intervention",
        required=True,
        choices=["none", "success", "failed"],
        help="none = baseline, success/failed = inject trace steps",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=0,
        help="Steps to inject (1,2 = first N; -1,-2 = all except last N). "
        "Ignored for --intervention none.",
    )
    parser.add_argument(
        "--task-selection",
        type=str,
        default=None,
        help="Path to task_selection.json (used for baseline runs)",
    )
    parser.add_argument(
        "--trace-registry",
        type=str,
        default=None,
        help="Path to trace_registry.json (used for intervention runs)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=TRIALS_PER_CONDITION,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
    )
    args = parser.parse_args()

    # Validate: baseline needs task-selection, intervention needs trace-registry
    if args.intervention == "none" and not args.task_selection:
        parser.error("--task-selection is required for baseline (--intervention none)")
    if args.intervention != "none" and not args.trace_registry:
        parser.error("--trace-registry is required for intervention runs")

    return args


def build_run_name(args) -> str:
    parts = [args.env, args.agent, args.intervention]
    if args.intervention != "none":
        parts.append(f"steps{args.num_steps}")
    return "_".join(parts)


def load_baseline_task_ids(task_selection_path: str, env: str, agent: str) -> list[str]:
    """Load task IDs from task_selection.json for baseline runs."""
    with Path(task_selection_path).open() as f:
        selection = json.load(f)
    key = f"{env}/{agent}"
    entry = selection.get(key, {})
    return entry.get("task_ids", [])


def load_intervention_task_ids_and_trace_map(
    registry_path: str, env: str, agent: str, intervention: str
) -> tuple[list[str], dict[str, str]]:
    """Load task IDs and trace_map from trace_registry.json for intervention runs."""
    with Path(registry_path).open() as f:
        registry = json.load(f)

    task_ids = []
    trace_map = {}

    for entry in registry.values():
        if entry["environment"] != env or entry["agent_type"] != agent:
            continue

        task_id = entry["task_id"]
        task_ids.append(task_id)

        if intervention == "success":
            trace_map[task_id] = entry["success_trace"]
        elif intervention == "failed":
            trace_map[task_id] = entry["failed_trace"]

    return task_ids, trace_map


def main():
    load_dotenv()
    litellm.set_verbose = False

    args = parse_args()
    env_config = ENVIRONMENTS[args.env]
    run_name = build_run_name(args)

    logger.info(f"Run: {run_name}")
    logger.info(f"Working directory: {Path.cwd()}")

    # Load task IDs (and trace map for intervention)
    trace_map = {}
    if args.intervention == "none":
        task_ids = load_baseline_task_ids(args.task_selection, args.env, args.agent)
    else:
        task_ids, trace_map = load_intervention_task_ids_and_trace_map(
            args.trace_registry, args.env, args.agent, args.intervention
        )

    if not task_ids:
        logger.error(f"No tasks found for {args.env}/{args.agent}")
        sys.exit(1)

    logger.info(f"Tasks: {task_ids}")

    # Set up hooks
    hooks = None
    if args.intervention != "none" and trace_map:
        hooks = AgentHooks()
        hook = create_trace_intervention_hook(
            trace_map,
            num_steps=args.num_steps,
            execute_tools=True,
        )
        hooks.register(HookPoint.BEFORE_TASK, hook)
        logger.info(
            f"Intervention hook: {args.intervention} trace, "
            f"num_steps={args.num_steps}, execute_tools=True"
        )

    # Create interface — each agent type gets its own server port
    port = env_config["port"][args.agent]
    interface = CorralRouter(base_url=f"http://localhost:{port}")

    # WandB
    wandb_logger = CorralWandbLogger(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=run_name,
    )

    # Agent
    agent_kwargs = {
        "model": args.model,
        "max_iterations": env_config.get("max_iterations", 20),
        "temperature": TEMPERATURE,
    }
    if "bedrock" in args.model:
        agent_kwargs["aws_profile_name"] = AWS_PROFILE
        agent_kwargs["aws_region_name"] = AWS_REGION

    if args.agent == "react":
        agent = ReActAgent(**agent_kwargs)
    else:
        agent = ToolCallingAgent(**agent_kwargs)

    runner = CorralRunner(interface, agent, logger=wandb_logger)

    # Run
    logger.info(f"Starting: {run_name}")
    # Cap k_values at trials count
    k_values = [k for k in K_VALUES if k <= args.trials]
    result = runner.bench(
        task_ids=task_ids,
        trials_per_task=args.trials,
        k_values=k_values,
        verbose=True,
        tool_verbosity=TOOL_VERBOSITY,
        hooks=hooks,
    )

    report_name = f"{run_name}_report.json"
    result.generate_report(report_name)
    logger.info(f"Report: {report_name}")
    logger.info("Done")


if __name__ == "__main__":
    main()
