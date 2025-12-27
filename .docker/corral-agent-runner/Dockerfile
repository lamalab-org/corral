# syntax=docker/dockerfile:1

# Corral Agent Runner Docker Image
# Runs agent benchmarks against Corral environments using different LLM models and agent classes
#
# This image installs corral from GitHub and includes the benchmark runner script

FROM python:3.11-slim

LABEL maintainer="Corral Team"
LABEL description="Corral Agent Runner for benchmarking"

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/corral-workspace

# Install uv (native installation - much faster than pip install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install corral core from GitHub
RUN uv pip install --system git+https://github.com/lamalab-org/corral.git@main

# Install runtime dependencies used by the agent runner
RUN uv pip install --system litellm python-dotenv loguru wandb

# Create entrypoint script inline (self-contained, no external file needed)
RUN cat <<'EOF' > /opt/corral-workspace/run_agent.py
#!/usr/bin/env python3
"""Wrapper runner script that exposes the main options via env vars and CLI args.

This script mirrors the flow in `reports/afm/run_ReActAgent _claude.py` but allows
configuration of model, agent, temperature, k values, base url and more.
"""
import os
import argparse
import subprocess
from dotenv import load_dotenv
from loguru import logger

import litellm

import inspect
import json
from importlib import import_module
from pathlib import Path

from corral import CorralRouter, CorralRunner
from corral.report import CorralWandbLogger

# Results directory - can be overridden via environment variable
RESULTS_DIR = os.getenv("RESULTS_DIR", "/opt/corral-workspace/results")


def check_volume_mounted():
    """Warn if results directory is not a mounted volume."""
    results_path = Path(RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)

    # Check if path is a mount point (simple heuristic)
    try:
        result = subprocess.run(
            ["mountpoint", "-q", str(results_path)],
            capture_output=True
        )
        if result.returncode != 0:
            logger.warning(
                f"⚠️  Results directory '{RESULTS_DIR}' is not a mounted volume. "
                "Results will be lost when the container exits! "
                "Use -v $(pwd)/results:/opt/corral-workspace/results to persist results."
            )
    except Exception:
        pass  # mountpoint command not available, skip check


def setup_litellm():
    litellm.set_verbose = True


def discover_agent_classes() -> dict[str, type]:
    """Discover classes defined in `corral.agents` that inherit from BaseAgent.

    Returns a mapping of class name -> class object.
    """
    try:
        agent_mod = import_module("corral.agents")
    except Exception:
        return {}

    classes: dict[str, type] = {}
    for attr_name in dir(agent_mod):
        attr = getattr(agent_mod, attr_name)
        if inspect.isclass(attr):
            # detect BaseAgent in MRO by name to avoid heavy imports
            mro_names = [c.__name__ for c in getattr(attr, "__mro__", [])]
            if "BaseAgent" in mro_names and attr.__module__.startswith("corral.agents"):
                classes[attr.__name__] = attr
    return classes


AGENT_CLASSES = discover_agent_classes()


def parse_k_values(k_str: str):
    return [int(x) for x in k_str.split(",") if x.strip()]


def parse_json_or_file(value: str | None) -> dict | None:
    """Parse a JSON string or @file reference into a dict. Return None if empty."""
    if not value:
        return None
    value = value.strip()
    try:
        if value.startswith("@"):
            path = value[1:]
            with Path(path).open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return json.loads(value)
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON or file for agent params: {exc}") from exc


def run_benchmark(
    model: str,
    base_url: str,
    temperature: float,
    run_name: str,
    trials_per_task: int,
    k_values: list,
    agent_class_name: str,
    max_iterations: int,
    agent_kwargs: dict | None = None,
    api_endpoint: str | None = None,
    default_verbosity: str = "brief",
    checkpoint_dir: str | None = None,
    checkpoint_name: str | None = None,
    session_id: str | None = None,
    configure_timeout: float | None = None,
    enable_surrender: bool = False,
    wandb_project: str = "corral",
    wandb_entity: str | None = None,
    wandb_group: str = "agent-runner",
    task_ids: list | None = None,
    verbose: bool = False,
    tool_verbosity: str = "brief",
):
    # Ensure results directory exists
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Use RESULTS_DIR for checkpoints if not specified
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(RESULTS_DIR, "benchmark_checkpoints")

    interface = CorralRouter(base_url, default_verbosity=default_verbosity)
    # configure wandb logger if api key is present, otherwise fallback to simple logger
    wandb_logger = None
    if os.getenv("WANDB_API_KEY"):
        wandb_logger = CorralWandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            name=run_name
        )

    agent_cls = AGENT_CLASSES.get(agent_class_name)
    if agent_cls is None:
        raise ValueError(f"Unknown agent class: {agent_class_name}. Available: {list(AGENT_CLASSES.keys())}")

    # Prepare kwargs for agent constructor. Allow full control via `agent_kwargs` dict.
    agent_init_kwargs = dict(agent_kwargs or {})
    # provide sensible defaults if not specified
    agent_init_kwargs.setdefault("model", model)
    agent_init_kwargs.setdefault("max_iterations", max_iterations)
    agent_init_kwargs.setdefault("temperature", temperature)
    if api_endpoint is not None:
        agent_init_kwargs.setdefault("api_endpoint", api_endpoint)

    agent = agent_cls(**agent_init_kwargs)
    runner = CorralRunner(
        interface,
        agent,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        logger=wandb_logger,
        enable_surrender=enable_surrender,
    )

    logger.info(f"Starting benchmark with model: {model}, base_url: {base_url}")
    result = runner.bench(
        task_ids=task_ids,
        trials_per_task=trials_per_task,
        k_values=k_values,
        verbose=verbose,
        session_id=session_id,
        tool_verbosity=tool_verbosity,
        configure_timeout=configure_timeout,
    )
    # Save report to RESULTS_DIR
    out_path = os.path.join(RESULTS_DIR, f"{run_name}.json")
    result.generate_report(out_path)
    logger.info(f"Benchmark completed — report saved to {out_path}")


def main():
    load_dotenv()
    setup_litellm()

    # Check if results directory is mounted (warn if not)
    check_volume_mounted()

    parser = argparse.ArgumentParser()
    # Agent params
    parser.add_argument("--model", default=os.getenv("MODEL", "claude-3-5-sonnet-20241022"))
    parser.add_argument("--agent-class", default=os.getenv("AGENT_CLASS", "ReActAgent"))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("MAX_ITERATIONS", "10")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.0")))
    parser.add_argument("--api-endpoint", default=os.getenv("API_ENDPOINT", None))
    parser.add_argument("--agent-params", default=os.getenv("AGENT_PARAMS", None),
                        help="JSON string or @path to JSON file with additional kwargs for the agent constructor")
    parser.add_argument("--list-agents", action="store_true", help="List discovered agent classes and exit")
    # Router params
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000"))
    parser.add_argument("--default-verbosity", default=os.getenv("DEFAULT_VERBOSITY", "brief"))
    # Runner params
    parser.add_argument("--run-name", default=os.getenv("RUN_NAME", "corral_run"))
    parser.add_argument("--trials-per-task", type=int, default=int(os.getenv("TRIALS_PER_TASK", "5")))
    parser.add_argument("--k-values", default=os.getenv("K_VALUES", "1,2,3,4,5"))
    parser.add_argument("--tool-verbosity", default=os.getenv("TOOL_VERBOSITY", "brief"))
    parser.add_argument("--verbose", type=lambda x: x.lower() == "true", default=os.getenv("VERBOSE", "false"))
    parser.add_argument("--task-ids", default=os.getenv("TASK_IDS", None), help="Comma-separated task ids")
    parser.add_argument("--checkpoint-dir", default=os.getenv("CHECKPOINT_DIR", None))
    parser.add_argument("--checkpoint-name", default=os.getenv("CHECKPOINT_NAME", None))
    parser.add_argument("--session-id", default=os.getenv("SESSION_ID", None))
    parser.add_argument("--configure-timeout", type=float, default=float(os.getenv("CONFIGURE_TIMEOUT", "60.0")) if os.getenv("CONFIGURE_TIMEOUT") else None)
    parser.add_argument("--enable-surrender", type=lambda x: x.lower() == "true", default=os.getenv("ENABLE_SURRENDER", "false"))
    # WandB logger params
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "corral"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", None))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_GROUP", "agent-runner"))
    args = parser.parse_args()

    if getattr(args, "list_agents", False):
        logger.info("Available agent classes:")
        for name in sorted(AGENT_CLASSES.keys()):
            logger.info(f" - {name}")
        return

    k_values = parse_k_values(args.k_values)
    task_ids = None
    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]

    # parse agent params (JSON string or @file). This allows full control over agent ctor.
    parsed_agent_params = parse_json_or_file(args.agent_params)
    agent_kwargs: dict | None = parsed_agent_params or {}
    # merge common shortcuts into agent_kwargs if not provided explicitly
    agent_kwargs.setdefault("model", args.model)
    agent_kwargs.setdefault("max_iterations", args.max_iterations)
    agent_kwargs.setdefault("temperature", args.temperature)
    if args.api_endpoint and "api_endpoint" not in agent_kwargs:
        agent_kwargs.setdefault("api_endpoint", args.api_endpoint)

    run_benchmark(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        run_name=args.run_name,
        trials_per_task=args.trials_per_task,
        k_values=k_values,
        agent_class_name=args.agent_class,
        max_iterations=args.max_iterations,
        agent_kwargs=agent_kwargs,
        api_endpoint=args.api_endpoint,
        default_verbosity=args.default_verbosity,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        session_id=args.session_id,
        configure_timeout=args.configure_timeout,
        enable_surrender=args.enable_surrender,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        task_ids=task_ids,
        verbose=args.verbose,
        tool_verbosity=args.tool_verbosity,
    )


if __name__ == "__main__":
    main()
EOF

RUN chmod +x /opt/corral-workspace/run_agent.py

# Create entrypoint wrapper
RUN cat <<'EOF' > /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

# Corral Agent Runner Entrypoint
# Forwards arguments to run_agent.py

exec uv run python3 /opt/corral-workspace/run_agent.py "$@"
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PATH="/root/.local/bin:${PATH}"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
