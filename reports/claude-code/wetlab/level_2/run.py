#!/usr/bin/env python
"""Run the Claude Code agent on wetlab corral-mini, level 2.

Runs every *task* the environment-mini/level_2 server is serving (the 2
corral-mini level-1 tasks; see ../serve_mini_env.py and
sampler/data/corral_mini_manifest_budget20.json for the selection), with
``--trials`` trials per task, reporting pass@k for k in ``--k``.

The Claude Code agent delegates the reasoning loop to the Claude Code harness
(via the Claude Agent SDK) and exposes the corral task tools to it through the
environment server's task-scoped MCP endpoint. The harness returns its answer
through a dedicated in-process `submit_answer` tool call. It therefore needs
the ``claude-agent-sdk`` package (``pip install 'corral[claude]'``) and an
ANTHROPIC_API_KEY, loaded here from the repo-root ``.env``.

wetlab keeps its active reaktoro chemical system in shared module globals
(`DEFAULT_CONCURRENCY = "process"` on `QualitativeAnalysisEnvironment`), so
the corral scheduler auto-serialises wetlab trials regardless of
`--max-concurrency` -- this run is effectively sequential no matter what that
flag is set to (there's no process-worker pool backing this simple launcher).

This uses ``CorralRunner.abench`` (the async/concurrent runner -- see
docs/documentation/how_tos/concurrent_benchmarking.md) with an
``AsyncCorralRouter`` and a per-trial ``agent_factory``.

Prerequisites
-------------
1. Set up the wetlab conda environment once (see tasks/wetlab/README.md)::

       cd tasks/wetlab
       conda env create -f environment.yml
       # reaktoro is conda-forge-only (not on PyPI), so this cannot use uv/pip.

2. Start the corral-mini level_1 server in a separate terminal::

       cd tasks/wetlab
       /opt/homebrew/Caskroom/miniconda/base/envs/wetlab/bin/python \\
           ../../reports/claude-code/wetlab/serve_mini_env.py --level 2

3. Run this script (repo-root .venv, with the `claude` extra installed)::

       uv run --extra claude python reports/claude-code/wetlab/level_2/run.py

All outputs (report JSON, agent_logs/, benchmark_checkpoints/) are written
into this directory, regardless of the invocation's working directory.

Everything is overridable from the command line -- see ``--help``.
"""

import argparse
import os
import sys
from pathlib import Path

import anyio
from dotenv import load_dotenv
from loguru import logger

from corral import AsyncCorralRouter, CorralRunner, TrialContext
from corral.agents import BaseAgent, ClaudeCodeAgent
from corral.run import sanitize_model_name

LEVEL = 2
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run the Claude Code agent on wetlab corral-mini level {LEVEL}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="localhost", help="Env server host.")
    parser.add_argument("--port", type=int, default=8000, help="Env server port.")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5",
        help="Model string for the Claude Code harness.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=(
            "Task IDs to run (space-separated). Defaults to ALL tasks the "
            "environment server is currently serving."
        ),
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="Trials (k runs) per task."
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="k values for pass@k metrics.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help=(
            "Max trials running concurrently across the whole benchmark "
            "(async scheduler). wetlab is declared process-concurrency, so "
            "the scheduler auto-serialises it regardless of this value "
            "(no process-worker pool is set up here)."
        ),
    )
    parser.add_argument(
        "--max-concurrency-per-task",
        type=int,
        default=1,
        help=(
            "Max simultaneous trials of the SAME task. Values >1 require the "
            "env server to support per-trial runtimes."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=40,
        help="Max harness turns (mapped to the SDK max_turns) per trial.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort passed to the harness.",
    )
    parser.add_argument(
        "--tool-timeout",
        type=float,
        default=300.0,
        help="Recorded in run metadata for provenance (not enforced per-tool).",
    )
    parser.add_argument(
        "--wall-clock-timeout",
        type=float,
        default=1800.0,
        help="Hard wall-clock deadline for a whole harness run, in seconds.",
    )
    parser.add_argument(
        "--tool-verbosity",
        default="brief",
        choices=[
            "minimal",
            "brief",
            "detailed",
            "procedural",
            "contextual",
            "workflow",
            "syntactical",
            "comprehensive",
            "full",
        ],
        help=(
            "Verbosity level for the MCP tool descriptions the agent sees "
            "(controls how much of each tool's docstring is exposed)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Save per-trial agent transcripts to ./agent_logs-.../.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Name for the benchmark run (used as the report filename). "
            f"Defaults to 'wetlab-l{LEVEL}-claude_code-<model>.json'."
        ),
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")

    # Every output this script produces (report JSON, agent_logs/,
    # benchmark_checkpoints/) is written relative to CWD -- anchor to this
    # file's own directory so the script is self-contained no matter where
    # it's invoked from.
    os.chdir(THIS_DIR)

    interface = AsyncCorralRouter(f"http://{args.host}:{args.port}")

    # Pre-flight: make sure the server is up and the requested tasks exist.
    try:
        available = set(await interface.get_available_tasks())
    except Exception as exc:
        logger.error(
            f"Could not reach the env server at {args.host}:{args.port} ({exc}). "
            "Start it with:  cd tasks/wetlab && "
            "/opt/homebrew/Caskroom/miniconda/base/envs/wetlab/bin/python "
            "../../reports/claude-code/wetlab/serve_mini_env.py --level "
            f"{LEVEL}"
        )
        return 1

    task_ids = args.tasks if args.tasks is not None else sorted(available)
    if not task_ids:
        logger.error("The environment server is not serving any tasks.")
        return 1

    missing = [task_id for task_id in task_ids if task_id not in available]
    if missing:
        logger.error(
            f"These task IDs are not served by the environment: {missing}. "
            f"Available tasks: {sorted(available)}"
        )
        return 1

    # Concurrent trials cannot share one agent (it accumulates per-run state),
    # so the async scheduler mints a fresh agent per trial via this factory. The
    # shared `agent` is kept only for run metadata / report labeling.
    agent_kwargs = {
        "model": args.model,
        "max_iterations": args.max_iterations,
        "reasoning_effort": args.reasoning_effort,
        "tool_timeout_s": args.tool_timeout,
        "wall_clock_timeout_s": args.wall_clock_timeout,
    }

    def make_agent(_context: TrialContext) -> BaseAgent:
        return ClaudeCodeAgent(**agent_kwargs)

    agent = ClaudeCodeAgent(**agent_kwargs)
    runner = CorralRunner(interface, agent=agent, agent_factory=make_agent)

    run_name = args.run_name or (
        f"wetlab-l{LEVEL}-claude_code-{sanitize_model_name(args.model)}"
    )

    logger.info(
        f"Running Claude Code ({args.model}) on {len(task_ids)} corral-mini "
        f"level-{LEVEL} task(s) with {args.trials} trial(s) each; "
        f"pass@k for k={args.k}; max_concurrency={args.max_concurrency} "
        "(wetlab is process-concurrency, so trials auto-serialise "
        f"regardless), per_task={args.max_concurrency_per_task}."
    )

    result = await runner.abench(
        task_ids=task_ids,
        trials_per_task=args.trials,
        k_values=args.k,
        verbose=args.verbose,
        tool_verbosity=args.tool_verbosity,
        run_name=run_name,
        max_concurrency=args.max_concurrency,
        max_concurrency_per_task=args.max_concurrency_per_task,
    )

    calculated = result.calculate_metrics()
    logger.info(f"Calculated metrics: {calculated}")
    return 0


if __name__ == "__main__":
    sys.exit(anyio.run(main))
