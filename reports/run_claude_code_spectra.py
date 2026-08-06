"""Run the Claude Code agent on the spectra_elucidation benchmark.

This runs every served *task* (the full, top-level tasks — not subtasks) with k=3
trials per task, reporting pass@k for k in {1, 2, 3}.

The Claude Code agent delegates the reasoning loop to the Claude Code harness
(via the Claude Agent SDK) and exposes the corral task tools to it through an
in-process MCP server. It therefore needs the ``claude-agent-sdk`` package and
an ANTHROPIC_API_KEY.

Prerequisites
-------------
1. Install corral and the spectra_elucidation environment (once)::

       uv pip install -e .
       cd tasks/spectra_elucidation && uv venv && uv pip install -e .

2. Start the environment server in a separate terminal. Use the DEFAULT flags
   (no ``--subtask_level``) so the server serves the top-level *tasks* from
   ``environments/level_1/tasks_json`` rather than the subtasks::

       cd tasks/spectra_elucidation/spectra_elucidation
       python env.py --level 1              # serves on http://localhost:8000

3. Export your ANTHROPIC_API_KEY, then run this script::

       export ANTHROPIC_API_KEY=sk-ant-...
       python run_scripts/run_claude_code_spectra.py

Everything is overridable from the command line — see ``--help``.
"""

import argparse
import sys

import anyio
from loguru import logger

from corral import AsyncCorralRouter, CorralRunner, TrialContext
from corral.agents import BaseAgent, ClaudeCodeAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Claude Code agent on spectra_elucidation tasks.",
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
        "--trials", type=int, default=3, help="Trials (k runs) per task."
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="k values for pass@k metrics.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help=(
            "Max trials running concurrently across the whole benchmark "
            "(async scheduler). 1 runs the byte-for-byte serial path."
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
        default=30,
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
        help="Per-tool execution timeout in seconds.",
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
        help="Save per-trial agent transcripts.",
    )
    parser.add_argument(
        "--run-name",
        default="claude_code_spectra_tasks",
        help="Name for the benchmark run (used in the report filename).",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    interface = AsyncCorralRouter(f"http://{args.host}:{args.port}")

    # Pre-flight: make sure the server is up and the requested tasks exist.
    try:
        available = set(await interface.get_available_tasks())
    except Exception as exc:
        logger.error(
            f"Could not reach the env server at {args.host}:{args.port} ({exc}). "
            "Start it with:  cd tasks/spectra_elucidation/spectra_elucidation && "
            "python env.py --level 1"
        )
        return 1

    # Default to every task the server is serving, not a hardcoded pair.
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

    logger.info(
        f"Running Claude Code ({args.model}) on {len(task_ids)} task(s) "
        f"with {args.trials} trial(s) each; pass@k for k={args.k}; "
        f"max_concurrency={args.max_concurrency}, "
        f"per_task={args.max_concurrency_per_task}."
    )

    result = await runner.abench(
        task_ids=task_ids,
        trials_per_task=args.trials,
        k_values=args.k,
        verbose=args.verbose,
        tool_verbosity=args.tool_verbosity,
        run_name=args.run_name,
        max_concurrency=args.max_concurrency,
        max_concurrency_per_task=args.max_concurrency_per_task,
    )

    logger.info(f"Overall score: {result.total_score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(anyio.run(main))
