#!/usr/bin/env python
"""Generic Corral benchmark runner.

Connects to an already-running Corral environment server, runs ONE task with a
chosen agent (ReAct or ToolCalling) on Bedrock DeepSeek, and writes a JSON report.

Usage:
    python run_benchmark.py \
        --port 8001 \
        --agent react \
        --model bedrock/deepseek.v3.2 \
        --out reports/spectra_elucidation/react \
        --run-name spectra-react-smoke \
        --max-tasks 1 \
        --max-iterations 12

This script is intentionally self-contained so each per-env agent can copy/run it
from inside that env's virtualenv (where `corral` is importable).
"""

import argparse
import sys
from pathlib import Path


def build_agent(agent_kind: str, model: str, max_iterations: int, temperature: float):
    from corral.agents import ReActAgent, ToolCallingAgent

    kinds = {"react": ReActAgent, "toolcalling": ToolCallingAgent}
    if agent_kind not in kinds:
        raise SystemExit(f"--agent must be one of {list(kinds)}, got {agent_kind!r}")
    return kinds[agent_kind](
        model=model,
        max_iterations=max_iterations,
        temperature=temperature,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="localhost")
    p.add_argument("--agent", required=True, choices=["react", "toolcalling"])
    p.add_argument("--model", default="bedrock/deepseek.v3.2")
    p.add_argument("--out", required=True, help="Output directory for the report")
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-tasks", type=int, default=1,
                   help="Limit number of tasks (use a small number for a smoke run)")
    p.add_argument("--trials-per-task", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--tool-verbosity", default="brief")
    args = p.parse_args()

    from corral.run import CorralRunner
    from corral.router import CorralRouter

    base_url = f"http://{args.host}:{args.port}"
    interface = CorralRouter(base_url=base_url)

    all_tasks = interface.get_available_tasks()
    if not all_tasks:
        print(f"No tasks available at {base_url}", file=sys.stderr)
        return 2
    task_ids = all_tasks[: args.max_tasks] if args.max_tasks > 0 else all_tasks
    print(f"Connected to {base_url}; running {len(task_ids)}/{len(all_tasks)} task(s): {task_ids}")

    agent = build_agent(args.agent, args.model, args.max_iterations, args.temperature)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = CorralRunner(
        interface,
        agent,
        checkpoint_dir=str(out_dir / "checkpoints"),
    )

    run_name = args.run_name or f"{args.agent}-run"
    report_path = out_dir / f"{run_name}.json"

    result = runner.bench(
        task_ids=task_ids,
        trials_per_task=args.trials_per_task,
        verbose=True,
        tool_verbosity=args.tool_verbosity,
        run_name=str(report_path),
    )

    print("\n==== RESULT ====")
    try:
        metrics = result.calculate_metrics()
        for name in sorted(metrics):
            value = metrics[name]
            if isinstance(value, (int, float)):
                print(f"{name}: {value:.3f}")
    except Exception as e:
        print(f"(metric summary unavailable: {e})")
    print(f"Report written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
