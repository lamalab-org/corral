"""Generate the full experimental sweep matrix.

Reads trace_registry.json and outputs all valid (env, agent, intervention, num_steps)
conditions, skipping conditions where num_steps exceeds the trace length.

Can output as:
  - JSON list of condition dicts (default)
  - Shell commands for launch_sweep.sh (--shell)
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INTERVENTION_ROOT, NUM_STEPS_VALUES


def _count_assistant_steps(trace_path: str) -> int:
    """Count assistant messages in a trace file."""
    with Path(trace_path).open() as f:
        trace = json.load(f)
    return sum(1 for m in trace.get("messages", []) if m.get("role") == "assistant")


def load_registry() -> dict:
    registry_path = INTERVENTION_ROOT / "trace_registry.json"
    with registry_path.open() as f:
        return json.load(f)


def get_usable_num_steps(n_assistant_steps: int) -> list[int]:
    """Determine which num_steps values are usable for a given trace length."""
    return [
        ns
        for ns in NUM_STEPS_VALUES
        if (ns >= 0 and ns <= n_assistant_steps)
        or (ns < 0 and abs(ns) < n_assistant_steps)
    ]


def generate_conditions(registry: dict) -> list[dict]:
    """Generate all valid experimental conditions."""
    conditions = []

    # Group by (env, agent) to get task lists
    groups = {}
    for entry in registry.values():
        group_key = (entry["environment"], entry["agent_type"])
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(entry)

    for (env, agent), entries in sorted(groups.items()):
        # Baseline condition (one per env/agent)
        conditions.append(
            {
                "env": env,
                "agent": agent,
                "intervention": "none",
                "num_steps": 0,
                "dir_name": "baseline",
            }
        )

        # Intervention conditions per trace type
        for intervention_type in ["success", "failed"]:
            traces_key = f"{intervention_type}_traces"

            # Use the max trace steps across all tasks/traces to determine
            # which num_steps values are possible. Short traces are filtered
            # out at sampling time in the hook, so we only need at least one
            # trace per task that supports the step count.
            max_steps = 0
            for e in entries:
                for trace_path in e.get(traces_key, []):
                    steps = _count_assistant_steps(trace_path)
                    if steps > max_steps:
                        max_steps = steps
            if max_steps == 0:
                continue
            usable = get_usable_num_steps(max_steps)

            for ns in usable:
                ns_label = f"n{abs(ns)}" if ns < 0 else str(ns)
                dir_name = f"{intervention_type}_step{ns_label}"
                if ns < 0:
                    dir_name = f"{intervention_type}_stepn{abs(ns)}"

                conditions.append(
                    {
                        "env": env,
                        "agent": agent,
                        "intervention": intervention_type,
                        "num_steps": ns,
                        "dir_name": dir_name,
                    }
                )

    return conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", action="store_true", help="Output as shell commands")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--env", type=str, default=None, help="Filter by environment")
    parser.add_argument("--agent", type=str, default=None, help="Filter by agent type")
    args = parser.parse_args()

    registry = load_registry()
    conditions = generate_conditions(registry)

    # Apply filters
    if args.env:
        conditions = [c for c in conditions if c["env"] == args.env]
    if args.agent:
        conditions = [c for c in conditions if c["agent"] == args.agent]

    if args.shell:
        intervention_root = INTERVENTION_ROOT
        trace_registry = intervention_root / "trace_registry.json"
        runner_script = intervention_root / "run_intervention.py"

        for c in conditions:
            run_dir = intervention_root / "runs" / c["env"] / c["agent"] / c["dir_name"]
            cmd_parts = [
                f"mkdir -p {run_dir}",
                f"cd {run_dir}",
                f"nohup uv run python {runner_script}",
                f"  --env {c['env']}",
                f"  --agent {c['agent']}",
                f"  --intervention {c['intervention']}",
                f"  --num-steps {c['num_steps']}",
                f"  --trace-registry {trace_registry}",
                "  > run.log 2>&1 &",
            ]
            logger.info(" && ".join(cmd_parts[:2]) + " && \\")
            logger.info(" \\\n".join(cmd_parts[2:]))
            logger.info("")
    elif args.json:
        print(json.dumps(conditions, indent=2))
    else:
        # Summary
        logger.info(f"Total conditions: {len(conditions)}")
        logger.info("")
        for env in sorted({c["env"] for c in conditions}):
            env_conds = [c for c in conditions if c["env"] == env]
            for agent in sorted({c["agent"] for c in env_conds}):
                agent_conds = [c for c in env_conds if c["agent"] == agent]
                logger.info(f"  {env}/{agent}: {len(agent_conds)} conditions")
                for c in agent_conds:
                    label = c["dir_name"]
                    logger.info(f"    {label}")


if __name__ == "__main__":
    main()
