"""Validate all traces in the trace registry.

Run after build_trace_registry.py. Checks that every trace file:
- Exists and is loadable as JSON
- Has assistant messages in the correct format
- Reports which num_steps values are usable per trace
"""

import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INTERVENTION_ROOT, NUM_STEPS_VALUES


def validate_trace(trace_path: str, agent_type: str) -> dict:
    path = Path(trace_path)
    if not path.exists():
        return {"valid": False, "error": f"File not found: {trace_path}"}

    try:
        with path.open() as f:
            trace = json.load(f)
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON: {e}"}

    messages = trace.get("messages")
    if not messages:
        return {"valid": False, "error": "No 'messages' key or empty"}

    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    n = len(assistant_msgs)
    if n == 0:
        return {"valid": False, "error": "No assistant messages"}

    if agent_type == "react":
        if not any("<thought>" in m.get("content", "") for m in assistant_msgs):
            return {"valid": False, "error": "ReAct trace missing <thought> tags"}
    elif agent_type == "toolcalling":
        has_any = any(m.get("tool_calls") or m.get("content") for m in assistant_msgs)
        if not has_any:
            return {"valid": False, "error": "ToolCalling trace has no content"}

    usable = [
        ns
        for ns in NUM_STEPS_VALUES
        if (ns >= 0 and ns <= n) or (ns < 0 and abs(ns) < n)
    ]

    return {
        "valid": True,
        "error": None,
        "assistant_steps": n,
        "usable_num_steps": usable,
    }


def main():
    registry_path = INTERVENTION_ROOT / "trace_registry.json"
    if not registry_path.exists():
        logger.info(
            f"ERROR: {registry_path} not found. Run build_trace_registry.py first."
        )
        sys.exit(1)

    with registry_path.open() as f:
        registry = json.load(f)

    total = valid_count = 0
    all_valid = True

    for key, entry in registry.items():
        for trace_type in ["success_trace", "failed_trace"]:
            total += 1
            result = validate_trace(entry[trace_type], entry["agent_type"])
            prefix = f"{key} [{trace_type}]"

            if not result["valid"]:
                logger.info(f"  FAIL: {prefix} - {result['error']}")
                all_valid = False
            else:
                valid_count += 1
                logger.info(
                    f"  OK:   {prefix} - {result['assistant_steps']} steps, usable: {result['usable_num_steps']}"
                )

    logger.info(
        f"\nValidated {total} traces: {valid_count} OK, {total - valid_count} FAILED"
    )
    if not all_valid:
        sys.exit(1)
    logger.info("All traces valid.")


if __name__ == "__main__":
    main()
