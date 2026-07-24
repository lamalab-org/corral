#!/usr/bin/env python
"""Benchmark: resistor_network level_1 / ReActAgent on Bedrock Claude Opus 4.8.

Run FROM INSIDE this directory so artifacts stay local:
    cd reports/claude-opus-4.8/resistor_network/level_1/react
    python run.py                  # (use the env's interpreter)

Writes here:
  - resistor-l1-react-opus48.json   (report)
  - checkpoints/                                    (per-trial)
  - agent_logs-ReActAgent-*/                             (traces)

Prereq: the resistor_network server for LEVEL 1 must be running on port 8130.

NOTE: Opus 4.8 rejects the `temperature` param; `_opus_shim` (imported below)
strips it from litellm.completion calls for this model.
"""

import sys
from pathlib import Path

# Make the shared Opus 4.8 shim importable (reports/claude-opus-4.8/_opus_shim.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import _opus_shim  # noqa: E402,F401  (monkeypatches litellm for Opus 4.8)

from loguru import logger  # noqa: E402

from corral.agents import ReActAgent  # noqa: E402
from corral.router import CorralRouter  # noqa: E402
from corral.run import CorralRunner  # noqa: E402

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
PORT = 8130
LEVEL = 1
VERBOSITY = "workflow"
RUN_NAME = "resistor-l1-react-opus48"


def main() -> int:
    interface = CorralRouter(base_url=f"http://localhost:{PORT}")
    all_tasks = interface.get_available_tasks()
    if not all_tasks:
        logger.error(f"No tasks on port {PORT} - is the resistor_network level 1 server running?")
        return 2
    task_ids = all_tasks  # ALL tasks for this level
    logger.info(f"Running {RUN_NAME} on {len(task_ids)} task(s)")

    agent = ReActAgent(model=MODEL, max_iterations=20, temperature=0.0)
    runner = CorralRunner(interface, agent, checkpoint_dir="./checkpoints")

    result = runner.bench(
        task_ids=task_ids,
        trials_per_task=5,
        k_values=[1, 2, 3, 4, 5],
        verbose=True,
        tool_verbosity=VERBOSITY,
        run_name=RUN_NAME,
    )

    logger.info("Benchmark completed")
    try:
        metrics = result.calculate_metrics()
        for name in sorted(metrics):
            v = metrics[name]
            if isinstance(v, (int, float)):
                logger.info(f"{name}: {v:.3f}")
    except Exception as e:
        logger.warning(f"metric summary unavailable: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
