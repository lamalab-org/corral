#!/usr/bin/env python
"""Benchmark: retrosynthesis level_1 / ReActAgent on Bedrock DeepSeek v3.2.

Run FROM INSIDE this directory so artifacts stay local:
    cd reports/retrosynthesis/level_1/react
    python run.py                  # (use the env's interpreter)

Writes here:
  - retro-l1-react-workflow.json   (report)
  - checkpoints/                                    (per-trial)
  - agent_logs-ReActAgent-*/                        (traces)

Prereq: the retrosynthesis server for LEVEL 1 must be running on port 8110.
"""

from loguru import logger

from corral.agents import ReActAgent
from corral.router import CorralRouter
from corral.run import CorralRunner

MODEL = "bedrock/deepseek.v3.2"
PORT = 8110
LEVEL = 1
VERBOSITY = "workflow"
RUN_NAME = "retro-l1-react-workflow"


def main() -> int:
    interface = CorralRouter(base_url=f"http://localhost:{PORT}")
    all_tasks = interface.get_available_tasks()
    if not all_tasks:
        logger.error(f"No tasks on port {PORT} - is the retrosynthesis level 1 server running?")
        return 2
    task_ids = all_tasks  # ALL tasks for this level
    logger.info(f"Running {RUN_NAME} on {len(task_ids)} task(s)")

    agent = ReActAgent(model=MODEL, max_iterations=30, temperature=0.0)
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
