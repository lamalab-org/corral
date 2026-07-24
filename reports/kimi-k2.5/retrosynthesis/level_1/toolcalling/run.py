#!/usr/bin/env python
import os
from loguru import logger
from corral.agents import ToolCallingAgent
from corral.router import CorralRouter
from corral.run import CorralRunner

MODEL = "bedrock/moonshotai.kimi-k2.5"
PORT = 8110
RUN_NAME = "retro-l1-toolcalling-kimi"

def main() -> int:
    interface = CorralRouter(base_url=f"http://localhost:{PORT}")
    all_tasks = interface.get_available_tasks()
    if not all_tasks:
        logger.error(f"No tasks on port {PORT}")
        return 2
    task_ids = all_tasks
    logger.info(f"Running {RUN_NAME} on {len(task_ids)} tasks")
    agent = ToolCallingAgent(model=MODEL, max_iterations=30, temperature=0.0)
    runner = CorralRunner(interface, agent, checkpoint_dir="./checkpoints")
    bench_kwargs = {}
    sid = os.environ.get("CORRAL_SESSION_ID")
    if sid:
        bench_kwargs["session_id"] = sid
        logger.info(f"Resuming from session_id={sid}")
    result = runner.bench(
        task_ids=task_ids,
        trials_per_task=5,
        k_values=[1, 2, 3, 4, 5],
        verbose=True,
        tool_verbosity="workflow",
        run_name=RUN_NAME,
        **bench_kwargs,
    )
    logger.info("completed")
    try:
        for name in sorted(result.calculate_metrics()):
            v = result.calculate_metrics()[name]
            if isinstance(v, (int, float)):
                logger.info(f"{name}: {v:.3f}")
    except Exception as e:
        logger.warning(f"metric summary unavailable: {e}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
