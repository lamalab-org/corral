"""
Run kinetic fitting benchmark with ReActAgent.

Simple script that connects to a running corral server and executes
the kinetic fitting benchmark with the specified LLM model.
"""

import os
from pathlib import Path

import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents import ReActAgent
from corral.router.routes import CorralRouter
from corral.run import CorralRunner
from corral.report import CorralWandbLogger


def main():
    """Run the kinetic fitting benchmark"""
    load_dotenv()

    # Configuration
    model = "claude-sonnet-4-5"
    max_iterations = 20
    temperature = 0.0
    trials_per_task = 5
    port = 8004

    # Setup
    litellm.set_verbose = True

    # Setup work directory
    work_dir = Path(__file__).parent / "benchmark_workspace"
    work_dir.mkdir(exist_ok=True)
    os.environ["CORRAL_WORK_DIR"] = str(work_dir)

    logger.info("🧪 Running kinetic fitting benchmark")
    logger.info(f"Model: {model}")
    logger.info(f"Max iterations: {max_iterations}, Temperature: {temperature}")
    logger.info("=" * 60)

    try:
        # Connect to server
        logger.info(f"Connecting to corral server on port {port}...")
        interface = CorralRouter(base_url=f"http://localhost:{port}")

        # Verify connection
        tasks = interface.get_available_tasks()
        logger.info(f"Connected! Available tasks: {tasks}")

        if "kinetic_fitting" not in tasks:
            logger.error(f"Task 'kinetic_fitting' not found. Available: {tasks}")
            return False

    except Exception as e:
        logger.error(f"Failed to connect to server: {e}")
        logger.info("Make sure the server is running:")
        logger.info("  python start_server.py")
        return False

    # Create agent
    agent = ReActAgent(
        model=model, max_iterations=max_iterations, temperature=temperature
    )
    logger.info(f"Created ReActAgent with {model}")

    # Setup W&B logging (optional)
    wandb_logger = None
    try:
        wandb_logger = CorralWandbLogger(
            project="kinetic_fitting_benchmark",
            name=f"kinetic_fitting_{model.replace('-', '_')}",
        )
        logger.info("W&B logging enabled")
    except Exception as e:
        logger.warning(f"W&B logging disabled: {e}")

    # Create benchmark runner
    runner = CorralRunner(
        interface=interface,
        agent=agent,
        checkpoint_dir=str(work_dir / "checkpoints"),
        logger=wandb_logger,
    )

    # Run benchmark
    logger.info("Starting benchmark...")
    try:
        result = runner.bench(
            task_ids=["kinetic_fitting"],
            trials_per_task=trials_per_task,
            verbose=True,
            tool_verbosity="comprehensive",
        )

        logger.info("🎉 Benchmark completed successfully!")
        logger.info(f"Results: {result}")
        return True

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
