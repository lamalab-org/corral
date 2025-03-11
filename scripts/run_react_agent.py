import os

import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents.react import ReActAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def run_benchmark(model: str = "gpt-4", task_ids: list | None = None):
    """Run the benchmark with specified model and tasks"""

    interface = BenchmarkInterface()
    agent = ReActAgent(model=model)
    runner = MatAgentBenchmark(interface, agent)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(task_ids, trials_per_task=2, k_values=[1, 2])
    result.generate_report("results.json")

    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    try:
        run_benchmark()

    except Exception as e:
        logger.error(f"Benchmark failed: {e!s}")
        raise
