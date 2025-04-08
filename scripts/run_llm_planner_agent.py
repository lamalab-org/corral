import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents.llm_planner import LLMPlanner
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(model: str = "gpt-4o", task_ids: list | None = None):
    """Run the benchmark with specified model and tasks"""

    interface = BenchmarkInterface()
    agent = LLMPlanner(model=model)
    runner = MatAgentBenchmark(interface, agent)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(task_ids, trials_per_task=1, k_values=[1], verbose=True)
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
