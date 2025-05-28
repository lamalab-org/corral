import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents.react import ReActAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark

load_dotenv("../.env", override=True)


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "openai/gpt-4o",
    task_ids: list | None = None,
    temperature: float = 0.0,
):
    """Run the benchmark with specified model and tasks"""

    interface = BenchmarkInterface()
    agent = ReActAgent(model=model, max_iterations=10, temperature=temperature)
    runner = MatAgentBenchmark(interface, agent)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(task_ids, trials_per_task=2, k_values=[1, 2], verbose=True)
    result.generate_report("results_ocp_passk5.json")

    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    try:
        run_benchmark()

    except Exception as e:
        logger.error(f"Benchmark failed: {e!s}")
        raise
