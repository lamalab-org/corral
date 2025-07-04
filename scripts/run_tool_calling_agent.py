import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents import ToolCallingAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark
import os

def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(model: str = "gpt-4o", task_ids: list | None = None, temperature: float = 0.0, work_dir = "./results"):
    """Run the benchmark wsith specified model and tasks"""

    os.makedirs(work_dir, exist_ok=True)

    interface = BenchmarkInterface()
    agent = ToolCallingAgent(model=model, max_iterations=20 , temperature=temperature)
    runner = MatAgentBenchmark(interface, agent)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(task_ids, trials_per_task=5, k_values=[1, 2, 3, 4, 5], verbose=True)

    # Save result to the specified work_dir
    result_path = os.path.join(work_dir, "results.json")
    result.generate_report(result_path)

    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    try:
        task_type = f"task_5"
        # model = "gpt-4o"
        model = "anthropic/claude-3-7-sonnet-20250219"
        agent = "tool_calling"
        path = "./19_June_2025_AFM"
        # path = "./test"
        work_dir = f"{path}/{agent}/{model}/{task_type}"
        run_benchmark(model = model, work_dir = work_dir)

    except Exception as e:
        logger.error(f"Benchmark failed: {e!s}")
        raise

#Set P gain to 100, I gain to 6000, and D gain to 10 in the AFM software, and then capture an image