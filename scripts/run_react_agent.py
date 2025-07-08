from pathlib import Path

import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents import ReActAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark

load_dotenv("../.env", override=True)


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "anthropic/claude-3-7-sonnet-20250219",
    task_ids: list | None = None,
    temperature: float = 0.0,
    work_dir="./results",
):
    """Run the benchmark with specified model and tasks"""
    Path(work_dir).mkdir(exist_ok=True)

    interface = BenchmarkInterface()
    agent = ReActAgent(model=model, max_iterations=20, temperature=temperature)
    runner = MatAgentBenchmark(interface, agent)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        task_ids,
        trials_per_task=2,
        k_values=[1, 2],
        verbose=True,
        tool_verbosity="minimal",
    )

    # Save result to the specified work_dir
    result_path = Path(work_dir) / "results.json"
    result.generate_report(str(result_path))

    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    try:
        task_type = "task_6"
        model = "gpt-4o"
        # model = "anthropic/claude-3-7-sonnet-20250219"
        agent = "react"
        path = "./19_June_2025_AFM"
        work_dir = f"{path}/{agent}/{model}/{task_type}"
        run_benchmark(model=model, work_dir=work_dir)

    except Exception as e:
        logger.error(f"Benchmark failed: {e!s}")
        raise
