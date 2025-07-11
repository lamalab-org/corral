import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents import ToolCallingAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark
from corral.report import CorralWandbLogger


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "gpt-4o",
    task_ids: list | None = None,
    temperature: float = 0.0,
    run_name: str = "corral_benchmark_run",
):
    """Run the benchmark with specified model and tasks"""

    interface = BenchmarkInterface("http://0.0.0.0:8000")
    wandblogger = CorralWandbLogger(
        project="corral-test",
        group="tool_description_ablation",
        name=run_name,
    )
    agent = ToolCallingAgent(model=model, max_iterations=20, temperature=temperature)
    runner = MatAgentBenchmark(interface, agent, logger=wandblogger)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        task_ids,
        trials_per_task=5,
        k_values=[1, 2, 3, 4, 5],
        verbose=True,
        tool_verbosity="workflow",
    )
    result.generate_report("gpt-toolcalling-ml_env-workflow_verbosity.json")
    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    try:
        model = "gpt-4o"
        run_name = "gpt-toolcalling-ml_env-workflow_verbosity"
        run_benchmark(model=model, run_name=run_name)

    except Exception as e:
        logger.error(f"Benchmark failed: {e!s}")
        raise
