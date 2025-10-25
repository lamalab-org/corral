import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents import ToolCallingAgent
from corral import CorralRunner, CorralRouter
from corral.report import CorralWandbLogger


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "claude-3-5-sonnet-20241022",
    task_ids: list | None = None,
    temperature: float = 0.0,
    run_name: str = "corral_benchmark_run_ml_toolcalling",
    verbose: str = "brief",
):
    """Run the benchmark with specified model and tasks"""

    interface = CorralRouter("http://localhost:8000")
    wandblogger = CorralWandbLogger(
        project="corral",
        group="tool_description_ablation",
        name=run_name,
    )
    agent = ToolCallingAgent(model=model, max_iterations=20, temperature=temperature)
    runner = CorralRunner(interface, agent, logger=wandblogger)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        task_ids,
        trials_per_task=5,
        k_values=[1,2,3,4,5],
        verbose=True,
        tool_verbosity=verbose,
    )
    result.generate_report(f"{run_name}.json")
    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    verboses = [
        # "brief",
        # "workflow",
        "comprehensive",
    ]
    for verbose in verboses:
        logger.info(f"Running benchmark with verbosity: {verbose}")
        try:
            model = "claude-3-5-sonnet-20241022"
            run_name = f"claude-3-5-sonnet-20241022_Tool_calling-test_afm-{verbose}_verbosity"
            run_benchmark(model=model, run_name=run_name, verbose=verbose)

        except Exception as e:
            logger.error(f"Benchmark failed: {e!s}")
            raise
