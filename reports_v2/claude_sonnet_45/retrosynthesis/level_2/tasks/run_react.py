import litellm
from dotenv import load_dotenv
from loguru import logger

from corral import CorralRouter, CorralRunner
from corral.agents import ReActAgent
from corral.report import CorralWandbLogger


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "claude-3-5-sonnet-20241022",
    task_ids: list | None = None,
    temperature: float = 0.0,
    run_name: str = "corral_benchmark_tests",
    verbose: str = "brief",
):
    """Run the benchmark with specified model and tasks"""

    interface = CorralRouter()
    wandblogger = CorralWandbLogger(
        project="corral",
        group="tool_description_ablation",
        name=run_name,
    )
    agent = ReActAgent(model=model, max_iterations=20, temperature=temperature)
    runner = CorralRunner(interface, agent, logger=wandblogger)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        task_ids,
        trials_per_task=5,
        k_values=[1, 2, 3, 4, 5],
        verbose=True,
        tool_verbosity=verbose,
    )
    result.generate_report(f"{run_name}.json")
    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    verboses = [
        "brief",
        # "workflow",
        # "comprehensive",
    ]
    for verbose in verboses:
        logger.info(f"Running benchmark with verbosity: {verbose}")
        try:
            model = "claude-sonnet-4-5-20250929"
            run_name = f"claude_45_sonnet-react-retro_lvl2_env-{verbose}_verbosity"
            run_benchmark(model=model, run_name=run_name, verbose=verbose)

        except Exception as e:
            logger.error(f"Benchmark failed: {e!s}")
            raise
