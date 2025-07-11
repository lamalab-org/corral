import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents import ToolCallingAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark
from corral.report import CorralWandbLogger

load_dotenv("../.env", override=True)


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "gpt-4o-2024-08-06",
    task_ids: list | None = None,
    temperature: float = 0.0,
    run_name: str = "corral_benchmark_run",
    ablation: str | None = None,
):
    """Run the benchmark with specified model and tasks"""
    if ablation is None:
        raise ValueError("Ablation must be specified")

    interface = BenchmarkInterface()
    wandblogger = CorralWandbLogger(
        project="corral",
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
        tool_verbosity=ablation,
    )
    result.generate_report(f"{run_name}.json")
    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()
    ablations = [
        "brief",
        "workflow",
        "comprehensive",
    ]
    for ablation in ablations:
        try:
            # claude-3-5-sonnet-20241022
            model = "gpt-4o-2024-08-06"
            run_name = f"gpt-tool_calling-spectra_easy-{ablation}_verbosity"
            run_benchmark(model=model, run_name=run_name, ablation=ablation)

        except Exception as e:
            logger.error(f"Benchmark failed: {e!s}")
            raise
