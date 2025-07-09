import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents.react import ReActAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark
from corral.report import WandbLogger

load_dotenv("../.env", override=True)


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "gpt-4o",
    task_ids: list | None = None,
    temperature: float = 0.0,
):
    """Run the benchmark with specified model and tasks"""
    wandb_project = "corral_test"  # Required: Your wandb project name
    # wandb_entity="your_entity_name", # Optional: Your wandb entity (username or team)
    wandb_group = "test"  # Optional: Group related runs
    wandb_name = "testrun-ml-functional"  # Optional: Custom run name)
    interface = BenchmarkInterface()
    agent = ReActAgent(model=model, max_iterations=20, temperature=temperature)
    wandblogger = WandbLogger(
        project=wandb_project,
        group=wandb_group,
        name=wandb_name,
    )
    runner = MatAgentBenchmark(interface, agent, logger=wandblogger)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        task_ids,
        trials_per_task=2,
        k_values=[
            1,
            2,
        ],
        verbose=True,
        tool_verbosity="full",
    )
    result.generate_report("results_react.json")

    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    try:
        model = "gpt-4o"
        run_benchmark(model=model)

    except Exception as e:
        logger.error(f"Benchmark failed: {e!s}")
        raise
