import os
import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents import ToolCallingAgent
from corral import CorralRouter, CorralRunner
from corral.report import CorralWandbLogger


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "gpt-4o-2024-08-06",
    task_ids: list | None = None,
    temperature: float = 0.0,
    run_name: str = "corral_benchmark_run",
    verbose: str = "workflow",
):
    """Run the benchmark with specified model and tasks"""

    host = os.environ.get("CORRAL_HOST","0.0.0.0")
    port = os.environ.get("CORRAL_PORT","8000")
    interface = CorralRouter(base_url=f"http://{host}:{port}")
    # wandblogger = CorralWandbLogger(
    #     project="corral",
    #     group="tool_description_ablation",
    #     name=run_name,
    # )
    agent = ToolCallingAgent(model=model, max_iterations=30, temperature=temperature)
    runner = CorralRunner(interface, agent)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        task_ids,
        trials_per_task=1,
        k_values=[1],
        verbose=True,
        tool_verbosity=verbose,
    )
    result.generate_report(f"{run_name}.json")
    logger.info("Benchmark completed")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    verbosities = [
        #"brief",
        "workflow",
        #"comprehensive",
    ]

    models = [
        "gpt-4o-2024-08-06",
        #"claude-sonnet-4-5-20250929"
    ]
    for model in models:
        for verbosity in verbosities:
            logger.info(f"Running benchmark with verbosity: {verbosity}")
            try: 
                model_name = "gpt_4o" if model.startswith("gpt-4o") else ("claude_45" if model.startswith("claude") else None)
                if model_name is not None:
                    run_name = f"{model_name}-Tool_Calling-WetLab-{verbosity}"
                    run_benchmark(model=model, run_name=run_name, verbose=verbosity)
                else: 
                    logger.error(f"Invalid model: {model!r}")

            except Exception as e:
                logger.error(f"Benchmark failed: {e!s}")
                raise
