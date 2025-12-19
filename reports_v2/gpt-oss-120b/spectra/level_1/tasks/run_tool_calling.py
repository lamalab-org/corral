import os

import litellm
from dotenv import load_dotenv
from loguru import logger

from corral import CorralRouter, CorralRunner
from corral.agents import ToolCallingAgent


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "claude-3-5-sonnet-20241022",
    task_ids: list | None = None,
    temperature: float = 0.0,
    run_name: str = "corral_benchmark_run",
    verbose: str = "brief",
):
    """Run the benchmark with specified model and tasks"""

    interface = CorralRouter(base_url="http://localhost:1231")
    agent = ToolCallingAgent(
        model=model,
        max_iterations=20,
        temperature=temperature,
        api_endpoint="https://api.helmholtz-blablador.fz-juelich.de/v1",
    )
    runner = CorralRunner(interface, agent)

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
    load_dotenv("../../../../../.env", override=True)
    setup_litellm()
    os.environ["OPENAI_API_KEY"] = os.getenv("BLABLADOR_API_KEY_TEST", "")

    verboses = [
        # "brief",
        "workflow",
        # "comprehensive",
    ]
    for verbose in verboses:
        logger.info(f"Running benchmark with verbosity: {verbose}")
        try:
            model = "openai/1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
            run_name = f"gpt_oss_120-tool_calling-spectra_lvl1_env-{verbose}_verbosity"
            run_benchmark(model=model, run_name=run_name, verbose=verbose)

        except Exception as e:
            logger.error(f"Benchmark failed: {e!s}")
            raise
