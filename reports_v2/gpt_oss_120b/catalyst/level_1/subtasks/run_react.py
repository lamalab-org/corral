import os
from pathlib import Path

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
    _task_ids: list | None = None,
    temperature: float = 0,
    run_name: str = "corral_benchmark_run",
    verbose: str = "brief",
):
    """Run the benchmark with specified model and tasks"""

    interface = CorralRouter(base_url="http://localhost:8000")
    wandblogger = CorralWandbLogger(
        project="corral_catalyst_oss",
        group="gpt_oss",
        name=run_name,
    )
    agent = ReActAgent(
        model=model,
        max_iterations=20,
        temperature=temperature,
        api_endpoint="https://api.helmholtz-blablador.fz-juelich.de/v1",
    )
    runner = CorralRunner(
        interface,
        agent,
        logger=wandblogger,
    )

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        trials_per_task=5,
        k_values=[1, 2, 3, 4, 5],
        verbose=True,
        tool_verbosity=verbose,
    )
    result.generate_report(f"{run_name}_try.json")
    logger.info("Benchmark completed")


def rename_output_dirs(verbosity: str, agent: str):
    for dir_name in ["logprobs", "metrics"]:
        src_dir = Path(dir_name)
        if src_dir.exists() and src_dir.is_dir():
            dest_dir = Path(f"{dir_name}_{verbosity}_{agent}")
            src_dir.rename(dest_dir)
            logger.info(f"Renamed {src_dir} to {dest_dir}")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()
    os.environ["OPENAI_API_KEY"] = os.getenv("BLABLADOR_API_KEY_TEST", "")
    agent_type = "react"
    verbosities = ["brief", "workflow", "comprehensive"]
    for verbose in verbosities:
        run_name = f"gpt_oss_120-{agent_type}-catalyst-{verbose}_verbosity-single"
        model = (
            "openai/1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
        )
        run_benchmark(model=model, run_name=run_name, verbose=verbose)
        # utility function that would rename directory logprobs and metrics to include verbosity level
        rename_output_dirs(verbose, agent_type)
