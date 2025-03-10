from __future__ import annotations

import os
import argparse

import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents.react import ReActAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True

    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def run_benchmark(model: str = "gpt-4", task_ids: list | None = None, results_dir: str = "./results", local_results_dir: str = "./results", app = "simagent", bash_command = "run_bash_command", dir_command = None):
    """Run the benchmark with specified model and tasks"""

    interface = BenchmarkInterface()
    agent = ReActAgent(model=model, max_iterations = 10, temperature = 0.1)
    runner = MatAgentBenchmark(interface, agent, k=3, results_dir=results_dir, local_results_dir = local_results_dir, app = app, bash_command=bash_command, dir_command=dir_command)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}, results_dir: {results_dir}")
    result = runner.bench(task_ids)
    os.makedirs(local_results_dir, exist_ok=True)
    result.report(os.path.join(local_results_dir, "results.json"))

    logger.info("Benchmark completed")

if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    # **Argument Parsing**
    parser = argparse.ArgumentParser(description="Run the ReAct agent benchmark.")
    parser.add_argument("--model", type=str, default="gpt-4", help="The model to use (e.g., gpt-4)")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to store results")
    parser.add_argument("--local_results_dir", type=str, default="./results", help="Directory to store results")
    parser.add_argument("--app", type=str, default="simagent", help="Application name")
    parser.add_argument("--bash_command", type=str, default="run_bash_command", help="Command for running bash")
    args = parser.parse_args()

    try:
        run_benchmark(
            model=args.model,
            results_dir=args.results_dir,
            local_results_dir=args.local_results_dir,
            app=args.app,
            bash_command=args.bash_command,
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e!s}")
        raise


# if __name__ == "__main__":
#     load_dotenv()
#     setup_litellm()

#     try:
#         # run_benchmark(model: str = "gpt-4", task_ids: list | None = None, results_dir: str = "./results", app = "simagent", bash_command = "run_bash_command", dir_command = "change_directory")
#         run_benchmark(model = "gpt-4", results_dir='/results/results_lammps_test', app = "simagent", bash_command = "run_bash_command", dir_command = "change_directory")

#     except Exception as e:
#         logger.error(f"Benchmark failed: {e!s}")
#         raise
