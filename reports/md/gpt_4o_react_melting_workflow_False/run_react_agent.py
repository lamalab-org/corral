import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents.react import ReActAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark
from corral.report import CorralWandbLogger
from corral import setup_logging

load_dotenv("../../.env", override=True)


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "gpt-4o-2024-08-06",
    task_ids: list | None = None,
    temperature: float = 0.0,
    run_name: str = "corral_benchmark_run",
    ablation: str | None = None,
    port: int = 8000,
    host: str = "0.0.0.0",
):
    """Run the benchmark with specified model and tasks"""
    if ablation is None:
        raise ValueError("Ablation must be specified")

    interface = BenchmarkInterface(f"http://{host}:{port}")
    wandblogger = CorralWandbLogger(
        project="corral",
        group="tool_description_ablation",
        name=run_name,
    )
    agent = ReActAgent(model=model, max_iterations=20, temperature=temperature)
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
    setup_logging()
    load_dotenv()
    setup_litellm()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--verbosity", required=True)
    parser.add_argument(
        "--subtask_level", type=lambda x: x.lower() == "true", required=True
    )
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.llm == "gpt_4o":
        model = "gpt-4o-2024-08-06"
    elif args.llm == "claude_35":
        model = "claude-3-5-sonnet-20241022"
    environment = args.environment
    host = "0.0.0.0"
    port = args.port
    verbosity = args.verbosity
    agent = args.agent
    task = "subtask" if args.subtask_level else "task"
    try:
        run_name = f"{args.llm}-{agent}-{environment}-{verbosity}-{task}"
        run_benchmark(
            model=model, run_name=run_name, ablation=verbosity, port=port, host=host
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e!s}")
        raise
