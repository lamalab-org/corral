import os
from dotenv import load_dotenv
import litellm
from loguru import logger

from corral.agents.react import ReActAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark

def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def run_benchmark(model: str = "gpt-4", task_ids: list = None):
    """Run the benchmark with specified model and tasks"""

    interface = BenchmarkInterface()
    agent = ReActAgent(model=model)
    runner = MatAgentBenchmark(interface, agent)

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(task_ids)

    # Print summary
    print("\n=== Benchmark Results ===")
    print(f"Model used: {model}")
    print(f"Average score: {result.average_score:.2f}")
    print(f"Tasks completed: {result.successful_tasks}/{result.total_tasks}")

    # Print detailed results
    print("\n=== Detailed Results ===")
    for task_id, task_result in result.task_results.items():
        print(f"\nTask {task_id}:")
        print(f"Score: {task_result.score}")
        print("Tool Statistics:")
        for tool, stats in task_result.tool_statistics.items():
            print(f"  - {tool}: {stats}")

if __name__ == "__main__":
    load_dotenv()
    setup_litellm()

    try:
        run_benchmark()

    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise
