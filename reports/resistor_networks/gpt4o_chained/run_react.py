import litellm
from dotenv import load_dotenv
from loguru import logger
import os
import shutil
from corral import CorralRouter, CorralRunner
from corral.agents import ReActAgent
from corral.report import CorralWandbLogger


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True

def move_work_dir_files(work_dir: str, target_subdir: str):
    """
    Moves all contents of work_dir into a new subdirectory within work_dir,
    and creates the target subdirectory if it doesn't exist.
    """
    if not work_dir:
        logger.warning("CORRAL_WORK_DIR is not set. Skipping file movement.")
        return

    target_path = os.path.join(work_dir, target_subdir)
    os.makedirs(target_path, exist_ok=True)
    logger.info(f"Moving work directory contents to: {target_path}")

    # Move all files and folders *except* the target subdirectory itself
    for item_name in os.listdir(work_dir):
        item_path = os.path.join(work_dir, item_name)
        if item_name != target_subdir:
            try:
                shutil.move(item_path, target_path)
            except Exception as e:
                logger.error(f"Failed to move {item_path}: {e!s}")

    logger.success("Work directory files moved successfully.")

def run_benchmark(
    model: str = "none",
    task_ids: list | None = None,
    temperature: float = 0.0,
    run_name: str = "resistor_network_gpt_react_run_chained",
    verbose: str = "brief",
):
    """Run the benchmark with specified model and tasks"""

    interface = CorralRouter(base_url="http://localhost:8000")
    wandblogger = CorralWandbLogger(
        project="corral_ml",
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
    
    corral_work_dir = os.getenv("CORRAL_WORK_DIR")
    if not corral_work_dir:
        logger.error("CORRAL_WORK_DIR environment variable is not set!")

    verboses = ["brief", "workflow", "comprehensive"]
    for verbose in verboses:
        logger.info(f"Running benchmark with verbosity: {verbose}")
        try:
            model = "gpt-4o-2024-08-06"
            run_name = f"gpt4o-react-resistor_network-{verbose}_verbosity_chained"
            run_benchmark(model=model, run_name=run_name, verbose=verbose)
            
            # 2. Move the files after the benchmark completes successfully
            move_work_dir_files(corral_work_dir, verbose)

        except Exception as e:
            logger.error(f"Benchmark failed: {e!s}")
            raise
