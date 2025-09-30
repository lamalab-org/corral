"""
Run kinetic fitting benchmark with actual LLM agents.

This script runs the kinetic fitting environment as a proper benchmark
using the Corral framework with ReActAgent and various LLM models.
"""

import os
from pathlib import Path

import litellm
from dotenv import load_dotenv
from loguru import logger

from corral.agents import ReActAgent
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark
from corral.report import CorralWandbLogger


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True
    # Add any model-specific configurations here
    logger.info("LiteLLM configured")


def setup_environment():
    """Setup environment variables and paths"""
    # Set up work directory for kinetic fitting
    work_dir = Path(__file__).parent / "benchmark_workspace"
    work_dir.mkdir(exist_ok=True)
    os.environ["CORRAL_WORK_DIR"] = str(work_dir)
    
    logger.info(f"Work directory set to: {work_dir}")


def validate_data_files():
    """Validate that required data files exist"""
    base_dir = Path(__file__).parent
    
    required_files = [
        base_dir / "data" / "experimental_data.h5",
        base_dir / "tasks" / "kinetic_fitting_tasks.json",
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            logger.error(f"Required file missing: {file_path}")
            return False
            
    logger.info("All required data files found")
    return True


def create_initial_network_file():
    """Create an initial reaction network file for the benchmark"""
    import json
    
    network_path = Path(__file__).parent / "benchmark_workspace" / "initial_network.json"
    
    # Create initial network if it doesn't exist
    if not network_path.exists():
        initial_network = {
            "reactions": [
                {
                    "equation": "RuII + hv -> RuII*",
                    "type": "light",
                    "quantum_yield": [0.8, 1.0],
                    "description": "Photoexcitation of Ru catalyst"
                },
                {
                    "equation": "RuII* + S2O8 -> RuIII + SO4_rad + SO4",
                    "type": "dark", 
                    "k_range": [1e7, 1e9],
                    "description": "Excited Ru oxidation by persulfate"
                },
                {
                    "equation": "2 RuIII + H2O -> 2 RuII + 0.5 O2 + 2 H",
                    "type": "dark",
                    "k_range": [1e3, 1e5], 
                    "description": "Water oxidation by Ru(III)"
                }
            ],
            "metadata": {
                "created_by": "run_benchmark.py",
                "description": "Initial network for benchmark",
                "version": "1.0"
            }
        }
        
        with open(network_path, 'w') as f:
            json.dump(initial_network, f, indent=2)
            
        logger.info(f"Created initial network: {network_path}")
    
    return str(network_path)


def run_kinetic_fitting_benchmark(
    model: str = "claude-3-5-sonnet-20241022",
    task_ids: list | None = None, 
    temperature: float = 0.0,
    run_name: str = "kinetic_fitting_benchmark",
    verbose: str = "comprehensive",
    max_iterations: int = 25,
    trials_per_task: int = 1,
):
    """Run the kinetic fitting benchmark with specified model and parameters"""
    
    # Setup
    setup_environment()
    if not validate_data_files():
        logger.error("Data validation failed")
        return False
    
    network_path = create_initial_network_file()
    
    # Configure benchmark interface
    try:
        interface = BenchmarkInterface(base_url="http://localhost:8004")
    except Exception as e:
        logger.warning(f"Could not connect to benchmark interface: {e}")
        logger.info("Running without benchmark interface - will use local environment")
        interface = None
    
    # Setup W&B logging (optional)
    try:
        wandb_logger = CorralWandbLogger(
            project="kinetic_fitting_benchmark",
            group="photocatalysis_optimization", 
            name=run_name,
            tags=["kinetic_fitting", "photocatalysis", "reaction_network"]
        )
    except Exception as e:
        logger.warning(f"W&B logging not available: {e}")
        wandb_logger = None
    
    # Create ReAct agent
    agent = ReActAgent(
        model=model, 
        max_iterations=max_iterations, 
        temperature=temperature
    )
    
    logger.info(f"Created ReAct agent with model: {model}")
    logger.info(f"Max iterations: {max_iterations}, Temperature: {temperature}")
    
    # If we have benchmark interface, use it
    if interface:
        runner = MatAgentBenchmark(interface, agent, logger=wandb_logger)
        
        # Define task IDs for kinetic fitting
        if task_ids is None:
            task_ids = ["kinetic_fitting"]  # Use our task ID
        
        logger.info(f"Starting benchmark with task IDs: {task_ids}")
        
        # Run benchmark
        result = runner.bench(
            task_ids,
            trials_per_task=trials_per_task,
            k_values=[1],
            verbose=True,
            tool_verbosity=verbose,
        )
        
        # Generate report
        report_path = f"{run_name}_report.json"
        result.generate_report(report_path)
        logger.info(f"Benchmark completed. Report saved to: {report_path}")
        
    else:
        # Run directly with environment for testing
        logger.info("Running direct environment test...")
        
        from kinetic_fitting.env import create_environments
        
        # Create environment
        task_json_path = Path(__file__).parent / "tasks" / "kinetic_fitting_tasks.json"
        envs = create_environments(task_json_path)
        env = envs['kinetic_fitting']
        
        # Simple test task for the agent
        test_prompt = """
You are tasked with optimizing a kinetic reaction network for photocatalytic water oxidation.

Your goal is to:
1. Examine the experimental data to understand the system
2. Evaluate the current reaction network
3. Fit the network to experimental data  
4. Analyze phenomenological trends
5. Suggest improvements to achieve a phenomenological score >0.75

The experimental system involves:
- Catalyst: Ru(bpy)₃²⁺ (0-25 µM range)
- Oxidant: Persulfate (S₂O₈²⁻) (0-8000 µM range) 
- Product: O₂ evolution (measured experimentally)
- 61 real experiments with varying conditions

Start by examining the experimental data.
"""
        
        # Run agent on the task
        logger.info("Running agent on kinetic fitting task...")
        try:
            result = agent.run(env, test_prompt)
            logger.info(f"Agent completed task. Result: {result}")
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    logger.info("Kinetic fitting benchmark completed successfully")
    return True


def main():
    """Main entry point for benchmark"""
    load_dotenv()
    setup_litellm()
    
    # Configuration options
    models_to_test = [
        "claude-3-5-sonnet-20241022",
        # "gpt-4o",
        # "claude-3-5-haiku-20241022", 
    ]
    
    verbosity_levels = [
        "comprehensive",
        # "workflow", 
        # "brief",
    ]
    
    # Run benchmarks
    for model in models_to_test:
        for verbose in verbosity_levels:
            run_name = f"kinetic_fitting_{model.replace('-', '_')}_{verbose}_verbosity"
            
            logger.info(f"🧪 Running benchmark: {run_name}")
            logger.info(f"Model: {model}")
            logger.info(f"Verbosity: {verbose}")
            logger.info("=" * 60)
            
            try:
                success = run_kinetic_fitting_benchmark(
                    model=model,
                    run_name=run_name,
                    verbose=verbose,
                    max_iterations=25,
                    trials_per_task=1,  # Reduce for testing
                    temperature=0.0
                )
                
                if success:
                    logger.info(f"✅ Benchmark {run_name} completed successfully")
                else:
                    logger.error(f"❌ Benchmark {run_name} failed")
                    
            except Exception as e:
                logger.error(f"💥 Benchmark {run_name} crashed: {e}")
                import traceback
                traceback.print_exc()
            
            logger.info("=" * 60 + "\\n")


if __name__ == "__main__":
    main()