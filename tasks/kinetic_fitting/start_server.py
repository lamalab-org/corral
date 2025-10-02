#!/usr/bin/env python3
"""
Start the Corral server for kinetic fitting benchmark.

This script starts the FastAPI server that serves the kinetic fitting environment
for use with ReActAgent and CorralRunner.
"""

import os
from pathlib import Path

# Configure matplotlib to use non-GUI backend before any imports that might use it
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''  # Disable display
import matplotlib
matplotlib.use('Agg')

from loguru import logger

from corral.backend.server import run_server
from kinetic_fitting.env import create_environments


def main():
    """Start the kinetic fitting server"""
    
    # Setup environment
    work_dir = Path(__file__).parent / "benchmark_workspace"
    work_dir.mkdir(exist_ok=True)
    os.environ["CORRAL_WORK_DIR"] = str(work_dir)
    
    # Create environments
    task_json_path = Path(__file__).parent / "tasks" / "kinetic_fitting_tasks.json"
    logger.info(f"Loading tasks from: {task_json_path}")
    
    envs = create_environments(task_json_path)
    logger.info(f"Created environments: {list(envs.keys())}")
    
    # Start server
    port = 8004
    logger.info(f"Starting Corral server on port {port}")
    logger.info("Available endpoints:")
    logger.info("  GET /tasks - List available tasks")
    logger.info("  GET /tasks/{task_id}/guide - Get task guide") 
    logger.info("  POST /tasks/{task_id}/tools/execute - Execute tools")
    logger.info("  POST /tasks/{task_id}/submit - Submit answers")
    logger.info("")
    logger.info("Use Ctrl+C to stop the server")
    
    try:
        run_server(envs, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    main()