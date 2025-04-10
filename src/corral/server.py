import os
from collections.abc import Mapping
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from loguru import logger

from corral.base import Environment, ToolRequest
from corral.graph import GraphTrackerFactory


def create_benchmark_server(
    environments: dict[str, Environment], graph_output_dir: str = "./graph_output"
) -> FastAPI:
    """CORRAL Benchmark Server"""
    app = FastAPI()

    graph_factory = GraphTrackerFactory(output_dir=graph_output_dir)

    # Convert standard environments to graph-tracked environments if needed
    for task_id, env in environments.items():
        if env.graph_factory is None:
            env.graph_factory = graph_factory
            # Create new tracker for the current state if needed
            if env.graph_tracker is None and hasattr(env, "state"):
                env.graph_tracker = graph_factory.create_tracker(
                    task_id=task_id,
                    agent_type="Environment",
                    trial_id=env.state.trial_id
                    if hasattr(env, "state") and env.state
                    else "0",
                )
    Path(graph_output_dir).mkdir(parents=True, exist_ok=True)

    app.mount("/graphs", StaticFiles(directory=graph_output_dir), name="graphs")

    @app.get("/tasks")
    def get_available_tasks():
        """Get list of available task IDs"""
        return list(environments.keys())

    @app.get("/tasks/{task_id}/prompt")
    def get_task_prompt(task_id: str):
        """Get the task prompt for the agent"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"prompt": environments[task_id].get_task_prompt()}

    @app.get("/tasks/{task_id}/guide")
    def get_environment_guide(task_id: str):
        """Get the task prompt for the agent"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"prompt": environments[task_id].get_environment_guide()}

    @app.get("/tasks/{task_id}/tools/guide")
    def get_tools_guide(task_id: str):
        """Get the tools guide for the agent"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"prompt": environments[task_id].get_tools_guide()}

    @app.get("/tasks/{task_id}/tools")
    def get_available_tools(task_id: str):
        """Get available tools for this task"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"tools": environments[task_id].get_available_tools()}

    @app.post("/tasks/{task_id}/tools/execute")
    def execute_tool(task_id: str, request: ToolRequest):
        """Execute a tool in the environment"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        try:
            result = environments[task_id].call_tool(
                request.tool_name, request.arguments
            )
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/tasks/{task_id}/state")
    def get_state(task_id: str):
        """Get the current state of the task"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        return environments[task_id].state

    @app.post("/tasks/{task_id}/submit")
    def submit_answer(task_id: str, answer: dict):
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        score = env.submit_answer(answer["answer"])

        state_dict = env.state.__dict__  # Get state as dict
        tool_statistics = env.state.get_tool_statistics()

        # Add detailed tool calls to the statistics
        tool_statistics["tool_calls"] = [
            {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "result": call.result,
                "status": call.status.value,  # Convert enum to string
                "error_message": call.error_message,
                "timestamp": call.timestamp.isoformat() if call.timestamp else None,
            }
            for call in env.state.tool_calls
        ]
        state_dict["tool_statistics"] = tool_statistics
        finished_trail = env.reset_state()  # Reset the state for the next trail

        return {"score": score, "state": state_dict, "trial_id": finished_trail}

    @app.get("/tasks/{task_id}/status")
    def get_task_status(task_id: str):
        """Get task completion status"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        return {
            "is_completed": env.state.is_completed,
            "score": env.state.score,
            "submitted_answer": env.state.submitted_answer,
            "tool_statistics": env.state.get_tool_statistics(),
        }

    @app.get("/tasks/{task_id}/trials")
    def get_all_trials(task_id: str):
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        env = environments[task_id]
        return {"trials": env.trial_states}

    @app.get("/tasks/{task_id}/trials/{trial_id}")
    def get_trial_state(task_id: str, trial_id: str):
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        env = environments[task_id]
        trial_state = env.trial_states.get(trial_id)
        if trial_state is None:
            raise HTTPException(status_code=404, detail="Trial not found")
        return {"trial_state": trial_state}

    @app.get("/graphs")
    def list_available_graphs():
        """List all available graph visualizations"""
        graph_files = []

        # List JSON and PNG files in the graph output directory
        try:
            files = os.listdir(graph_output_dir)
            for file in files:
                if file.endswith((".json", ".png")):
                    file_path = Path(graph_output_dir) / file
                    file_info = {
                        "name": file,
                        "path": f"/graphs/{file}",
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                    }
                    graph_files.append(file_info)
        except Exception as e:
            return {"error": str(e), "graphs": []}

        return {
            "graphs": sorted(graph_files, key=lambda x: x["modified"], reverse=True)
        }

    @app.get("/graphs/{task_id}/{trial_id}/statistics")
    def get_graph_statistics(task_id: str, trial_id: str):
        """Get statistics for a specific graph"""
        graph_file = f"{task_id}_{trial_id}.json"
        graph_path = Path(graph_output_dir) / graph_file

        if not graph_path.exists():
            raise HTTPException(status_code=404, detail="Graph not found")

        try:
            # Load graph and get statistics
            graph = graph_factory.get_tracker(task_id, trial_id)
            if not graph:
                from corral.graph import GraphTracker  # Ensure GraphTracker is imported

                graph = GraphTracker.load_from_file(graph_path)

            stats = graph.get_statistics()
            return {"statistics": stats}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error loading graph: {e!s}"
            ) from e

    return app


def run_server(
    environments: Mapping[str, Environment], host: str = "0.0.0.0", port: int = 8000
):
    """Run the benchmark server with the provided environments

    Args:
        environments: dictionary of environments
        host: Server host
        port: Server port
    """
    app = create_benchmark_server(dict(environments))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
