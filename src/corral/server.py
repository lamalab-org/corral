from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException

from corral.base import Environment, ToolRequest


def create_benchmark_server(environments: Dict[str, Environment]) -> FastAPI:
    app = FastAPI()

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
            raise HTTPException(status_code=400, detail=str(e))

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
        state_dict["tool_statistics"] = (
            env.state.get_tool_statistics()
        )  # Add tool statistics

        return {"score": score, "state": state_dict}

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

    # add endpoint for scoring the task

    return app
