from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException

from corral.base import Environment, ToolRequest


def create_benchmark_server(environments: Dict[str, Environment]) -> FastAPI:
    app = FastAPI()

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

    #add endpoint for scoring the task

    return app
