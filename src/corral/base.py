from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Basic Tool Definition
class Tool(BaseModel):
    name: str
    description: str

    def execute(self, **kwargs) -> str:
        """Execute the tool functionality"""
        raise NotImplementedError


# Message format for tool requests
class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


# Environment State
class EnvironmentState(BaseModel):
    tool_history: list[ToolRequest] = []
    last_response: Optional[str] = None


# Base Environment
class BaseEnvironment(ABC):
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self.state = EnvironmentState()

    def add_tool(self, tool: Tool):
        """Register a new tool in the environment"""
        self.tools[tool.name] = tool

    def get_available_tools(self) -> list[Tool]:
        """Get list of available tools"""
        return list(self.tools.values())

    def execute_tool(self, request: ToolRequest) -> str:
        """Execute a tool and update state"""
        if request.tool_name not in self.tools:
            raise ValueError(f"Tool {request.tool_name} not found")

        tool = self.tools[request.tool_name]
        response = tool.execute(**request.arguments)

        # Update state
        self.state.tool_history.append(request)
        self.state.last_response = response

        return response


# Example Calculator Tool
class CalculatorTool(Tool):
    name :str = "calculator"
    description : str = "Performs basic arithmetic operations"

    async def execute(self, operation: str, x: float, y: float) -> str:
        operations = {
            "add": lambda: x + y,
            "subtract": lambda: x - y,
            "multiply": lambda: x * y,
            "divide": lambda: x / y if y != 0 else "Error: Division by zero",
        }

        if operation not in operations:
            return f"Invalid operation: {operation}"

        result = operations[operation]()
        return str(result)


# FastAPI Environment Server
def create_environment_server(env: BaseEnvironment) -> FastAPI:
    app = FastAPI()

    @app.get("/tools")
    def list_tools():
        return {
            "tools": [
                {"name": tool.name, "description": tool.description}
                for tool in env.get_available_tools()
            ]
        }

    @app.post("/execute")
    def execute_tool(request: ToolRequest):
        try:
            response = env.execute_tool(request)
            return {"response": response, "state": env.state}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/state")
    def get_state():
        return env.state

    return app
