from collections.abc import Mapping

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from loguru import logger

from corral.backend.env import Environment
from corral.backend.schema import ToolRequest
from corral.router.verbosity import (
    ToolVerbosity,
    VerbosityConfig,
    get_tools_guide_with_verbosity,
)


def create_benchmark_server(environments: dict[str, Environment]) -> FastAPI:
    app = FastAPI()

    @app.get("/tasks")
    def get_available_tasks():
        """Get list of available task IDs"""
        return list(environments.keys())

    @app.get("/dependency_chain")
    def get_dependency_chain_setting():
        has_chained_tasks = any(
            hasattr(env, "task_group")
            and env.task_group
            and env.task_group.chained_tasks
            for env in environments.values()
        )
        return {"dependency_chain": has_chained_tasks}

    @app.get("/tasks/{task_id}/prompt")
    def get_task_prompt(task_id: str):
        """Get the task prompt for the agent"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"prompt": environments[task_id].get_task_prompt()}

    @app.get("/tasks/{task_id}/guide")
    def get_environment_guide(
        task_id: str,
        verbosity: ToolVerbosity | None = None,
    ):
        if verbosity is None:
            verbosity = Query(
                ToolVerbosity.FULL, description="Tool description verbosity level"
            )
        """Get the complete environment guide with specified tool verbosity"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        task_prompt = env.get_task_prompt()
        tools_guide = get_tools_guide_with_verbosity(env, verbosity)

        return {"prompt": f"Task: {task_prompt}\n\n{tools_guide}"}

    @app.get("/tasks/{task_id}/tools/guide")
    def get_tools_guide(
        task_id: str,
        verbosity: ToolVerbosity | None = None,
    ):
        if verbosity is None:
            verbosity = Query(
                ToolVerbosity.FULL, description="Tool description verbosity level"
            )
        """Get the tools guide with specified verbosity level"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        tools_guide = get_tools_guide_with_verbosity(env, verbosity)

        return {"prompt": tools_guide}

    @app.get("/tasks/{task_id}/tools")
    def get_available_tools(
        task_id: str,
        verbosity: ToolVerbosity | None = None,
    ):
        """
        Get available tools for a task, returning a structured JSON object.

        This endpoint provides tool definitions in a format compatible with modern
        LLM function-calling APIs. The structure of the returned argument
        dictionaries will vary based on the requested verbosity level.
        """
        if verbosity is None:
            verbosity = Query(
                ToolVerbosity.FULL, description="Tool description verbosity level"
            )
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        tools_info = []

        for tool in env.tools.values():
            filtered_description = VerbosityConfig.filter_tool_description(
                tool.description, verbosity
            )

            structured_args = []
            for arg in tool.arguments:
                # Conditionally build the argument dictionary based on verbosity.
                if verbosity == ToolVerbosity.MINIMAL:
                    # For MINIMAL, provide only the essential keys.
                    structured_args.append(
                        {
                            "name": arg.name,
                            "type": arg.type,
                            "required": arg.required,
                        }
                    )
                else:
                    # For FULL (or other levels), provide all details.
                    filtered_arg_desc = VerbosityConfig.filter_argument_description(
                        arg.description, verbosity
                    )
                    structured_args.append(
                        {
                            "name": arg.name,
                            "type": arg.type,
                            "description": filtered_arg_desc,
                            "required": arg.required,
                            "default": arg.default,
                            "choices": arg.choices,
                        }
                    )

            tools_info.append(
                {
                    "name": tool.name,
                    "description": filtered_description,
                    "arguments": structured_args,
                }
            )

        return {"tools": tools_info}

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

        # 1. Submit answer and score
        score = env.submit_answer(answer["answer"])

        # 2. Get completed trial data (before any reset)
        completed_trial = env.get_completed_trial_data()

        # 3. Reset for next trial
        finished_trial_id = env.reset_state()

        return {
            "score": score,
            "state": completed_trial["state"],
            "trial_id": finished_trial_id,
        }

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

    # add endpoint for scoring the task

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
