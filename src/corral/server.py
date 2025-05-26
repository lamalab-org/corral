import os
import traceback
from collections.abc import Mapping
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from loguru import logger

from corral.agents.utils import AgentTrackingRequest
from corral.base import Environment, ToolRequest
from corral.graph import GraphTracker, GraphTrackerFactory, NodeType


def create_benchmark_server(
    environments: dict[str, Environment], graph_output_dir: str = "./graph_output"
) -> FastAPI:
    """CORRAL Benchmark Server"""
    app = FastAPI()

    graph_factory = GraphTrackerFactory(output_dir=graph_output_dir)

    # Dictionary to store processed messages for each task
    processed_messages = {}

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

    @app.get("/tasks/{task_id}/guide")
    def get_environment_guide(task_id: str):
        """Get the task prompt for the agent"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"prompt": environments[task_id].get_environment_guide()}

    @app.get("/tasks/{task_id}/prompt")
    def get_task_prompt(task_id: str):
        """Get the task prompt for the agent"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"prompt": environments[task_id].get_task_prompt()}

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

    @app.post("/tasks/{task_id}/submit")
    def submit_answer(task_id: str, answer: dict):
        """Submit an answer for evaluation"""
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

    @app.post("/tasks/{task_id}/track")
    def track_agent_activity(task_id: str, tracking_data: AgentTrackingRequest):
        """Track agent activity in the environment's graph"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]

        # Check if environment has tracking method
        if not hasattr(env, "track_agent_activity"):
            # Add compatibility function if not present
            def track_agent_activity(node_type_str, content, metadata=None):
                """Compatibility method for environments without tracking"""
                if not hasattr(env, "graph_tracker") or not env.graph_tracker:
                    return None

                metadata = metadata or {}
                metadata["component"] = "agent"

                try:
                    node_type = NodeType(node_type_str)
                    return env.graph_tracker.add_node(
                        node_type=node_type, content=content, metadata=metadata
                    )
                except Exception as e:
                    logger.error(f"Error tracking agent activity: {e}")
                    return None

            env.track_agent_activity = track_agent_activity

        # Track the activity
        node_id = env.track_agent_activity(
            tracking_data.node_type, tracking_data.content, tracking_data.metadata
        )

        if node_id:
            return {"node_id": node_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to track activity")

    @app.post("/tasks/{task_id}/messages")
    def build_graph_from_messages(task_id: str, messages: list[dict]):
        """Process agent messages and build interaction graph incrementally

        This endpoint receives a list of message objects from the agent interface
        and creates nodes and edges in the environment's graph tracker.
        Only new messages that haven't been processed before will be added to the graph.

        Args:
            task_id (str): The ID of the task to process messages for.
            messages (list[dict]): List of message objects with keys like 'role', 'content',
                                   'timestamp', and optionally 'tool_call_id'.
        """
        try:
            if task_id not in environments:
                logger.error(f"Task {task_id} not found in environments")
                raise HTTPException(status_code=404, detail="Task not found")

            env = environments[task_id]

            if not hasattr(env, "graph_tracker") or not env.graph_tracker:
                logger.error(
                    f"Environment for task {task_id} doesn't have a graph tracker"
                )
                if hasattr(env, "graph_factory") and env.graph_factory:
                    logger.info(f"Creating new graph tracker for task {task_id}")
                    trial_id = "0"
                    if hasattr(env, "state") and hasattr(env.state, "trial_id"):
                        trial_id = env.state.trial_id
                    env.graph_tracker = env.graph_factory.create_tracker(
                        task_id=task_id, agent_type="Environment", trial_id=trial_id
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Environment doesn't support graph tracking",
                    )

            if task_id not in processed_messages:
                processed_messages[task_id] = {}

            task_processed_messages = processed_messages[task_id]
            logger.debug(f"Received {len(messages)} messages for task {task_id}")

            new_messages = []
            for i, msg in enumerate(messages):
                try:
                    if not isinstance(msg, dict):
                        logger.warning(
                            f"Skipping non-dict message at index {i}: {type(msg)}"
                        )
                        continue

                    msg_content = msg.get("content", "")
                    if not isinstance(msg_content, str) and msg_content is not None:
                        try:
                            msg_content = str(msg_content)
                        except Exception as e:
                            logger.warning(
                                f"Failed to convert message content to string at index {i}: {e}"
                            )
                            msg_content = ""

                    msg_role = msg.get("role", "")
                    msg_timestamp = msg.get("timestamp", "")
                    tool_call_id = msg.get("tool_call_id", "")
                    msg_key = f"{msg_role}:{tool_call_id}:{msg_timestamp}:{hash(str(msg_content))}"

                    if msg_key not in task_processed_messages:
                        new_messages.append(msg)
                        task_processed_messages[msg_key] = True
                except Exception as msg_error:
                    logger.error(f"Error processing message at index {i}: {msg_error}")

            if not new_messages:
                return {"message": "No new messages to process", "node_ids": []}

            prev_node_id = None
            node_ids = []
            if (
                hasattr(env.graph_tracker, "last_node_id")
                and env.graph_tracker.last_node_id
            ):
                prev_node_id = env.graph_tracker.last_node_id

            for i, msg in enumerate(new_messages):
                try:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    metadata = {"component": "agent"}
                    if "name" in msg:
                        metadata["name"] = msg["name"]
                    if "timestamp" in msg:
                        metadata["timestamp"] = msg["timestamp"]

                    if role in ["system", "user", "agent", "environment"]:
                        node_type = NodeType.LLM_PROMPT
                    elif role == "assistant":
                        node_type = NodeType.LLM_RESPONSE
                    elif role == "tool":
                        node_type = NodeType.TOOL_RESPONSE
                        if "tool_call_id" in msg:
                            metadata["tool_call_id"] = msg["tool_call_id"]
                    else:
                        logger.warning(
                            f"Unrecognized message role: {role}, treating as LLM_PROMPT"
                        )
                        node_type = NodeType.LLM_PROMPT

                    try:
                        node_id = env.graph_tracker.add_node(
                            node_type=node_type, content=content, metadata=metadata
                        )
                    except Exception as node_error:
                        logger.error(
                            f"Error adding node for message at index {i}: {node_error}"
                        )
                        continue

                    if prev_node_id is not None:
                        try:
                            env.graph_tracker.add_edge(
                                source=prev_node_id,
                                target=node_id,
                                edge_type="sequence",
                                metadata={"component": "agent_conversation"},
                            )
                        except Exception as edge_error:
                            logger.error(
                                f"Error adding edge between {prev_node_id} and {node_id}: {edge_error}"
                            )

                    prev_node_id = node_id
                    node_ids.append(node_id)
                except Exception as msg_error:
                    logger.error(f"Error processing message {i}: {msg_error}")
                    logger.error(f"Message content: {msg}")

            return {
                "message": f"Created {len(node_ids)} nodes in graph",
                "node_ids": node_ids,
                "new_message_count": len(new_messages),
                "total_message_count": len(messages),
            }
        except Exception as e:
            logger.error(f"Error processing messages for task {task_id}: {e}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            return {
                "message": "Error processing messages",
                "error": str(e),
                "node_ids": [],
                "new_message_count": 0,
                "total_message_count": len(messages)
                if isinstance(messages, list)
                else 0,
            }

    @app.post("/tasks/{task_id}/reset_tracking")
    def reset_message_tracking(task_id: str):
        """Reset the message tracking for a specific task

        This clears the record of processed messages, which will allow
        reprocessing messages that were previously sent.

        Args:
            task_id (str): The ID of the task to reset tracking for.
        """
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        if task_id in processed_messages:
            processed_messages[task_id] = {}

        return {"message": f"Message tracking reset for task {task_id}"}

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

    @app.post("/graphs/visualize")
    def visualize_all_graphs(directory_path: str | Path | None = None):
        """Generate and save visualizations for all graphs in the specified directory

        Args:
            directory_path: Optional path to the directory containing graph JSON files.
                        If not provided, uses the default graph_output_dir.

        Returns:
            List of paths to generated visualization files
        """
        # Use default directory if none provided
        directory = directory_path or graph_output_dir
        directory_path = Path(directory)

        if not directory_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Directory not found: {directory}"
            )

        # Find all JSON graph files
        graph_files = list(directory_path.glob("*.json"))

        if not graph_files:
            return {"message": "No graph files found", "visualizations": []}

        # Generate visualizations for each file
        visualizations = []
        for graph_file in graph_files:
            try:
                # Load the graph
                graph = GraphTracker.load_from_file(graph_file)

                # Create visualization filename
                viz_path = graph_file.with_suffix(".png")

                # Generate visualization
                graph.visualize(save_path=viz_path)

                visualizations.append(
                    {
                        "original_file": str(graph_file),
                        "visualization": str(viz_path),
                        "task_id": graph.task_id,
                        "trial_id": graph.trial_id,
                    }
                )

                logger.info(f"Generated visualization: {viz_path}")

            except Exception as e:
                logger.error(f"Error visualizing graph {graph_file}: {e}")

        return {
            "message": f"Generated {len(visualizations)} visualizations",
            "visualizations": visualizations,
        }

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
