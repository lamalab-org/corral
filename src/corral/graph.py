import json
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger


class NodeType(Enum):
    """Types of nodes in the interaction graph"""

    LLM_PROMPT = "llm_prompt"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    FINAL_SUBMISSION = "final_submission"
    PLANNER = "planner"
    EXECUTOR = "executor"


@dataclass
class Node:
    """Represents a node in the interaction graph"""

    id: str
    type: NodeType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Edge:
    """Represents an edge in the interaction graph"""

    source: str  # Node ID
    target: str  # Node ID
    type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert edge to dictionary for serialization"""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "metadata": self.metadata,
        }


class GraphTracker:
    """Tracks agent interactions as a graph"""

    def __init__(self, task_id: str, agent_type: str, trial_id: str | None = None):
        """Initialize the graph tracker

        Args:
            task_id: ID of the task being solved
            agent_type: Type of agent being used (e.g., "ReActAgent")
            trial_id: Optional trial ID, will be generated if not provided
        """
        self.task_id = task_id
        self.agent_type = agent_type
        self.trial_id = trial_id or str(uuid.uuid4())

        # Core graph components
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

        # Track the last node for easy linking
        self.last_node_id: str | None = None

        # Track specific important nodes
        self.final_submission_node: str | None = None

        # For NetworkX graph generation
        self._graph = None

    def add_node(
        self, node_type: NodeType, content: Any, metadata: dict[str, Any] | None = None
    ) -> str:
        """Add a node to the graph

        Args:
            node_type: Type of the node
            content: Content of the node (e.g., message text)
            metadata: Additional node metadata

        Returns:
            The ID of the newly created node
        """
        node_id = str(uuid.uuid4())
        self.nodes[node_id] = Node(
            id=node_id,
            type=node_type,
            content=content,
            timestamp=datetime.now(tz=timezone.utc),
            metadata=metadata or {},
        )

        # Link to previous node if it exists
        if self.last_node_id:
            self.add_edge(
                source=self.last_node_id, target=node_id, edge_type="sequence"
            )

        # Update last node reference
        self.last_node_id = node_id

        # Mark if this is a final submission
        if node_type == NodeType.FINAL_SUBMISSION:
            self.final_submission_node = node_id

        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an edge between two nodes

        Args:
            source: ID of the source node
            target: ID of the target node
            edge_type: Type of the edge (e.g., "calls", "responds")
            metadata: Additional edge metadata
        """
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(
                f"Both source ({source}) and target ({target}) must exist in the graph"
            )

        self.edges.append(
            Edge(source=source, target=target, type=edge_type, metadata=metadata or {})
        )

    def track_llm_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Track an LLM prompt

        Args:
            messages: list of messages sent to the LLM

        Returns:
            ID of the created node
        """
        return self.add_node(
            node_type=NodeType.LLM_PROMPT,
            content=messages,
            metadata={"message_count": len(messages)},
        )

    def track_llm_response(
        self, content: str, source_node_id: str | None = None
    ) -> str:
        """Track an LLM response

        Args:
            content: Content of the LLM response
            source_node_id: Optional ID of the prompt node that generated this response

        Returns:
            ID of the created node
        """
        node_id = self.add_node(
            node_type=NodeType.LLM_RESPONSE,
            content=content,
            metadata={"length": len(content)},
        )

        # If we have a specific source node, create a direct edge
        if source_node_id and source_node_id in self.nodes:
            self.add_edge(
                source=source_node_id, target=node_id, edge_type="responds_to"
            )

        return node_id

    def track_tool_call(self, tool_call) -> str:
        """Track a tool call

        Args:
            tool_call: The tool call to track

        Returns:
            ID of the created node
        """
        return self.add_node(
            node_type=NodeType.TOOL_CALL,
            content={
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
            },
            metadata={
                "status": tool_call.status.value,
                "error_message": tool_call.error_message,
            },
        )

    def track_tool_response(
        self, result: str, tool_call_node_id: str | None = None
    ) -> str:
        """Track a tool response

        Args:
            result: Result of the tool call
            tool_call_node_id: Optional ID of the tool call node

        Returns:
            ID of the created node
        """
        node_id = self.add_node(node_type=NodeType.TOOL_RESPONSE, content=result)

        # Link to the tool call if provided
        if tool_call_node_id and tool_call_node_id in self.nodes:
            self.add_edge(
                source=tool_call_node_id, target=node_id, edge_type="results_in"
            )

        return node_id

    def track_final_submission(self, answer: str, score: float | None = None) -> str:
        """Track the final submission

        Args:
            answer: The final answer submitted
            score: Optional score of the submission

        Returns:
            ID of the created node
        """
        return self.add_node(
            node_type=NodeType.FINAL_SUBMISSION,
            content=answer,
            metadata={"score": score},
        )

    def track_planner_executor_interaction(
        self, planner_content: str, executor_messages: list[dict[str, Any]]
    ) -> tuple[str, str]:
        """Track an interaction between a planner and executor

        Args:
            planner_content: Content from the high-level planner
            executor_messages: Messages from the executor

        Returns:
            tuple of (planner_node_id, executor_node_id)
        """
        planner_node_id = self.add_node(
            node_type=NodeType.PLANNER, content=planner_content
        )

        executor_node_id = self.add_node(
            node_type=NodeType.EXECUTOR, content=executor_messages
        )

        self.add_edge(
            source=planner_node_id, target=executor_node_id, edge_type="executes"
        )

        return planner_node_id, executor_node_id

    def to_dict(self) -> dict[str, Any]:
        """Convert the graph to a dictionary for serialization

        Returns:
            dictionary representation of the graph
        """
        return {
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "trial_id": self.trial_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "final_submission_node": self.final_submission_node,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert the graph to a JSON string

        Args:
            indent: Indentation level for JSON formatting

        Returns:
            JSON string representation of the graph
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_to_file(self, file_path: str | Path) -> None:
        """Save the graph to a file

        Args:
            file_path: Path to save the graph to
        """
        with Path(file_path).open("w") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> "GraphTracker":
        """Load a graph from a file

        Args:
            file_path: Path to load the graph from

        Returns:
            Loaded GraphTracker instance
        """
        with Path(file_path).open() as f:
            data = json.loads(f.read())

        graph = cls(
            task_id=data["task_id"],
            agent_type=data["agent_type"],
            trial_id=data["trial_id"],
        )

        # Recreate nodes
        for node_id, node_data in data["nodes"].items():
            node = Node(
                id=node_id,
                type=NodeType(node_data["type"]),
                content=node_data["content"],
                timestamp=datetime.fromisoformat(node_data["timestamp"]),
                metadata=node_data["metadata"],
            )
            graph.nodes[node_id] = node

        # Recreate edges
        for edge_data in data["edges"]:
            edge = Edge(
                source=edge_data["source"],
                target=edge_data["target"],
                type=edge_data["type"],
                metadata=edge_data["metadata"],
            )
            graph.edges.append(edge)

        graph.final_submission_node = data.get("final_submission_node")

        # Set last node to the final submission if it exists
        if graph.final_submission_node:
            graph.last_node_id = graph.final_submission_node

        return graph

    def to_networkx(self) -> nx.DiGraph:
        """Convert the graph to a NetworkX DiGraph

        Returns:
            NetworkX DiGraph representation of the graph
        """
        G = nx.DiGraph()

        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            # Fix: Rename 'type' to 'node_type' to avoid conflict with NetworkX parameter
            G.add_node(
                node_id,
                node_type=node.type.value,  # Changed from 'type' to 'node_type'
                content=str(node.content)[:100] + "..."
                if len(str(node.content)) > 100
                else str(node.content),
                timestamp=node.timestamp.isoformat(),
                **{
                    k: v for k, v in node.metadata.items() if k != "type"
                },  # Exclude 'type' from metadata
            )

        # Add edges with attributes
        for edge in self.edges:
            G.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.type,  # Changed from 'type' to 'edge_type'
                **{
                    k: v for k, v in edge.metadata.items() if k != "type"
                },  # Exclude 'type' from metadata
            )

        self._graph = G
        return G

    def visualize(self, figsize=(15, 10), save_path: str | None = None) -> None:
        """Visualize the graph in a thread-safe manner

        Args:
            figsize: Size of the figure
            save_path: Optional path to save the visualization
        """
        # Only proceed if we have a save path in non-interactive mode
        if not save_path:
            logger.info(
                "Warning: Skipping interactive visualization which requires the main thread"
            )
            return

        if not self._graph:
            self.to_networkx()

        G = self._graph

        # Use non-interactive Agg backend for thread safety
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        # Create a color map for node types
        node_types = [data["node_type"] for _, data in G.nodes(data=True)]
        unique_types = list(set(node_types))
        colors = list(mcolors.TABLEAU_COLORS)[: len(unique_types)]
        color_map = dict(zip(unique_types, colors, strict=False))

        # Get node colors
        node_colors = [color_map[data["node_type"]] for _, data in G.nodes(data=True)]

        # Position nodes
        pos = nx.spring_layout(G, seed=42)

        # Create figure without display
        plt.figure(figsize=figsize)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)

        # Draw labels
        labels = {
            node: f"{data['node_type']}\n{data['content']}"
            for node, data in G.nodes(data=True)
        }
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=node_type,
                markerfacecolor=color,
                markersize=10,
            )
            for node_type, color in color_map.items()
        ]
        plt.legend(handles=legend_elements, loc="upper right")

        plt.title(f"Interaction Graph for Task: {self.task_id}, Trial: {self.trial_id}")
        plt.axis("off")

        # Save figure
        plt.savefig(save_path, bbox_inches="tight")

        # Close figure to prevent memory leaks
        plt.close()

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the graph

        Returns:
            dictionary of statistics
        """
        if not self._graph:
            self.to_networkx()

        G = self._graph

        # Count nodes by type - using node_type instead of type
        node_types = {}
        for _, data in G.nodes(data=True):
            node_type = data["node_type"]  # Changed from 'type' to 'node_type'
            node_types[node_type] = node_types.get(node_type, 0) + 1

        # Count edges by type - using edge_type instead of type
        edge_types = {}
        for _, _, data in G.edges(data=True):
            edge_type = data.get(
                "edge_type", "unknown"
            )  # Changed from 'type' to 'edge_type'
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        # Calculate path length to final submission
        path_length = None
        if self.final_submission_node:
            # Find all simple paths from each node to the final submission
            path_lengths = []
            for node in G.nodes():
                if node != self.final_submission_node:
                    try:
                        paths = list(
                            nx.all_simple_paths(G, node, self.final_submission_node)
                        )
                        if paths:
                            path_lengths.append(min(len(path) for path in paths))
                    except nx.NetworkXNoPath:
                        pass

            if path_lengths:
                path_length = sum(path_lengths) / len(path_lengths)

        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "average_path_length_to_submission": path_length,
            "final_score": self.nodes[self.final_submission_node].metadata.get("score")
            if self.final_submission_node
            else None,
        }


class GraphTrackerFactory:
    """Factory for creating and managing graph trackers"""

    def __init__(self, output_dir: str = "./graph_output"):
        """Initialize the graph tracker factory

        Args:
            output_dir: Directory to save graph data
        """
        self.output_dir = output_dir
        self.active_trackers: dict[str, GraphTracker] = {}

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def create_tracker(
        self, task_id: str, agent_type: str, trial_id: str | None = None
    ) -> GraphTracker:
        """Create a new graph tracker

        Args:
            task_id: ID of the task
            agent_type: Type of agent
            trial_id: Optional trial ID

        Returns:
            New GraphTracker instance
        """
        tracker = GraphTracker(
            task_id=task_id, agent_type=agent_type, trial_id=trial_id
        )

        # Generate a unique key for this tracker
        key = f"{task_id}_{trial_id or 'default'}"
        self.active_trackers[key] = tracker

        return tracker

    def get_tracker(
        self, task_id: str, trial_id: str | None = None
    ) -> GraphTracker | None:
        """Get an existing graph tracker

        Args:
            task_id: ID of the task
            trial_id: Optional trial ID

        Returns:
            GraphTracker instance or None if not found
        """
        key = f"{task_id}_{trial_id or 'default'}"
        return self.active_trackers.get(key)

    def save_trackers(self) -> None:
        """Save all active trackers

        Args:
            visualize: Whether to generate visualization images
        """
        for key, tracker in self.active_trackers.items():
            try:
                # Save JSON data
                json_path = Path(self.output_dir) / f"{key}.json"
                tracker.save_to_file(json_path)

                # Generate visualization if requested
                viz_path = Path(self.output_dir) / f"{key}.png"
                try:
                    tracker.visualize(save_path=viz_path)
                except Exception as e:
                    logger.info(
                        f"Warning: Failed to generate visualization for {key}: {e}"
                    )
            except Exception as e:
                logger.info(f"Error saving tracker {key}: {e}")
                logger.info(traceback.format_exc())

    def save_tracker(
        self, task_id: str, trial_id: str | None = None, visualize: bool = False
    ) -> None:
        """Save a specific tracker with error handling

        Args:
            task_id: ID of the task
            trial_id: Optional trial ID
            visualize: Whether to generate visualization image
        """
        key = f"{task_id}_{trial_id or 'default'}"
        tracker = self.active_trackers.get(key)

        if tracker:
            try:
                # Save JSON data
                json_path = Path(self.output_dir) / f"{key}.json"
                tracker.save_to_file(json_path)

                # Generate visualization if requested
                if visualize:
                    viz_path = Path(self.output_dir / f"{key}.png")
                    try:
                        tracker.visualize(save_path=viz_path)
                    except Exception as e:
                        logger.info(
                            f"Warning: Failed to generate visualization for {key}: {e}"
                        )
                        logger.info(traceback.format_exc())
            except Exception as e:
                logger.info(f"Error saving tracker {key}: {e}")
                logger.exception("Error saving tracker")
