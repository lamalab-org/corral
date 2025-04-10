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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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

    def __init__(self, task_id: str, trial_id: str | None = None):
        """Initialize the graph tracker

        Args:
            task_id: ID of the task being solved
            trial_id: Optional trial ID, will be generated if not provided
        """
        self.task_id = task_id
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
            content: Content of the node
            metadata: Additional metadata including component information

        Returns:
            ID of the created node
        """
        node_id = str(uuid.uuid4())

        # Initialize metadata if needed
        metadata = metadata or {}

        # Create the node
        self.nodes[node_id] = Node(
            id=node_id,
            type=node_type,
            content=content,
            timestamp=datetime.now(tz=timezone.utc),
            metadata=metadata,
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

    def track_llm_prompt(
        self, messages: list[dict[str, Any]], metadata: dict[str, Any] | None = None
    ) -> str:
        """Track an LLM prompt

        Args:
            messages: list of messages sent to the LLM
            metadata: Additional metadata including component information

        Returns:
            ID of the created node
        """
        # Initialize metadata
        meta = metadata or {}
        # Add message count if not already present
        if "message_count" not in meta:
            meta["message_count"] = len(messages)

        return self.add_node(
            node_type=NodeType.LLM_PROMPT,
            content=messages,
            metadata=meta,
        )

    def track_llm_response(
        self,
        content: str,
        source_node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Track an LLM response

        Args:
            content: Content of the LLM response
            source_node_id: Optional ID of the prompt node that generated this response
            metadata: Additional metadata including component information

        Returns:
            ID of the created node
        """
        # Initialize metadata
        meta = metadata or {}
        # Add content length if not already present
        if "length" not in meta:
            meta["length"] = len(content)

        node_id = self.add_node(
            node_type=NodeType.LLM_RESPONSE,
            content=content,
            metadata=meta,
        )

        # If we have a specific source node, create a direct edge
        if source_node_id and source_node_id in self.nodes:
            self.add_edge(
                source=source_node_id, target=node_id, edge_type="responds_to"
            )

        return node_id

    def track_tool_call(self, tool_call, metadata: dict[str, Any] | None = None) -> str:
        """Track a tool call

        Args:
            tool_call: The tool call to track
            metadata: Additional metadata including component information

        Returns:
            ID of the created node
        """
        # Initialize metadata
        meta = metadata or {}
        # Add tool status and error message if not already present
        if "status" not in meta and hasattr(tool_call, "status"):
            meta["status"] = tool_call.status.value
        if "error_message" not in meta and hasattr(tool_call, "error_message"):
            meta["error_message"] = tool_call.error_message

        return self.add_node(
            node_type=NodeType.TOOL_CALL,
            content={
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
            },
            metadata=meta,
        )

    def track_tool_response(
        self,
        result: str,
        tool_call_node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Track a tool response

        Args:
            result: Result of the tool call
            tool_call_node_id: Optional ID of the tool call node
            metadata: Additional metadata including component information

        Returns:
            ID of the created node
        """
        node_id = self.add_node(
            node_type=NodeType.TOOL_RESPONSE,
            content=result,
            metadata=metadata,
        )

        # Link to the tool call if provided
        if tool_call_node_id and tool_call_node_id in self.nodes:
            self.add_edge(
                source=tool_call_node_id, target=node_id, edge_type="results_in"
            )

        return node_id

    def track_final_submission(
        self,
        answer: str,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Track the final submission

        Args:
            answer: The final answer submitted
            score: Optional score of the submission
            metadata: Additional metadata including component information

        Returns:
            ID of the created node
        """
        # Initialize metadata
        meta = metadata or {}
        # Add score if provided and not already present
        if score is not None and "score" not in meta:
            meta["score"] = score

        return self.add_node(
            node_type=NodeType.FINAL_SUBMISSION,
            content=answer,
            metadata=meta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the graph to a dictionary for serialization

        Returns:
            dictionary representation of the graph
        """
        return {
            "task_id": self.task_id,
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

    def visualize(
        self, figsize=(15, 10), save_path: str | Path | None = None
    ) -> None:  # TODO: move to seperate func
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

        # Create color maps for node types and components
        node_types = [data["node_type"] for _, data in G.nodes(data=True)]
        unique_types = list(set(node_types))
        colors = list(mcolors.TABLEAU_COLORS)[: len(unique_types)]
        color_map = dict(zip(unique_types, colors, strict=False))

        # Determine component types for visualization
        components = []
        for _, data in G.nodes(data=True):
            component = "environment"  # Default
            if "component" in data:
                component = data["component"]
            elif "metadata" in data and isinstance(data["metadata"], dict):
                component = data["metadata"].get("component", "environment")
            components.append(component)

        # Get node colors based on type
        node_colors = [color_map[data["node_type"]] for _, data in G.nodes(data=True)]

        # Determine node shapes based on component
        node_shapes = []
        for _, data in G.nodes(data=True):
            component = "environment"  # Default
            if "component" in data:
                component = data["component"]
            elif "metadata" in data and isinstance(data["metadata"], dict):
                component = data["metadata"].get("component", "environment")

            # Use different shapes for different components
            if component == "agent":
                node_shapes.append("s")  # square for agent
            else:
                node_shapes.append("o")  # circle for environment

        # Position nodes
        pos = nx.spring_layout(G, seed=42)

        # Create figure without display
        plt.figure(figsize=figsize)

        # Draw nodes by component shape
        unique_shapes = set(node_shapes)
        for shape in unique_shapes:
            # Get indices of nodes with this shape
            indices = [i for i, s in enumerate(node_shapes) if s == shape]
            if not indices:
                continue

            # Get the node IDs for these indices
            nodes = [list(G.nodes())[i] for i in indices]

            # Draw nodes with this shape
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes,
                node_color=[node_colors[i] for i in indices],
                node_shape=shape,
                node_size=800,
                alpha=0.8,
            )

        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)

        # Draw labels
        labels = {}
        for _i, (node, data) in enumerate(G.nodes(data=True)):
            component = "environment"
            if "component" in data:
                component = data["component"]
            elif "metadata" in data and isinstance(data["metadata"], dict):
                component = data["metadata"].get("component", "environment")

            # Create label with component information
            labels[node] = f"{data['node_type']} ({component})\n{data['content']}"

        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

        # Add legend for node types
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{node_type}",
                markerfacecolor=color,
                markersize=10,
            )
            for node_type, color in color_map.items()
        ]

        # Add legend for component types
        legend_elements.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Environment component",
                    markerfacecolor="gray",
                    markersize=10,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    label="Agent component",
                    markerfacecolor="gray",
                    markersize=10,
                ),
            ]
        )

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

        # Count nodes by type
        node_types = {}
        for _, data in G.nodes(data=True):
            node_type = data["node_type"]
            node_types[node_type] = node_types.get(node_type, 0) + 1

        # Count nodes by component
        components = {}
        for _, data in G.nodes(data=True):
            component = "environment"  # Default
            if "component" in data:
                component = data["component"]
            elif "metadata" in data and isinstance(data["metadata"], dict):
                component = data["metadata"].get("component", "environment")

            components[component] = components.get(component, 0) + 1

        # Count edges by type
        edge_types = {}
        for _, _, data in G.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
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
            "components": components,
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

    def get_or_create_tracker(
        self, task_id: str, trial_id: str | None = None
    ) -> GraphTracker:
        """Get an existing tracker or create a new one if it doesn't exist

        This is the primary method to use for obtaining a tracker - it ensures
        a single tracker exists per task-trial pair.

        Args:
            task_id: ID of the task
            trial_id: Optional trial ID

        Returns:
            The graph tracker for this task-trial pair
        """
        trial_id = trial_id or "default"
        key = f"{task_id}_{trial_id}"

        if key not in self.active_trackers:
            # Create a new tracker if one doesn't exist
            logger.info(f"Creating new graph tracker for {key}")
            self.active_trackers[key] = GraphTracker(task_id=task_id, trial_id=trial_id)

        return self.active_trackers[key]

    def create_tracker(
        self, task_id: str, agent_type: str, trial_id: str | None = None
    ) -> GraphTracker:
        """Create or retrieve an existing graph tracker

        This method is maintained for backward compatibility but delegates to get_or_create_tracker.
        The agent_type parameter is stored in metadata when adding nodes.

        Args:
            task_id: ID of the task
            agent_type: Type of agent (used in metadata)
            trial_id: Optional trial ID

        Returns:
            GraphTracker instance
        """
        logger.info(
            f"Request for tracker with task_id={task_id}, agent_type={agent_type}, trial_id={trial_id}"
        )
        return self.get_or_create_tracker(task_id, trial_id)

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
        trial_id = trial_id or "default"
        key = f"{task_id}_{trial_id}"
        return self.active_trackers.get(key)

    def save_tracker(
        self, task_id: str, trial_id: str | None = None, visualize: bool = False
    ) -> None:
        """Save a specific tracker with error handling

        Args:
            task_id: ID of the task
            trial_id: Optional trial ID
            visualize: Whether to generate visualization image
        """
        trial_id = trial_id or "default"
        key = f"{task_id}_{trial_id}"
        tracker = self.active_trackers.get(key)

        if tracker:
            try:
                # Save JSON data
                json_path = Path(self.output_dir) / f"{key}.json"
                tracker.save_to_file(json_path)
                logger.info(f"Saved graph data to {json_path}")

                # Generate visualization if requested
                if visualize:
                    viz_path = Path(self.output_dir) / f"{key}.png"
                    try:
                        tracker.visualize(save_path=viz_path)
                        logger.info(f"Generated visualization at {viz_path}")
                    except Exception as e:
                        logger.warning(f"Failed to generate visualization: {e}")
                        logger.debug(traceback.format_exc())
            except Exception as e:
                logger.error(f"Error saving tracker {key}: {e}")
                logger.debug(traceback.format_exc())
        else:
            logger.warning(f"No tracker found for {key}")

    def save_trackers(self) -> None:
        """Save all active trackers"""
        for key in self.active_trackers:
            task_id, trial_id = key.split("_", 1)
            self.save_tracker(task_id, trial_id, visualize=True)
