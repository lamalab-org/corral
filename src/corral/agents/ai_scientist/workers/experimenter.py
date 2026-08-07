"""Turn planned nodes into observed experiment nodes."""

from corral.agents.ai_scientist.search.nodes import ExperimentNode, NodeStatus
from corral.agents.ai_scientist.tools.corral_executor import CorralExecutor


class Experimenter:
    def execute(self, node: ExperimentNode, executor: CorralExecutor) -> ExperimentNode:
        node.status = NodeStatus.RUNNING
        # Aggregation nodes intentionally synthesize existing evidence and need
        # no environment action. Every other empty plan is invalid.
        if not node.plan:
            node.status = (
                NodeStatus.SUCCESSFUL
                if node.node_type.value == "aggregation"
                else NodeStatus.INVALID
            )
            return node

        node.observations = executor.execute_plan(node.plan)
        node.status = (
            NodeStatus.SUCCESSFUL
            if node.observations and all(item.success for item in node.observations)
            else NodeStatus.FAILED
        )
        return node
