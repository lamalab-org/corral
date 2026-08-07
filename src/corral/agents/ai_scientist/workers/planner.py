"""Task formulation and stage-aware experiment planning."""

import json

from corral.agents.ai_scientist.prompts import render_prompt
from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeProposal,
    NodeType,
    PlanningBatch,
    ResearchStage,
)
from corral.agents.ai_scientist.state import TaskFormulation
from corral.agents.ai_scientist.workers.base import StructuredModel


class TaskFormulator:
    def __init__(self, model: StructuredModel, *, max_tool_schema_chars: int) -> None:
        self.model = model
        self.max_tool_schema_chars = max_tool_schema_chars

    def formulate(
        self,
        *,
        task_prompt: str,
        tools: list[dict],
        examples: list[str] | None = None,
    ) -> TaskFormulation:
        tool_text = json.dumps(tools, indent=2, ensure_ascii=False)
        if len(tool_text) > self.max_tool_schema_chars:
            tool_text = tool_text[: self.max_tool_schema_chars] + "\n... truncated ..."
        prompt = render_prompt(
            "formulate",
            task_prompt=task_prompt,
            tools=tool_text,
            examples=json.dumps(examples or [], ensure_ascii=False),
        )
        return self.model.generate(prompt, TaskFormulation, purpose="task_formulation")


class NodePlanner:
    def __init__(
        self,
        model: StructuredModel,
        *,
        max_actions_per_node: int,
        max_journal_chars: int,
        max_tool_schema_chars: int,
    ) -> None:
        self.model = model
        self.max_actions_per_node = max_actions_per_node
        self.max_journal_chars = max_journal_chars
        self.max_tool_schema_chars = max_tool_schema_chars

    def propose(
        self,
        *,
        stage: ResearchStage,
        node_type: NodeType,
        count: int,
        task_prompt: str,
        formulation: TaskFormulation,
        tools: list[dict],
        journal_context: str,
        parent: ExperimentNode | None,
        branch_workspaces: list[str],
    ) -> list[NodeProposal]:
        prompt_name = {
            ResearchStage.PRELIMINARY: "preliminary",
            ResearchStage.TUNING: "tuning",
            ResearchStage.RESEARCH: "research",
            ResearchStage.VERIFICATION: "ablation",
        }[stage]
        parent_text = (
            json.dumps(parent.model_dump(mode="json"), indent=2, ensure_ascii=False)
            if parent is not None
            else "No parent; create independent root approaches."
        )
        tool_text = json.dumps(tools, indent=2, ensure_ascii=False)
        if len(tool_text) > self.max_tool_schema_chars:
            tool_text = tool_text[: self.max_tool_schema_chars] + "\n... truncated ..."
        prompt = render_prompt(
            prompt_name,
            task_prompt=task_prompt,
            formulation=formulation.model_dump_json(indent=2),
            tools=tool_text,
            journal=journal_context[: self.max_journal_chars],
            parent=parent_text,
            branch_workspaces=json.dumps(branch_workspaces, ensure_ascii=False),
            node_type=node_type.value,
            count=count,
            max_actions=self.max_actions_per_node,
        )
        batch = self.model.generate(
            prompt, PlanningBatch, purpose=f"plan_{stage.value}_{node_type.value}"
        )
        proposals = batch.proposals[:count]
        # The manager, not the model, owns the tree semantics. Coerce the type
        # requested for this expansion and bound actions deterministically.
        return [
            proposal.model_copy(
                update={
                    "node_type": node_type,
                    "plan": proposal.plan[: self.max_actions_per_node],
                }
            )
            for proposal in proposals
        ]
