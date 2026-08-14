"""Task formulation and stage-aware experiment planning."""

import json

from corral.agents.ai_scientist.prompts import render_prompt
from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeProposal,
    NodeType,
    PlanningBatch,
    ResearchStage,
    SubstagePlan,
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
        siblings: list[ExperimentNode],
        branch_workspaces: list[str],
        substage: SubstagePlan | None = None,
        proposal_offset: int = 0,
    ) -> list[NodeProposal]:
        if node_type == NodeType.CONTINUE:
            if parent is None:
                raise ValueError("A continuation proposal requires a parent node")
            # Continuation is an execution detail, not a new hypothesis. Keep
            # the parent's scientific identity instead of inviting the planner
            # to silently redefine an unfinished experiment.
            return [
                NodeProposal(
                    node_type=NodeType.CONTINUE,
                    hypothesis=parent.hypothesis,
                    rationale=parent.rationale,
                    experiment_goal=parent.experiment_goal,
                    success_criteria=list(parent.success_criteria),
                )
            ]

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
        sibling_text = (
            json.dumps(
                [
                    sibling.model_dump(
                        mode="json",
                        include={
                            "id",
                            "node_type",
                            "hypothesis",
                            "experiment_goal",
                            "success_criteria",
                            "conclusions",
                        },
                    )
                    for sibling in siblings
                ],
                indent=2,
                ensure_ascii=False,
            )
            if siblings
            else "No existing children from this checkpoint."
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
            siblings=sibling_text,
            branch_workspaces=json.dumps(branch_workspaces, ensure_ascii=False),
            proposal_slots=json.dumps(
                list(range(proposal_offset + 1, proposal_offset + count + 1))
            ),
            node_type=node_type.value,
            count=count,
            max_actions=self.max_actions_per_node,
            substage=(
                substage.model_dump_json(indent=2)
                if substage is not None
                else "No narrower substage agenda."
            ),
        )
        batch = self.model.generate(
            prompt, PlanningBatch, purpose=f"plan_{stage.value}_{node_type.value}"
        )
        proposals = batch.proposals[:count]
        # The manager, not the model, owns the tree semantics. The proposal is
        # deliberately high-level; the experiment worker chooses actions only
        # after seeing the preceding observations.
        return [
            proposal.model_copy(update={"node_type": node_type})
            for proposal in proposals
        ]

    def propose_substage(
        self,
        *,
        stage: ResearchStage,
        task_prompt: str,
        formulation: TaskFormulation,
        journal_context: str,
        seed: ExperimentNode | None,
        previous: SubstagePlan,
        stage_nodes: list[ExperimentNode],
        substage_number: int,
    ) -> SubstagePlan:
        """Replan a main-stage agenda from evidence gathered so far."""
        prompt = render_prompt(
            "substage",
            task_prompt=task_prompt,
            formulation=formulation.model_dump_json(indent=2),
            stage=stage.value,
            seed=(
                json.dumps(seed.model_dump(mode="json"), indent=2)
                if seed is not None
                else "No inherited seed in the preliminary stage."
            ),
            previous_substage=previous.model_dump_json(indent=2),
            stage_results=json.dumps(
                [
                    node.model_dump(
                        mode="json",
                        include={
                            "id",
                            "substage_id",
                            "node_type",
                            "hypothesis",
                            "experiment_goal",
                            "status",
                            "conclusions",
                            "open_questions",
                            "evaluation",
                            "visual_artifacts",
                        },
                    )
                    for node in stage_nodes
                ],
                indent=2,
                ensure_ascii=False,
            ),
            journal=journal_context[: self.max_journal_chars],
            substage_number=substage_number,
        )
        return self.model.generate(
            prompt,
            SubstagePlan,
            purpose=f"manage_{stage.value}_substage_{substage_number}",
        )
