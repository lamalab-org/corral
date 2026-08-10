"""Evidence-oriented LLM evaluation of a completed experiment node."""

import json

from corral.agents.ai_scientist.prompts import render_prompt
from corral.agents.ai_scientist.search.evaluator import node_ranking_key
from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeEvaluation,
    ResearchStage,
    StageWinnerSelection,
    SubstageCompletion,
    SubstagePlan,
)
from corral.agents.ai_scientist.state import TaskFormulation
from corral.agents.ai_scientist.workers.base import StructuredModel


class ScientificCritic:
    def __init__(
        self,
        model: StructuredModel,
        *,
        evaluator_model: str,
        max_journal_chars: int,
    ) -> None:
        self.model = model
        self.evaluator_model = evaluator_model
        self.max_journal_chars = max_journal_chars

    def evaluate(
        self,
        *,
        task_prompt: str,
        formulation: TaskFormulation,
        node: ExperimentNode,
        journal_context: str,
    ) -> NodeEvaluation:
        prompt = render_prompt(
            "evaluate_node",
            task_prompt=task_prompt,
            formulation=formulation.model_dump_json(indent=2),
            node=json.dumps(node.model_dump(mode="json"), indent=2),
            journal=journal_context[: self.max_journal_chars],
            visual_artifacts=json.dumps(node.visual_artifacts, indent=2),
        )
        multimodal_generate = getattr(self.model, "generate_multimodal", None)
        if node.visual_artifacts and callable(multimodal_generate):
            return multimodal_generate(
                prompt,
                NodeEvaluation,
                image_paths=node.visual_artifacts,
                model=self.evaluator_model,
                purpose=f"evaluate_visual_{node.id}",
            )
        return self.model.generate(
            prompt,
            NodeEvaluation,
            model=self.evaluator_model,
            purpose=f"evaluate_{node.id}",
        )

    def select_best(
        self,
        *,
        task_prompt: str,
        formulation: TaskFormulation,
        stage: ResearchStage,
        seed_node_id: str | None,
        candidates: list[ExperimentNode],
        journal_context: str,
    ) -> StageWinnerSelection:
        """Choose a stage winner by comparing candidates in one context."""
        prompt = render_prompt(
            "select_stage_winner",
            task_prompt=task_prompt,
            formulation=formulation.model_dump_json(indent=2),
            stage=stage.value,
            seed_node_id=seed_node_id or "No inherited seed",
            candidates=json.dumps(
                [candidate.model_dump(mode="json") for candidate in candidates],
                indent=2,
                ensure_ascii=False,
            ),
            journal=journal_context[: self.max_journal_chars],
        )
        return self.model.generate(
            prompt,
            StageWinnerSelection,
            model=self.evaluator_model,
            purpose=f"select_best_{stage.value}",
        )

    def is_substage_complete(
        self,
        *,
        task_prompt: str,
        formulation: TaskFormulation,
        stage: ResearchStage,
        substage: SubstagePlan,
        stage_nodes: list[ExperimentNode],
        journal_context: str,
    ) -> SubstageCompletion:
        """Decide from observations whether the current agenda may advance."""
        best_node = max(stage_nodes, key=node_ranking_key) if stage_nodes else None
        prompt = render_prompt(
            "evaluate_substage",
            task_prompt=task_prompt,
            formulation=formulation.model_dump_json(indent=2),
            stage=stage.value,
            substage=substage.model_dump_json(indent=2),
            best_node=(
                json.dumps(best_node.model_dump(mode="json"), indent=2)
                if best_node is not None
                else "No evaluated experiment is available."
            ),
            stage_results=json.dumps(
                [node.model_dump(mode="json") for node in stage_nodes],
                indent=2,
                ensure_ascii=False,
            ),
            journal=journal_context[: self.max_journal_chars],
        )
        return self.model.generate(
            prompt,
            SubstageCompletion,
            model=self.evaluator_model,
            purpose=f"evaluate_{stage.value}_substage",
        )
