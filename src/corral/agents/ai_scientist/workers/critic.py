"""Evidence-oriented LLM evaluation of a completed experiment node."""

import json

from corral.agents.ai_scientist.prompts import render_prompt
from corral.agents.ai_scientist.search.nodes import ExperimentNode, NodeEvaluation
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
        )
        return self.model.generate(
            prompt,
            NodeEvaluation,
            model=self.evaluator_model,
            purpose=f"evaluate_{node.id}",
        )
