"""Submission-ready answer generation from collected evidence."""

import json

from pydantic import BaseModel, ConfigDict

from corral.agents.ai_scientist.prompts import render_prompt
from corral.agents.ai_scientist.search.nodes import ExperimentNode
from corral.agents.ai_scientist.workers.base import StructuredModel


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer: str


class FinalSynthesizer:
    def __init__(self, model: StructuredModel, *, max_journal_chars: int) -> None:
        self.model = model
        self.max_journal_chars = max_journal_chars

    def generate(
        self,
        *,
        task_prompt: str,
        journal_context: str,
        best_nodes: list[ExperimentNode],
    ) -> str:
        prompt = render_prompt(
            "final_answer",
            task_prompt=task_prompt,
            journal=journal_context[: self.max_journal_chars],
            best_nodes=json.dumps(
                [node.model_dump(mode="json") for node in best_nodes],
                indent=2,
                ensure_ascii=False,
            ),
        )
        result = self.model.generate(prompt, FinalAnswer, purpose="final_synthesis")
        return result.final_answer.strip()
