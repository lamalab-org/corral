"""Task formulation and complete manager state."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from corral.agents.ai_scientist.journal import ResearchJournal
from corral.agents.ai_scientist.search.nodes import ResearchStage
from corral.agents.ai_scientist.search.tree import ExperimentTree


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    rationale: str


class TaskFormulation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str
    required_answer: str
    known_information: list[str] = Field(default_factory=list)
    unknown_information: list[str] = Field(default_factory=list)
    candidate_hypotheses: list[Hypothesis] = Field(default_factory=list)
    observable_quantities: list[str] = Field(default_factory=list)
    possible_experiments: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    tunable_parameters: list[str] = Field(default_factory=list)

    @property
    def has_tunable_parameters(self) -> bool:
        return bool(self.tunable_parameters)


class ScientistState:
    """Mutable orchestration state; the tree and journal stay explicit."""

    def __init__(
        self,
        *,
        task_prompt: str,
        tools: list[dict[str, Any]],
        formulation: TaskFormulation,
    ) -> None:
        self.task_prompt = task_prompt
        self.tools = tools
        self.formulation = formulation
        self.tree = ExperimentTree()
        self.journal = ResearchJournal()
        self.current_stage = ResearchStage.FORMULATION
        self.tool_calls = 0
        self.llm_calls = 0
        self.llm_tokens = 0

        for hypothesis in formulation.candidate_hypotheses:
            self.journal.register_hypothesis(hypothesis.statement)

    @property
    def best_nodes(self):
        return self.tree.best()
