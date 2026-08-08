"""Task formulation and complete manager state."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from corral.agents.ai_scientist.journal import ResearchJournal
from corral.agents.ai_scientist.search.nodes import ResearchStage, SubstagePlan
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


class SubstageState(BaseModel):
    """One manager-created agenda inside a main research stage."""

    model_config = ConfigDict(extra="forbid")

    id: str
    plan: SubstagePlan
    node_ids: list[str] = Field(default_factory=list)


class StageProgress(BaseModel):
    """Explicit handoff and validation state for a main research stage."""

    model_config = ConfigDict(extra="forbid")

    stage: ResearchStage
    seed_node_id: str | None = None
    best_node_id: str | None = None
    improved_over_seed: bool = False
    completion_criteria_met: bool = False
    comparison_reason: str | None = None
    substages: list[SubstageState] = Field(default_factory=list)
    replication_node_ids: list[str] = Field(default_factory=list)
    aggregation_node_id: str | None = None

    @property
    def current_substage(self) -> SubstageState | None:
        return self.substages[-1] if self.substages else None


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
        self.stages: dict[ResearchStage, StageProgress] = {}
        self.tool_calls = 0
        self.scientific_tool_calls = 0
        self.replay_tool_calls = 0
        self.trial_runtimes_created = 0
        self.peak_simultaneous_trials = 0
        self.replay_results = []
        self.artifact_source_workspace: str | None = None
        self.artifact_destination_workspace: str | None = None
        self.promoted_artifacts: list[str] = []
        self.llm_calls = 0
        self.llm_tokens = 0

        for hypothesis in formulation.candidate_hypotheses:
            self.journal.register_hypothesis(hypothesis.statement)

    @property
    def best_nodes(self):
        return self.tree.best()

    def stage_progress(self, stage: ResearchStage) -> StageProgress:
        return self.stages[stage]
