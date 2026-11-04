"""Task formulation and complete manager state."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from corral.agents.ai_scientist.journal import ResearchJournal
from corral.agents.ai_scientist.search.nodes import (
    ExecutedAction,
    MeasuredMetric,
    ResearchStage,
    SubstagePlan,
)
from corral.agents.ai_scientist.search.tree import ExperimentTree
from corral.agents.ai_scientist.tools.execution_pool import ReplayResult


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
    measured_metric: MeasuredMetric | None = None

    @property
    def has_tunable_parameters(self) -> bool:
        return bool(self.tunable_parameters)


class SubstageState(BaseModel):
    """One manager-created agenda inside a main research stage."""

    model_config = ConfigDict(extra="forbid")

    id: str
    plan: SubstagePlan
    node_ids: list[str] = Field(default_factory=list)
    completion_criteria_met: bool = False
    completion_reason: str | None = None
    last_completion_check_node_count: int = Field(default=0, ge=0)


class StageProgress(BaseModel):
    """Explicit handoff and validation state for a main research stage."""

    model_config = ConfigDict(extra="forbid")

    stage: ResearchStage
    seed_node_id: str | None = None
    best_node_id: str | None = None
    improved_over_seed: bool = False
    completion_criteria_met: bool = False
    search_budget_exhausted: bool = False
    boundary_validation_reason: str | None = None
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
        self.executions_created = 0
        self.executions_cloned = 0
        self.peak_simultaneous_executions = 0
        self.replay_results = []
        self.artifact_source_workspace: str | None = None
        self.artifact_destination_workspace: str | None = None
        self.promoted_artifacts: list[str] = []
        self.llm_calls = 0
        self.llm_tokens = 0
        self.experimental_search_terminated_reason: str | None = None

        for hypothesis in formulation.candidate_hypotheses:
            self.journal.register_hypothesis(hypothesis.statement)

    @property
    def best_nodes(self):
        return self.tree.best()

    def stage_progress(self, stage: ResearchStage) -> StageProgress:
        return self.stages[stage]

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, Any],
        *,
        task_prompt: str,
        tools: list[dict[str, Any]],
    ) -> "ScientistState":
        state = cls(
            task_prompt=task_prompt,
            tools=tools,
            formulation=TaskFormulation.model_validate(snapshot["formulation"]),
        )
        state.tree = ExperimentTree.from_dict(snapshot["tree"])
        state.journal = ResearchJournal.from_dict(snapshot["journal"])
        state.current_stage = ResearchStage(snapshot["current_stage"])
        state.stages = {
            ResearchStage(stage): StageProgress.model_validate(progress)
            for stage, progress in snapshot["stages"].items()
        }
        usage = snapshot.get("usage", {})
        for name in (
            "tool_calls",
            "scientific_tool_calls",
            "replay_tool_calls",
            "executions_created",
            "executions_cloned",
            "peak_simultaneous_executions",
            "llm_calls",
            "llm_tokens",
        ):
            setattr(state, name, int(usage.get(name, 0)))
        state.replay_results = [
            ReplayResult(
                branch_id=item["branch_id"],
                exact=item["exact"],
                equivalent=item["equivalent"],
                original=ExecutedAction.model_validate(item["original"]),
                replayed=ExecutedAction.model_validate(item["replayed"]),
            )
            for item in snapshot.get("replay_results", [])
        ]
        artifacts = snapshot.get("artifacts", {})
        state.artifact_source_workspace = artifacts.get("source_workspace")
        state.artifact_destination_workspace = artifacts.get("destination_workspace")
        state.promoted_artifacts = list(artifacts.get("promoted", []))
        state.experimental_search_terminated_reason = snapshot.get(
            "experimental_search_terminated_reason"
        )
        return state
