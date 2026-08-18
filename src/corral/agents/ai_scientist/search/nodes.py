"""Typed records stored in the scientific experiment tree."""

import json
from enum import Enum
from math import isfinite
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_validator,
    model_validator,
)


class ResearchStage(str, Enum):
    FORMULATION = "formulation"
    PRELIMINARY = "preliminary"
    TUNING = "tuning"
    RESEARCH = "research"
    VERIFICATION = "verification"
    COMPLETE = "complete"


class NodeType(str, Enum):
    DRAFT = "draft"
    CONTINUE = "continue"
    REFINE = "refine"
    DEBUG = "debug"
    PARAMETER_SEARCH = "parameter_search"
    RESEARCH = "research"
    ABLATION = "ablation"
    REPLICATION = "replication"
    COUNTERFACTUAL = "counterfactual"
    AGGREGATION = "aggregation"


class NodeStatus(str, Enum):
    PROPOSED = "proposed"
    RUNNING = "running"
    SUCCESSFUL = "successful"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID = "invalid"


class ExperimentTermination(str, Enum):
    """Why an experiment worker stopped producing actions."""

    AGGREGATED = "aggregated"
    REPLICATION_REPLAYED = "replication_replayed"
    WORKER_FINISHED = "worker_finished"
    ACTION_BUDGET_EXHAUSTED = "action_budget_exhausted"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    ACTION_FAILED = "action_failed"


class Recommendation(str, Enum):
    CONTINUE = "continue"
    REFINE = "refine"
    DEBUG = "debug"
    FINALIZE = "finalize"
    ABANDON = "abandon"


# Providers such as Anthropic reject a structured-output schema containing an
# arbitrary object (`additionalProperties: true`). Expose arguments as a JSON
# string to the provider while retaining a dict in Python.
JsonArguments = Annotated[
    dict[str, Any],
    WithJsonSchema(
        {
            "type": "string",
            "description": (
                "Tool arguments encoded as one JSON object string. "
                'Use "{}" when the tool takes no arguments.'
            ),
        }
    ),
]


class PlannedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: str
    tool_name: str
    arguments: JsonArguments = Field(default_factory=dict)
    expected_information: str

    @field_validator("arguments", mode="before")
    @classmethod
    def _parse_arguments(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return json.loads(value) if value else {}
        return value


class ExperimentDecision(BaseModel):
    """One observe-reason-act decision made inside an experiment node."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["act", "finish"]
    rationale: str
    purpose: str | None = None
    tool_name: str | None = None
    arguments: JsonArguments = Field(default_factory=dict)
    expected_information: str | None = None
    conclusion: str | None = None

    @field_validator("arguments", mode="before")
    @classmethod
    def _parse_arguments(cls, value: Any) -> Any:
        return PlannedAction._parse_arguments(value)

    @model_validator(mode="after")
    def _consistent_decision(self) -> "ExperimentDecision":
        action_fields = (self.purpose, self.tool_name, self.expected_information)
        if self.decision == "act" and not all(action_fields):
            raise ValueError(
                "An act decision requires purpose, tool_name, and expected_information"
            )
        if self.decision == "finish" and any(action_fields):
            raise ValueError("A finish decision cannot contain an action")
        return self

    def as_action(self) -> PlannedAction | None:
        if self.decision == "finish":
            return None
        return PlannedAction(
            purpose=self.purpose or "",
            tool_name=self.tool_name or "",
            arguments=self.arguments,
            expected_information=self.expected_information or "",
        )


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_index: int = Field(ge=0)
    purpose: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool
    result: str | None = None
    error: str | None = None


class ExecutedAction(BaseModel):
    """A physical tool call and the observation it originally produced."""

    model_config = ConfigDict(extra="forbid")

    action: PlannedAction
    observation: Observation


class ExperimentStep(BaseModel):
    """A durable worker decision and, for actions, its resulting observation."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(ge=0)
    decision: ExperimentDecision
    observation: Observation | None = None


class NodeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validity: float = Field(ge=0.0, le=1.0)
    task_progress: float = Field(ge=0.0, le=1.0)
    evidence_strength: float = Field(ge=0.0, le=1.0)
    information_gain: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    recommendation: Recommendation
    reason: str
    conclusions: list[str] = Field(default_factory=list)
    supported_claims: list[str] = Field(default_factory=list)
    contradicted_claims: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    visual_feedback: list[str] = Field(default_factory=list)


class MeasuredMetric(BaseModel):
    """A task-internal scalar that can objectively rank experiments."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    maximize: bool
    unit: str | None = None


class MeasuredOutcome(MeasuredMetric):
    """A scalar parsed from an actual tool observation, never benchmark score."""

    value: float
    provenance: str = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def _finite_value(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("measured outcome must be finite")
        return value

    @property
    def directional_value(self) -> float:
        """Return a value where larger is always scientifically preferable."""
        return self.value if self.maximize else -self.value


class ReplicationSummary(BaseModel):
    """Deterministic statistics over stage-boundary replication outcomes.

    `std` is the sample standard deviation and `stderr` is computed from
    the number of comparable scalar values. Non-scalar Corral experiments keep
    the run/success/seed accounting while leaving metric statistics unset.
    """

    model_config = ConfigDict(extra="forbid")

    n_runs: int = Field(ge=0)
    n_successful: int = Field(ge=0)
    metric_name: str | None = None
    values: list[float] = Field(default_factory=list)
    mean: float | None = None
    std: float | None = None
    stderr: float | None = None
    seeds: list[int | None] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_counts(self) -> "ReplicationSummary":
        if self.n_successful > self.n_runs:
            raise ValueError("n_successful cannot exceed n_runs")
        if len(self.seeds) != self.n_runs:
            raise ValueError("seeds must contain one entry per replication run")
        return self


class NodeProposal(BaseModel):
    """One high-level scientific experiment, without a precomputed action list."""

    model_config = ConfigDict(extra="forbid")

    node_type: NodeType
    hypothesis: str
    rationale: str
    experiment_goal: str
    success_criteria: list[str] = Field(default_factory=list)
    related_node_ids: list[str] = Field(default_factory=list)
    visual_artifacts: list[str] = Field(default_factory=list)


class PlanningBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[NodeProposal]


class SubstagePlan(BaseModel):
    """An evidence-dependent research agenda within one main stage."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    rationale: str
    objectives: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)


class SubstageCompletion(BaseModel):
    """Evidence-based decision about whether a substage may advance."""

    model_config = ConfigDict(extra="forbid")

    complete: bool
    reason: str
    satisfied_criteria: list[str] = Field(default_factory=list)
    unmet_criteria: list[str] = Field(default_factory=list)


class StageWinnerSelection(BaseModel):
    """Listwise comparison used when handing one stage into the next."""

    model_config = ConfigDict(extra="forbid")

    selected_node_id: str
    reason: str
    candidate_comparison: list[str] = Field(default_factory=list)


class ExperimentNode(BaseModel):
    """A stable, serializable checkpoint in the experiment tree."""

    model_config = ConfigDict(extra="forbid")

    id: str
    parent_id: str | None = None
    branch_id: str | None = None
    execution_id: str | None = None
    branch_workspace: str | None = None
    stage: ResearchStage
    stage_seed_id: str | None = None
    substage_id: str | None = None
    boundary_validation: bool = False
    physical_state_inherited: bool = False
    node_type: NodeType
    hypothesis: str
    rationale: str
    experiment_goal: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    related_node_ids: list[str] = Field(default_factory=list)
    visual_artifacts: list[str] = Field(default_factory=list)
    # `plan` is the realized action sequence. Actions are appended only after
    # the worker has observed every preceding result; it is never precomputed.
    plan: list[PlannedAction] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    trajectory: list[ExperimentStep] = Field(default_factory=list)
    worker_conclusion: str | None = None
    allocated_action_budget: int = Field(default=0, ge=0)
    termination_reason: ExperimentTermination | None = None
    executed_actions: list[ExecutedAction] = Field(default_factory=list)
    replication_seed: int | None = None
    replication_seed_overrides: list[str] = Field(default_factory=list)
    replication_summary: ReplicationSummary | None = None
    conclusions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.PROPOSED
    debug_depth: int = Field(default=0, ge=0)
    depth: int = Field(default=0, ge=0)
    measured_outcome: MeasuredOutcome | None = None
    evaluation: NodeEvaluation | None = None
