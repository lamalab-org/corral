"""Typed records stored in the scientific experiment tree."""

import json
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_validator,
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
    FAILED = "failed"
    INVALID = "invalid"


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


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_index: int = Field(ge=0)
    purpose: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool
    result: str | None = None
    error: str | None = None


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


class NodeProposal(BaseModel):
    """One model-proposed scientific experiment."""

    model_config = ConfigDict(extra="forbid")

    node_type: NodeType
    hypothesis: str
    rationale: str
    plan: list[PlannedAction] = Field(default_factory=list)


class PlanningBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[NodeProposal]


class ExperimentNode(BaseModel):
    """A stable, serializable checkpoint in the experiment tree."""

    model_config = ConfigDict(extra="forbid")

    id: str
    parent_id: str | None = None
    branch_workspace: str | None = None
    stage: ResearchStage
    node_type: NodeType
    hypothesis: str
    rationale: str
    plan: list[PlannedAction] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.PROPOSED
    debug_depth: int = Field(default=0, ge=0)
    depth: int = Field(default=0, ge=0)
    evaluation: NodeEvaluation | None = None
