"""Provider-independent contracts and serialization for epistemic analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict

REPORT_MESSAGE_NAME = "epistemic_report"
SUBMISSION_REVIEW_MESSAGE_NAME = "epistemic_submission_review"


GRAPH_SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "corral-messages-exact-instructions-v2"
ANNOTATION_VERSION = "9"
ANSWER_PROVENANCE_VERSION = "answer-provenance-v1"

# Frozen catalog for schema-1 graphs saved before evaluation_shortcut was added.
# Reading them must not invent an assessment for a check that was never run.
_LEGACY_TRACE_CHECK_NAMES = (
    "cheating_consideration",
    "cheating_attempt",
    "answer_contamination",
    "fabricated_evidence",
    "false_reporting",
    "generation_breakdown",
    "contradiction_ignored",
    "repeated_failure",
    "insufficient_validation",
    "validation_submission_mismatch",
    "known_invalid_submission",
    "submission_with_concerns",
)


class Message(TypedDict):
    role: str
    content: str


class SupportOffsets(TypedDict, total=False):
    start: int
    end: int


class Support(SupportOffsets):
    msg_idx: int
    quote: str


class Node(TypedDict):
    node_id: str
    type: str
    time: int
    text: str
    support: list[Support]


class Edge(TypedDict):
    src: str
    dst: str
    relation: str
    time: int
    support: list[Support]


Status = Literal[
    "complete",
    "unspecified",
    "incomplete",
    "unsupported",
    "failed",
    "missing",
    "uninformative",
]

# Execution failure (including exhausting the interaction budget) does not
# invalidate a fully reconstructed trace. Reconstruction and annotation status
# independently determine whether its evidence can be analyzed.
ANALYZABLE_COMPLETION_STATUSES = frozenset({"complete", "unspecified", "failed"})


def content_hash(value: Any) -> str:
    """Canonical JSON hash; never includes machine-specific repr strings."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class GroupingMetadata:
    agent_model: str | None = None
    environment: str | None = None
    level: str | None = None
    task: str | None = None
    trial: str | None = None


@dataclass(frozen=True)
class SourceReference:
    identity: str
    locator: str
    content_hash: str
    provider: str = "path"
    normalization_version: str = NORMALIZATION_VERSION
    resolved_locator: str | None = None
    list_path: str | None = None
    line_number: int | None = None
    snapshots: tuple[str, ...] = ()
    execution_id: str | None = None
    commit_hash: str | None = None
    branch_id: str | None = None
    agent_run_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedTrace:
    messages: list[Message]
    source: SourceReference
    message_sources: list[dict[str, Any]] = field(default_factory=list)
    grouping: GroupingMetadata = field(default_factory=GroupingMetadata)
    completion_status: Status = "unspecified"
    reconstruction_status: Status = "complete"
    # Source adapters retain the exact input for local snapshots. The pipeline
    # does not read it, and it is not duplicated inside annotation documents.
    snapshot: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.messages, list) or any(
            not isinstance(m, dict)
            or not isinstance(m.get("role"), str)
            or not isinstance(m.get("content"), str)
            for m in self.messages
        ):
            raise ValueError("Normalized messages must contain string role and content")
        if not self.message_sources:
            self.message_sources = [{"index": i} for i in range(len(self.messages))]
        if len(self.message_sources) != len(self.messages):
            raise ValueError("Every normalized message requires a source mapping")


@dataclass(frozen=True)
class AnnotationConfig:
    model: str
    window: int = 254
    overlap: int = 1
    max_nodes_per_window: int = 1000
    strict_support: bool = True
    tool_output_max_chars: int = 0
    tool_output_truncation: str = "head-tail"
    max_request_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("An annotation model is required")
        for name in (
            "tool_output_max_chars",
            "max_request_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if 0 < self.tool_output_max_chars < 16:
            raise ValueError("tool_output_max_chars must be zero or at least 16")
        if self.tool_output_truncation != "head-tail":
            raise ValueError("tool_output_truncation must be head-tail")
        for name in ("window", "max_nodes_per_window"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.overlap, int)
            or isinstance(self.overlap, bool)
            or not 0 <= self.overlap < self.window
        ):
            raise ValueError("Require 0 <= overlap < window")


@dataclass
class AnnotatedGraph:
    nodes: list[Node]
    edges: list[Edge]
    provenance: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=lambda: {"warnings": []})
    input_file: str = ""
    source: SourceReference | None = None
    grouping: GroupingMetadata = field(default_factory=GroupingMetadata)
    messages: list[Message] = field(default_factory=list)
    message_sources: list[dict[str, Any]] = field(default_factory=list)
    completion_status: Status = "unspecified"
    reconstruction_status: Status = "complete"
    annotation_status: Status = "complete"
    schema_version: int = GRAPH_SCHEMA_VERSION
    # Optional in v1 graphs; embedded for portable inspection of annotation inputs.
    annotation_view: dict[str, Any] | None = None
    # Exact annotation exchanges, including absent decisions; no auth headers.
    annotation_requests: list[dict[str, Any]] = field(default_factory=list)
    # Read checks embedded by the retired combined pipeline in historical graphs.
    # New behavioral labels are saved independently by CheckPipeline.
    trace_checks: list[dict[str, Any]] = field(default_factory=list)
    trace_checks_status: Status = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_graph(document: dict[str, Any]) -> AnnotatedGraph:
    """Read an annotation in the current graph format."""
    if not isinstance(document, dict):
        raise ValueError("An annotation must be a JSON object")
    version = document.get("schema_version")
    if version != GRAPH_SCHEMA_VERSION:
        raise ValueError(f"Unsupported graph schema version: {version}")
    for key in ("provenance", "qc", "grouping"):
        if key in document and not isinstance(document[key], dict):
            raise ValueError(f"Annotation {key} must be an object")
    for key in (
        "completion_status",
        "reconstruction_status",
        "annotation_status",
        "trace_checks_status",
    ):
        if key in document and document[key] not in {
            "complete",
            "unspecified",
            "incomplete",
            "unsupported",
            "failed",
            "missing",
            "uninformative",
        }:
            raise ValueError(f"Unsupported {key}: {document[key]}")
    if not isinstance(document.get("nodes"), list) or not isinstance(
        document.get("edges"), list
    ):
        raise ValueError("Annotation requires nodes and edges lists")
    ids = set()
    for node in document["nodes"]:
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("node_id"), str)
            or node.get("type") not in {"H", "T", "E", "J", "C", "N", "F"}
        ):
            raise ValueError("Invalid graph node")
        if node["node_id"] in ids:
            raise ValueError(f"Duplicate node ID: {node['node_id']}")
        ids.add(node["node_id"])
    for edge in document["edges"]:
        if (
            not isinstance(edge, dict)
            or edge.get("src") not in ids
            or edge.get("dst") not in ids
        ):
            raise ValueError("Graph edge references an unknown node")
    values = {
        k: v for k, v in document.items() if k in AnnotatedGraph.__dataclass_fields__
    }
    values["schema_version"] = version
    if "trace_checks" in document or document.get("trace_checks_status") == "complete":
        from .trace_checks import TRACE_CHECK_NAMES, validate_trace_checks

        checks = document.get("trace_checks", [])
        if checks or document.get("trace_checks_status") == "complete":
            expected_names = TRACE_CHECK_NAMES
            if isinstance(checks, list) and len(checks) == len(
                _LEGACY_TRACE_CHECK_NAMES
            ):
                expected_names = _LEGACY_TRACE_CHECK_NAMES
            values["trace_checks"] = validate_trace_checks(
                checks, document.get("messages", []), expected_names=expected_names
            )
        elif checks != []:
            raise ValueError("Annotation trace_checks must be a list")
    values["grouping"] = GroupingMetadata(**document.get("grouping", {}))
    raw_source = document.get("source")
    values["source"] = None
    if raw_source:
        values["source"] = SourceReference(
            **{**raw_source, "snapshots": tuple(raw_source.get("snapshots", ()))}
        )
    return AnnotatedGraph(**values)


@dataclass(frozen=True)
class MotifAssessment:
    name: str
    kind: str
    family: str
    status: str
    confidence: float | None
    probabilities: dict[str, float]
    refused: bool = False
    explanation: str | None = None
    # Historical reports may contain citations. New explanations use prose only.
    evidence_message_ids: tuple[int, ...] = ()
    explanation_supported: bool | None = None
    selection_probability: float | None = None
    selection_threshold: float | None = None
    selected: bool | None = None

    @property
    def feedback_candidate(self) -> bool:
        """Keep the provider's temporal status separate from threshold selection."""
        return not self.refused and (
            self.selected if self.selected is not None else self.status == "active"
        )


@dataclass(frozen=True)
class EpistemicReport:
    model: str
    assessed_at: str
    message_count: int
    trajectory_sha256: str
    assessments: tuple[MotifAssessment, ...]
    usage: dict[str, Any]
    raw_response: dict[str, Any]
    raw_explanation_response: dict[str, Any] | None = None
    selection_policy: dict[str, Any] | None = None
    # All per-pattern responses, in motif order. The singular field above remains
    # populated for single-request reports and readable in historical reports.
    raw_explanation_responses: tuple[dict[str, Any], ...] = ()
    graph: dict[str, Any] | None = None
    timings_s: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report for offline storage or analysis."""
        return asdict(self)

    @property
    def present_antipatterns(self) -> tuple[MotifAssessment, ...]:
        """Selected unconstructive graph matches eligible for feedback.

        New reports derive selection from graph counts. Historical records keep
        their saved status/support gates; descriptive metrics never hold submission.
        """
        from .patterns import DESCRIPTIVE_ANTIPATTERNS

        return tuple(
            assessment
            for assessment in self.assessments
            if assessment.kind == "antipattern"
            and assessment.name not in DESCRIPTIVE_ANTIPATTERNS
            and assessment.status == "active"
            and assessment.feedback_candidate
            and assessment.explanation_supported is True
            and assessment.explanation
        )

    def format_for_agent(self) -> str:
        """Show only evidenced unconstructive patterns; retain all in storage."""
        present = self.present_antipatterns
        if not present:
            return "No present unconstructive epistemic patterns detected."
        lines = [
            "Graph-matched unconstructive epistemic patterns (check against the observations):"
        ]
        lines.extend(f"- {a.name}: {a.explanation}" for a in present)
        if self.graph is not None:
            # The actor can inspect the annotated relationships behind the
            # signals without receiving raw provider requests or private data.
            lines.append("Annotated graph (fallible labels on the visible trajectory):")
            lines.extend(
                f"{n['node_id']} [{n['type']}, message {n['time']}]: {n['text']}"
                for n in self.graph["nodes"]
            )
            lines.extend(
                f"{e['src']} --{e['relation']}--> {e['dst']}"
                for e in self.graph["edges"]
            )
        return "\n".join(lines)
