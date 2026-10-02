"""Provider-independent contracts and serialization for epistemic analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict

GRAPH_SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "corral-messages-v1"
ANNOTATION_VERSION = "4"
ANALYSIS_VERSION = "2"
PATTERN_VERSION = "1"


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
    window: int = 2
    overlap: int = 1
    max_nodes_per_window: int = 100
    strict_support: bool = True
    tool_output_max_chars: int = 4000
    tool_output_truncation: str = "head-tail"
    max_request_tokens: int = 0
    # Used only by the generative backend when budgeting is enabled.
    llm_output_reserve_tokens: int = 4096

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("An annotation model is required")
        for name in (
            "tool_output_max_chars",
            "max_request_tokens",
            "llm_output_reserve_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if 0 < self.tool_output_max_chars < 16:
            raise ValueError("tool_output_max_chars must be zero or at least 16")
        if self.tool_output_truncation != "head-tail":
            raise ValueError("tool_output_truncation must be head-tail")
        if self.llm_output_reserve_tokens < 1:
            raise ValueError("llm_output_reserve_tokens must be positive")
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
    # Optional in v1 graphs; embedded for portable, provider-free reanalysis.
    annotation_view: dict[str, Any] | None = None
    # Exact Jev requests and responses, including absent decisions; no auth headers.
    annotation_requests: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    row: dict[str, Any]
    source: SourceReference | None
    grouping: GroupingMetadata
    configuration_id: str
    annotation_configuration: dict[str, Any]
    status: Status
    eligible: bool
    analysis_version: str = field(default_factory=lambda: ANALYSIS_VERSION)
    pattern_version: str = field(default_factory=lambda: PATTERN_VERSION)


@dataclass
class AggregateResult:
    configurations: dict[str, dict[str, Any]]
    n_selected: int
    analysis_version: str = field(default_factory=lambda: ANALYSIS_VERSION)
    pattern_version: str = field(default_factory=lambda: PATTERN_VERSION)

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
    for key in ("completion_status", "reconstruction_status", "annotation_status"):
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
    values["grouping"] = GroupingMetadata(**document.get("grouping", {}))
    raw_source = document.get("source")
    values["source"] = None
    if raw_source:
        values["source"] = SourceReference(
            **{**raw_source, "snapshots": tuple(raw_source.get("snapshots", ()))}
        )
    return AnnotatedGraph(**values)
