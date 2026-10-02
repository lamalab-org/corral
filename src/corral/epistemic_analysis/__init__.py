"""Epistemic annotation and analysis, independent of trace sources and providers."""

from .aggregation import aggregate
from .analysis import analyze_graph
from .annotators.base import AnnotationBackend
from .pipeline import annotate_trace
from .schema import (
    AggregateResult,
    AnalysisResult,
    AnnotatedGraph,
    AnnotationConfig,
    GroupingMetadata,
    NormalizedTrace,
    SourceReference,
    read_graph,
)

__all__ = [
    "AggregateResult",
    "AnalysisResult",
    "AnnotatedGraph",
    "AnnotationBackend",
    "AnnotationConfig",
    "GroupingMetadata",
    "NormalizedTrace",
    "SourceReference",
    "aggregate",
    "analyze_graph",
    "annotate_trace",
    "read_graph",
]
