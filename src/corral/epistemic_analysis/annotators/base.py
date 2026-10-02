"""Annotators base for epistemic trace analysis."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass(frozen=True)
class AnnotationBackend:
    """Node and edge extraction passes with reproducibility metadata."""

    extract_nodes: Callable[..., Awaitable[tuple[list[dict[str, Any]], list[str]]]]
    extract_edges: Callable[..., Awaitable[tuple[list[dict[str, Any]], list[str]]]]
    provenance: dict[str, Any]

    def resolved_models(self) -> set[str]:
        return _resolved_models.get() or set()


_resolved_models: ContextVar[set[str] | None] = ContextVar(
    "annotation_models", default=None
)


def record_model(model: str | None) -> None:
    if model:
        _resolved_models.set((_resolved_models.get() or set()) | {model})


def reset_models() -> None:
    _resolved_models.set(set())
