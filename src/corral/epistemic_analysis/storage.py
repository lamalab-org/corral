"""Storage for epistemic trace analysis."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_annotation_artifacts(graph, output: Path) -> None:
    """Write portable views and exact Jev exchanges for inspection."""
    from .schema import content_hash

    if graph.annotation_view is not None:
        digest = content_hash(graph.annotation_view)
        artifact = f"annotation_views/{digest}.json"
        graph.provenance["annotation_view_hash"] = digest
        graph.provenance["annotation_view_artifact"] = artifact
        safe_write_json(output / artifact, graph.annotation_view)
    if graph.annotation_requests:
        digest = content_hash(graph.annotation_requests)
        artifact = f"annotation_requests/{digest}.json"
        graph.provenance["annotation_requests_hash"] = digest
        graph.provenance["annotation_requests_artifact"] = artifact
        safe_write_json(output / artifact, graph.annotation_requests)
