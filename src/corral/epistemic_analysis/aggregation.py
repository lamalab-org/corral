"""Aggregation for epistemic trace analysis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from corral.epistemic_analysis.patterns import (
    ANTIPATTERN_FAMILY_NAMES,
    ANTIPATTERN_NAMES,
    SUBGRAPH_FAMILY_NAMES,
    SUBGRAPH_NAMES,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .schema import AggregateResult, AnalysisResult


def _mean(values: list[float | None]) -> float | None:
    clean = [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    return sum(clean) / len(clean) if clean else None


def compute_aggregate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    subgraph_local: dict[str, dict[str, Any]] = {}
    subgraph_global: dict[str, dict[str, Any]] = {}
    for sg in SUBGRAPH_NAMES:
        loc = [bool(r.get(f"sg_local_{sg}")) for r in rows]
        loc_raw = [int(r.get(f"sg_local_count_{sg}", 0)) for r in rows]
        glb = [bool(r.get(f"sg_global_{sg}")) for r in rows]
        glb_raw = [int(r.get(f"sg_global_count_{sg}", 0)) for r in rows]
        subgraph_local[sg] = {
            "count": sum(loc),
            "fraction": sum(loc) / n if n else None,
            "raw_total": sum(loc_raw),
            "raw_mean": _mean([float(v) for v in loc_raw]),
        }
        subgraph_global[sg] = {
            "count": sum(glb),
            "fraction": sum(glb) / n if n else None,
            "raw_total": sum(glb_raw),
            "raw_mean": _mean([float(v) for v in glb_raw]),
        }

    antipattern_local: dict[str, dict[str, Any]] = {}
    antipattern_global: dict[str, dict[str, Any]] = {}
    for ap in ANTIPATTERN_NAMES:
        loc = [bool(r.get(f"ap_local_{ap}")) for r in rows]
        loc_raw = [int(r.get(f"ap_local_count_{ap}", 0)) for r in rows]
        glb = [bool(r.get(f"ap_global_{ap}")) for r in rows]
        glb_raw = [int(r.get(f"ap_global_count_{ap}", 0)) for r in rows]
        antipattern_local[ap] = {
            "count": sum(loc),
            "fraction": sum(loc) / n if n else None,
            "raw_total": sum(loc_raw),
            "raw_mean": _mean([float(v) for v in loc_raw]),
        }
        antipattern_global[ap] = {
            "count": sum(glb),
            "fraction": sum(glb) / n if n else None,
            "raw_total": sum(glb_raw),
            "raw_mean": _mean([float(v) for v in glb_raw]),
        }

    ap_family_local: dict[str, dict[str, Any]] = {}
    ap_family_global: dict[str, dict[str, Any]] = {}
    for fam_name in ANTIPATTERN_FAMILY_NAMES:
        loc = [bool(r.get(f"ap_family_local_{fam_name}")) for r in rows]
        loc_raw = [int(r.get(f"ap_family_local_count_{fam_name}", 0)) for r in rows]
        glb = [bool(r.get(f"ap_family_global_{fam_name}")) for r in rows]
        glb_raw = [int(r.get(f"ap_family_global_count_{fam_name}", 0)) for r in rows]
        ap_family_local[fam_name] = {
            "count": sum(loc),
            "fraction": sum(loc) / n if n else None,
            "raw_total": sum(loc_raw),
            "raw_mean": _mean([float(v) for v in loc_raw]),
        }
        ap_family_global[fam_name] = {
            "count": sum(glb),
            "fraction": sum(glb) / n if n else None,
            "raw_total": sum(glb_raw),
            "raw_mean": _mean([float(v) for v in glb_raw]),
        }

    sg_family_local: dict[str, dict[str, Any]] = {}
    sg_family_global: dict[str, dict[str, Any]] = {}
    for fam_name in SUBGRAPH_FAMILY_NAMES:
        loc = [bool(r.get(f"sg_family_local_{fam_name}")) for r in rows]
        loc_raw = [int(r.get(f"sg_family_local_count_{fam_name}", 0)) for r in rows]
        glb = [bool(r.get(f"sg_family_global_{fam_name}")) for r in rows]
        glb_raw = [int(r.get(f"sg_family_global_count_{fam_name}", 0)) for r in rows]
        sg_family_local[fam_name] = {
            "count": sum(loc),
            "fraction": sum(loc) / n if n else None,
            "raw_total": sum(loc_raw),
            "raw_mean": _mean([float(v) for v in loc_raw]),
        }
        sg_family_global[fam_name] = {
            "count": sum(glb),
            "fraction": sum(glb) / n if n else None,
            "raw_total": sum(glb_raw),
            "raw_mean": _mean([float(v) for v in glb_raw]),
        }

    return {
        "n_traces": n,
        "subgraph_presence_local": subgraph_local,
        "subgraph_presence_global": subgraph_global,
        "antipattern_presence_local": antipattern_local,
        "antipattern_presence_global": antipattern_global,
        "antipattern_family_local": ap_family_local,
        "antipattern_family_global": ap_family_global,
        "subgraph_family_local": sg_family_local,
        "subgraph_family_global": sg_family_global,
    }


def aggregate(
    results: Iterable[AnalysisResult],
    group_by: tuple[str, ...] = ("agent_model", "environment", "level"),
) -> AggregateResult:
    """Combine an explicit result set; partition all annotation configurations.

    Failed, missing, incomplete, unsupported and empty graphs are excluded from
    measurement denominators, with counts and source references retained.
    Different snapshots of one source must be selected in separate invocations.
    """
    from collections import Counter, defaultdict
    from dataclasses import asdict

    from .schema import AggregateResult, GroupingMetadata

    if set(group_by) - set(GroupingMetadata.__dataclass_fields__):
        raise ValueError("Unknown grouping field")
    selected = list(results)
    measurement_versions = {(r.analysis_version, r.pattern_version) for r in selected}
    if len(measurement_versions) > 1:
        raise ValueError(
            "Cannot aggregate different analysis or pattern versions; reanalyze the selected graphs"
        )
    versions = {}
    seen = set()
    configurations = defaultdict(list)
    for result in selected:
        if result.source:
            identity = result.source.identity
            version = (result.source.content_hash, result.source.normalization_version)
            if identity in versions and versions[identity] != version:
                raise ValueError(
                    f"Multiple snapshots selected for {identity}; select one snapshot explicitly"
                )
            versions[identity] = version
            key = (identity, version, result.configuration_id)
            if key in seen:
                continue
            seen.add(key)
        configurations[result.configuration_id].append(result)

    def statistics(items):
        return {
            **compute_aggregate_stats([r.row for r in items if r.eligible]),
            "n_selected": len(items),
            "n_excluded": sum(not r.eligible for r in items),
            "status_counts": dict(Counter(r.status for r in items)),
            "n_reduced_context": sum(bool(r.row.get("reduced_context")) for r in items),
            "clipped_outputs": sum(r.row.get("clipped_outputs", 0) for r in items),
            "retained_tool_chars": sum(
                r.row.get("retained_tool_chars") or 0 for r in items
            ),
            "omitted_tool_chars": sum(
                r.row.get("omitted_tool_chars", 0) for r in items
            ),
        }

    partitions = {}
    for identity, items in sorted(configurations.items()):
        groups = defaultdict(list)
        for item in items:
            groups[tuple(getattr(item.grouping, key) for key in group_by)].append(item)
        partitions[identity] = {
            "annotation_configuration": items[0].annotation_configuration,
            "overall": statistics(items),
            "groups": [
                {
                    "metadata": dict(zip(group_by, key, strict=False)),
                    "statistics": statistics(group),
                }
                for key, group in sorted(groups.items(), key=lambda item: str(item[0]))
            ],
            "traces": [
                {
                    "source": asdict(r.source) if r.source else None,
                    "grouping": asdict(r.grouping),
                    "status": r.status,
                    "eligible": r.eligible,
                    "statistics": r.row,
                }
                for r in items
            ],
        }
    summary = AggregateResult(
        configurations=partitions,
        n_selected=sum(len(items) for items in configurations.values()),
    )
    if selected:
        summary.analysis_version, summary.pattern_version = next(
            iter(measurement_versions)
        )
    return summary
