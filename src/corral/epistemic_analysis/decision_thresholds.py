"""Versioned offline probability thresholds for Decisions candidate selection.

These are fixed settings, not an online calibration step. They were fitted on
328 completed trials, pooled across agents, with 16-message windows/4 overlap.
The epistemic agent applies them to each full observed prefix. These historical
occurrence thresholds do not calibrate current submission risk: feedback also
requires an active classifier status and an evidence-supported unresolved defect.
Descriptive trace properties remain stored but do not hold submissions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from corral.epistemic_analysis.motifs import DESCRIPTIVE_ANTIPATTERNS, MOTIFS

CALIBRATED_THRESHOLD_POLICY = "positive-mass-20261020-v1"

# P(active) + P(resolved) >= threshold. Zero deliberately accepts even zero
# probability. Selection alone cannot bypass the active-status, descriptive-
# pattern and independent evidence checks on feedback.
CALIBRATED_PATTERN_THRESHOLDS = {
    "refutation_driven_belief_revision": 0.73,
    "fixed_hypothesis_test_tuning": 0.14,
    "explore_then_test_transition": 0.32,
    "hypothesis_reranking": 0.51,
    "evidence_led_hypothesis_generation": 0.04,
    "convergent_multi_test_evidence": 0.08,
    "evidence_guided_test_redesign": 0.38,
    "untested_claim": 0.15,
    "evidence_non_uptake": 0.0,
    "unsupported_judgment": 0.18,
    "stalled_revision": 0.39,
    "contradiction_without_repair": 0.06,
    "premature_commitment": 0.32,
    "uninformative_test": 0.0,
    "fixed_belief_trace": 0.0,
    "disconnected_evidence": 0.0,
    "one_sided_confirmation": 0.23,
    "precommitted_test_plan": 0.10,
}


def resolve_pattern_thresholds(
    value: str | Mapping[str, float] | None,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Validate and detach settings so callers cannot mutate a running policy."""
    feedback_policy = {
        "feedback_gate_version": "current-unresolved-v2",
        "feedback_gate": (
            "choice == active; descriptive antipatterns excluded; explanation "
            "supports a current unresolved defect with message evidence"
        ),
        "descriptive_antipatterns": sorted(DESCRIPTIVE_ANTIPATTERNS),
    }
    if value is None:
        return None, {
            "name": "active-status-v1",
            "rule": "choice == active",
            **feedback_policy,
        }
    calibration = None
    if isinstance(value, str):
        if value != CALIBRATED_THRESHOLD_POLICY:
            raise ValueError(f"Unknown Decisions threshold policy: {value!r}")
        name = value
        thresholds = dict(CALIBRATED_PATTERN_THRESHOLDS)
        calibration = {
            "analysis": "decisions_threshold_analysis_20261020",
            "model": "gpt-6-luna",
            "paired_trials": 328,
            "window": 16,
            "overlap": 4,
            "agents": "pooled",
            "snapshot_sha256": "aaf0ebb64f7fd9b4e790a0c9546801d6202a0b9a643604d2d7707bdbeddabd3c",
            "objective": "mean positive-class macro F1 against both graph annotators",
        }
    elif isinstance(value, Mapping):
        name = "custom-positive-mass-v1"
        thresholds = dict(value)
    else:
        raise ValueError("pattern_thresholds must be a policy name, mapping, or None")
    if set(thresholds) != {motif.name for motif in MOTIFS}:
        raise ValueError(
            "pattern_thresholds must specify exactly the 18 registered motifs"
        )
    if any(
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        for threshold in thresholds.values()
    ):
        raise ValueError(
            "Every pattern threshold must be a finite probability in [0, 1]"
        )
    thresholds = {key: float(threshold) for key, threshold in thresholds.items()}
    provenance = {
        "name": name,
        "rule": "round(P(active) + P(resolved), 12) >= threshold; refusals excluded",
        "scope": "current supplied prefix; no accumulation of previous reports",
        **feedback_policy,
        "thresholds": dict(thresholds),
    }
    if calibration is not None:
        provenance["calibration"] = calibration
    return thresholds, provenance
