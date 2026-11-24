"""Probability selection, evidence checks and saved threshold provenance."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.epistemic_analysis.test_decisions import (
    decision_payload,
    explanation_payload,
)

from corral.epistemic_analysis.decision_thresholds import (
    CALIBRATED_PATTERN_THRESHOLDS,
    CALIBRATED_THRESHOLD_POLICY,
    resolve_pattern_thresholds,
)
from corral.epistemic_analysis.decisions import DecisionsAnnotator
from corral.epistemic_analysis.motifs import MOTIFS


@pytest.fixture
def anyio_backend():
    return "asyncio"


def set_probabilities(answer, active, resolved):
    """Keep a valid distribution while deliberately preserving the chosen status."""
    for entry in answer["probabilities"]:
        entry["probability"] = {
            "active": active,
            "resolved": resolved,
            "absent": 1 - active - resolved,
        }.get(entry["value"], 0)


@pytest.mark.anyio
async def test_threshold_selection_changes_candidates_but_preserves_provider_output():
    payload = decision_payload()
    first, rejected, resolved, refused, unsupported = [m.name for m in MOTIFS[:5]]
    payload["answers"][0]["confidence"] = 0.01
    set_probabilities(payload["answers"][0], 0.20, 0.15)
    payload["answers"][1]["choice"] = "active"
    payload["answers"][1]["confidence"] = 0.99
    set_probabilities(payload["answers"][1], 0.20, 0.14)
    payload["answers"][2]["choice"] = "resolved"
    set_probabilities(payload["answers"][2], 0.05, 0.80)
    payload["answers"][3] = {"type": "refusal", "name": refused}
    set_probabilities(payload["answers"][4], 0.20, 0.15)
    thresholds = dict.fromkeys((m.name for m in MOTIFS), 1.0)
    thresholds.update(dict.fromkeys((first, rejected, resolved, unsupported), 0.35))
    thresholds[refused] = 0
    explanations = {
        first: {
            "supported": True,
            "explanation": "The observed test prompted a revision.",
        },
        resolved: {
            "supported": True,
            "explanation": "Exploration led to a hypothesis and its completed test.",
        },
        unsupported: {
            "supported": False,
            "explanation": "The required test is missing.",
        },
    }

    async def respond(**kwargs):
        items = {
            name: explanations[name]
            for name in kwargs["text"]["format"]["schema"]["required"]
        }
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(items),
            model_dump=lambda **kwargs: explanation_payload(items),
        )

    create = AsyncMock(side_effect=respond)
    client = SimpleNamespace(
        post=AsyncMock(return_value=payload), responses=SimpleNamespace(create=create)
    )
    annotator = DecisionsAnnotator(client=client, pattern_thresholds=thresholds)
    thresholds[first] = 1  # Caller mutation must not change the configured policy.
    report = await annotator.annotate(
        [{"role": "assistant", "content": "Observed test and revision."}]
    )
    assert create.await_count == 3
    assert {
        call.kwargs["text"]["format"]["schema"]["required"][0]
        for call in create.await_args_list
    } == {
        first,
        resolved,
        unsupported,
    }
    assert "were classified as active" not in create.await_args.kwargs["instructions"]
    assert report.raw_response == payload
    assert report.assessments[0].status == "absent"
    assert report.assessments[0].selection_probability == 0.35
    assert report.assessments[0].selection_threshold == 0.35
    assert report.assessments[0].selected is True
    assert report.assessments[1].selected is False
    assert report.assessments[2].status == "resolved"
    assert report.assessments[2].selected is True
    # Completion preserves a supported productive occurrence, but never turns
    # it into corrective feedback or changes the provider's temporal status.
    assert report.assessments[2].explanation_supported is True
    assert report.assessments[3].selected is False
    assert report.assessments[3].selection_probability is None
    assert first not in report.format_for_agent()  # Productive feedback is omitted.
    assert all(
        name not in report.format_for_agent()
        for name in (rejected, resolved, refused, unsupported)
    )
    saved = json.loads(json.dumps(report.to_dict()))
    assert saved["selection_policy"]["thresholds"][first] == 0.35
    assert saved["assessments"][0]["selected"] is True
    assert saved["assessments"][2]["status"] == "resolved"
    assert saved["assessments"][2]["explanation_supported"] is True
    assert report.usage["llm_calls"] == 4
    annotator.selection_policy["thresholds"][first] = 0.9
    assert report.selection_policy["thresholds"][first] == 0.35


@pytest.mark.anyio
async def test_zero_threshold_accepts_zero_probability_but_never_a_refusal():
    payload = decision_payload()
    for answer in payload["answers"]:
        set_probabilities(answer, 0, 0)
    refused = "evidence_non_uptake"
    index = next(i for i, m in enumerate(MOTIFS) if m.name == refused)
    payload["answers"][index] = {"type": "refusal", "name": refused}
    report = await DecisionsAnnotator(
        client=SimpleNamespace(post=AsyncMock(return_value=payload)),
        pattern_thresholds=CALIBRATED_THRESHOLD_POLICY,
        include_explanations=False,
    ).annotate([])
    selected = {a.name for a in report.assessments if a.selected}
    assert selected == {
        "uninformative_test",
        "fixed_belief_trace",
        "disconnected_evidence",
    }
    # Selection alone never presents an unsupported accusation to the actor.
    assert (
        report.format_for_agent()
        == "No present unconstructive epistemic patterns detected."
    )
    assert report.selection_policy["name"] == CALIBRATED_THRESHOLD_POLICY
    assert report.selection_policy["calibration"]["paired_trials"] == 328
    assert report.selection_policy["thresholds"] == CALIBRATED_PATTERN_THRESHOLDS


@pytest.mark.anyio
async def test_replayed_failed_ir_assessment_stays_saved_without_current_feedback():
    # Retry job 52090bde96fbaa41, conversation index 25: the assessor chose
    # absent, but active + resolved = .20 cleared the zero selection threshold.
    target = "uninformative_test"
    payload = decision_payload()
    answer = next(a for a in payload["answers"] if a["name"] == target)
    probabilities = {
        "absent": 0.74,
        "active": 0.07,
        "emerging": 0.05,
        "resolved": 0.13,
        "uncertain": 0.01,
    }
    answer["probabilities"] = [
        {"value": name, "probability": probability}
        for name, probability in probabilities.items()
    ]

    async def explain(**kwargs):
        explanations = {
            name: {
                "supported": name == target,
                "explanation": (
                    "The IR test returned only 'IR prediction failed'; the agent "
                    "excluded it from structural inference."
                ),
            }
            for name in kwargs["text"]["format"]["schema"]["required"]
        }
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(explanations),
            model_dump=lambda **_: explanation_payload(explanations),
        )

    report = await DecisionsAnnotator(
        client=SimpleNamespace(
            post=AsyncMock(return_value=payload),
            responses=SimpleNamespace(create=explain),
        ),
        pattern_thresholds=CALIBRATED_THRESHOLD_POLICY,
    ).annotate(
        [
            {
                "role": "assistant",
                "content": "Obtain IR to distinguish the candidates.",
            },
            {"role": "tool", "content": "IR prediction failed"},
            {
                "role": "assistant",
                "content": "IR supplied no evidence; use the independent NMR observations.",
            },
        ]
    )
    saved = next(a for a in report.to_dict()["assessments"] if a["name"] == target)
    assert saved["probabilities"] == probabilities
    assert saved["status"] == "absent"
    assert saved["selection_probability"] == 0.20
    assert saved["selection_threshold"] == 0
    assert saved["selected"] is True
    assert saved["explanation_supported"] is True
    assert report.present_antipatterns == ()
    assert (
        report.format_for_agent()
        == "No present unconstructive epistemic patterns detected."
    )


@pytest.mark.anyio
async def test_each_prefix_is_reassessed_without_sticky_detections():
    present, absent = decision_payload(), decision_payload()
    set_probabilities(present["answers"][0], 0.1, 0.2)
    set_probabilities(absent["answers"][0], 0, 0)
    thresholds = dict.fromkeys((m.name for m in MOTIFS), 1.0)
    thresholds[MOTIFS[0].name] = 0.3
    annotator = DecisionsAnnotator(
        client=SimpleNamespace(post=AsyncMock(side_effect=[present, absent])),
        pattern_thresholds=thresholds,
        include_explanations=False,
    )
    first = await annotator.annotate(
        [{"role": "assistant", "content": "First prefix."}]
    )
    second = await annotator.annotate(
        [{"role": "assistant", "content": "Later prefix."}]
    )
    assert first.assessments[0].selection_probability == 0.3
    assert first.assessments[0].selected is True
    assert second.assessments[0].selected is False
    assert second.assessments[0].selection_probability == 0


@pytest.mark.parametrize(
    "value", [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"]
)
def test_invalid_threshold_values_fail_before_any_request(value):
    thresholds = dict(CALIBRATED_PATTERN_THRESHOLDS)
    thresholds[MOTIFS[0].name] = value
    with pytest.raises(ValueError, match="finite probability"):
        DecisionsAnnotator(pattern_thresholds=thresholds)


@pytest.mark.parametrize("value", [{}, {"invented": 0.5}, "unknown-policy", 0.5])
def test_incomplete_or_unknown_threshold_policies_are_rejected(value):
    with pytest.raises(ValueError):
        DecisionsAnnotator(pattern_thresholds=value)


def test_named_policy_is_detached_and_standalone_default_stays_legacy():
    thresholds, provenance = resolve_pattern_thresholds(CALIBRATED_THRESHOLD_POLICY)
    assert set(thresholds) == {m.name for m in MOTIFS}
    assert thresholds == CALIBRATED_PATTERN_THRESHOLDS
    thresholds[MOTIFS[0].name] = 0
    assert provenance["thresholds"][MOTIFS[0].name] == 0.73
    assert CALIBRATED_PATTERN_THRESHOLDS[MOTIFS[0].name] == 0.73
    assert DecisionsAnnotator().pattern_thresholds is None
