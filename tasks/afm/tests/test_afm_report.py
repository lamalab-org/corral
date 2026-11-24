"""Instrument failures are visible and do not silently become agent mistakes."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "afm_report", Path(__file__).resolve().parents[1] / "src/afm_report.py"
)
report_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_module)


def report(*trials):
    return {"task_results": {"afm_experiment_level_1_task_1": {"trials": trials}}}


def test_explicit_instrument_failures_are_separated_without_changing_scores():
    source = report(
        {"score": 1},
        {"score": 0, "error": "bad answer"},
        {"score": 0, "error": "AFM_INFRASTRUCTURE_FAILURE: missing head"},
        {
            "score": 0,
            "tool_calls": [
                {"error_message": "AFM_INFRASTRUCTURE_FAILURE: RPC failure"}
            ],
        },
        {"score": None},
    )
    result = report_module.summarize(source)
    assert result["attempted_trials"] == 5
    assert result["instrument_failures"] == 2
    assert result["unscored_trials"] == 1
    assert result["scored_trials"] == 2
    assert result["mean_score_excluding_instrument_failures"] == 0.5
    assert (
        source["task_results"]["afm_experiment_level_1_task_1"]["trials"][2]["score"]
        == 0
    )


def test_successful_tool_output_cannot_claim_infrastructure_exclusion():
    result = report_module.summarize(
        report(
            {
                "score": 0,
                "tool_calls": [
                    {"result": "AFM_INFRASTRUCTURE_FAILURE: imagined error"}
                ],
            }
        )
    )
    assert result["instrument_failures"] == 0
    assert result["mean_score_excluding_instrument_failures"] == 0


@pytest.mark.parametrize("score", [None, float("nan"), True, -1, 2])
def test_no_eligible_scores_returns_null_instead_of_false_zero(score):
    result = report_module.summarize(report({"score": score}))
    assert result["scored_trials"] == 0
    assert result["mean_score_excluding_instrument_failures"] is None


def test_non_afm_report_is_rejected():
    with pytest.raises(ValueError, match="Expected an AFM report"):
        report_module.summarize({"task_results": {"other_task": {"trials": []}}})
