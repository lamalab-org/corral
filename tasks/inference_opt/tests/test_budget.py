"""Tests for the Corral-committed task ledger."""

import pytest
from inference_opt.api import BudgetExhausted
from inference_opt.budget import BudgetSpec, RunRecord, StateLedger


def test_state_ledger_is_json_shaped_corral_state():
    state = {}
    ledger = StateLedger(state, BudgetSpec(max_experiments=2, max_student_calls=10))
    ledger.reserve(experiments=1, calls=3)
    ledger.mark_revealed(["q1"])
    assert state["experiments"] == 1
    assert state["student_calls"] == 3
    assert state["revealed_ids"] == ["q1"]
    assert state["submissions"] == 0 and state["final"] is None


def test_reserving_past_a_limit_is_refused():
    ledger = StateLedger({}, BudgetSpec(max_student_calls=5))
    ledger.reserve(calls=5)
    with pytest.raises(BudgetExhausted, match="student calls"):
        ledger.reserve(calls=1)


def test_a_refund_returns_what_a_run_did_not_use():
    ledger = StateLedger({}, BudgetSpec(max_student_calls=10))
    ledger.reserve(experiments=1, calls=10)
    ledger.refund(calls=4, experiments=1)
    assert ledger.remaining()["student_calls"] == 4
    assert ledger.experiments == 0


def test_submissions_left_are_reported():
    ledger = StateLedger({"submissions": 1}, BudgetSpec(max_submissions=3))
    assert ledger.remaining()["submissions"] == 2


def test_state_ledger_keeps_zero_delta_above_negative_delta():
    ledger = StateLedger({}, BudgetSpec())
    ledger.record_run(RunRecord("zero", "experiment", "policy", delta=0.0))
    ledger.record_run(RunRecord("negative", "experiment", "policy", delta=-0.01))
    assert ledger.best_run_id == "zero"
