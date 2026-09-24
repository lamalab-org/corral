"""Tests for local metering and Corral session state."""

import pytest
from inference_opt.api import BudgetExhausted
from inference_opt.budget import (
    Budget,
    BudgetSpec,
    QuestionAllocator,
    RunRecord,
    StateLedger,
)


def test_budget_charges_student_calls():
    budget = Budget(max_student_calls=2)
    budget.reserve_student_call(2)
    with pytest.raises(BudgetExhausted):
        budget.reserve_student_call()


def test_output_token_ceiling():
    budget = Budget(max_student_calls=10, max_output_tokens=100)
    budget.record_tokens(60)
    with pytest.raises(BudgetExhausted, match="output-token"):
        budget.record_tokens(50)


def test_allocator_protects_each_question():
    allocator = QuestionAllocator(total_calls=100, questions=10, per_question_cap=8)
    allocator.charge("q1", 8)
    assert allocator.allowance("q9") > 0


def test_allocator_enforces_per_question_cap():
    allocator = QuestionAllocator(total_calls=1000, questions=10, per_question_cap=4)
    allocator.charge("q1", 4)
    with pytest.raises(BudgetExhausted, match="per-question cap"):
        allocator.charge("q1", 1)


def test_state_ledger_is_json_shaped_corral_state():
    state = {}
    ledger = StateLedger(state, BudgetSpec(max_experiments=2, max_student_calls=10))
    ledger.reserve(experiments=1, calls=3)
    ledger.mark_revealed(["q1"])
    assert state["experiments"] == 1
    assert state["student_calls"] == 3
    assert state["revealed_ids"] == ["q1"]


def test_state_ledger_keeps_zero_delta_above_negative_delta():
    ledger = StateLedger({}, BudgetSpec())
    ledger.record_run(RunRecord("zero", "experiment", "policy", delta=0.0))
    ledger.record_run(RunRecord("negative", "experiment", "policy", delta=-0.01))
    assert ledger.best_run_id == "zero"
