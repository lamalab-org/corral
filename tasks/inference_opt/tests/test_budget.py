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


def test_the_best_run_is_chosen_by_the_tasks_own_metric():
    """On level 2 a run with better accuracy but worse consistency is not the best."""
    ledger = StateLedger({}, BudgetSpec())
    ledger.record_run(RunRecord("exp-1", "experiment", "policy", delta=0.10, target_delta=0.00))
    ledger.record_run(RunRecord("exp-2", "experiment", "policy", delta=0.02, target_delta=0.07))
    ledger.record_run(RunRecord("exp-3", "experiment", "policy", delta=0.20, target_delta=-0.03))
    assert ledger.best_run_id == "exp-2"


def test_a_run_recorded_without_a_target_delta_is_ranked_by_its_delta():
    ledger = StateLedger({}, BudgetSpec())
    ledger.record_run(RunRecord("exp-1", "experiment", "policy", delta=0.10))
    ledger.record_run(RunRecord("exp-2", "experiment", "policy", delta=0.05, target_delta=0.05))
    assert ledger.best_run_id == "exp-1"


# -- one budget shared by a trial's branches ----------------------------------


def _branches(tmp_path, spec, parent_state=None, count=2):
    """Ledgers for ``count`` branches forked from one parent state."""
    from copy import deepcopy

    from inference_opt.budget import SharedLedger

    parent = parent_state or {}
    StateLedger(parent, spec)  # fill in the defaults, as the main session would
    path = tmp_path / "shared_ledger.json"
    return [
        StateLedger(deepcopy(parent), spec, SharedLedger(path, branch=f"b{i}"))
        for i in range(count)
    ]


def test_branches_draw_on_one_budget(tmp_path):
    first, second = _branches(tmp_path, BudgetSpec(max_student_calls=10))
    first.reserve(calls=6)
    assert second.remaining()["student_calls"] == 4
    with pytest.raises(BudgetExhausted, match="across all branches"):
        second.reserve(calls=5)
    second.reserve(calls=4)
    assert first.state["student_calls"] == 6
    assert second.state["student_calls"] == 4


def test_a_branch_refund_returns_calls_to_every_branch(tmp_path):
    first, second = _branches(tmp_path, BudgetSpec(max_student_calls=10, max_experiments=2))
    first.reserve(experiments=1, calls=10)
    first.refund(calls=7)
    assert second.remaining()["student_calls"] == 7
    assert second.remaining()["experiments"] == 1


def test_branches_share_reveals_and_see_each_others_examples(tmp_path):
    first, second = _branches(tmp_path, BudgetSpec(max_reveals=1))
    first.reserve(reveals=1)
    first.mark_revealed(["q1", "q2"])
    with pytest.raises(BudgetExhausted, match="reveals"):
        second.reserve(reveals=1)
    assert second.revealed_ids == ["q1", "q2"]
    assert second.state["revealed_ids"] == []


def test_run_ids_are_unique_across_branches_and_runs_are_pooled(tmp_path):
    first, second = _branches(tmp_path, BudgetSpec())
    assert first.next_run_id("exp") == "exp-1"
    assert second.next_run_id("exp") == "exp-2"
    second.record_run(RunRecord(run_id="exp-2", kind="experiment", policy_dir="p"))
    pooled = second.shared.read(second.state)
    assert [(run["run_id"], run["branch"]) for run in pooled["runs"]] == [("exp-2", "b1")]


def test_the_shared_ledger_starts_from_what_the_parent_had_used(tmp_path):
    parent = {"student_calls": 8, "revealed_ids": ["q1"]}
    (branch,) = _branches(tmp_path, BudgetSpec(max_student_calls=10), parent, count=1)
    assert branch.remaining()["student_calls"] == 2
    assert branch.revealed_ids == ["q1"]


def test_concurrent_branches_never_overspend(tmp_path):
    import threading

    branches = _branches(tmp_path, BudgetSpec(max_student_calls=50), count=8)
    granted = []

    def spend(ledger):
        for _ in range(20):
            try:
                ledger.reserve(calls=1)
                granted.append(1)
            except BudgetExhausted:
                pass

    threads = [threading.Thread(target=spend, args=(ledger,)) for ledger in branches]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(granted) == 50
    assert branches[0].remaining()["student_calls"] == 0


def test_without_a_shared_ledger_branches_are_independent():
    from copy import deepcopy

    parent = {}
    StateLedger(parent, BudgetSpec(max_student_calls=10))
    first = StateLedger(deepcopy(parent), BudgetSpec(max_student_calls=10))
    second = StateLedger(deepcopy(parent), BudgetSpec(max_student_calls=10))
    first.reserve(calls=10)
    second.reserve(calls=10)
    assert "shared_across_branches" not in first.snapshot()
