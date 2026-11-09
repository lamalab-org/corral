"""Selection tie-break behavior for the corral-mini builder."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from corral_mini_common import Stratum, rank_cost_tie_counts, select


def test_rank_cost_uses_cost_to_break_equal_rank_sets():
    stratum = Stratum(
        tasks=np.array(["a", "b", "c", "d"]),
        passes=np.zeros((4, 2)),
        costs=np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float),
        solved=np.zeros((4, 2), dtype=bool),
    )
    agents = np.arange(2)
    counts = rank_cost_tie_counts(stratum, 2, agents)
    assert counts == {
        "eligible_sets": 6,
        "sets_tied_at_best_rank": 6,
        "sets_tied_after_cost": 1,
        "sets_tied_after_score_error": 1,
    }
    assert list(
        stratum.tasks[select(stratum, 2, "rank_cost", agents, exclude_saturated=True)]
    ) == ["a", "b"]


def test_rank_cost_uses_score_error_when_cost_also_ties():
    stratum = Stratum(
        tasks=np.array(["a", "b", "c", "d"]),
        passes=np.array([[0, 0], [1, 0], [0.4, 0], [1, 1]], dtype=float),
        costs=np.ones((4, 2)),
        solved=np.zeros((4, 2), dtype=bool),
    )
    agents = np.arange(2)
    assert rank_cost_tie_counts(stratum, 1, agents) == {
        "eligible_sets": 4,
        "sets_tied_at_best_rank": 2,
        "sets_tied_after_cost": 2,
        "sets_tied_after_score_error": 1,
    }
    assert list(stratum.tasks[select(stratum, 1, "rank_cost", agents)]) == ["c"]
