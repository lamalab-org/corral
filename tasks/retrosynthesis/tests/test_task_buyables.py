"""Regression tests for task-leaf coverage and level-3 price budgets."""

import json
import runpy
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

import pytest
from retrosynthesis.price_db import lookup_prices

ENVIRONMENTS = Path(__file__).parents[1] / "environments"


def _task(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))[0]


def _reference_leaves(task_number: int) -> list[str]:
    path = ENVIRONMENTS / "level_2" / "tasks_json" / f"make_{task_number}.json"
    return _task(path)["output"][0]["target"]


@pytest.mark.parametrize("task_number", range(1, 11))
def test_all_reference_route_leaves_are_buyable(task_number):
    leaves = _reference_leaves(task_number)
    prices = lookup_prices(leaves)

    assert all(prices[smiles] is not None for smiles in leaves)


@pytest.mark.parametrize("task_number", [1, 2, 3, 4, 9, 10])
def test_tight_budgets_are_ten_percent_above_reference_cost(task_number):
    leaves = _reference_leaves(task_number)
    prices = lookup_prices(leaves)
    reference_cost = sum(prices[smiles]["price_usd_per_g"] for smiles in leaves)
    expected_budget = (Decimal(str(reference_cost)) * Decimal("1.10")).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )

    task_path = ENVIRONMENTS / "level_3" / "tasks_json" / f"make_{task_number}.json"
    actual_budget = _task(task_path)["output"][0]["target"]["prize"]

    assert Decimal(str(actual_budget)) == expected_budget


@pytest.mark.parametrize("level", [2, 3])
def test_existing_generators_include_all_ten_tasks(level, tmp_path):
    directory = ENVIRONMENTS / f"level_{level}"
    generator = runpy.run_path(str(directory / "generate_tasks.py"))
    generator["main"](tmp_path)

    assert len(list(tmp_path.glob("*.json"))) == 10
    for number in (9, 10):
        generated = _task(tmp_path / f"make_{number}.json")
        committed = _task(directory / "tasks_json" / f"make_{number}.json")
        for key in ("id", "input", "output", "scoring_function"):
            assert generated[key] == committed[key]
