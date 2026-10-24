"""A valid route must also solve the task that was asked."""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from retrosynthesis.env import load_tasks_from_json
from retrosynthesis.route_contract import validate_contract
from retrosynthesis.score import check_reactants, score_final, score_final_without_price

ROOT = Path(__file__).resolve().parents[1] / "environments"
CONTRACT = {"product_smiles": "CCO", "min_reactions": 1, "max_reactions": 1}


def route(product="CCO"):
    return json.dumps(
        {
            "type": "mol",
            "smiles": product,
            "children": [
                {
                    "type": "reaction",
                    "template_id": 123,
                    "children": [{"type": "mol", "smiles": "CC=O"}],
                }
            ],
        }
    )


@pytest.mark.parametrize("scorer", [check_reactants, score_final_without_price])
def test_unreacted_expected_precursor_cannot_solve_the_task(scorer):
    answer = json.dumps({"type": "mol", "smiles": "CC=O"})
    assert scorer(answer, ["CC=O"], route_constraints=CONTRACT) == 0.0


def test_unrelated_cheap_molecule_rejected_before_price_lookup():
    answer = json.dumps({"type": "mol", "smiles": "C"})
    with patch("retrosynthesis.score.check_price") as prices:
        assert (
            score_final(
                answer,
                {"prize": 165, "max_steps": 1},
                route_constraints={"product_smiles": "CCO"},
            )
            == 0.0
        )
        prices.assert_not_called()


def test_wrong_target_rejected_even_with_a_valid_catalog_reaction():
    with patch("retrosynthesis.retrosynthesis_utils.search_by_template") as catalog:
        catalog.return_value = {
            "mapped_rxn": "[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]"
        }
        constraints = {**CONTRACT, "product_smiles": "CCCO"}
        assert check_reactants(route(), ["CC=O"], route_constraints=constraints) == 0.0


@pytest.mark.parametrize("product", ["CCO", "OCC", "[CH3:10][CH2:20][OH:30]"])
def test_correct_product_and_real_reaction_still_pass(product):
    with patch("retrosynthesis.retrosynthesis_utils.search_by_template") as catalog:
        catalog.return_value = {
            "mapped_rxn": "[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]"
        }
        assert (
            check_reactants(route(product), ["CC=O"], route_constraints=CONTRACT) == 1.0
        )
        with patch(
            "retrosynthesis.score.check_price", return_value={"CC=O": [{"Price": 2}]}
        ):
            assert (
                score_final(
                    route(product),
                    {"prize": 2, "max_steps": 1},
                    route_constraints=CONTRACT,
                )
                == 1.0
            )


def test_exact_reaction_count_is_enforced():
    constraints = {**CONTRACT, "min_reactions": 2, "max_reactions": 2}
    with patch("retrosynthesis.score.validate_reactions_with_products") as validate:
        assert check_reactants(route(), ["CC=O"], route_constraints=constraints) == 0.0
        validate.assert_not_called()


def test_configuration_errors_are_not_scored_as_model_failures():
    with pytest.raises(ValueError, match="Inconsistent"):
        check_reactants(
            route(), ["CC=O"], route_constraints={**CONTRACT, "min_reactions": 3}
        )


def test_all_shipped_route_contracts_match_existing_questions():
    checked = 0
    for file in ROOT.glob("*/tasks_json/*.json"):
        for task in json.loads(file.read_text()):
            if task["scoring_function"] not in {
                "check_reactants",
                "score_final",
                "score_final_without_price",
            }:
                continue
            contract = task["route_constraints"]
            validate_contract(contract)
            prompt = task["input"]["prompt"]
            assert f"with SMILES {contract['product_smiles']}." in prompt
            if task["scoring_function"] == "score_final":
                assert (
                    contract["max_reactions"]
                    == task["output"][0]["target"]["max_steps"]
                )
                assert "min_reactions" not in contract
            else:
                steps = re.search(r"must have (at least )?(\d+) reactions", prompt)
                assert steps
                assert contract["min_reactions"] == int(steps[2])
                assert contract.get("max_reactions") == (
                    None if steps[1] else int(steps[2])
                )
            checked += 1
    assert checked == 28


def test_loader_binds_product_constraint_and_fails_closed_if_missing(tmp_path):
    original = json.loads((ROOT / "level_3/tasks_json/make_1.json").read_text())
    file = tmp_path / "task.json"
    file.write_text(json.dumps(original))
    task = load_tasks_from_json(tmp_path)[original[0]["id"]]
    with patch("retrosynthesis.score.check_price") as prices:
        assert task.scoring_fn(json.dumps({"type": "mol", "smiles": "C"})) == 0.0
        prices.assert_not_called()
    del original[0]["route_constraints"]
    file.write_text(json.dumps(original))
    with pytest.raises(KeyError, match="route_constraints"):
        load_tasks_from_json(tmp_path)


def test_assembled_subtask_routes_keep_the_requested_product():
    checked = 0
    for file in ROOT.glob("*/subtasks_json/*.json"):
        main = json.loads((file.parent.parent / "tasks_json" / file.name).read_text())[
            0
        ]
        tasks = load_tasks_from_json(file.parent)
        for task in json.loads(file.read_text()):
            if task["scoring_function"] != "check_reactants":
                continue
            assert task["route_constraints"] == {
                "product_smiles": main["route_constraints"]["product_smiles"]
            }
            with patch("retrosynthesis.score.check_price") as prices:
                assert (
                    tasks[task["id"]].scoring_fn(
                        json.dumps({"type": "mol", "smiles": "C"})
                    )
                    == 0.0
                )
                prices.assert_not_called()
            checked += 1
    assert checked == 16
