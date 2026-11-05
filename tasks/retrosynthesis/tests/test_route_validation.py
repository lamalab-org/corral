"""Regressions for passing routes whose templates do not connect their molecules."""

import json
from unittest.mock import patch

import psycopg2
import pytest

from retrosynthesis.retrosynthesis_utils import validate_route_reactions
from retrosynthesis.score import check_reactants, score_final, score_final_without_price
from retrosynthesis.tools import verify_route


def route(template_id=123, precursors=("CC=O",), product="CCO"):
    return {
        "type": "mol", "smiles": product,
        "children": [{
            "type": "reaction", "template_id": template_id,
            "children": [{"type": "mol", "smiles": s} for s in precursors],
        }],
    }


@pytest.fixture
def catalog():
    # Real template extraction/application, with only database access replaced.
    with patch("retrosynthesis.retrosynthesis_utils.search_by_template") as lookup:
        lookup.return_value = {
            "mapped_rxn": "[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]"
        }
        yield lookup


def test_real_catalog_reaction_passes_all_validators(catalog):
    answer = json.dumps(route())
    assert validate_route_reactions(route())[0]
    assert check_reactants(answer, ["CC=O"]) == 1.0
    assert score_final_without_price(answer, ["CC=O"]) == 1.0
    with patch("retrosynthesis.score.check_price", return_value={"CC=O": [{"Price": 2}]}):
        assert score_final(answer, {"prize": 2, "max_steps": 1}) == 1.0
    result = verify_route.execute(route=answer)
    assert result.startswith("(True,")
    assert "catalog reaction steps" in result
    assert "have not been checked" in result


@pytest.mark.parametrize("precursors", [("C",), ("CC=O", "O"), ("CC=O", "CC=O")])
def test_correct_expected_leaves_cannot_hide_wrong_reaction(catalog, precursors):
    answer = json.dumps(route(precursors=precursors))
    assert check_reactants(answer, list(precursors)) == 0.0
    assert score_final_without_price(answer, list(precursors)) == 0.0
    with patch("retrosynthesis.score.check_price") as prices:
        assert score_final(answer, {"prize": 100, "max_steps": 1}) == 0.0
        prices.assert_not_called()
    assert verify_route.execute(route=answer).startswith("(False,")


def test_checks_nested_reactions_after_valid_root():
    tree = route()
    tree["children"][0]["children"][0] = route(456, ("C",), "CC=O")
    with patch("retrosynthesis.retrosynthesis_utils.apply_template_retro") as apply:
        apply.side_effect = [(("CC=O",),), ()]
        valid, error = validate_route_reactions(tree)
    assert not valid
    assert "root.children[0].children[0].children[0]" in error
    assert "456" in error
    assert apply.call_count == 2


@pytest.mark.parametrize("template_id", [0, -1, True, "123", "CCO>>CC=O"])
def test_rejects_placeholder_and_non_catalog_identifiers(template_id):
    with patch("retrosynthesis.retrosynthesis_utils.search_by_template") as lookup:
        assert not validate_route_reactions(route(template_id))[0]
        lookup.assert_not_called()


def test_missing_catalog_entry_fails(catalog):
    catalog.return_value = None
    valid, error = validate_route_reactions(route())
    assert not valid
    assert "not found in database" in error


def test_template_that_does_not_apply_fails(catalog):
    valid, error = validate_route_reactions(route(product="c1ccccc1"))
    assert not valid
    assert "does not apply" in error


def test_atom_maps_and_explicit_empty_leaf_children_are_supported(catalog):
    tree = route(precursors=("[CH3:21][CH:22]=[O:23]",))
    tree["children"][0]["children"][0]["children"] = []
    assert validate_route_reactions(tree)[0]


def test_missing_reactant_is_rejected_even_if_remaining_leaf_matches():
    with patch("retrosynthesis.retrosynthesis_utils.apply_template_retro") as apply:
        apply.return_value = (("CC=O", "O"),)
        assert not validate_route_reactions(route())[0]


@pytest.mark.parametrize("bad_node", [
    {"type": "mol", "smiles": "invalid"},
    {"type": "reaction", "template_id": 123, "children": []},
    {"type": "mol", "smiles": "C", "children": [{"type": "mol", "smiles": "CC=O"}]},
])
def test_malformed_nodes_cannot_bypass_reaction_checks(bad_node):
    assert not validate_route_reactions(bad_node)[0]


def test_database_outage_is_not_a_failed_synthesis(catalog):
    catalog.side_effect = psycopg2.OperationalError("catalog unavailable")
    answer = json.dumps(route())
    with pytest.raises(psycopg2.OperationalError):
        check_reactants(answer, ["CC=O"])
    with pytest.raises(psycopg2.OperationalError):
        score_final(answer, {"prize": 100, "max_steps": 1})
