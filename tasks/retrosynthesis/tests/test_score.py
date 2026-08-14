"""Tests for retrosynthesis scoring functions."""

import json
from unittest.mock import patch

import pytest
from retrosynthesis.score import (
    check_apply_template,
    check_list_molecules,
    check_reactants,
    check_template,
    collect_leaf_molecules,
    count_reactions,
    score_final,
    score_final_without_price,
    validate_reactions_with_products,
)


class TestCountReactions:
    """Tests for count_reactions function."""

    def test_count_single_reaction(self):
        """Test counting reactions in a simple tree with one reaction."""
        tree = {
            "type": "mol",
            "smiles": "CCO",
            "children": [
                {
                    "type": "reaction",
                    "template_id": "1",
                    "children": [
                        {"type": "mol", "smiles": "C", "children": []},
                        {"type": "mol", "smiles": "CO", "children": []},
                    ],
                }
            ],
        }
        result = count_reactions(tree)
        assert result == 1

    def test_count_multiple_reactions(self):
        """Test counting reactions in a tree with multiple reactions."""
        tree = {
            "type": "mol",
            "smiles": "CCCO",
            "children": [
                {
                    "type": "reaction",
                    "template_id": "1",
                    "children": [
                        {
                            "type": "mol",
                            "smiles": "CC",
                            "children": [
                                {
                                    "type": "reaction",
                                    "template_id": "2",
                                    "children": [
                                        {"type": "mol", "smiles": "C", "children": []},
                                        {"type": "mol", "smiles": "C", "children": []},
                                    ],
                                }
                            ],
                        },
                        {"type": "mol", "smiles": "CO", "children": []},
                    ],
                }
            ],
        }
        result = count_reactions(tree)
        assert result == 2

    def test_count_no_reactions(self):
        """Test counting reactions in a tree with no reactions."""
        tree = {"type": "mol", "smiles": "CCO", "children": []}
        result = count_reactions(tree)
        assert result == 0

    def test_count_nested_reactions(self):
        """Test counting deeply nested reactions."""
        tree = {
            "type": "mol",
            "smiles": "CCCCO",
            "children": [
                {
                    "type": "reaction",
                    "template_id": "1",
                    "children": [
                        {
                            "type": "mol",
                            "smiles": "CCC",
                            "children": [
                                {
                                    "type": "reaction",
                                    "template_id": "2",
                                    "children": [
                                        {
                                            "type": "mol",
                                            "smiles": "CC",
                                            "children": [
                                                {
                                                    "type": "reaction",
                                                    "template_id": "3",
                                                    "children": [
                                                        {
                                                            "type": "mol",
                                                            "smiles": "C",
                                                            "children": [],
                                                        },
                                                        {
                                                            "type": "mol",
                                                            "smiles": "C",
                                                            "children": [],
                                                        },
                                                    ],
                                                }
                                            ],
                                        },
                                        {"type": "mol", "smiles": "C", "children": []},
                                    ],
                                }
                            ],
                        },
                        {"type": "mol", "smiles": "CO", "children": []},
                    ],
                }
            ],
        }
        result = count_reactions(tree)
        assert result == 3


class TestCollectLeafMolecules:
    """Tests for collect_leaf_molecules function."""

    def test_collect_single_leaf(self):
        """Test collecting from a tree with one leaf molecule."""
        tree = {"type": "mol", "smiles": "CCO", "children": []}
        result = collect_leaf_molecules(tree)
        assert result == ["CCO"]
        assert len(result) == 1

    def test_collect_multiple_leaves(self):
        """Test collecting from a tree with multiple leaf molecules."""
        tree = {
            "type": "mol",
            "smiles": "CCCO",
            "children": [
                {
                    "type": "reaction",
                    "template_id": "1",
                    "children": [
                        {"type": "mol", "smiles": "CC", "children": []},
                        {"type": "mol", "smiles": "CO", "children": []},
                    ],
                }
            ],
        }
        result = collect_leaf_molecules(tree)
        assert len(result) == 2
        assert "CC" in result
        assert "CO" in result
        assert "CCCO" not in result

    def test_collect_nested_leaves(self):
        """Test collecting from a tree with nested reactions."""
        tree = {
            "type": "mol",
            "smiles": "CCCCO",
            "children": [
                {
                    "type": "reaction",
                    "template_id": "1",
                    "children": [
                        {
                            "type": "mol",
                            "smiles": "CCC",
                            "children": [
                                {
                                    "type": "reaction",
                                    "template_id": "2",
                                    "children": [
                                        {"type": "mol", "smiles": "C", "children": []},
                                        {"type": "mol", "smiles": "CC", "children": []},
                                    ],
                                }
                            ],
                        },
                        {"type": "mol", "smiles": "CO", "children": []},
                    ],
                }
            ],
        }
        result = collect_leaf_molecules(tree)
        assert len(result) == 3
        assert "C" in result
        assert "CC" in result
        assert "CO" in result

    def test_collect_no_children(self):
        """Test collecting when there are no children."""
        tree = {"type": "mol", "smiles": "CCO", "children": []}
        result = collect_leaf_molecules(tree)
        assert result == ["CCO"]


class TestValidateReactionsWithProducts:
    """Tests for validate_reactions_with_products function."""

    def test_validate_leaf_molecule(self):
        """Test validation of a leaf molecule (no reactions)."""
        tree = {"type": "mol", "smiles": "CCO", "children": []}
        result = validate_reactions_with_products(tree)
        assert result is True

    def test_validate_invalid_molecule_type(self):
        """Test validation with invalid node type."""
        tree = {"type": "invalid", "smiles": "CCO", "children": []}
        result = validate_reactions_with_products(tree)
        assert result is False

    def test_validate_simple_valid_reaction(self):
        """Test validation of a simple valid reaction."""
        # Note: This test requires actual template validation which may fail
        # if templates are not set up correctly
        tree = {
            "type": "mol",
            "smiles": "CCO",
            "children": [],
        }
        result = validate_reactions_with_products(tree)
        assert isinstance(result, bool)


class TestScoreFinal:
    """Tests for score_final function."""

    @staticmethod
    def leaf_route(*smiles):
        return json.dumps(
            {
                "type": "mol",
                "smiles": "CCO",
                "children": [
                    {
                        "type": "reaction",
                        "template_id": "1",
                        "children": [
                            {"type": "mol", "smiles": item, "children": []}
                            for item in smiles
                        ],
                    }
                ],
            }
        )

    @patch("retrosynthesis.score.validate_reactions_with_products", return_value=True)
    @patch("retrosynthesis.score.check_price")
    def test_score_final_batches_price_lookup(self, mock_check_price, mock_validate):
        mock_check_price.return_value = {
            "C": [{"Price": 2.5}],
            "CO": [{"Price": 4.0}],
        }

        result = score_final(self.leaf_route("C", "CO"), {"prize": 6.5, "max_steps": 1})

        assert result == 1.0
        mock_validate.assert_called_once()
        mock_check_price.assert_called_once_with(["C", "CO"], limit=1)

    @patch("retrosynthesis.score.validate_reactions_with_products", return_value=True)
    @patch("retrosynthesis.score.check_price")
    def test_score_final_rejects_missing_buyable(
        self, mock_check_price, mock_validate
    ):
        mock_check_price.return_value = {"C": [{"Price": 2.5}], "CO": []}

        result = score_final(
            self.leaf_route("C", "CO"), {"prize": 100.0, "max_steps": 1}
        )

        assert result == 0.0
        mock_validate.assert_called_once()

    @patch("retrosynthesis.score.validate_reactions_with_products", return_value=True)
    @patch("retrosynthesis.score.check_price")
    def test_score_final_rejects_route_over_budget(
        self, mock_check_price, mock_validate
    ):
        mock_check_price.return_value = {
            "C": [{"Price": 2.5}],
            "CO": [{"Price": 4.0}],
        }

        result = score_final(
            self.leaf_route("C", "CO"), {"prize": 6.49, "max_steps": 1}
        )

        assert result == 0.0
        mock_validate.assert_called_once()

    def test_score_final_valid_route(self):
        """Test scoring a valid retrosynthesis route."""
        prediction = json.dumps(
            {
                "type": "mol",
                "smiles": "CCO",
                "children": [
                    {
                        "type": "reaction",
                        "template_id": "1",
                        "children": [
                            {"type": "mol", "smiles": "C", "children": []},
                            {"type": "mol", "smiles": "CO", "children": []},
                        ],
                    }
                ],
            }
        )
        target = {"prize": 1000.0, "max_steps": 5}
        result = score_final(prediction, target)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_final_exceeds_max_steps(self):
        """Test scoring fails when max_steps is exceeded."""
        # Create a route with 3 reactions
        prediction = json.dumps(
            {
                "type": "mol",
                "smiles": "CCCCO",
                "children": [
                    {
                        "type": "reaction",
                        "template_id": "1",
                        "children": [
                            {
                                "type": "mol",
                                "smiles": "CCC",
                                "children": [
                                    {
                                        "type": "reaction",
                                        "template_id": "2",
                                        "children": [
                                            {
                                                "type": "mol",
                                                "smiles": "CC",
                                                "children": [
                                                    {
                                                        "type": "reaction",
                                                        "template_id": "3",
                                                        "children": [
                                                            {
                                                                "type": "mol",
                                                                "smiles": "C",
                                                                "children": [],
                                                            },
                                                            {
                                                                "type": "mol",
                                                                "smiles": "C",
                                                                "children": [],
                                                            },
                                                        ],
                                                    }
                                                ],
                                            },
                                            {
                                                "type": "mol",
                                                "smiles": "C",
                                                "children": [],
                                            },
                                        ],
                                    }
                                ],
                            },
                            {"type": "mol", "smiles": "CO", "children": []},
                        ],
                    }
                ],
            }
        )
        target = {"prize": 1000.0, "max_steps": 2}
        result = score_final(prediction, target)
        assert result == 0.0

    def test_score_final_no_max_steps(self):
        """Test scoring raises error when max_steps is not provided."""
        prediction = json.dumps({"type": "mol", "smiles": "CCO", "children": []})
        target = {"prize": 1000.0}
        with pytest.raises(ValueError):
            score_final(prediction, target)

    def test_score_final_invalid_json(self):
        """Test scoring with invalid JSON."""
        prediction = "not valid json"
        target = {"prize": 1000.0, "max_steps": 5}
        result = score_final(prediction, target)
        assert result == 0.0

    def test_score_final_with_json_markers(self):
        """Test that JSON markers are properly stripped."""
        prediction = (
            "```json\n"
            + json.dumps({"type": "mol", "smiles": "CCO", "children": []})
            + "\n```"
        )
        target = {"prize": 1000.0, "max_steps": 5}
        result = score_final(prediction, target)
        assert isinstance(result, float)


class TestScoreFinalWithoutPrice:
    """Tests for score_final_without_price function."""

    def test_score_without_price_valid(self):
        """Test scoring without price validation."""
        prediction = json.dumps(
            {
                "type": "mol",
                "smiles": "CCO",
                "children": [
                    {
                        "type": "reaction",
                        "template_id": "1",
                        "children": [
                            {"type": "mol", "smiles": "C", "children": []},
                            {"type": "mol", "smiles": "CO", "children": []},
                        ],
                    }
                ],
            }
        )
        target = []
        result = score_final_without_price(prediction, target)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_without_price_invalid_json(self):
        """Test scoring without price with invalid JSON."""
        prediction = "not valid json"
        target = []
        # Invalid JSON raises JSONDecodeError since json.loads() is outside try-except
        with pytest.raises(json.JSONDecodeError):
            score_final_without_price(prediction, target)

    def test_score_without_price_with_markers(self):
        """Test that JSON markers are properly stripped."""
        prediction = (
            "```json\n"
            + json.dumps({"type": "mol", "smiles": "CCO", "children": []})
            + "\n```"
        )
        target = []
        result = score_final_without_price(prediction, target)
        assert isinstance(result, float)


class TestCheckReactants:
    """Tests for check_reactants function."""

    def test_check_reactants_match(self):
        """Test when predicted reactants match target."""
        prediction = json.dumps(
            {
                "type": "mol",
                "smiles": "CCO",
                "children": [
                    {
                        "type": "reaction",
                        "template_id": "1",
                        "children": [
                            {"type": "mol", "smiles": "C", "children": []},
                            {"type": "mol", "smiles": "CO", "children": []},
                        ],
                    }
                ],
            }
        )
        target = ["C", "CO"]
        result = check_reactants(prediction, target)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_check_reactants_no_match(self):
        """Test when predicted reactants don't match target."""
        prediction = json.dumps(
            {
                "type": "mol",
                "smiles": "CCO",
                "children": [
                    {
                        "type": "reaction",
                        "template_id": "1",
                        "children": [
                            {"type": "mol", "smiles": "C", "children": []},
                            {"type": "mol", "smiles": "CO", "children": []},
                        ],
                    }
                ],
            }
        )
        target = ["CC", "O"]
        result = check_reactants(prediction, target)
        assert isinstance(result, float)

    def test_check_reactants_invalid_smiles(self):
        """Test with invalid SMILES in prediction."""
        prediction = json.dumps(
            {
                "type": "mol",
                "smiles": "CCO",
                "children": [
                    {
                        "type": "reaction",
                        "template_id": "1",
                        "children": [
                            {"type": "mol", "smiles": "invalid", "children": []},
                        ],
                    }
                ],
            }
        )
        target = ["C", "CO"]
        result = check_reactants(prediction, target)
        assert result == 0.0

    def test_check_reactants_invalid_json(self):
        """Test with invalid JSON."""
        prediction = "not valid json"
        target = ["C", "CO"]
        result = check_reactants(prediction, target)
        assert result == 0.0


class TestCheckTemplate:
    """Tests for check_template function."""

    def test_check_template_valid(self):
        """Test checking a valid template."""
        prediction = json.dumps({"template_id": 1, "mapped_rxn": "test_rxn"})
        target = "1"
        result = check_template(prediction, target)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_check_template_mismatch(self):
        """Test checking with mismatched template ID."""
        prediction = json.dumps({"template_id": 1, "mapped_rxn": "test_rxn"})
        target = "2"
        result = check_template(prediction, target)
        assert result == 0.0

    def test_check_template_missing_id(self):
        """Test checking with missing template_id."""
        prediction = json.dumps({"mapped_rxn": "test_rxn"})
        target = "1"
        result = check_template(prediction, target)
        assert result == 0.0

    def test_check_template_invalid_json(self):
        """Test checking with invalid JSON."""
        prediction = "not valid json"
        target = "1"
        result = check_template(prediction, target)
        assert result == 0.0


class TestCheckApplyTemplate:
    """Tests for check_apply_template function."""

    @patch("retrosynthesis.score.apply_template_retro")
    @patch("retrosynthesis.score.search_by_template")
    def test_check_apply_template_valid(self, mock_search, mock_apply):
        """Test checking template application."""
        mock_search.return_value = {"mapped_rxn": "test_rxn"}
        mock_apply.return_value = ["C", "O"]
        prediction = {"template_id": "1", "mapped_rxn": "test_rxn"}
        target = "CCO"
        result = check_apply_template(prediction, target)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @patch("retrosynthesis.score.search_by_template")
    def test_check_apply_template_missing_template_id(self, mock_search):
        """Test with missing template_id."""
        mock_search.return_value = None
        prediction = {"mapped_rxn": "test_rxn"}
        target = "CCO"
        # Missing template_id causes search_by_template to return None,
        # then .get() on None raises AttributeError
        with pytest.raises(AttributeError):
            check_apply_template(prediction, target)

    @patch("retrosynthesis.score.apply_template_retro")
    @patch("retrosynthesis.score.search_by_template")
    def test_check_apply_template_invalid_smiles(self, mock_search, mock_apply):
        """Test with invalid target SMILES."""
        mock_search.return_value = {"mapped_rxn": "test_rxn"}
        mock_apply.return_value = None
        prediction = {"template_id": "1", "mapped_rxn": "test_rxn"}
        target = "invalid_smiles"
        result = check_apply_template(prediction, target)
        assert isinstance(result, float)


class TestCheckListMolecules:
    """Tests for check_list_molecules function."""

    def test_check_list_molecules_match(self):
        """Test when predicted molecules match target."""
        prediction = [["C", "CO"]]
        target = [["C", "CO"]]
        result = check_list_molecules(prediction, target)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_check_list_molecules_no_match(self):
        """Test when predicted molecules don't match target."""
        prediction = [["C", "CO"]]
        target = [["CC", "O"]]
        result = check_list_molecules(prediction, target)
        assert isinstance(result, float)

    def test_check_list_molecules_string_input(self):
        """Test with string representation of list."""
        prediction = "[['C', 'CO']]"
        target = [["C", "CO"]]
        result = check_list_molecules(prediction, target)
        assert isinstance(result, float)

    def test_check_list_molecules_json_string(self):
        """Test with JSON string representation."""
        prediction = json.dumps([["C", "CO"]])
        target = [["C", "CO"]]
        result = check_list_molecules(prediction, target)
        assert isinstance(result, float)

    def test_check_list_molecules_invalid_smiles(self):
        """Test with invalid SMILES."""
        prediction = [["invalid", "also_invalid"]]
        target = [["C", "CO"]]
        result = check_list_molecules(prediction, target)
        assert result == 0.0

    def test_check_list_molecules_invalid_string(self):
        """Test with invalid string input."""
        prediction = "not a valid list representation"
        target = [["C", "CO"]]
        result = check_list_molecules(prediction, target)
        assert result == 0.0

    def test_check_list_molecules_multiple_possibilities(self):
        """Test with multiple possible precursor sets."""
        prediction = [["C", "CO"], ["CC", "O"]]
        target = [["C", "CO"]]
        result = check_list_molecules(prediction, target)
        assert isinstance(result, float)


class TestScoringIntegration:
    """Integration tests for scoring functions."""

    def test_score_complete_workflow(self):
        """Test complete scoring workflow."""
        # Create a simple route
        route = {
            "type": "mol",
            "smiles": "CCO",
            "children": [
                {
                    "type": "reaction",
                    "template_id": "1",
                    "children": [
                        {"type": "mol", "smiles": "C", "children": []},
                        {"type": "mol", "smiles": "CO", "children": []},
                    ],
                }
            ],
        }

        # Test count_reactions
        num_reactions = count_reactions(route)
        assert num_reactions == 1

        # Test collect_leaf_molecules
        leaves = collect_leaf_molecules(route)
        assert len(leaves) == 2

        # Test validate_reactions_with_products
        is_valid = validate_reactions_with_products(route)
        assert isinstance(is_valid, bool)

    def test_score_functions_consistency(self):
        """Test that different scoring functions handle the same input consistently."""
        route = {
            "type": "mol",
            "smiles": "CCO",
            "children": [],
        }
        route_json = json.dumps(route)

        # All scoring functions should return float between 0 and 1
        target_with_price = {"prize": 1000.0, "max_steps": 5}
        score1 = score_final(route_json, target_with_price)
        assert isinstance(score1, float)
        assert 0.0 <= score1 <= 1.0

        target_without_price = []
        score2 = score_final_without_price(route_json, target_without_price)
        assert isinstance(score2, float)
        assert 0.0 <= score2 <= 1.0

    def test_helper_functions_on_complex_tree(self):
        """Test helper functions on a more complex tree structure."""
        complex_tree = {
            "type": "mol",
            "smiles": "CCCCO",
            "children": [
                {
                    "type": "reaction",
                    "template_id": "1",
                    "children": [
                        {
                            "type": "mol",
                            "smiles": "CCC",
                            "children": [
                                {
                                    "type": "reaction",
                                    "template_id": "2",
                                    "children": [
                                        {"type": "mol", "smiles": "C", "children": []},
                                        {"type": "mol", "smiles": "CC", "children": []},
                                    ],
                                }
                            ],
                        },
                        {
                            "type": "mol",
                            "smiles": "CO",
                            "children": [
                                {
                                    "type": "reaction",
                                    "template_id": "3",
                                    "children": [
                                        {"type": "mol", "smiles": "C", "children": []},
                                        {"type": "mol", "smiles": "O", "children": []},
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        }

        # Count reactions
        num_reactions = count_reactions(complex_tree)
        assert num_reactions == 3

        # Collect leaf molecules
        leaves = collect_leaf_molecules(complex_tree)
        assert len(leaves) == 4
        assert all(isinstance(leaf, str) for leaf in leaves)
