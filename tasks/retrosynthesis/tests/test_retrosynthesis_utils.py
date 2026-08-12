"""Tests for retrosynthesis_utils.py functions."""

from unittest.mock import patch

from rdkit import Chem
from retrosynthesis.retrosynthesis_utils import (
    _fragment_mapped_smiles,
    _get_full_mapped_smiles,
    _is_buyable,
    check_price,
    detect_functional_groups_in_molecule,
    get_functional_groups,
    return_matching,
    species_match,
    valid_smiles,
    validate_molecule,
    validate_reaction,
)


class TestValidSmiles:
    """Tests for valid_smiles function."""

    def test_valid_simple_smiles(self):
        """Test validation of simple valid SMILES."""
        assert valid_smiles("CCO") is True
        assert valid_smiles("c1ccccc1") is True
        assert valid_smiles("CC(=O)O") is True

    def test_invalid_smiles(self):
        """Test validation of invalid SMILES."""
        assert valid_smiles("invalid_smiles") is False
        assert valid_smiles("C1CCC") is False
        assert valid_smiles("XYZ") is False

    def test_empty_smiles(self):
        """Test validation of empty SMILES."""
        # RDKit actually considers empty string as valid (creates an empty molecule)
        assert valid_smiles("") is True

    def test_complex_valid_smiles(self):
        """Test validation of complex valid SMILES."""
        assert valid_smiles("CC(=O)Oc1ccccc1C(=O)O") is True  # Aspirin
        assert valid_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C") is True  # Caffeine


class TestSpeciesMatch:
    """Tests for species_match and _species_match_single functions."""

    def test_exact_match(self):
        """Test exact match of species."""
        ground_truth = ["CCO", "CC"]
        predicted = ["CCO", "CC"]
        assert species_match(ground_truth, predicted) is True

    def test_order_independent_match(self):
        """Test that order doesn't matter."""
        ground_truth = ["CCO", "CC"]
        predicted = ["CC", "CCO"]
        assert species_match(ground_truth, predicted) is True

    def test_stereochemistry_ignored(self):
        """Test that stereochemistry is ignored in matching."""
        ground_truth = ["C[C@H](O)C"]
        predicted = ["C[C@@H](O)C"]
        assert species_match(ground_truth, predicted) is True

    def test_different_length_lists(self):
        """Test mismatch when lists have different lengths."""
        ground_truth = ["CCO", "CC"]
        predicted = ["CCO"]
        assert species_match(ground_truth, predicted) is False

    def test_different_molecules(self):
        """Test mismatch when molecules are different."""
        ground_truth = ["CCO", "CC"]
        predicted = ["CCO", "CCC"]
        assert species_match(ground_truth, predicted) is False

    def test_tuple_of_tuples_predicted(self):
        """Test with predicted as tuple of tuples (multiple outcomes)."""
        ground_truth = ["CCO", "CC"]
        predicted = (("CC", "CCO"), ("CCC", "C"))
        assert species_match(ground_truth, predicted) is True

    def test_tuple_of_tuples_no_match(self):
        """Test with predicted as tuple of tuples with no match."""
        ground_truth = ["CCO", "CC"]
        predicted = (("CCC", "C"), ("CCCC", "C"))
        assert species_match(ground_truth, predicted) is False

    def test_invalid_smiles_in_ground_truth(self):
        """Test with invalid SMILES in ground truth."""
        ground_truth = ["CCO", "invalid"]
        predicted = ["CCO", "CC"]
        assert species_match(ground_truth, predicted) is False

    def test_invalid_smiles_in_predicted(self):
        """Test with invalid SMILES in predicted."""
        ground_truth = ["CCO", "CC"]
        predicted = ["CCO", "invalid"]
        assert species_match(ground_truth, predicted) is False


class TestIsBuyable:
    """Tests for availability backed by the local lookup."""

    @patch("retrosynthesis.retrosynthesis_utils.lookup_prices")
    def test_is_buyable_preserves_input_order(self, mock_lookup):
        mock_lookup.return_value = {
            "CCO": {
                "canonical_smiles": "CCO",
                "price_usd_per_g": 3.2,
                "sources": ["test"],
            },
            "CCCCCCCCCC": None,
        }

        assert _is_buyable(["CCO", "CCCCCCCCCC"]) == [True, False]
        mock_lookup.assert_called_once_with(["CCO", "CCCCCCCCCC"])


class TestCheckPrice:
    """Tests for the compatibility shape returned by check_price."""

    @patch("retrosynthesis.retrosynthesis_utils.lookup_prices")
    def test_check_price_returns_one_gram_usd_entry(self, mock_lookup):
        mock_lookup.return_value = {
            "C1=CC=CC=C1": {
                "canonical_smiles": "c1ccccc1",
                "price_usd_per_g": 4.25,
                "sources": ["chemcost", "coprinet_mcule"],
            },
            "not-smiles": None,
        }

        result = check_price(["C1=CC=CC=C1", "not-smiles"], limit=5)

        assert result["C1=CC=CC=C1"] == [
            {
                "SMILES": "c1ccccc1",
                "Supplier": "chemcost, coprinet_mcule",
                "Purity": None,
                "Amount": 1.0,
                "Measure": "g",
                "Price": 4.25,
            }
        ]
        assert result["not-smiles"] == []

    @patch("retrosynthesis.retrosynthesis_utils.lookup_prices")
    def test_check_price_zero_limit_still_returns_frozen_offer(self, mock_lookup):
        mock_lookup.return_value = {
            "CCO": {
                "canonical_smiles": "CCO",
                "price_usd_per_g": 3.2,
                "sources": ["test"],
            }
        }

        assert len(check_price(["CCO"], limit=0)["CCO"]) == 1


class TestDetectFunctionalGroups:
    """Tests for detect_functional_groups_in_molecule function."""

    def test_detect_alcohol(self):
        """Test detection of alcohol functional group."""
        result = detect_functional_groups_in_molecule("CCO")
        # The result depends on FG_PATTERNS, but alcohol should be detected
        assert isinstance(result, list)

    def test_detect_benzene(self):
        """Test detection of aromatic ring."""
        result = detect_functional_groups_in_molecule("c1ccccc1")
        assert isinstance(result, list)

    def test_detect_carbonyl(self):
        """Test detection of carbonyl group."""
        result = detect_functional_groups_in_molecule("CC(=O)C")
        assert isinstance(result, list)

    def test_invalid_smiles(self):
        """Test with invalid SMILES."""
        result = detect_functional_groups_in_molecule("invalid_smiles")
        assert result == []

    def test_complex_molecule(self):
        """Test with complex molecule (aspirin)."""
        result = detect_functional_groups_in_molecule("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(result, list)
        # Aspirin has multiple functional groups
        assert len(result) > 0


class TestGetFunctionalGroups:
    """Tests for get_functional_groups function."""

    def test_get_functional_groups_single_molecule(self):
        """Test getting functional groups from a single molecule."""
        result = get_functional_groups("CCO")
        assert isinstance(result, list)

    def test_get_functional_groups_multiple_molecules(self):
        """Test getting functional groups from multiple molecules (dot-separated)."""
        result = get_functional_groups("CCO.c1ccccc1")
        assert isinstance(result, list)

    def test_get_functional_groups_complex(self):
        """Test with complex molecule."""
        result = get_functional_groups("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(result, list)

    def test_get_functional_groups_invalid(self):
        """Test with invalid SMILES."""
        # Invalid SMILES returns empty list instead of raising exception
        result = get_functional_groups("invalid_smiles")
        assert result == []


class TestFragmentMappedSmiles:
    """Tests for _fragment_mapped_smiles function."""

    def test_fragment_simple_molecule(self):
        """Test fragmenting a simple molecule."""
        mol = Chem.MolFromSmiles("CCO")
        # Fragment containing first two carbons (indices 0, 1)
        result = _fragment_mapped_smiles(mol, (0, 1))
        assert isinstance(result, str)
        # Result should contain atom maps
        assert ":" in result

    def test_fragment_benzene(self):
        """Test fragmenting benzene ring."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        # Fragment containing first three atoms
        result = _fragment_mapped_smiles(mol, (0, 1, 2))
        assert isinstance(result, str)

    def test_fragment_preserves_mapping(self):
        """Test that atom mapping is preserved."""
        mol = Chem.MolFromSmiles("CCCO")
        result = _fragment_mapped_smiles(mol, (0, 1, 2))
        # Check that map numbers are present
        assert ":1]" in result or ":2]" in result or ":3]" in result


class TestGetFullMappedSmiles:
    """Tests for _get_full_mapped_smiles function."""

    def test_full_mapping_single_group(self):
        """Test full mapping with single functional group."""
        mol = Chem.MolFromSmiles("CCO")
        all_atom_indices = [(0, 1, 2)]
        result = _get_full_mapped_smiles(mol, all_atom_indices)
        assert isinstance(result, str)
        # Should have atom maps for all three atoms
        assert ":1]" in result
        assert ":2]" in result
        assert ":3]" in result

    def test_full_mapping_multiple_groups(self):
        """Test full mapping with multiple functional groups."""
        mol = Chem.MolFromSmiles("CC(=O)C")
        all_atom_indices = [(0, 1), (1, 2, 3)]
        result = _get_full_mapped_smiles(mol, all_atom_indices)
        assert isinstance(result, str)
        # Should have atom maps
        assert ":" in result

    def test_full_mapping_overlapping_groups(self):
        """Test with overlapping atom indices."""
        mol = Chem.MolFromSmiles("CCCO")
        all_atom_indices = [(0, 1), (1, 2), (2, 3)]
        result = _get_full_mapped_smiles(mol, all_atom_indices)
        assert isinstance(result, str)
        # All atoms should be mapped
        assert ":1]" in result
        assert ":2]" in result
        assert ":3]" in result
        assert ":4]" in result


class TestReturnMatching:
    """Tests for return_matching function."""

    @patch("retrosynthesis.retrosynthesis_utils.apply_template_retro")
    def test_return_matching_with_matches(self, mock_apply_template):
        """Test when template produces matches."""
        # Mock apply_template_retro to return some reactants
        mock_apply_template.return_value = (("CCO", "CC"),)

        result = return_matching("CCCO", "1")
        assert result is True

    @patch("retrosynthesis.retrosynthesis_utils.apply_template_retro")
    def test_return_matching_no_matches(self, mock_apply_template):
        """Test when template produces no matches."""
        # Mock apply_template_retro to return empty tuple
        mock_apply_template.return_value = ()

        result = return_matching("CCCO", "1")
        assert result is False


class TestValidateMolecule:
    """Tests for validate_molecule function."""

    def test_valid_simple_molecule(self):
        """Test validation of a simple valid molecule."""
        mol = {"type": "mol", "smiles": "CCO"}
        is_valid, error = validate_molecule(mol)
        assert is_valid is True
        assert error == ""

    def test_valid_molecule_with_children(self):
        """Test validation of molecule with reaction children."""
        mol = {
            "type": "mol",
            "smiles": "CCO",
            "children": [
                {
                    "type": "reaction",
                    "template_id": 1,
                    "children": [
                        {"type": "mol", "smiles": "CC"},
                        {"type": "mol", "smiles": "CO"},
                    ],
                }
            ],
        }
        is_valid, error = validate_molecule(mol)
        assert is_valid is True

    def test_invalid_type(self):
        """Test validation with wrong type."""
        mol = {"type": "reaction", "smiles": "CCO"}
        is_valid, error = validate_molecule(mol)
        assert is_valid is False
        assert "type 'reaction'" in error

    def test_missing_smiles(self):
        """Test validation with missing SMILES."""
        mol = {"type": "mol"}
        is_valid, error = validate_molecule(mol)
        assert is_valid is False
        assert "missing required 'smiles' field" in error

    def test_empty_smiles(self):
        """Test validation with empty SMILES."""
        mol = {"type": "mol", "smiles": "  "}
        is_valid, error = validate_molecule(mol)
        assert is_valid is False
        assert "empty SMILES" in error

    def test_non_string_smiles(self):
        """Test validation with non-string SMILES."""
        mol = {"type": "mol", "smiles": 123}
        is_valid, error = validate_molecule(mol)
        assert is_valid is False
        assert "non-string 'smiles'" in error

    def test_invalid_children_type(self):
        """Test validation with non-list children."""
        mol = {"type": "mol", "smiles": "CCO", "children": "not a list"}
        is_valid, error = validate_molecule(mol)
        assert is_valid is False
        assert "non-list 'children'" in error

    def test_wrong_number_of_children(self):
        """Test validation with wrong number of children."""
        mol = {
            "type": "mol",
            "smiles": "CCO",
            "children": [
                {"type": "reaction", "template_id": 1, "children": []},
                {"type": "reaction", "template_id": 2, "children": []},
            ],
        }
        is_valid, error = validate_molecule(mol)
        assert is_valid is False
        assert "2 children, expected exactly 1" in error

    def test_not_dict(self):
        """Test validation with non-dict input."""
        is_valid, error = validate_molecule("not a dict")
        assert is_valid is False
        assert "not a dictionary" in error


class TestValidateReaction:
    """Tests for validate_reaction function."""

    def test_valid_reaction(self):
        """Test validation of a valid reaction."""
        reaction = {
            "type": "reaction",
            "template_id": 1,
            "children": [
                {"type": "mol", "smiles": "CC"},
                {"type": "mol", "smiles": "CO"},
            ],
        }
        is_valid, error = validate_reaction(reaction)
        assert is_valid is True
        assert error == ""

    def test_invalid_type(self):
        """Test validation with wrong type."""
        reaction = {"type": "mol", "template_id": 1, "children": []}
        is_valid, error = validate_reaction(reaction)
        assert is_valid is False
        assert "type 'mol'" in error

    def test_missing_template_id(self):
        """Test validation with missing template_id."""
        reaction = {"type": "reaction", "children": []}
        is_valid, error = validate_reaction(reaction)
        assert is_valid is False
        assert "missing required 'template_id'" in error

    def test_non_integer_template_id(self):
        """Test validation with non-integer template_id."""
        reaction = {"type": "reaction", "template_id": "not an int", "children": []}
        is_valid, error = validate_reaction(reaction)
        assert is_valid is False
        assert "non-integer 'template_id'" in error

    def test_missing_children(self):
        """Test validation with missing children."""
        reaction = {"type": "reaction", "template_id": 1}
        is_valid, error = validate_reaction(reaction)
        assert is_valid is False
        assert "missing required 'children'" in error

    def test_non_list_children(self):
        """Test validation with non-list children."""
        reaction = {"type": "reaction", "template_id": 1, "children": "not a list"}
        is_valid, error = validate_reaction(reaction)
        assert is_valid is False
        assert "non-list 'children'" in error

    def test_invalid_child_molecule(self):
        """Test validation with invalid child molecule."""
        reaction = {
            "type": "reaction",
            "template_id": 1,
            "children": [
                {"type": "mol", "smiles": "CC"},
                {"type": "mol"},  # Missing smiles
            ],
        }
        is_valid, error = validate_reaction(reaction)
        assert is_valid is False
        assert "missing required 'smiles' field" in error

    def test_not_dict(self):
        """Test validation with non-dict input."""
        is_valid, error = validate_reaction("not a dict")
        assert is_valid is False
        assert "not a dictionary" in error

    def test_nested_reaction_structure(self):
        """Test validation with nested structure."""
        reaction = {
            "type": "reaction",
            "template_id": 1,
            "children": [
                {
                    "type": "mol",
                    "smiles": "CCO",
                    "children": [
                        {
                            "type": "reaction",
                            "template_id": 2,
                            "children": [
                                {"type": "mol", "smiles": "CC"},
                                {"type": "mol", "smiles": "CO"},
                            ],
                        }
                    ],
                }
            ],
        }
        is_valid, error = validate_reaction(reaction)
        assert is_valid is True
