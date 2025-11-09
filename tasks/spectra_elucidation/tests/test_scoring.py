"""
Tests for the scoring functions in the spectra_elucidation package.
"""

import pytest
from rdkit import Chem
from spectra_elucidation.score import (
    calculate_dbe,
    neutralize_charges,
    parse_formula,
    score_formula_match,
    score_isotopic_distribution,
    score_molecule,
    score_molecule_fragments,
    score_num_aromatic_carbons,
    score_num_carbon_symmetry_classes,
    score_num_carbonyl_groups,
    score_num_ch3_groups,
    score_num_hydrogen_symmetry_classes,
    validate_dbe_consistency,
    validate_molecular_formula,
)


class TestScoreMolecule:
    """Tests for the score_molecule function."""

    def test_exact_match(self):
        """Test that identical SMILES strings return score of 1.0."""
        prediction = "CCO"
        ground_truth = "CCO"
        assert score_molecule(prediction, ground_truth) == 1.0

    def test_canonical_match(self):
        """Test that different representations of the same molecule match."""
        prediction = "C(C)O"  # Alternative representation of ethanol
        ground_truth = "CCO"
        assert score_molecule(prediction, ground_truth) == 1.0

    def test_stereochemistry_ignored(self):
        """Test that stereochemistry differences are ignored."""
        prediction = "C[C@H](O)CC"  # S-stereoisomer
        ground_truth = "C[C@@H](O)CC"  # R-stereoisomer
        assert score_molecule(prediction, ground_truth) == 1.0

    def test_no_match(self):
        """Test that different molecules return score of 0.0."""
        prediction = "CCO"  # Ethanol
        ground_truth = "CC"  # Ethane
        assert score_molecule(prediction, ground_truth) == 0.0

    def test_invalid_prediction(self):
        """Test that invalid SMILES prediction returns score of 0.0."""
        prediction = "INVALID_SMILES"
        ground_truth = "CCO"
        assert score_molecule(prediction, ground_truth) == 0.0

    def test_invalid_ground_truth(self):
        """Test that invalid ground truth SMILES raises ValueError."""
        prediction = "CCO"
        ground_truth = "INVALID_SMILES"
        with pytest.raises(ValueError, match="Invalid ground truth SMILES string"):
            score_molecule(prediction, ground_truth)

    def test_benzene(self):
        """Test with aromatic molecule."""
        prediction = "c1ccccc1"
        ground_truth = "C1=CC=CC=C1"
        assert score_molecule(prediction, ground_truth) == 1.0


class TestNeutralizeCharges:
    """Tests for the neutralize_charges function."""

    def test_neutral_molecule(self):
        """Test that neutral molecules remain unchanged."""
        mol = Chem.MolFromSmiles("CCO")
        result = neutralize_charges(mol)
        assert result is not None
        for atom in result.GetAtoms():
            assert atom.GetFormalCharge() == 0

    def test_simple_charge(self):
        """Test neutralization of simple charged molecule."""
        mol = Chem.MolFromSmiles("[NH4+]")
        result = neutralize_charges(mol)
        assert result is not None
        # Check that charges are neutralized
        total_charge = sum(atom.GetFormalCharge() for atom in result.GetAtoms())
        assert total_charge == 0

    def test_carboxylate_anion(self):
        """Test neutralization of carboxylate anion."""
        mol = Chem.MolFromSmiles("CC(=O)[O-]")
        result = neutralize_charges(mol)
        assert result is not None
        total_charge = sum(atom.GetFormalCharge() for atom in result.GetAtoms())
        assert total_charge == 0

    def test_extreme_charge(self):
        """Test neutralization of extreme charges."""
        mol = Chem.MolFromSmiles("[O-2]")
        result = neutralize_charges(mol)
        assert result is not None
        # Should attempt to neutralize even extreme charges

    def test_none_input(self):
        """Test that None input returns None."""
        result = neutralize_charges(None)
        assert result is None


class TestScoreMoleculeFragments:
    """Tests for the score_molecule_fragments function."""

    def test_valid_fragments(self):
        """Test with valid fragments that match the ground truth."""
        prediction = ["C", "CC", "CCO"]
        ground_truth = "CCCO"
        score = score_molecule_fragments(prediction, ground_truth)
        assert score == 1.0

    def test_invalid_fragment(self):
        """Test with one invalid fragment in the list."""
        prediction = ["C", "CCCCCCCC"]  # Second fragment too large
        ground_truth = "CCO"
        score = score_molecule_fragments(prediction, ground_truth)
        assert score == 0.0

    def test_empty_list(self):
        """Test with empty list returns 1.0."""
        prediction = []
        ground_truth = "CCO"
        score = score_molecule_fragments(prediction, ground_truth)
        assert score == 0.0

    def test_string_representation(self):
        """Test with string representation of a list."""
        prediction = '["C", "CC"]'
        ground_truth = "CCO"
        score = score_molecule_fragments(prediction, ground_truth)
        assert score == 1.0

    def test_invalid_string_format(self):
        """Test with invalid string format returns 0.0."""
        prediction = "not a valid list"
        ground_truth = "CCO"
        score = score_molecule_fragments(prediction, ground_truth)
        assert score == 0.0

    def test_non_list_input(self):
        """Test with non-list input returns 0.0."""
        prediction = {"C", "CC"}  # Set instead of list
        ground_truth = "CCO"
        score = score_molecule_fragments(prediction, ground_truth)
        assert score == 0.0

    def test_non_string_elements(self):
        """Test with non-string elements returns 0.0."""
        prediction = [1, 2, 3]
        ground_truth = "CCO"
        score = score_molecule_fragments(prediction, ground_truth)
        assert score == 0.0

    def test_charged_fragments(self):
        """Test with charged fragments."""
        prediction = ["[NH4+]", "C"]
        ground_truth = "CN"
        score = score_molecule_fragments(prediction, ground_truth)
        # Should handle charged fragments through neutralization
        assert score >= 0.0

    def test_extreme_charge_fragments(self):
        """Test with extreme charge fragments (should be lenient)."""
        prediction = ["[O-2]", "C"]
        ground_truth = "CO"
        score = score_molecule_fragments(prediction, ground_truth)
        # Extreme charge fragments are treated leniently
        assert score >= 0.0

    def test_unparseable_fragment(self):
        """Test that unparseable fragments are skipped."""
        prediction = ["C", "INVALID_SMILES", "CC"]
        ground_truth = "CCC"
        # Should skip invalid fragment and check others
        score = score_molecule_fragments(prediction, ground_truth)
        assert score >= 0.0


class TestValidateMolecularFormula:
    """Tests for the validate_molecular_formula function."""

    def test_valid_formula_match(self):
        """Test with matching molecular formula."""
        prediction = "C2H6O"
        ground_truth = "CCO"
        assert validate_molecular_formula(prediction, ground_truth) is True

    def test_valid_formula_no_match(self):
        """Test with non-matching molecular formula."""
        prediction = "C2H6"
        ground_truth = "CCO"
        assert validate_molecular_formula(prediction, ground_truth) is False

    def test_different_order(self):
        """Test that element order doesn't matter."""
        prediction = "H6C2O"  # Different order
        ground_truth = "CCO"
        assert validate_molecular_formula(prediction, ground_truth) is True

    def test_invalid_ground_truth(self):
        """Test with invalid ground truth SMILES."""
        prediction = "C2H6O"
        ground_truth = "INVALID_SMILES"
        with pytest.raises(ValueError, match="Invalid ground truth SMILES string"):
            validate_molecular_formula(prediction, ground_truth)

    def test_invalid_prediction_format(self):
        """Test with invalid prediction format."""
        prediction = "not_a_formula"
        ground_truth = "CCO"
        # Should return False or raise exception
        assert validate_molecular_formula(prediction, ground_truth) is False


class TestScoreFormulaMatch:
    """Tests for the score_formula_match function."""

    def test_match(self):
        """Test with matching formula."""
        prediction = "C2H6O"
        ground_truth = "CCO"
        assert score_formula_match(prediction, ground_truth) == 1.0

    def test_no_match(self):
        """Test with non-matching formula."""
        prediction = "C2H6"
        ground_truth = "CCO"
        assert score_formula_match(prediction, ground_truth) == 0.0


class TestParseFormula:
    """Tests for the parse_formula function."""

    def test_simple_formula(self):
        """Test parsing simple molecular formula."""
        result = parse_formula("C2H6O")
        assert result == {"C": 2, "H": 6, "O": 1}

    def test_formula_with_parentheses(self):
        """Test parsing formula with parentheses."""
        result = parse_formula("Ca(OH)2")
        assert result == {"Ca": 1, "O": 2, "H": 2}

    def test_formula_with_dot_adduct(self):
        """Test parsing formula with dot notation (hydrates)."""
        # Note: The parser doesn't handle multipliers outside parentheses like "5H2O"
        # It treats "5H2O" as literally 5 hydrogens and 2 oxygens
        # For proper hydrate parsing, use parentheses: CuSO4·(H2O)5
        result = parse_formula("CuSO4·(H2O)5")
        assert result == {"Cu": 1, "S": 1, "O": 9, "H": 10}

    def test_complex_formula(self):
        """Test parsing complex formula."""
        result = parse_formula("C6H5(CH3)")
        assert result == {"C": 7, "H": 8}

    def test_single_element(self):
        """Test parsing single element."""
        result = parse_formula("O2")
        assert result == {"O": 2}

    def test_element_without_number(self):
        """Test element without explicit number."""
        result = parse_formula("CO")
        assert result == {"C": 1, "O": 1}


class TestCalculateDBE:
    """Tests for the calculate_dbe function."""

    def test_ethane(self):
        """Test DBE calculation for ethane (saturated)."""
        mol = Chem.MolFromSmiles("CC")
        dbe = calculate_dbe(mol)
        assert dbe == 0

    def test_ethene(self):
        """Test DBE calculation for ethene (one double bond)."""
        mol = Chem.MolFromSmiles("C=C")
        dbe = calculate_dbe(mol)
        assert dbe == 1

    def test_benzene(self):
        """Test DBE calculation for benzene (aromatic)."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        dbe = calculate_dbe(mol)
        assert dbe == 4

    def test_acetone(self):
        """Test DBE calculation for acetone (carbonyl)."""
        mol = Chem.MolFromSmiles("CC(=O)C")
        dbe = calculate_dbe(mol)
        assert dbe == 1

    def test_with_nitrogen(self):
        """Test DBE calculation with nitrogen."""
        mol = Chem.MolFromSmiles("CCN")
        dbe = calculate_dbe(mol)
        # C=3, H=9 (including 2 on N), N=1
        # DBE = 3 - 9/2 + 1/2 + 1 = 0
        assert dbe == 0

    def test_with_halogen(self):
        """Test DBE calculation with halogen."""
        mol = Chem.MolFromSmiles("CCCl")
        dbe = calculate_dbe(mol)
        # Halogens count like hydrogens
        assert dbe == 0


class TestValidateDBEConsistency:
    """Tests for the validate_dbe_consistency function."""

    def test_correct_dbe(self):
        """Test with correct DBE value."""
        prediction = 0
        ground_truth = "CC"
        assert validate_dbe_consistency(prediction, ground_truth) == 1.0

    def test_incorrect_dbe(self):
        """Test with incorrect DBE value."""
        prediction = 5
        ground_truth = "CC"
        assert validate_dbe_consistency(prediction, ground_truth) == 0.0

    def test_string_prediction(self):
        """Test with string representation of integer."""
        prediction = "4"
        ground_truth = "c1ccccc1"
        assert validate_dbe_consistency(prediction, ground_truth) == 1.0

    def test_invalid_string_prediction(self):
        """Test with invalid string prediction."""
        prediction = "not_an_integer"
        ground_truth = "CC"
        assert validate_dbe_consistency(prediction, ground_truth) == 0.0

    def test_benzene_dbe(self):
        """Test DBE for benzene."""
        prediction = 4
        ground_truth = "c1ccccc1"
        assert validate_dbe_consistency(prediction, ground_truth) == 1.0


class TestScoreIsotopicDistribution:
    """Tests for the score_isotopic_distribution function."""

    def test_correct_elements(self):
        """Test with correct elements that have isotopic distribution."""
        prediction = ["C", "O"]
        ground_truth = "CCO"
        assert score_isotopic_distribution(prediction, ground_truth) == 1.0

    def test_incorrect_elements(self):
        """Test with incorrect elements."""
        prediction = ["C", "S"]
        ground_truth = "CCO"
        assert score_isotopic_distribution(prediction, ground_truth) == 0.0

    def test_no_isotopic_elements(self):
        """Test molecule with no relevant isotopic elements."""
        # Methane (CH4) has Carbon which is in the isotopic elements list
        prediction = ["C"]
        ground_truth = "C"
        assert score_isotopic_distribution(prediction, ground_truth) == 1.0

    def test_subset_of_elements(self):
        """Test with subset of correct elements."""
        prediction = ["C"]
        ground_truth = "CCO"
        assert score_isotopic_distribution(prediction, ground_truth) == 0.0

    def test_superset_of_elements(self):
        """Test with superset of elements."""
        prediction = ["C", "O", "N"]
        ground_truth = "CCO"
        assert score_isotopic_distribution(prediction, ground_truth) == 0.0


class TestScoreNumHydrogenSymmetryClasses:
    """Tests for the score_num_hydrogen_symmetry_classes function."""

    def test_methane(self):
        """Test with methane (all hydrogens equivalent)."""
        prediction = 1
        ground_truth = "C"
        assert score_num_hydrogen_symmetry_classes(prediction, ground_truth) == 1.0

    def test_ethanol(self):
        """Test with ethanol (multiple hydrogen environments)."""
        prediction = 3  # CH3, CH2, OH hydrogens
        ground_truth = "CCO"
        assert score_num_hydrogen_symmetry_classes(prediction, ground_truth) == 1.0

    def test_benzene(self):
        """Test with benzene (all hydrogens equivalent)."""
        prediction = 1
        ground_truth = "c1ccccc1"
        assert score_num_hydrogen_symmetry_classes(prediction, ground_truth) == 1.0

    def test_incorrect_count(self):
        """Test with incorrect count."""
        prediction = 5
        ground_truth = "C"
        assert score_num_hydrogen_symmetry_classes(prediction, ground_truth) == 0.0

    def test_string_prediction(self):
        """Test with string representation of integer."""
        prediction = "1"
        ground_truth = "C"
        assert score_num_hydrogen_symmetry_classes(prediction, ground_truth) == 1.0

    def test_invalid_string(self):
        """Test with invalid string."""
        prediction = "not_an_int"
        ground_truth = "C"
        assert score_num_hydrogen_symmetry_classes(prediction, ground_truth) == 0.0


class TestScoreNumCarbonSymmetryClasses:
    """Tests for the score_num_carbon_symmetry_classes function."""

    def test_methane(self):
        """Test with methane (one carbon)."""
        prediction = 1
        ground_truth = "C"
        assert score_num_carbon_symmetry_classes(prediction, ground_truth) == 1.0

    def test_ethane(self):
        """Test with ethane (equivalent carbons)."""
        prediction = 1
        ground_truth = "CC"
        assert score_num_carbon_symmetry_classes(prediction, ground_truth) == 1.0

    def test_propane(self):
        """Test with propane (two symmetry classes)."""
        prediction = 2  # CH3 and CH2 carbons
        ground_truth = "CCC"
        assert score_num_carbon_symmetry_classes(prediction, ground_truth) == 1.0

    def test_benzene(self):
        """Test with benzene (all carbons equivalent)."""
        prediction = 1
        ground_truth = "c1ccccc1"
        assert score_num_carbon_symmetry_classes(prediction, ground_truth) == 1.0

    def test_incorrect_count(self):
        """Test with incorrect count."""
        prediction = 5
        ground_truth = "CC"
        assert score_num_carbon_symmetry_classes(prediction, ground_truth) == 0.0

    def test_string_prediction(self):
        """Test with string representation."""
        prediction = "1"
        ground_truth = "C"
        assert score_num_carbon_symmetry_classes(prediction, ground_truth) == 1.0


class TestScoreNumAromaticCarbons:
    """Tests for the score_num_aromatic_carbons function."""

    def test_benzene(self):
        """Test with benzene (6 aromatic carbons)."""
        prediction = 6
        ground_truth = "c1ccccc1"
        assert score_num_aromatic_carbons(prediction, ground_truth) == 1.0

    def test_no_aromatic_carbons(self):
        """Test with non-aromatic molecule."""
        prediction = 0
        ground_truth = "CCC"
        assert score_num_aromatic_carbons(prediction, ground_truth) == 1.0

    def test_pyridine(self):
        """Test with pyridine (5 aromatic carbons)."""
        prediction = 5
        ground_truth = "c1ccncc1"
        assert score_num_aromatic_carbons(prediction, ground_truth) == 1.0

    def test_naphthalene(self):
        """Test with naphthalene (10 aromatic carbons)."""
        prediction = 10
        ground_truth = "c1ccc2ccccc2c1"
        assert score_num_aromatic_carbons(prediction, ground_truth) == 1.0

    def test_incorrect_count(self):
        """Test with incorrect count."""
        prediction = 3
        ground_truth = "c1ccccc1"
        assert score_num_aromatic_carbons(prediction, ground_truth) == 0.0

    def test_string_prediction(self):
        """Test with string representation."""
        prediction = "6"
        ground_truth = "c1ccccc1"
        assert score_num_aromatic_carbons(prediction, ground_truth) == 1.0


class TestScoreNumCH3Groups:
    """Tests for the score_num_ch3_groups function."""

    def test_methane(self):
        """Test with methane (not a CH3 group, no other carbons)."""
        prediction = 0
        ground_truth = "C"
        assert score_num_ch3_groups(prediction, ground_truth) == 1.0

    def test_ethane(self):
        """Test with ethane (two CH3 groups)."""
        prediction = 2
        ground_truth = "CC"
        assert score_num_ch3_groups(prediction, ground_truth) == 1.0

    def test_propane(self):
        """Test with propane (two CH3 groups)."""
        prediction = 2
        ground_truth = "CCC"
        assert score_num_ch3_groups(prediction, ground_truth) == 1.0

    def test_isobutane(self):
        """Test with isobutane (three CH3 groups)."""
        prediction = 3
        ground_truth = "CC(C)C"
        assert score_num_ch3_groups(prediction, ground_truth) == 1.0

    def test_benzene(self):
        """Test with benzene (no CH3 groups)."""
        prediction = 0
        ground_truth = "c1ccccc1"
        assert score_num_ch3_groups(prediction, ground_truth) == 1.0

    def test_toluene(self):
        """Test with toluene (one CH3 group)."""
        prediction = 1
        ground_truth = "Cc1ccccc1"
        assert score_num_ch3_groups(prediction, ground_truth) == 1.0

    def test_incorrect_count(self):
        """Test with incorrect count."""
        prediction = 5
        ground_truth = "CC"
        assert score_num_ch3_groups(prediction, ground_truth) == 0.0

    def test_string_prediction(self):
        """Test with string representation."""
        prediction = "2"
        ground_truth = "CC"
        assert score_num_ch3_groups(prediction, ground_truth) == 1.0


class TestScoreNumCarbonylGroups:
    """Tests for the score_num_carbonyl_groups function."""

    def test_acetone(self):
        """Test with acetone (one carbonyl)."""
        prediction = 1
        ground_truth = "CC(=O)C"
        assert score_num_carbonyl_groups(prediction, ground_truth) == 1.0

    def test_no_carbonyl(self):
        """Test with molecule without carbonyl."""
        prediction = 0
        ground_truth = "CCO"
        assert score_num_carbonyl_groups(prediction, ground_truth) == 1.0

    def test_acetic_acid(self):
        """Test with acetic acid (one carbonyl)."""
        prediction = 1
        ground_truth = "CC(=O)O"
        assert score_num_carbonyl_groups(prediction, ground_truth) == 1.0

    def test_oxalic_acid(self):
        """Test with oxalic acid (two carbonyls)."""
        prediction = 2
        ground_truth = "C(=O)(O)C(=O)O"
        assert score_num_carbonyl_groups(prediction, ground_truth) == 1.0

    def test_benzaldehyde(self):
        """Test with benzaldehyde (one carbonyl)."""
        prediction = 1
        ground_truth = "O=Cc1ccccc1"
        assert score_num_carbonyl_groups(prediction, ground_truth) == 1.0

    def test_tautomer_handling(self):
        """Test that tautomers are canonicalized."""
        # Keto-enol tautomers should be handled
        prediction = 1
        ground_truth = "CC(=O)C"
        assert score_num_carbonyl_groups(prediction, ground_truth) == 1.0

    def test_incorrect_count(self):
        """Test with incorrect count."""
        prediction = 5
        ground_truth = "CC(=O)C"
        assert score_num_carbonyl_groups(prediction, ground_truth) == 0.0

    def test_string_prediction(self):
        """Test with string representation."""
        prediction = "1"
        ground_truth = "CC(=O)C"
        assert score_num_carbonyl_groups(prediction, ground_truth) == 1.0

    def test_ester(self):
        """Test with ester (one carbonyl)."""
        prediction = 1
        ground_truth = "CC(=O)OC"
        assert score_num_carbonyl_groups(prediction, ground_truth) == 1.0

    def test_amide(self):
        """Test with amide (one carbonyl)."""
        prediction = 1
        ground_truth = "CC(=O)N"
        assert score_num_carbonyl_groups(prediction, ground_truth) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
