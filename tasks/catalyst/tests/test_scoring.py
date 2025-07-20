import os
from pathlib import Path

import pytest

# Import the scoring functions to test
from catalyst.score import (
    check_adsorption_structure,
    check_co2_molecule_structure,
    check_mp_structure,
    check_slab_structure,
    check_valid_json_file,
)
from hypothesis import given
from hypothesis import strategies as st

os.environ["CORRAL_WORK_DIR"] = str(Path(__file__).parent / "test_score_files")
TEST_SCORE_FILES = Path(os.environ["CORRAL_WORK_DIR"])


class TestBasicFileValidation:
    """Test basic file validation functions."""

    @pytest.fixture()
    def sample_json_files(self):
        """Create sample JSON files for testing."""
        tmp_path = Path(TEST_SCORE_FILES, "json_files")
        files = {}

        # Valid JSON files
        valid_simple = tmp_path / "valid_simple.json"
        files["valid_simple"] = str(valid_simple)

        valid_complex = tmp_path / "valid_complex.json"
        files["valid_complex"] = str(valid_complex)

        # Invalid JSON files
        invalid_syntax = tmp_path / "invalid_syntax.json"
        files["invalid_syntax"] = str(invalid_syntax)

        # Empty file
        empty_file = tmp_path / "empty.json"
        files["empty"] = str(empty_file)

        # Null content
        null_content = tmp_path / "null_content.json"
        files["null_content"] = str(null_content)

        # Non-JSON file
        text_file = tmp_path / "not_json.txt"
        files["not_json"] = str(text_file)

        return files

    def test_check_valid_json_file_with_valid_files(self, sample_json_files):
        """Test check_valid_json_file with valid JSON files."""
        assert check_valid_json_file(sample_json_files["valid_simple"]) == 1.0
        assert check_valid_json_file(sample_json_files["valid_complex"]) == 1.0

    def test_check_valid_json_file_with_invalid_files(self, sample_json_files):
        """Test check_valid_json_file with invalid files."""
        assert check_valid_json_file(sample_json_files["invalid_syntax"]) == 0.0
        assert check_valid_json_file(sample_json_files["empty"]) == 0.0
        assert check_valid_json_file(sample_json_files["not_json"]) == 0.0

    def test_check_valid_json_file_with_null_content(self, sample_json_files):
        """Test check_valid_json_file with null JSON content."""
        # Null is valid JSON but should return 0.0 according to the function
        assert check_valid_json_file(sample_json_files["null_content"]) == 0.0

    def test_check_valid_json_file_with_nonexistent_file(self):
        """Test check_valid_json_file with nonexistent file."""
        assert check_valid_json_file("nonexistent_file.json") == 0.0

    def test_check_valid_json_file_with_empty_input(self):
        """Test check_valid_json_file with empty/invalid input."""
        assert check_valid_json_file("") == 0.0
        assert check_valid_json_file("   ") == 0.0
        assert check_valid_json_file(None) == 0.0

    def test_check_valid_json_file_with_directory(self, tmp_path):
        """Test check_valid_json_file with directory instead of file."""
        test_dir = tmp_path / "test_directory"
        test_dir.mkdir()
        assert check_valid_json_file(str(test_dir)) == 0.0

    @given(st.text())
    def test_check_valid_json_file_property_always_returns_float(self, input_text):
        """Property test: check_valid_json_file should always return a float."""
        result = check_valid_json_file(input_text)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestStructureValidation:
    """Test structure validation functions that work with CIF files."""

    @pytest.fixture()
    def sample_cif_files(self, tmp_path):
        """Create sample CIF files for testing."""
        files = {}

        # Valid bulk structure CIF
        tmp_path = Path(TEST_SCORE_FILES, "cif_files")
        valid_bulk = tmp_path / "valid_bulk.cif"
        files["valid_bulk"] = str(valid_bulk)

        # Valid slab structure CIF
        valid_slab = tmp_path / "valid_slab.cif"
        files["valid_slab"] = str(valid_slab)

        # Valid CO2 molecule CIF
        valid_co2 = tmp_path / "valid_co2.cif"
        files["valid_co2"] = str(valid_co2)

        # Valid adsorption system (Si + CO)
        valid_adsorption = tmp_path / "valid_adsorption.cif"
        files["valid_adsorption"] = str(valid_adsorption)

        # Valid adsorption system (Si + CO) with element instead of symbol
        valid_adsorption2 = tmp_path / "valid_adsorption2.cif"
        files["valid_adsorption2"] = str(valid_adsorption2)

        # Invalid CIF files
        invalid_malformed = tmp_path / "invalid_malformed.cif"
        files["invalid_malformed"] = str(invalid_malformed)

        empty_cif = tmp_path / "empty.cif"
        empty_cif.write_text("")
        files["empty"] = str(empty_cif)

        no_atoms = tmp_path / "no_atoms.cif"
        files["no_atoms"] = str(no_atoms)

        return files

    def test_check_mp_structure_with_valid_files(self, sample_cif_files):
        """Test check_mp_structure with valid CIF files."""
        assert check_mp_structure(sample_cif_files["valid_bulk"]) == 1.0
        assert check_mp_structure(sample_cif_files["valid_slab"]) == 1.0

    def test_check_mp_structure_with_invalid_files(self, sample_cif_files):
        """Test check_mp_structure with invalid CIF files."""
        assert check_mp_structure(sample_cif_files["invalid_malformed"]) == 0.0
        assert check_mp_structure(sample_cif_files["empty"]) == 0.0

    def test_check_mp_structure_with_cif_string(self):
        """Test check_mp_structure with CIF string instead of file."""
        cif_string = """# generated using pymatgen
data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.44370237
_cell_length_b   5.44370237
_cell_length_c   5.44370237
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   Si8
_cell_volume   161.31810739
_cell_formula_units_Z   8
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_type_symbol
 _atom_type_oxidation_number
  Si0+  0.0
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si0+  Si0  1  0.75000000  0.75000000  0.25000000  1
  Si0+  Si1  1  0.00000000  0.50000000  0.50000000  1
  Si0+  Si2  1  0.75000000  0.25000000  0.75000000  1
  Si0+  Si3  1  0.00000000  0.00000000  0.00000000  1
  Si0+  Si4  1  0.25000000  0.75000000  0.75000000  1
  Si0+  Si5  1  0.50000000  0.50000000  0.00000000  1
  Si0+  Si6  1  0.25000000  0.25000000  0.25000000  1
  Si0+  Si7  1  0.50000000  0.00000000  0.50000000  1
"""
        assert check_mp_structure(cif_string) == 1.0

    def test_check_slab_structure_with_valid_files(self, sample_cif_files):
        """Test check_slab_structure with valid files."""
        assert check_slab_structure(sample_cif_files["valid_slab"]) == 1.0
        assert check_slab_structure(sample_cif_files["valid_bulk"]) == 1.0

    def test_check_slab_structure_with_invalid_files(self, sample_cif_files):
        """Test check_slab_structure with invalid files."""
        assert check_slab_structure(sample_cif_files["invalid_malformed"]) == 0.0
        assert check_slab_structure(sample_cif_files["empty"]) == 0.0

    def test_check_co2_molecule_structure_with_valid_co2(self, sample_cif_files):
        """Test check_co2_molecule_structure with valid CO2."""
        assert check_co2_molecule_structure(sample_cif_files["valid_co2"]) == 1.0

    def test_check_co2_molecule_structure_with_wrong_stoichiometry(self, tmp_path):
        """Test check_co2_molecule_structure with wrong stoichiometry."""
        # CO molecule (not CO2)
        co_file = tmp_path / "co_molecule.cif"
        assert check_co2_molecule_structure(str(co_file)) == 0.0

    def test_check_co2_molecule_structure_missing_elements(self, tmp_path):
        """Test check_co2_molecule_structure with missing elements."""
        # Only Carbon, no Oxygen
        c_only = tmp_path / "carbon_only.cif"
        assert check_co2_molecule_structure(str(c_only)) == 0

    def test_check_adsorption_structure_factory(self, sample_cif_files):
        """Test check_adsorption_structure factory function."""
        # Create scoring function for Si slab + CO adsorbate
        scorer = check_adsorption_structure(
            slab_elements=["Si"], adsorbate_elements=["C", "O"]
        )

        # Test with valid adsorption system
        assert scorer(sample_cif_files["valid_adsorption"]) == 1.0

        assert scorer(sample_cif_files["valid_adsorption2"]) == 1.0

        # Test with slab only (no adsorbate)
        assert scorer(sample_cif_files["valid_slab"]) == 0.0

        # Test with invalid file
        assert scorer(sample_cif_files["invalid_malformed"]) == 0.0

    def test_adsorption_structure_with_missing_elements(self, sample_cif_files):
        """Test adsorption structure with missing required elements."""
        # Look for Pt slab (not present in Si structure)
        scorer = check_adsorption_structure(
            slab_elements=["Pt"], adsorbate_elements=["C", "O"]
        )
        assert scorer(sample_cif_files["valid_adsorption"]) == 0.0

    @given(st.text())
    def test_structure_functions_handle_arbitrary_input(self, input_text):
        """Property test: structure functions handle arbitrary input gracefully."""
        functions = [
            check_mp_structure,
            check_slab_structure,
            check_co2_molecule_structure,
        ]

        for func in functions:
            result = func(input_text)
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
