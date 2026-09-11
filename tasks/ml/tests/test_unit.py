"""
Unit tests for core ML package functions.

These tests focus on individual functions in isolation.
"""

import json
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given
from ml.score import (
    check_mp_structure,
    compare_with_ground_truth,
    ml_pipeline_score,
    resolve_path,
)

# Import the tool objects to test
from ml.tools import (
    consolidate_polymorph_datasets,
    filter_json_with_strategy,
    select_polymorphs_with_strategy,
    sort_and_get_first_from_json,
)
from ml.utils import extract_path_from_answer, smart_resolve_path

from corral.utils.code_tools import execute_python_code


class TestUtilityFunctions:
    """Test utility functions."""

    @given(st.text())
    def test_extract_path_from_answer_with_random_text(self, text):
        """Test path extraction with random text input."""
        result = extract_path_from_answer(text)
        assert isinstance(result, str)

    def test_extract_path_from_answer_with_paths(self):
        """Test path extraction with known path patterns."""
        test_cases = [
            ("answer: /path/to/file.json", "/path/to/file.json"),
            ("`data/results.csv`", "data/results.csv"),
            ('"output/model.pkl"', "output/model.pkl"),
            ("'test/data.json'", "test/data.json"),
            ("The file is located at file.txt", "file.txt"),
            ("simple_file.json", "simple_file.json"),
        ]

        for input_text, expected in test_cases:
            result = extract_path_from_answer(input_text)
            assert expected in result or result == expected

    def test_smart_resolve_path_existing_file(self, temp_dir):
        """Test path resolution for existing files."""
        # Create a test file
        test_file = Path(temp_dir) / "test.json"
        test_file.write_text('{"test": "data"}')

        result = smart_resolve_path(str(test_file))
        assert result == str(test_file)

    def test_resolve_path_with_prefix(self):
        """Test resolve_path function with various prefixes."""
        test_cases = [
            "answer: /path/to/file.json",
            "file.json",
            "/absolute/path/file.json",
        ]

        for test_case in test_cases:
            result = resolve_path(test_case)
            assert isinstance(result, str)
            assert "answer:" not in result


class TestJSONProcessing:
    """Test JSON processing functions."""

    def test_sort_and_get_first_from_json_valid_data(self, sample_polymorph_json):
        """Test sorting and extraction with valid data."""
        result = sort_and_get_first_from_json.execute(
            polymorph_data_json=sample_polymorph_json,
            sort_key="energy_above_hull",
            return_key="material_id",
        )
        assert result in ["mp-149", "mp-2657"]

    def test_sort_and_get_first_from_json_invalid_key(self, sample_polymorph_json):
        """Test with invalid sort key."""
        with pytest.raises(KeyError):
            sort_and_get_first_from_json.execute(
                polymorph_data_json=sample_polymorph_json,
                sort_key="nonexistent_key",
                return_key="material_id",
            )

    def test_sort_and_get_first_from_json_empty_data(self):
        """Test with empty JSON array."""
        empty_json = json.dumps([])
        with pytest.raises(IndexError):
            sort_and_get_first_from_json.execute(
                polymorph_data_json=empty_json, sort_key="key", return_key="return_key"
            )

    @given(
        st.lists(
            st.fixed_dictionaries(
                {
                    "energy": st.floats(min_value=-100, max_value=100),
                    "id": st.one_of(
                        st.floats(min_value=-100, max_value=100), st.text()
                    ),
                },
                optional={"value": st.one_of(st.floats(), st.text())},
            ),
            min_size=1,
        )
    )
    def test_sort_and_get_first_hypothesis(self, data_list):
        """Property-based test for sorting function."""
        json_data = json.dumps(data_list)
        result = sort_and_get_first_from_json.execute(
            polymorph_data_json=json_data, sort_key="energy", return_key="id"
        )

        # Result should be the id of the item with minimum energy
        min_energy_item = min(data_list, key=lambda x: x["energy"])
        assert result == str(min_energy_item["id"])


class TestPolymorphSelection:
    """Test polymorph selection functions."""

    def test_select_polymorphs_with_strategy_most_stable(self, sample_polymorph_json):
        """Test most stable selection strategy."""
        result_json = select_polymorphs_with_strategy.execute(
            polymorphs_data=sample_polymorph_json,
            selection_strategy="most_stable",
            max_polymorphs=1,
            energy_threshold=1.0,
            is_path=False,
        )

        result = json.loads(result_json)
        assert len(result) <= 1
        assert isinstance(result, list)
        if result:
            assert result[0]["energy_above_hull"] == 0.0

    def test_select_polymorphs_with_strategy_diverse_energy(
        self, sample_polymorph_json
    ):
        """Test diverse energy selection strategy."""
        result_json = select_polymorphs_with_strategy.execute(
            polymorphs_data=sample_polymorph_json,
            selection_strategy="diverse_energy",
            max_polymorphs=2,
            energy_threshold=1.0,
            is_path=False,
        )

        result = json.loads(result_json)
        assert len(result) <= 2
        assert isinstance(result, list)

    def test_select_polymorphs_energy_threshold_filtering(self, sample_polymorph_json):
        """Test energy threshold filtering."""
        # Test with very low threshold - should get no results
        result_json = select_polymorphs_with_strategy.execute(
            polymorphs_data=sample_polymorph_json,
            selection_strategy="most_stable",
            max_polymorphs=10,
            energy_threshold=-1.0,  # Below all energies
            is_path=False,
        )

        result = json.loads(result_json)
        assert len(result) == 0

    def test_select_polymorphs_invalid_strategy(self, sample_polymorph_json):
        """Test with invalid selection strategy."""
        result_json = select_polymorphs_with_strategy.execute(
            polymorphs_data=sample_polymorph_json,
            selection_strategy="invalid_strategy",
            max_polymorphs=5,
            energy_threshold=1.0,
            is_path=False,
        )

        # Should still work but use default behavior
        result = json.loads(result_json)
        assert isinstance(result, list)


class TestDatasetConsolidation:
    """Test dataset consolidation functions."""

    def test_consolidate_polymorph_datasets_success(self, test_data_files):
        """Test successful consolidation of polymorph datasets."""
        composition_files = {
            "Al2O3": test_data_files["Al2O3"],
            "TiO2": test_data_files["TiO2"],
        }

        output_path = str(
            Path(test_data_files["Al2O3"]).parent / "consolidated_test.json"
        )

        result_json = consolidate_polymorph_datasets.execute(
            composition_files=composition_files, output_path=output_path
        )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["statistics"]["compositions_included"] == 2
        assert Path(output_path).exists()

        # Check consolidated file content
        with Path(output_path).open() as f:
            consolidated_data = json.load(f)

        assert len(consolidated_data) > 0
        assert all("source_composition" in item for item in consolidated_data)

    def test_consolidate_polymorph_datasets_missing_files(self):
        """Test consolidation with missing files."""
        composition_files = {
            "Al2O3": "/nonexistent/path.json",
            "TiO2": "/another/nonexistent/path.json",
        }

        result_json = consolidate_polymorph_datasets.execute(
            composition_files=composition_files, output_path="output.json"
        )

        result = json.loads(result_json)
        assert result["success"] is True  # Should succeed but with no data
        assert result["statistics"]["compositions_included"] == 0


class TestFilteringFunctions:
    """Test data filtering functions."""

    def test_filter_json_with_strategy_success(self, temp_dir, sample_polymorph_data):
        """Test successful JSON filtering."""
        # Create input file
        input_path = Path(temp_dir) / "input.json"
        output_path = Path(temp_dir) / "output.json"

        with input_path.open("w") as f:
            json.dump(sample_polymorph_data, f)

        custom_code = "filtered_data = [x for x in data if x['band_gap'] > 2.0]"

        result_json = filter_json_with_strategy.execute(
            input_json_path=str(input_path),
            output_json_path=str(output_path),
            custom_code=custom_code,
        )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["filtered_count"] == 1  # Only TiO2 has band_gap > 2.0
        assert Path(output_path).exists()

    def test_filter_json_with_strategy_file_not_found(self):
        """Test filtering with non-existent input file."""
        result_json = filter_json_with_strategy.execute(
            input_json_path="/nonexistent/input.json",
            output_json_path="/tmp/output.json",
            custom_code="filtered_data = data",
        )

        result = json.loads(result_json)
        assert result["success"] is False
        assert "not found" in result["error"]


class TestStructureValidation:
    """Test structure validation functions."""

    def test_check_mp_structure_with_valid_cif(self):
        """Test structure validation with valid CIF string."""
        cif_string = """# generated using pymatgen
data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.84927840
_cell_length_b   3.84927941
_cell_length_c   3.84927800
_cell_angle_alpha   60.00001213
_cell_angle_beta   60.00000347
_cell_angle_gamma   60.00001098
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   Si2
_cell_volume   40.32952685
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si  Si0  1  0.87500000  0.87500000  0.87500000  1
  Si  Si1  1  0.12500000  0.12500000  0.12500000  1
"""
        result = check_mp_structure(cif_string)
        assert result == 1.0

    def test_check_mp_structure_with_invalid_cif(self):
        """Test structure validation with invalid CIF."""
        invalid_cif = "this is not a valid CIF string"
        result = check_mp_structure(invalid_cif)
        assert result == 0.0

    def test_check_mp_structure_with_file_path(self, temp_dir):
        """Test structure validation with file path."""
        cif_file = Path(temp_dir) / "test.cif"
        cif_content = """# generated using pymatgen
data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.84927840
_cell_length_b   3.84927941
_cell_length_c   3.84927800
_cell_angle_alpha   60.00001213
_cell_angle_beta   60.00000347
_cell_angle_gamma   60.00001098
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   Si2
_cell_volume   40.32952685
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si  Si0  1  0.87500000  0.87500000  0.87500000  1
  Si  Si1  1  0.12500000  0.12500000  0.12500000  1
"""
        cif_file.write_text(cif_content)

        result = check_mp_structure(str(cif_file))
        assert result == 1.0


class TestComparisonFunctions:
    """Test data comparison functions."""

    def test_compare_with_ground_truth_strict_equal(self, temp_dir):
        """Test strict comparison with equal data."""
        data = {"key1": "value1", "key2": 42}

        # Create both files with same data
        gen_file = Path(temp_dir) / "generated.json"
        gt_file = Path(temp_dir) / "ground_truth.json"

        with gen_file.open("w") as f:
            json.dump(data, f)
        with gt_file.open("w") as f:
            json.dump(data, f)

        result = compare_with_ground_truth(str(gen_file), str(gt_file), "strict")
        assert result == 1.0

    def test_compare_with_ground_truth_strict_different(self, temp_dir):
        """Test strict comparison with different data."""
        gen_data = {"key1": "value1", "key2": 42}
        gt_data = {"key1": "value1", "key2": 43}  # Different value

        gen_file = Path(temp_dir) / "generated.json"
        gt_file = Path(temp_dir) / "ground_truth.json"

        with gen_file.open("w") as f:
            json.dump(gen_data, f)
        with gt_file.open("w") as f:
            json.dump(gt_data, f)

        result = compare_with_ground_truth(str(gen_file), str(gt_file), "strict")
        assert result == 0.0

    def test_compare_with_ground_truth_numerical_tolerance(self, temp_dir):
        """Test numerical comparison with tolerance."""
        gen_data = {"value": 1.001}
        gt_data = {"value": 1.000}

        gen_file = Path(temp_dir) / "generated.json"
        gt_file = Path(temp_dir) / "ground_truth.json"

        with gen_file.open("w") as f:
            json.dump(gen_data, f)
        with gt_file.open("w") as f:
            json.dump(gt_data, f)

        # Should pass with 0.1% tolerance
        result = compare_with_ground_truth(
            str(gen_file), str(gt_file), "numerical", tolerance=0.01
        )
        assert result == 1.0

        # Should fail with 0.01% tolerance
        result = compare_with_ground_truth(
            str(gen_file), str(gt_file), "numerical", tolerance=0.0001
        )
        assert result == 0.0

    def test_compare_with_ground_truth_missing_files(self):
        """Test comparison with missing files."""
        result = compare_with_ground_truth(
            "/nonexistent/gen.json", "/nonexistent/gt.json", "strict"
        )
        assert result == 0.0


class TestMLPipelineScoring:
    """Test ML pipeline scoring functions."""

    def test_ml_pipeline_score_nonexistent_model(self):
        """Test scoring with non-existent model file."""
        result = ml_pipeline_score("/nonexistent/model.pkl")
        assert result == 0.0

    def test_ml_pipeline_score_with_model(self, sample_xgboost_model, temp_dir):
        """Test scoring with actual model file."""
        # Create some evaluation results
        eval_file = Path(temp_dir) / "evaluation.json"
        eval_data = {
            "evaluation_metrics": {"r2": 0.85, "mae": 0.25},
            "cross_validation_results": {"r2_mean": 0.80},
        }

        with eval_file.open("w") as f:
            json.dump(eval_data, f)

        result = ml_pipeline_score(sample_xgboost_model)
        assert 0.0 <= result <= 1.0
        assert result >= 0.3  # Base score for model existence


@given(
    st.lists(
        st.dictionaries(
            keys=st.sampled_from(["energy", "density", "volume", "formation_energy"]),
            values=st.floats(min_value=-100, max_value=100, allow_nan=False),
        ),
        min_size=1,
        max_size=10,
    )
)
def test_polymorph_selection_property_based(data_list):
    """Property-based test for polymorph selection."""
    # Ensure all items have required keys
    valid_data = []
    for item in data_list:
        if "energy" in item:
            item["energy_above_hull"] = item["energy"]
            valid_data.append(item)

    assume(len(valid_data) > 0)

    json_data = json.dumps(valid_data)

    result_json = select_polymorphs_with_strategy.execute(
        polymorphs_data=json_data,
        selection_strategy="most_stable",
        max_polymorphs=len(valid_data),
        energy_threshold=1000.0,  # High threshold to include all
        is_path=False,
    )

    result = json.loads(result_json)

    # Properties that should always hold
    assert isinstance(result, list)
    assert len(result) <= len(valid_data)

    # If we got results, they should be sorted by energy
    if len(result) > 1:
        energies = [item["energy_above_hull"] for item in result]
        assert energies == sorted(energies)


class TestPythonExecution:
    """Test Python code execution functions."""

    def test_execute_python_code_simple(self):
        """Test simple Python code execution."""
        code = "result = 2 + 2"
        result_json = execute_python_code.execute(python_code=code)

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["execution_result"]["result"] == 4

    def test_execute_python_code_with_input(self):
        """Test Python code execution with input data."""
        code = "output = sum(input_data)"
        input_data = json.dumps([1, 2, 3, 4, 5])

        result_json = execute_python_code.execute(
            python_code=code, input_data=input_data
        )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["execution_result"]["output"] == 15

    def test_execute_python_code_syntax_error(self):
        """Test Python code execution with syntax error."""
        code = "this is not valid python code {"
        result_json = execute_python_code.execute(python_code=code)

        result = json.loads(result_json)
        assert result["success"] is False
        assert "error" in result

    def test_execute_python_code_runtime_error(self):
        """Test Python code execution with runtime error."""
        code = "result = undefined_variable + 1"
        result_json = execute_python_code.execute(python_code=code)

        result = json.loads(result_json)
        assert result["success"] is False
        assert "error" in result
