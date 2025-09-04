import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import hypothesis.strategies as st
import pytest
from hypothesis import given

# Import your tool objects (these have .execute() methods)
from ml.tools import (
    evaluate_xgboost_model,
    get_bulk_polymorphs_data,
    get_structure_from_mp_text,
    sort_and_get_first_from_json,
    train_xgboost_model,
)

from corral.utils.code_tools import execute_python_code

# Mock data directory
MOCK_DATA_DIR = Path(__file__).parent / "mock_data"


def skip_if_no_api_key():
    """Skip test if MP_API_KEY is not available"""
    return pytest.mark.skipif(
        not os.getenv("MP_API_KEY"), reason="MP_API_KEY not available in environment"
    )


@pytest.fixture()
def mock_mp_rester(mocker):
    """Mock the Materials Project API for testing."""
    # Mock structure response
    mock_structure = MagicMock()
    mock_structure.to.return_value = """data_Si
_cell_length_a 5.468
_cell_length_b 5.468
_cell_length_c 5.468
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.125 0.125 0.125
"""

    mock_doc = MagicMock()
    mock_doc.structure = mock_structure

    mock_mpr_instance = MagicMock()
    mock_mpr_instance.materials.summary.search.return_value = [mock_doc]

    mock_mpr_class = MagicMock()
    mock_mpr_class.return_value.__enter__.return_value = mock_mpr_instance

    mocker.patch("ml.tools.MPRester", mock_mpr_class)
    return mock_mpr_instance


@pytest.fixture(scope="module")
def silicon_cif():
    """Sample Silicon CIF for testing."""
    return """data_Si
_cell_length_a 5.468
_cell_length_b 5.468
_cell_length_c 5.468
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'F d -3 m'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.125 0.125 0.125
"""


class TestMaterialsProjectTools:
    """Test tools that interact with Materials Project API."""

    @skip_if_no_api_key()
    def test_get_structure_from_mp_text_real_api(self):
        """Test structure retrieval with real API (requires API key)."""
        cif_str = get_structure_from_mp_text.execute(mp_id="mp-149")

        assert isinstance(cif_str, str)
        assert "data_Si" in cif_str

        # Validate that we can parse the CIF
        from pymatgen.core import Structure

        struct = Structure.from_str(cif_str, fmt="cif")
        assert struct.composition.reduced_formula == "Si"

    @skip_if_no_api_key()
    def test_get_bulk_polymorphs_data_real_api(self):
        """Test polymorph retrieval with real API."""
        polymorphs_json = get_bulk_polymorphs_data.execute(composition="TiO2")

        data = json.loads(polymorphs_json)
        assert isinstance(data, list)
        assert len(data) > 1

        # Check that data is sorted by energy_above_hull
        energies = [item["energy_above_hull"] for item in data]
        assert energies == sorted(energies)

        # Verify structure of polymorph data
        for item in data:
            assert "material_id" in item
            assert "cif" in item
            assert "energy_above_hull" in item
            assert "formation_energy_per_atom" in item


class TestJSONProcessingTools:
    """Test tools that process JSON data."""

    def test_sort_and_get_first_from_json_basic(self):
        """Test basic sorting and extraction."""
        test_data = [
            {"id": "A", "energy": 0.5},
            {"id": "B", "energy": 0.0},  # Should be first after sorting
            {"id": "C", "energy": 1.0},
        ]

        json_data = json.dumps(test_data)

        result = sort_and_get_first_from_json.execute(
            polymorph_data_json=json_data, sort_key="energy", return_key="id"
        )

        assert result == "B"  # Item with lowest energy

    def test_sort_and_get_first_from_json_with_polymorphs(self):
        """Test with realistic polymorph data."""
        polymorph_data = [
            {
                "material_id": "mp-2657",
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -9.4,
                "is_stable": True,
            },
            {
                "material_id": "mp-1234",
                "energy_above_hull": 0.1,
                "formation_energy_per_atom": -9.2,
                "is_stable": False,
            },
        ]

        json_data = json.dumps(polymorph_data)

        # Get most stable material
        stable_id = sort_and_get_first_from_json.execute(
            polymorph_data_json=json_data,
            sort_key="energy_above_hull",
            return_key="material_id",
        )

        assert stable_id == "mp-2657"

    def test_sort_and_get_first_empty_data(self):
        """Test error handling with empty data."""
        empty_json = json.dumps([])

        with pytest.raises(IndexError):
            sort_and_get_first_from_json.execute(
                polymorph_data_json=empty_json, sort_key="energy", return_key="id"
            )

    @given(
        st.lists(
            st.dictionaries(
                keys=st.sampled_from(["id", "energy", "value"]),
                values=st.one_of(
                    st.text(min_size=1, max_size=10),
                    st.floats(min_value=-100, max_value=100, allow_nan=False),
                ),
            ),
            min_size=1,
            max_size=20,
        )
    )
    def test_sort_and_get_first_property_based(self, data_list):
        """Property-based test for sorting function."""
        # Filter to ensure we have required keys with numeric values
        valid_data = []
        for item in data_list:
            if "id" in item and "energy" in item:
                try:
                    # Ensure energy is numeric
                    if isinstance(item["energy"], int | float):
                        valid_data.append(item)
                except (TypeError, ValueError):
                    pass

        if len(valid_data) == 0:
            return  # Skip if no valid data

        json_data = json.dumps(valid_data)

        result = sort_and_get_first_from_json.execute(
            polymorph_data_json=json_data, sort_key="energy", return_key="id"
        )

        # Result should be the ID of the item with minimum energy
        min_energy_item = min(valid_data, key=lambda x: x["energy"])
        assert str(result) == str(min_energy_item["id"])


class TestPythonExecutionTools:
    """Test tools that execute Python code."""

    def test_execute_python_code_simple(self):
        """Test simple code execution."""
        code = "result = 10 * 5"

        result_json = execute_python_code.execute(python_code=code)
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["execution_result"]["result"] == 50

    def test_execute_python_code_with_input_data(self):
        """Test code execution with input data."""
        code = """
# Process the input data
total = sum(input_data)
average = total / len(input_data)
output = {
    'total': total,
    'average': average,
    'count': len(input_data)
}
"""
        input_data = json.dumps([1, 2, 3, 4, 5])

        result_json = execute_python_code.execute(
            python_code=code, input_data=input_data
        )

        result = json.loads(result_json)
        assert result["success"] is True

        output = result["execution_result"]["output"]
        assert output["total"] == 15
        assert output["average"] == 3.0
        assert output["count"] == 5

    def test_execute_python_code_syntax_error(self):
        """Test error handling for syntax errors."""
        bad_code = "this is not valid python syntax {"

        result_json = execute_python_code.execute(python_code=bad_code)
        result = json.loads(result_json)

        assert result["success"] is False
        assert "error" in result

    def test_execute_python_code_runtime_error(self):
        """Test error handling for runtime errors."""
        code = "result = undefined_variable * 2"

        result_json = execute_python_code.execute(python_code=code)
        result = json.loads(result_json)

        assert result["success"] is False
        assert "error" in result

    def test_execute_python_code_with_timeout(self):
        """Test code execution with timeout."""
        # Code that should complete quickly
        fast_code = "result = sum(range(100))"

        result_json = execute_python_code.execute(
            python_code=fast_code,
            timeout=5,  # 5 second timeout
        )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["execution_result"]["result"] == sum(range(100))


class TestMLTools:
    """Test machine learning tools."""

    def test_train_xgboost_model(self, temp_dir):
        """Test XGBoost model training."""
        # Create sample training data
        import pandas as pd

        train_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
                "target": [1.1, 2.1, 3.1, 4.1, 5.1],
            }
        )

        test_data = pd.DataFrame(
            {"feature1": [6, 7], "feature2": [12, 14], "target": [6.1, 7.1]}
        )

        train_path = Path(temp_dir) / "train.csv"
        test_path = Path(temp_dir) / "test.csv"
        model_path = Path(temp_dir) / "model.pkl"

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        # Train model using the tool
        result_json = train_xgboost_model.execute(
            train_data_path=str(train_path),
            test_data_path=str(test_path),
            model_save_path=str(model_path),
            target_column="target",
            hyperparameters={"n_estimators": 5, "max_depth": 2},
        )

        result = json.loads(result_json)
        assert result["success"] is True
        assert model_path.exists()
        assert "test_metrics" in result

        # Verify model can be loaded
        import joblib

        model = joblib.load(model_path)
        assert hasattr(model, "predict")

    def test_evaluate_xgboost_model(self, temp_dir, sample_xgboost_model):
        """Test XGBoost model evaluation."""
        # Create test data
        import pandas as pd

        test_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [2, 4, 6, 8],
                "feature3": [0.5, 1.0, 1.5, 2.0],
                "feature4": [10, 20, 30, 40],
                "target": [1.1, 2.1, 3.1, 4.1],
            }
        )

        test_path = Path(temp_dir) / "test.csv"
        test_data.to_csv(test_path, index=False)

        # Evaluate model using the tool
        result_json = evaluate_xgboost_model.execute(
            model_path=sample_xgboost_model,
            test_data_path=str(test_path),
            target_column="target",
            detailed_analysis=True,
        )

        result = json.loads(result_json)
        assert result["success"] is True

        metrics = result["evaluation_metrics"]
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "feature_importance" in metrics


class TestWorkflowIntegration:
    """Test integration of multiple tools in workflows."""

    @skip_if_no_api_key()
    def test_materials_to_ml_workflow(self):
        """Test complete workflow from materials data to ML."""
        # Step 1: Get polymorph data
        polymorphs_json = get_bulk_polymorphs_data.execute(composition="TiO2")
        polymorphs = json.loads(polymorphs_json)

        assert len(polymorphs) > 0

        # Step 2: Select most stable polymorph
        stable_id = sort_and_get_first_from_json.execute(
            polymorph_data_json=polymorphs_json,
            sort_key="energy_above_hull",
            return_key="material_id",
        )

        assert stable_id in [p["material_id"] for p in polymorphs]

        # Step 3: Process data with Python code
        processing_code = """
# Extract properties for ML
features = []
targets = []

for material in input_data:
    if material.get('band_gap') is not None:
        feature_vector = [
            material.get('density', 0),
            material.get('volume', 0),
            material.get('nsites', 0),
            material.get('band_gap', 0)
        ]
        features.append(feature_vector)
        targets.append(material.get('formation_energy_per_atom', 0))

result = {
    'features': features,
    'targets': targets,
    'num_samples': len(features)
}
"""

        result_json = execute_python_code.execute(
            python_code=processing_code, input_data=polymorphs_json
        )

        result = json.loads(result_json)
        assert result["success"] is True

        processed_data = result["execution_result"]["result"]
        assert "features" in processed_data
        assert "targets" in processed_data
        assert processed_data["num_samples"] > 0
