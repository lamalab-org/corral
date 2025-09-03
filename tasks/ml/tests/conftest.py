# common fixtures and utilities for testing.
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

os.environ["CORRAL_WORK_DIR"] = str(Path(__file__).parent / "test_data")
TEMP_DIR = Path(os.environ["CORRAL_WORK_DIR"])


@pytest.fixture()
def temp_dir():
    """Create a temporary directory for tests."""
    return TEMP_DIR


@pytest.fixture()
def sample_polymorph_data():
    """Sample polymorph data for testing."""
    return [
        {
            "material_id": "mp-149",
            "cif": """# CIF sample data
# generated using pymatgen
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
""",
            "energy_above_hull": 0.0,
            "formation_energy_per_atom": -4.2,
            "band_gap": 1.2,
            "density": 2.33,
            "volume": 163.7,
            "nsites": 8,
            "space_group": "F d -3 m",
            "is_stable": True,
        },
        {
            "material_id": "mp-2657",
            "cif": """# CIF sample data for TiO2
data_TiO2
_cell_length_a 4.641
_cell_length_b 4.641
_cell_length_c 2.969
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 4_2/m n m'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ti1 0.0 0.0 0.0
O1 0.306 0.306 0.0
""",
            "energy_above_hull": 0.0,
            "formation_energy_per_atom": -8.1,
            "band_gap": 3.2,
            "density": 4.25,
            "volume": 62.1,
            "nsites": 6,
            "space_group": "P 4_2/m n m",
            "is_stable": True,
        },
    ]


@pytest.fixture()
def sample_polymorph_json(sample_polymorph_data):
    """Sample polymorph data as JSON string."""
    return json.dumps(sample_polymorph_data, indent=2)


@pytest.fixture()
def sample_csv_data():
    """Sample CSV dataset for ML testing."""
    data = {
        "density": [2.33, 4.25, 3.98, 2.65],
        "volume": [163.7, 62.1, 85.2, 120.4],
        "nsites": [8, 6, 4, 10],
        "band_gap": [1.2, 3.2, 2.1, 0.8],
        "formation_energy_per_atom": [-4.2, -8.1, -6.5, -3.8],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def mock_mp_api():
    """Mock Materials Project API responses."""
    with patch("ml.tools.MPRester") as mock_rester:
        mock_instance = Mock()
        mock_rester.return_value.__enter__.return_value = mock_instance

        # Mock structure response
        mock_structure = Mock()
        mock_structure.to.return_value = """# Sample CIF
data_test
_cell_length_a 5.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.0 0.0 0.0
"""

        mock_doc = Mock()
        mock_doc.structure = mock_structure
        mock_instance.materials.summary.search.return_value = [mock_doc]

        yield mock_instance


@pytest.fixture()
def test_data_files(temp_dir, sample_polymorph_data):
    """Create test data files similar to the existing test structure."""
    # Create directories
    batch_dir = Path(temp_dir) / "test_batch_data"

    ml_dir = Path(temp_dir) / "test_ml_data"

    # Create polymorph files
    compositions = ["Al2O3", "TiO2", "SiO2"]
    files = {}

    for comp in compositions:
        file_path = batch_dir / f"{comp}_polymorphs.json"
        files[comp] = str(file_path)

    # Create consolidated file
    consolidated_path = batch_dir / "consolidated_polymorphs.json"
    all_data = []
    for data in sample_polymorph_data:
        data_copy = data.copy()
        data_copy["source_composition"] = "TiO2"  # Add source composition
        all_data.append(data_copy)
    files["consolidated"] = str(consolidated_path)

    train_path = ml_dir / "train.csv"
    test_path = ml_dir / "test.csv"

    files["train_csv"] = str(train_path)
    files["test_csv"] = str(test_path)

    return files


@pytest.fixture()
def sample_xgboost_model(temp_dir):
    """Create a simple trained XGBoost model for testing."""
    import joblib
    import xgboost as xgb
    from sklearn.datasets import make_regression

    # Create sample data
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)

    # Train simple model
    model = xgb.XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Save model
    model_path = Path(temp_dir) / "test_model.pkl"
    joblib.dump(model, model_path)

    return str(model_path)
