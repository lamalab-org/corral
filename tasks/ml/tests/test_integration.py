"""
Integration tests for ML package workflows.

These tests verify complete workflows and function interactions.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Import the tool objects for integration testing
from ml.tools import (
    batch_retrieve_polymorphs,
    consolidate_polymorph_datasets,
    evaluate_xgboost_model,
    filter_json_with_strategy,
    perform_cross_validation,
    prepare_tabular_dataset,
    select_polymorphs_with_strategy_to_file,
    train_xgboost_model,
)

from corral.utils.code_tools import execute_python_code


class TestDataRetrievalWorkflow:
    """Test complete data retrieval and processing workflow."""

    @patch("ml.tools.get_bulk_polymorphs_data_func")
    def test_batch_retrieve_to_consolidation_workflow(self, mock_get_data, temp_dir):
        """Test workflow from batch retrieval to consolidation."""
        # Mock the API response
        mock_polymorph_data = [
            {
                "material_id": "mp-149",
                "cif": "mock_cif_data",
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -4.2,
                "band_gap": 1.2,
                "density": 2.33,
                "volume": 163.7,
                "nsites": 8,
                "space_group": "F d -3 m",
                "is_stable": True,
            }
        ]
        mock_get_data.return_value = json.dumps(mock_polymorph_data)

        # Step 1: Batch retrieve polymorphs
        compositions = ["TiO2", "Al2O3"]
        save_directory = str(Path(temp_dir) / "batch_data")

        batch_result_json = batch_retrieve_polymorphs.execute(
            compositions=compositions,
            max_energy_above_hull=0.5,
            max_per_composition=5,
            save_directory=save_directory,
        )

        batch_result = json.loads(batch_result_json)
        assert batch_result["total_polymorphs"] > 0
        assert len(batch_result["successful_compositions"]) > 0

        # Step 2: Consolidate the datasets
        consolidated_path = str(Path(temp_dir) / "consolidated.json")
        consolidation_result_json = consolidate_polymorph_datasets.execute(
            composition_files=batch_result["composition_files"],
            output_path=consolidated_path,
        )

        consolidation_result = json.loads(consolidation_result_json)
        assert consolidation_result["success"] is True
        assert Path(consolidated_path).exists()

        # Verify consolidated data has source composition info
        with Path(consolidated_path).open() as f:
            consolidated_data = json.load(f)

        assert len(consolidated_data) > 0
        assert all("source_composition" in item for item in consolidated_data)

    def test_selection_and_filtering_workflow(self, test_data_files, temp_dir):
        """Test polymorph selection and filtering workflow."""
        # Step 1: Select polymorphs with strategy
        selected_path = str(Path(temp_dir) / "selected.json")

        selected_file = select_polymorphs_with_strategy_to_file.execute(
            polymorphs_data=test_data_files["consolidated"],
            save_path=selected_path,
            selection_strategy="most_stable",
            max_polymorphs=1,
            energy_threshold=1.0,
            is_path=True,
        )

        assert Path(selected_file).exists()

        # Step 2: Apply custom filtering
        filtered_path = str(Path(temp_dir) / "filtered.json")
        custom_code = "filtered_data = [x for x in data if x.get('band_gap', 0) > 0]"

        filter_result_json = filter_json_with_strategy.execute(
            input_json_path=selected_file,
            output_json_path=filtered_path,
            custom_code=custom_code,
        )

        filter_result = json.loads(filter_result_json)
        assert filter_result["success"] is True
        assert Path(filtered_path).exists()

        # Verify filtering worked
        with Path(filtered_path).open() as f:
            filtered_data = json.load(f)

        assert all(item.get("band_gap", 0) > 0 for item in filtered_data)


class TestMLPipelineWorkflow:
    """Test complete ML pipeline workflow."""

    def test_dataset_preparation_to_training_workflow(self, test_data_files, temp_dir):
        """Test workflow from dataset preparation to model training."""
        # Step 1: Prepare tabular dataset
        output_path = str(Path(temp_dir) / "ml_dataset")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        dataset_result_json = prepare_tabular_dataset.execute(
            polymorphs_json_path=test_data_files["consolidated"],
            output_path=output_path,
            target_property="formation_energy_per_atom",
            feature_engineering="basic",
            test_split=0.2,
            normalize=True,
        )

        dataset_result = json.loads(dataset_result_json)
        assert dataset_result["success"] is True
        assert Path(dataset_result["train_path"]).exists()
        assert Path(dataset_result["test_path"]).exists()

        # Step 2: Train XGBoost model
        model_path = str(Path(temp_dir) / "model.pkl")

        training_result_json = train_xgboost_model.execute(
            train_data_path=dataset_result["train_path"],
            test_data_path=dataset_result["test_path"],
            model_save_path=model_path,
            target_column="formation_energy_per_atom",
            hyperparameters={"n_estimators": 10, "max_depth": 3},  # Small for testing
        )

        training_result = json.loads(training_result_json)
        assert training_result["success"] is True
        assert Path(model_path).exists()
        assert "test_metrics" in training_result
        assert "r2" in training_result["test_metrics"]

        # Step 3: Evaluate the model
        evaluation_result_json = evaluate_xgboost_model.execute(
            model_path=model_path,
            test_data_path=dataset_result["test_path"],
            target_column="formation_energy_per_atom",
            detailed_analysis=True,
        )

        evaluation_result = json.loads(evaluation_result_json)
        assert evaluation_result["success"] is True
        assert "evaluation_metrics" in evaluation_result

        metrics = evaluation_result["evaluation_metrics"]
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "feature_importance" in metrics

        # Step 4: Perform cross-validation
        cv_result_json = perform_cross_validation.execute(
            train_data_path=dataset_result["train_path"],
            target_column="formation_energy_per_atom",
            cv_folds=3,  # Small number for testing
            hyperparameters={"n_estimators": 5},
        )

        cv_result = json.loads(cv_result_json)
        assert cv_result["success"] is True
        assert "cross_validation_results" in cv_result

        cv_results = cv_result["cross_validation_results"]
        assert "r2_mean" in cv_results
        assert "r2_std" in cv_results
        assert "mae_mean" in cv_results
        assert len(cv_results["r2_scores"]) == 3

    def test_feature_engineering_workflow(self, temp_dir):
        """Test advanced feature engineering workflow."""
        # Create a more complex dataset for feature engineering
        complex_polymorphs = [
            {
                "material_id": "mp-149",
                "cif": """data_Si
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
Si2 0.875 0.875 0.875
""",
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -4.2,
                "band_gap": 1.2,
                "density": 2.33,
                "volume": 163.7,
                "nsites": 8,
                "space_group": "F d -3 m",
                "is_stable": True,
            }
        ]

        # Save complex data
        input_path = Path(temp_dir) / "complex_polymorphs.json"
        with input_path.open("w") as f:
            json.dump(complex_polymorphs, f)

        # Test advanced feature engineering
        output_path = str(Path(temp_dir) / "advanced_dataset")

        result_json = prepare_tabular_dataset.execute(
            polymorphs_json_path=str(input_path),
            output_path=output_path,
            target_property="formation_energy_per_atom",
            feature_engineering="advanced",
            test_split=0.5,  # 50% split for small dataset
            normalize=True,
        )

        result = json.loads(result_json)

        if result["success"]:
            # Check that advanced features were created
            metadata_path = result["metadata_path"]
            with Path(metadata_path).open() as f:
                metadata = json.load(f)

            features = metadata["features"]

            # Should have more features than basic mode
            assert len(features) > 5  # Basic mode has ~5 features

            # Check for advanced features
            advanced_feature_indicators = [
                "lattice_",
                "num_species",
                "space_group",
                "electronegativity",
            ]
            has_advanced = any(
                any(indicator in feature for feature in features)
                for indicator in advanced_feature_indicators
            )
            assert has_advanced

    def test_model_performance_validation_workflow(self, test_data_files, temp_dir):
        """Test model performance validation and comparison workflow."""
        # Prepare dataset
        output_path = str(Path(temp_dir) / "validation_dataset")

        dataset_result_json = prepare_tabular_dataset.execute(
            polymorphs_json_path=test_data_files["consolidated"],
            output_path=output_path,
            target_property="formation_energy_per_atom",
            feature_engineering="basic",
            test_split=0.3,
            normalize=True,
        )

        dataset_result = json.loads(dataset_result_json)
        assert dataset_result["success"] is True

        # Train multiple models with different hyperparameters
        model_configs = [
            {"n_estimators": 5, "max_depth": 2},
            {"n_estimators": 10, "max_depth": 3},
            {"n_estimators": 8, "max_depth": 4},
        ]

        model_results = []

        for i, config in enumerate(model_configs):
            model_path = str(Path(temp_dir) / f"model_{i}.pkl")

            training_result_json = train_xgboost_model.execute(
                train_data_path=dataset_result["train_path"],
                test_data_path=dataset_result["test_path"],
                model_save_path=model_path,
                target_column="formation_energy_per_atom",
                hyperparameters=config,
            )

            training_result = json.loads(training_result_json)
            if training_result["success"]:
                # Evaluate each model
                eval_result_json = evaluate_xgboost_model.execute(
                    model_path=model_path,
                    test_data_path=dataset_result["test_path"],
                    target_column="formation_energy_per_atom",
                    detailed_analysis=True,
                )

                eval_result = json.loads(eval_result_json)
                if eval_result["success"]:
                    model_results.append(
                        {"config": config, "metrics": eval_result["evaluation_metrics"]}
                    )

        # Should have at least one successful model
        assert len(model_results) > 0

        # All models should have reasonable metrics
        for model_result in model_results:
            metrics = model_result["metrics"]
            assert "r2" in metrics
            assert "mae" in metrics
            assert isinstance(metrics["r2"], int | float)
            assert isinstance(metrics["mae"], int | float)


class TestErrorHandlingWorkflow:
    """Test error handling in workflows."""

    def test_dataset_prep_with_missing_target(self, temp_dir):
        """Test dataset preparation with missing target property."""
        # Create data without the target property
        incomplete_data = [
            {
                "material_id": "mp-149",
                "density": 2.33,
                "volume": 163.7,
                # Missing formation_energy_per_atom
            }
        ]

        input_path = Path(temp_dir) / "incomplete.json"
        with Path(input_path).open("w") as f:
            json.dump(incomplete_data, f)

        output_path = str(Path(temp_dir) / "failed_dataset")

        result_json = prepare_tabular_dataset.execute(
            polymorphs_json_path=str(input_path),
            output_path=output_path,
            target_property="formation_energy_per_atom",
            feature_engineering="basic",
        )

        result = json.loads(result_json)
        assert result["success"] is False
        assert "No valid samples found" in result["error"]

    def test_model_training_with_invalid_data(self, temp_dir):
        """Test model training with invalid CSV data."""
        # Create invalid CSV file
        invalid_csv = Path(temp_dir) / "invalid.csv"
        invalid_csv.write_text("not,a,valid,csv,format\n1,2,text,4,5")

        model_path = str(Path(temp_dir) / "model.pkl")

        result_json = train_xgboost_model.execute(
            train_data_path=str(invalid_csv),
            test_data_path=str(invalid_csv),  # Same invalid file for both
            model_save_path=model_path,
            target_column="nonexistent_column",
        )

        result = json.loads(result_json)
        assert result["success"] is False
        assert "error" in result

    def test_evaluation_with_nonexistent_model(self, test_data_files):
        """Test model evaluation with non-existent model file."""
        result_json = evaluate_xgboost_model.execute(
            model_path="/nonexistent/model.pkl",
            test_data_path=test_data_files["test_csv"],
            target_column="formation_energy_per_atom",
        )

        result = json.loads(result_json)
        assert result["success"] is False
        assert "error" in result


class TestDataIntegrityWorkflow:
    """Test data integrity throughout workflows."""

    def test_data_consistency_through_pipeline(self, test_data_files, temp_dir):
        """Test that data remains consistent through the entire pipeline."""
        # Step 1: Load original data and count samples
        with Path(test_data_files["consolidated"]).open() as f:
            original_data = json.load(f)

        original_count = len(original_data)
        original_material_ids = {item["material_id"] for item in original_data}

        # Step 2: Select subset
        selected_path = str(Path(temp_dir) / "selected.json")
        select_polymorphs_with_strategy_to_file.execute(
            polymorphs_data=test_data_files["consolidated"],
            save_path=selected_path,
            selection_strategy="most_stable",
            max_polymorphs=min(2, original_count),  # Select at most 2
            energy_threshold=10.0,  # High threshold
            is_path=True,
        )

        with Path(selected_path).open() as f:
            selected_data = json.load(f)

        selected_count = len(selected_data)
        selected_material_ids = {item["material_id"] for item in selected_data}

        # Verify selection worked correctly
        assert selected_count <= min(2, original_count)
        assert selected_material_ids.issubset(original_material_ids)

        # Step 3: Prepare ML dataset
        if selected_count > 0:
            output_path = str(Path(temp_dir) / "integrity_dataset")

            result_json = prepare_tabular_dataset.execute(
                polymorphs_json_path=selected_path,
                output_path=output_path,
                target_property="formation_energy_per_atom",
                feature_engineering="basic",
                test_split=0.3,
                normalize=True,
            )

            result = json.loads(result_json)

            if result["success"]:
                # Check that total samples match
                total_ml_samples = (
                    result["dataset_info"]["train_samples"]
                    + result["dataset_info"]["test_samples"]
                )
                assert (
                    total_ml_samples <= selected_count
                )  # May be less due to invalid data filtering

                # Verify data types in CSV files
                train_df = pd.read_csv(result["train_path"])
                test_df = pd.read_csv(result["test_path"])

                # Check that target column exists and is numeric
                assert "formation_energy_per_atom" in train_df.columns
                assert "formation_energy_per_atom" in test_df.columns
                assert pd.api.types.is_numeric_dtype(
                    train_df["formation_energy_per_atom"]
                )
                assert pd.api.types.is_numeric_dtype(
                    test_df["formation_energy_per_atom"]
                )

    def test_feature_consistency_in_ml_pipeline(self, test_data_files, temp_dir):
        """Test that features remain consistent between train and test sets."""
        output_path = str(Path(temp_dir) / "feature_consistency_dataset")

        result_json = prepare_tabular_dataset.execute(
            polymorphs_json_path=test_data_files["consolidated"],
            output_path=output_path,
            target_property="formation_energy_per_atom",
            feature_engineering="basic",
            test_split=0.4,
            normalize=True,
        )

        result = json.loads(result_json)

        if result["success"]:
            train_df = pd.read_csv(result["train_path"])
            test_df = pd.read_csv(result["test_path"])

            # Feature columns should be identical
            train_features = set(train_df.columns) - {"formation_energy_per_atom"}
            test_features = set(test_df.columns) - {"formation_energy_per_atom"}

            assert train_features == test_features

            # Same number of features
            assert len(train_features) == len(test_features)
            assert len(train_features) > 0

            # No missing values in feature columns
            for col in train_features:
                assert not train_df[col].isna().any()
                assert not test_df[col].isna().any()

            # Feature values should be reasonable (after normalization)
            for col in train_features:
                train_values = train_df[col]
                _test_values = test_df[col]

                # Should be roughly normalized (mean ~0, std ~1) for training set
                if len(train_values) > 1:
                    assert abs(train_values.mean()) < 2.0  # Allow some deviation
                    assert 0.1 < train_values.std() < 5.0  # Reasonable std range


class TestPythonCodeIntegration:
    """Test Python code execution integration with other tools."""

    def test_python_code_with_data_processing(self, sample_polymorph_data, temp_dir):
        """Test Python code execution for data processing tasks."""
        # Use Python code to process polymorph data
        input_data = json.dumps(sample_polymorph_data)

        processing_code = """
import json

# Calculate average formation energy
avg_formation_energy = sum(item['formation_energy_per_atom'] for item in input_data) / len(input_data)

# Count stable materials
stable_count = sum(1 for item in input_data if item.get('is_stable', False))

# Find material with highest band gap
max_bandgap_material = max(input_data, key=lambda x: x.get('band_gap', 0))

result = {
    'avg_formation_energy': avg_formation_energy,
    'stable_count': stable_count,
    'max_bandgap_material_id': max_bandgap_material['material_id'],
    'max_band_gap': max_bandgap_material['band_gap']
}
"""

        output_file = str(Path(temp_dir) / "processing_results.json")

        result_json = execute_python_code.execute(
            python_code=processing_code,
            input_data=input_data,
            save_output_to=output_file,
        )

        result = json.loads(result_json)
        assert result["success"] is True

        exec_result = result["execution_result"]["result"]
        assert "avg_formation_energy" in exec_result
        assert "stable_count" in exec_result
        assert exec_result["stable_count"] == 2  # Both samples are stable
        assert (
            exec_result["max_bandgap_material_id"] == "mp-2657"
        )  # TiO2 has higher band gap

        # Check that output was saved
        assert Path(output_file).exists()
        with Path(output_file).open() as f:
            saved_result = json.load(f)
        assert saved_result["result"] == exec_result

    def test_python_code_for_custom_filtering(self, sample_polymorph_data):
        """Test using Python code for custom data filtering."""
        input_data = json.dumps(sample_polymorph_data)

        filtering_code = """
# Custom filtering logic
filtered_materials = []

for material in input_data:
    # Select materials with specific criteria
    if (material.get('band_gap', 0) > 1.0 and
        material.get('density', 0) > 2.0 and
        material.get('is_stable', False)):

        # Add some computed properties
        material_copy = material.copy()
        material_copy['volume_per_atom'] = material_copy['volume'] / material_copy['nsites']
        material_copy['energy_density_ratio'] = abs(material_copy['formation_energy_per_atom']) / material_copy['density']

        filtered_materials.append(material_copy)

output = {
    'filtered_materials': filtered_materials,
    'original_count': len(input_data),
    'filtered_count': len(filtered_materials)
}
"""

        result_json = execute_python_code.execute(
            python_code=filtering_code, input_data=input_data
        )

        result = json.loads(result_json)
        assert result["success"] is True

        output = result["execution_result"]["output"]
        assert "filtered_materials" in output
        assert output["original_count"] == 2
        assert output["filtered_count"] <= 2

        # Check that filtered materials have the computed properties
        for material in output["filtered_materials"]:
            assert "volume_per_atom" in material
            assert "energy_density_ratio" in material
            assert material["volume_per_atom"] > 0
            assert material["energy_density_ratio"] > 0
