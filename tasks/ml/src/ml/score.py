import json
import os
from pathlib import Path

import joblib
from loguru import logger
from pymatgen.core import Structure

if "CORRAL_WORK_DIR" not in os.environ:
    raise OSError("Environment variable 'CORRAL_WORK_DIR' is not set.")
BASE_WORK_DIR = os.environ["CORRAL_WORK_DIR"]


def resolve_path(path_or_str: str) -> str:
    """
    Resolves a path that might be relative to the base work directory.
    Also cleans up common input format issues.
    """
    # Handle various input issues
    if isinstance(path_or_str, str):
        # Remove "answer:" prefix if present
        if path_or_str.startswith("answer:"):
            path_or_str = path_or_str.replace("answer:", "", 1).strip()

        # Replace escaped quotes that might come from JSON strings
        path_or_str = path_or_str.replace('\\"', '"').replace("\\'", "'")

    try:
        # If it's an absolute path or already exists, return as is
        if Path(path_or_str).is_absolute() or Path(path_or_str).exists():
            return path_or_str

        # Try to resolve against base directory
        full_path = Path(BASE_WORK_DIR) / path_or_str
        if full_path.exists():
            return str(full_path)

        # If we can't resolve it, return the original
        return path_or_str
    except Exception:
        # If there's any error treating it as a path, return the original
        return path_or_str


def check_mp_structure(path_or_cif: str) -> float:
    """
    Check if the path points to a valid CIF file containing a structure from Materials Project.
    """
    logger.info("check_mp_structure")
    logger.info(f"Input path_or_cif: {path_or_cif}")
    try:
        # Try first as a CIF string since that's more common
        try:
            logger.info(f"Trying to parse as CIF string first: {path_or_cif}")
            structure = Structure.from_str(path_or_cif, fmt="cif")
            logger.info("Successfully parsed as CIF string")
        except Exception as e:
            logger.info(f"Could not parse as CIF string: {e}")
            # If that fails, try as a file path
            if Path(path_or_cif).exists():
                logger.info(f"Input is a valid file path: {path_or_cif}")
                structure = Structure.from_file(path_or_cif)
            else:
                logger.error(
                    f"Input is neither a valid CIF string nor a file path: {path_or_cif}"
                )
                return 0.0

        return 1.0 if structure and len(structure) > 0 else 0.0
    except Exception as e:
        logger.error(f"Error validating structure: {e}")
        return 0.0


def compare_with_ground_truth(
    generated_path, ground_truth_path, comparison_mode="strict", tolerance=0.05
):
    """
    Generic function to compare a generated JSON file with a ground truth JSON file.

    Args:
        generated_path: Path to the generated JSON file
        ground_truth_path: Path to the ground truth JSON file
        comparison_mode: Mode of comparison:
                        - "strict": Exact matching of structure and values
                        - "keys": Only check if all required keys exist
                        - "numerical": Compare numerical values with tolerance
                        - "subset": Check if generated contains at least a subset of ground truth
        tolerance: Tolerance for numerical comparisons (as a fraction)

    Returns:
        1.0 if generated matches ground truth according to the comparison mode, 0.0 otherwise
    """
    import json

    # Check if both files exist
    if not Path(generated_path).exists() or not Path(ground_truth_path).exists():
        return 0.0

    try:
        # Load both files
        with Path(generated_path).open("r") as f:
            generated = json.load(f)

        with Path(ground_truth_path).open("r") as f:
            ground_truth = json.load(f)

        # Handle different comparison modes
        if comparison_mode == "strict":
            # Direct equality check
            return 1.0 if generated == ground_truth else 0.0

        elif comparison_mode == "keys":
            # Check if all required keys from ground truth exist in generated
            if isinstance(ground_truth, dict) and isinstance(generated, dict):
                missing_keys = [key for key in ground_truth if key not in generated]
                return 1.0 if not missing_keys else 0.0
            elif isinstance(ground_truth, list) and isinstance(generated, list):
                # For lists, check if they have the same length
                if len(ground_truth) != len(generated):
                    return 0.0
                # If items are dictionaries, check keys for each item
                if all(isinstance(item, dict) for item in ground_truth):
                    for i, gt_item in enumerate(ground_truth):
                        if i >= len(generated):
                            return 0.0
                        missing_keys = [
                            key for key in gt_item if key not in generated[i]
                        ]
                        if missing_keys:
                            return 0.0
                return 1.0
            else:
                return 0.0

        elif comparison_mode == "numerical":
            # Compare numerical values with tolerance
            def compare_with_tolerance(val1, val2, tol):
                if isinstance(val1, int | float) and isinstance(val2, int | float):
                    # Use relative tolerance for non-zero values
                    if abs(val2) > 1e-10:
                        return abs((val1 - val2) / val2) <= tol
                    # Use absolute tolerance for values near zero
                    else:
                        return abs(val1 - val2) <= tol
                elif isinstance(val1, dict) and isinstance(val2, dict):
                    # Compare dictionaries recursively
                    return all(
                        k in val1 and compare_with_tolerance(val1[k], val2[k], tol)
                        for k in val2
                    )
                elif isinstance(val1, list) and isinstance(val2, list):
                    # Compare lists recursively
                    return len(val1) == len(val2) and all(
                        compare_with_tolerance(v1, v2, tol)
                        for v1, v2 in zip(val1, val2, strict=False)
                    )
                else:
                    # For non-numerical values, use strict equality
                    return val1 == val2

            return (
                1.0
                if compare_with_tolerance(generated, ground_truth, tolerance)
                else 0.0
            )

        elif comparison_mode == "subset":
            # Check if generated contains at least the required subset
            def is_subset(generated_val, ground_truth_val):
                if isinstance(ground_truth_val, dict) and isinstance(
                    generated_val, dict
                ):
                    # Check if all required keys and values match
                    for k, v in ground_truth_val.items():
                        if k not in generated_val or not is_subset(generated_val[k], v):
                            return False
                    return True
                elif isinstance(ground_truth_val, list) and isinstance(
                    generated_val, list
                ):
                    # For lists, check if all ground truth items are in generated
                    # This is a simplified approach that works for primitive values
                    for gt_item in ground_truth_val:
                        if isinstance(gt_item, dict | list):
                            # For complex items, check if any generated item is a superset
                            if not any(
                                is_subset(gen_item, gt_item)
                                for gen_item in generated_val
                            ):
                                return False
                        else:
                            # For simple items, just check if it's in the list
                            if gt_item not in generated_val:
                                return False
                    return True
                else:
                    # For primitive values, check equality
                    return generated_val == ground_truth_val

            return 1.0 if is_subset(generated, ground_truth) else 0.0

        else:
            raise ValueError(f"Unknown comparison mode: {comparison_mode}")

    except Exception:
        return 0.0


def ml_pipeline_score(model_path: str) -> float:
    """
    Comprehensive scoring function for the single-task ML pipeline.

    Evaluates the entire pipeline from data generation to model evaluation.
    This function looks for evidence of all pipeline steps and evaluates
    the final model quality.
    """
    try:
        if not Path(model_path).exists():
            return 0.0

        # Try to load the model
        try:
            import joblib

            model = joblib.load(model_path)
            if not hasattr(model, "predict"):
                return 0.2
        except Exception:
            return 0.1

        score = 0.3  # Base score for model existence

        # Look for evidence of dataset creation
        work_dir = Path(model_path).parent

        # Check for evaluation results
        eval_files = list(work_dir.glob("*evaluation*.json")) + list(
            work_dir.glob("*results*.json")
        )
        if eval_files:
            try:
                # Load the most recent evaluation file
                latest_eval = max(eval_files, key=lambda x: x.stat().st_mtime)
                with latest_eval.open() as f:
                    eval_results = json.load(f)

                # Check model performance
                if "evaluation_metrics" in eval_results:
                    metrics = eval_results["evaluation_metrics"]
                    r2 = metrics.get("r2", 0)
                    mae = metrics.get("mae", float("inf"))

                    if r2 >= 0.8 and mae <= 0.3:
                        score += 0.3
                    elif r2 >= 0.6 and mae <= 0.5:
                        score += 0.2
                    elif r2 >= 0.4:
                        score += 0.1

                # Check for cross-validation
                if "cross_validation_results" in eval_results:
                    score += 0.1

            except Exception:
                pass

        return min(1.0, score)

    except Exception as e:
        logger.error(f"Error scoring comprehensive ML pipeline: {e}")
        return 0.0


def polymorph_retrieval_success(retrieval_results_path: str) -> float:
    """
    Score the success of batch polymorph retrieval.

    Criteria:
    - At least 80% of compositions have polymorphs retrieved
    - Total polymorphs >= 50
    - Reasonable distribution across compositions
    """
    try:
        if not Path(retrieval_results_path).exists():
            return 0.0

        with Path(retrieval_results_path).open() as f:
            results = json.load(f)

        successful = len(results.get("successful_compositions", []))
        failed = len(results.get("failed_compositions", []))
        total_compositions = successful + failed
        total_polymorphs = results.get("total_polymorphs", 0)

        if total_compositions == 0:
            return 0.0

        success_rate = successful / total_compositions

        score = 0.0

        # Success rate scoring
        if success_rate >= 0.8:
            score += 0.4
        elif success_rate >= 0.6:
            score += 0.3
        elif success_rate >= 0.4:
            score += 0.2

        # Total polymorphs scoring
        if total_polymorphs >= 100:
            score += 0.3
        elif total_polymorphs >= 50:
            score += 0.2
        elif total_polymorphs >= 20:
            score += 0.1

        # Distribution check (average polymorphs per successful composition)
        if successful > 0:
            avg_per_comp = total_polymorphs / successful
            if 3 <= avg_per_comp <= 10:
                score += 0.3
            elif 2 <= avg_per_comp <= 12:
                score += 0.2

        return min(1.0, score)

    except Exception as e:
        logger.error(f"Error scoring polymorph retrieval: {e}")
        return 0.0


def score_polymorph_dataset(
    consolidated_json_path: str,
) -> float:
    """
    Analyzes a consolidated JSON file of polymorphs to identify if there are
    compositions with multiple polymorphs.

    Args:
        consolidated_json_path: Path to the consolidated JSON file
                                (e.g., created by consolidate_polymorph_datasets).

    Returns:
        A float: 1.0 if at least one composition with multiple polymorphs is found,
        otherwise 0.0. Returns 0.0 if the file is not found or an error occurs.
    """
    try:
        # Check if the consolidated JSON file exists
        # We need Path imported to check for file existence
        from pathlib import Path

        logger.info(f"score_polymorph_dataset: input={consolidated_json_path!r}")

        consolidated_json_file = Path(consolidated_json_path)
        if not consolidated_json_file.exists():
            logger.error(
                f"Consolidated JSON file not found at: {consolidated_json_path}"
            )
            return 0.0

        with consolidated_json_file.open("r") as f:
            all_polymorphs_data = json.load(f)

        # Dictionary to store polymorphs grouped by composition
        polymorphs_by_composition = {}
        for polymorph in all_polymorphs_data:
            composition = polymorph.get("source_composition")
            if composition:
                if composition not in polymorphs_by_composition:
                    polymorphs_by_composition[composition] = []
                polymorphs_by_composition[composition].append(polymorph)
            else:
                logger.warning(
                    f"Polymorph without 'source_composition' found: {polymorph.get('material_id', 'N/A')}"
                )

        # Check if any composition has multiple polymorphs
        for polymorphs in polymorphs_by_composition.values():
            if len(polymorphs) > 1:
                # If we find at least one composition with multiple polymorphs, return 1.0
                return 1.0

        # If the loop completes and no composition with multiple polymorphs is found
        return 0.0

    except Exception as e:
        logger.error(f"Error in score_polymorph_dataset: {e}", exc_info=True)
        return 0.0


def ml_dataset_preparation_quality_binary(ml_metadata_path: str) -> int:
    """
    Scores the quality of ML dataset preparation as binary (0 for fail, 1 for pass).

    A dataset preparation passes (1) if it meets the following criteria:
    - Both train and test files exist.
    - Sufficient sample sizes: at least 40 training samples and 10 test samples.
    - A reasonable number of features: at least 10 features.

    Args:
        ml_metadata_path: Path to the ML dataset metadata JSON file.

    Returns:
        1 if the dataset preparation quality meets the defined passing criteria,
        0 otherwise (including errors).
    """
    try:
        logger.info(f"ml_dataset_preparation: input={ml_metadata_path!r}")

        if not Path(ml_metadata_path).exists():
            logger.info(f"Metadata file not found: {ml_metadata_path}")
            return 0

        with Path(ml_metadata_path).open() as f:
            metadata = json.load(f)

        # Criteria 1: Check train/test files exist
        train_path = metadata.get("train_path", "")
        test_path = metadata.get("test_path", "")
        if not (
            train_path
            and Path(train_path).exists()
            and test_path
            and Path(test_path).exists()
        ):
            logger.info(
                "Binary check failed: Train or test files are missing or paths are empty."
            )
            # Added more specific logging for clarity
            if not train_path:
                logger.info("train_path is empty in metadata.")
            elif not Path(train_path).exists():
                logger.info("Train file does not exist at: {train_path}")
            if not test_path:
                logger.info("test_path is empty in metadata.")
            elif not Path(test_path).exists():
                logger.info("Test file does not exist at: {test_path}")

            return 0

        # Criteria 2: Check sufficient sample sizes
        train_samples = metadata.get("train_samples", 0)
        test_samples = metadata.get("test_samples", 0)
        if not (train_samples >= 20 and test_samples >= 5):
            logger.info(
                f"Binary check failed: Insufficient sample sizes (Train: {train_samples}, Test: {test_samples})."
            )
            return 0

        # Criteria 3: Check reasonable feature count
        feature_count = metadata.get("feature_count", 0)
        if not (feature_count >= 10):
            logger.info(
                f"Binary check failed: Insufficient feature count ({feature_count})."
            )
            return 0

        logger.info("Binary check passed: ML dataset preparation quality is good.")
        return 1

    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in {ml_metadata_path}")
        return 0
    except Exception as e:
        logger.error(f"Error scoring ML dataset preparation quality: {e}")
        return 0


def model_training_success_binary(model_path: str) -> int:
    """
    Scores the success of XGBoost model training as binary (0 for fail, 1 for pass).

    Model training passes (1) if it meets the following criteria:
    - A valid model file exists and can be loaded.
    - Associated training results indicate successful training.
    - Reasonable performance metrics: R-squared (r2) >= 0.6 and Mean Absolute Error (mae) <= 0.5.

    Args:
        model_path: Path to the trained model file (e.g., .pkl).

    Returns:
        1 if the model training success meets the defined passing criteria,
        0 otherwise (including errors).
    """
    try:
        logger.info(f"model_training_success_binary: input={model_path!r}")

        if not Path(model_path).exists():
            logger.info(f"Model file not found: {model_path}")
            return 0

        # Criteria 1: Try to load the model
        try:
            model = joblib.load(model_path)
            if not hasattr(model, "predict"):
                logger.info(
                    "Binary check failed: Loaded model does not have a 'predict' method."
                )
                return 0
        except Exception as e:
            logger.info(
                f"Binary check failed: Could not load the model from {model_path}. Error: {e}"
            )
            return 0

        logger.info("Binary check passed: Model training success criteria met.")
        return 1

    except Exception as e:
        logger.error(f"Error scoring model training success: {e}")
        return 0


def model_evaluation_completeness_binary(evaluation_results_path: str) -> int:
    """
    Scores the completeness of model evaluation as binary (0 for fail, 1 for pass).

    Model evaluation completeness passes (1) if it meets the following criteria:
    - An evaluation results file exists.
    - All required basic evaluation metrics (mae, rmse, r2) are present.
    - Performance quality: R-squared (r2) in evaluation metrics is at least 0.7.
    - Cross-validation results are present, including mean and standard deviation for r2.
    - Feature importance analysis is included in the evaluation metrics.

    Args:
        evaluation_results_path: Path to the model evaluation results JSON file.

    Returns:
        1 if the model evaluation completeness meets the defined passing criteria,
        0 otherwise (including errors).
    """
    try:
        logger.info(f"ml_dataset_preparation: input={evaluation_results_path!r}")

        if not Path(evaluation_results_path).exists():
            logger.info(f"Evaluation results file not found: {evaluation_results_path}")
            return 0

        with Path(evaluation_results_path).open() as f:
            results = json.load(f)

        # Criteria 1 & 2: Check for basic evaluation metrics and all required metrics
        if "test_set_evaluation" not in results:
            logger.info("Binary check failed: 'test_set_evaluation' not found.")
            return 0
        metrics = results["test_set_evaluation"]
        required_metrics = ["mae", "rmse", "r2"]
        if not all(metric in metrics for metric in required_metrics):
            logger.info(
                "Binary check failed: Not all required metrics (mae, rmse, r2) are present."
            )
            return 0

        # Criteria 3: Check performance quality (r2 >= 0.7)
        if not ("r2" in metrics and metrics["r2"] >= 0.7):
            logger.info(
                f"Binary check failed: R2 ({metrics.get('r2', 'N/A')}) is below 0.7."
            )
            return 0

        # Criteria 4: Check for cross-validation results
        if "cross_validation_results" not in results:
            logger.info("Binary check failed: 'cross_validation_results' not found.")
            return 0
        cv_results = results["cross_validation_results"]
        if not ("r2_mean" in cv_results and "r2_std" in cv_results):
            logger.info(
                "Binary check failed: Cross-validation results missing 'r2_mean' or 'r2_std'."
            )
            return 0

        # Criteria 5: Check for feature importance
        if "feature_importance" not in metrics:
            logger.info(
                "Binary check failed: 'feature_importance' not found in evaluation metrics."
            )
            return 0

        logger.info("Binary check passed: Model evaluation completeness criteria met.")
        return 1

    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in {evaluation_results_path}")
        return 0
    except Exception as e:
        logger.error(f"Error scoring model evaluation completeness: {e}")
        return 0
