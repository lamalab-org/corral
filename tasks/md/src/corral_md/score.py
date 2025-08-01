"""
MD Tutorials Scoring Module

This module provides evaluation and scoring functions for molecular dynamics simulation
tasks and benchmarks. It contains various validation methods to assess the correctness
of simulation results, including numerical comparisons, structural validations, and
analysis of simulation parameters.
"""

# import modal
import contextlib
import json
import re
import tempfile
from pathlib import Path

import modal
from loguru import logger

# from utils import extract_lattice_coordinates


def check_potential_file(target: str):
    target_name = Path(target).name  # Just the filename (e.g., "Al.data")

    def score_fn(result: str) -> float:
        try:
            result_path = Path(result)
            result_name = result_path.name

            # Check filename match
            if result_name != target_name:
                return 0.0

            # Check file existence using Modal
            try:
                info = modal.Function.lookup("simagent", "file_info").remote(
                    str(result_path)
                )
                logger.info(f"File info: {info}")
                return 1.0  # Both checks passed
            except RuntimeError as e:
                logger.warning(f"File existence check failed: {e}")
                return 0.0

        except Exception as e:
            logger.warning(f"Error in check_potential_file: {e}, result was: {result}")
            return 0.0

    return score_fn


def check_numerical(target: float, tolerance: float):
    def score_fn(result: str) -> float:
        try:
            # Initialize variables
            answer = None
            file_path = None

            # Parse result depending on type
            if isinstance(result, str):
                result = result.strip()
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict):
                        # Extract numerical answer from keys
                        for key in ("BULK ENERGY", "SLAB ENERGY", "density"):
                            if key in parsed_result:
                                answer = float(parsed_result[key])
                                break
                        # Extract file path
                        file_path = parsed_result["path to relaxed structure"]
                    else:
                        # If not a dict, treat as direct numeric
                        answer = float(parsed_result)
                except json.JSONDecodeError:
                    # Not JSON: try regex extraction of number only
                    match = re.search(r"[-+]?\d*\.\d+|\d+", result)
                    if match:
                        answer = float(match.group())
                    else:
                        return 0.0
            else:
                # If result already parsed as dict
                if isinstance(result, dict):
                    for key in ("BULK ENERGY", "SLAB ENERGY", "density"):
                        if key in result:
                            answer = float(result[key])
                            break
                    if "path to relaxed structure" in result:
                        file_path = result["path to relaxed structure"]
                    elif "Relaxed BULK Structure_path" in result:
                        file_path = result["Relaxed BULK Structure_path"]
                else:
                    answer = float(result)

            # Check numerical score
            if target is not None and tolerance is not None:
                tol = tolerance * abs(target)
                numerical_ok = (target - tol) <= answer <= (target + tol)
            else:
                numerical_ok = False

            # Check if file exists using file_info
            if file_path is not None:
                try:
                    info = modal.Function.lookup("simagent", "file_info").remote(
                        file_path
                    )
                    logger.info(f"File info: {info}")
                    file_ok = True
                except RuntimeError as e:
                    logger.warning(f"File existence check failed: {e}")
                    file_ok = False
            else:
                # If no file path provided, fail file check
                file_ok = True

            # Return 1.0 only if both pass
            return 1.0 if (numerical_ok and file_ok) else 0.0

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Error parsing result for addition_score: {e}, result was: {result}"
            )
            return 0.0

    return score_fn


def check_structure(target, atom_style):
    def score_fn(result: str) -> float:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.io.lammps.data import LammpsData

        if result is None:
            logger.warning("Received None as result in check_structure")
            return 0.0

        tmp_path = None  # Predefine in case of early exception

        try:
            # Load target structure
            ld1 = LammpsData.from_file(target, atom_style=atom_style)

            # Read remote result file content
            read_file = modal.Function.lookup("simagent", "read_file")
            content = read_file.remote(result)

            # Write content to a unique temp file
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".data", delete=False
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Load predicted structure
            ld2 = LammpsData.from_file(tmp_path, atom_style=atom_style)

            # Compare structures
            matcher = StructureMatcher()
            are_equal = matcher.fit(ld1.structure, ld2.structure)

            return 1.0 if are_equal else 0.0

        except Exception as e:
            logger.warning(f"Error in check_structure: {e}")
            return 0.0

        finally:
            # Clean up temp file
            if tmp_path:
                with contextlib.suppress(Exception):
                    Path(tmp_path).unlink()

    return score_fn
