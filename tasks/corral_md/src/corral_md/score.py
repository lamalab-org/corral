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
                info = modal.Function.from_name("simagent", "file_info").remote(
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
    """
    Returns a scoring function score_fn(result) -> float in {0.0, 1.0}.

    Behavior:
      - If result is a plain numeric (int/float) or a numeric string -> only numerical check (no file check).
      - If result is a JSON/dict -> numeric check AND file existence check according to valid key pairs:
            density         -> trajectory_file
            BULK ENERGY     -> path to relaxed structure (or Relaxed BULK Structure_path)
            SLAB ENERGY     -> path to relaxed structure (or Relaxed BULK Structure_path)
    """

    def score_fn(result) -> float:
        try:
            answer = None
            file_path = None

            # Helper: numeric-only check
            def numeric_ok(val):
                tol = tolerance * abs(target)
                return (target - tol) <= val <= (target + tol)

            # 1) If input is a string, try JSON first; else try regex numeric
            if isinstance(result, str):
                s = result.strip()
                # Try to parse JSON (could be a dict or a primitive JSON value)
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError:
                    # Not JSON: try to extract a number with regex (numeric-only mode)
                    m = re.search(r"[-+]?\d*\.\d+|\d+", s)
                    if not m:
                        return 0.0
                    answer = float(m.group())
                    return 1.0 if numeric_ok(answer) else 0.0
                else:
                    # parsed is the JSON value (dict, number, or string)
                    if isinstance(parsed, dict):
                        parsed_result = parsed
                    else:
                        # primitive JSON value (number or numeric string) -> numeric-only mode
                        try:
                            answer = float(parsed)
                        except (ValueError, TypeError):
                            return 0.0
                        return 1.0 if numeric_ok(answer) else 0.0

            else:
                # result is not a string
                parsed_result = result

            # 2) If we reach here and parsed_result is a dict -> dict-mode
            if isinstance(parsed_result, dict):
                # density -> trajectory_file
                if "density" in parsed_result:
                    try:
                        answer = float(parsed_result["density"])
                    except (ValueError, TypeError):
                        return 0.0
                    file_path = parsed_result.get(
                        "trajectory_file"
                    ) or parsed_result.get("log_file")
                # BULK ENERGY -> path to relaxed structure (or fallback)
                elif "BULK ENERGY" in parsed_result:
                    try:
                        answer = float(parsed_result["BULK ENERGY"])
                    except (ValueError, TypeError):
                        return 0.0
                    file_path = parsed_result.get(
                        "path to relaxed structure"
                    ) or parsed_result.get("Relaxed BULK Structure_path")
                # SLAB ENERGY -> path to relaxed structure (or fallback)
                elif "SLAB ENERGY" in parsed_result:
                    try:
                        answer = float(parsed_result["SLAB ENERGY"])
                    except (ValueError, TypeError):
                        return 0.0
                    file_path = parsed_result.get(
                        "path to relaxed structure"
                    ) or parsed_result.get("Relaxed BULK Structure_path")
                else:
                    # No recognized key present -> fail
                    return 0.0

                # Numeric check
                if not numeric_ok(answer):
                    return 0.0

                # File existence check (required in dict mode)
                if not file_path:
                    logger.warning(
                        "Expected file path key not found in dict submission."
                    )
                    return 0.0

                # call the platform file existence checker (keeps original behavior)
                try:
                    info = modal.Function.from_name("simagent", "file_info").remote(
                        file_path
                    )
                    logger.info(f"File info: {info}")
                    return 1.0
                except RuntimeError as e:
                    logger.warning(f"File existence check failed for {file_path}: {e}")
                    return 0.0

            # 3) If parsed_result is a plain numeric (non-string passed in)
            if isinstance(parsed_result, (int | float)):
                answer = float(parsed_result)
                return 1.0 if numeric_ok(answer) else 0.0

            # Fallback: unknown type -> fail
            return 0.0

        except Exception as exc:
            logger.warning(
                f"Unexpected error in check_numerical: {exc}, result was: {result}"
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
            read_file = modal.Function.from_name("simagent", "read_file")
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
