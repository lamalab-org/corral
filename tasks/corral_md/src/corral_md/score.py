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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import modal
from loguru import logger

# from utils import extract_lattice_coordinates


def check_potential_file(target: str) -> Callable[[str], float]:
    """
    Create a scoring function that checks whether a given file path matches
    a target filename and exists according to a remote Modal file check.

    The returned function compares the filename (not the full path) of its
    input against the filename of `target`. If the filenames match, it then
    attempts to verify the file's existence using the Modal `file_info`
    function. A score of 1.0 is returned if both checks pass; otherwise 0.0.

    Args:
        target (str): Path to the target file whose filename will be used
            for comparison (e.g., "/data/files/Al.data").

    Returns:
        Callable[[str], float]: A scoring function that takes a file path
        (`result`) as input and returns 1.0 if the filename matches the
        target and the file exists remotely, otherwise 0.0.
    """
    target_name = Path(target).name  # Just the filename (e.g., "Al.data")

    def score_fn(result: str) -> float:
        """
        Score a potential file path against the target filename and existence.

        Args:
            result (str): Path to a candidate file to be checked.

        Returns:
            float: 1.0 if the filename matches the target filename and the
            file exists according to the Modal file check; otherwise 0.0.
        """
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
                return 1.0
            except RuntimeError as e:
                logger.warning(f"File existence check failed: {e}")
                return 0.0

        except Exception as e:
            logger.warning(f"Error in check_potential_file: {e}, result was: {result}")
            return 0.0

    return score_fn


def check_numerical(target: float, tolerance: float) -> Callable[[Any], float]:
    """
    Create a scoring function that validates numerical results against a target
    value within a relative tolerance, optionally requiring an associated file
    existence check.

    The returned function evaluates an input `result` and returns a score in
    {0.0, 1.0} according to the following rules:

    - Numeric-only mode:
        * If `result` is a number (int/float), a numeric string, or a JSON
          primitive (number or numeric string), only a numerical tolerance
          check is performed.

    - Dictionary (JSON) mode:
        * If `result` is a dict (or a JSON string that parses to a dict), a
          numerical check is performed AND a file existence check is required.
        * Recognized key pairs:
            - "density"        -> requires "trajectory_file" or "log_file"
            - "BULK ENERGY"    -> requires "path to relaxed structure" or
                                  "Relaxed BULK Structure_path"
            - "SLAB ENERGY"    -> requires "path to relaxed structure" or
                                  "Relaxed BULK Structure_path"

    Numerical validation succeeds if:
        |result - target| <= tolerance * |target|

    Args:
        target (float): The reference numerical value to compare against.
        tolerance (float): Relative tolerance factor applied to `target`.

    Returns:
        Callable[[Any], float]: A scoring function that takes a result object
        (string, number, or dict) and returns:
            - 1.0 if all required numerical (and file, if applicable) checks pass
            - 0.0 otherwise
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


def check_structure(target: str, atom_style: str) -> Callable[[str], float]:
    """
    Create a scoring function that compares a predicted atomic structure
    against a target structure using pymatgen's StructureMatcher.

    The returned function expects a remote file path to a LAMMPS data file.
    It reads the file content remotely, writes it to a temporary local file,
    loads both the target and predicted structures using pymatgen, and checks
    structural equivalence.

    A score of 1.0 is returned if the structures are considered equivalent by
    `StructureMatcher.fit`; otherwise 0.0.

    Args:
        target (str): Local filesystem path to the reference LAMMPS data file
            containing the target structure.
        atom_style (str): LAMMPS atom style used to parse both the target and
            predicted structure files (e.g., "atomic", "charge").

    Returns:
        Callable[[str], float]: A scoring function that takes a remote file path
        to a predicted LAMMPS data file and returns:
            - 1.0 if the predicted structure matches the target structure
            - 0.0 otherwise or if any error occurs
    """

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
