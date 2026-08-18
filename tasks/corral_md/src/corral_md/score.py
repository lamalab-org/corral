"""
MD Tutorials Scoring Module

This module provides evaluation and scoring functions for molecular dynamics simulation
tasks and benchmarks. It contains various validation methods to assess the correctness
of simulation results, including numerical comparisons, structural validations, and
analysis of simulation parameters.
"""

# import modal
import json
import re
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import modal
from loguru import logger

# from utils import extract_lattice_coordinates


def check_potential_file(target: str):
    """
    Returns a scoring function score_fn(result) -> float in {0.0, 1.0}.

    Behavior: This is a higher-order function that returns `score_fn`, a callable which:
        - Accepts a single argument `result` (str or None).
        - Logs a warning and returns 0.0 if `result` is None.
        - Otherwise, compares the selected catalog path with the expected path locally.

    Args:
        target (str): The target identifier or path to evaluate results against.

    Returns:
        Callable[[str | None], float]: A function that takes a result string (or None)
        and returns 1.0 for the expected path or 0.0 otherwise.
    """

    def score_fn(result: str | None = None) -> float:
        if result is None:
            logger.warning("Received None as result in check_potential_file")
            return 0.0

        try:
            return 1.0 if Path(result).resolve() == Path(target).resolve() else 0.0
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"Could not validate potential path {result!r}: {exc}")
            return 0.0

    return score_fn


def check_log(variable: str | list, target: float, tolerance: float, window: int):
    """
    Returns a scoring function score_fn(result) -> float in {0.0, 1.0}.

    Behavior:
      - If variable is a string:
          * Extracts that variable from the LAMMPS log file.
          * Computes the average of the last 'window' entries.
          * Compares the average to the target value within tolerance.
      - If variable is a list/tuple [var_to_check, var_must_exist]:
          * Checks var_must_exist exists in the log header.
          * Checks var_to_check numerically as above.
          * If either fails, returns 0.0.
    """
    import json

    import numpy as np

    def read_log_from_text(log_text: str, column: str):
        steps = []
        values = []

        lines = log_text.splitlines()

        header = None
        col_index = None
        step_index = None

        # Allow aliases for certain columns
        column_aliases = {
            "Temp": ["Temp", "Temperature"],
            "Temperature": ["Temp", "Temperature"],
        }

        for raw_line in lines:
            line = raw_line.strip()

            if line.startswith("Step"):
                header = line.split()

                # Resolve column name (handle Temp / Temperature alias)
                possible_names = column_aliases.get(column, [column])

                found_col = None
                for name in possible_names:
                    if name in header:
                        found_col = name
                        break

                if found_col is None:
                    raise ValueError(f"Column '{column}' not found in header: {header}")

                col_index = header.index(found_col)
                step_index = header.index("Step")
                continue

            if header and line:
                tokens = line.split()
                if len(tokens) != len(header):
                    continue
                try:
                    step = int(tokens[step_index])
                    value = float(tokens[col_index])
                except ValueError:
                    continue

                steps.append(step)
                values.append(value)

        return np.array(steps), np.array(values), header

    def header_has_column(header, column: str) -> bool:
        # Handle aliases here too
        column_aliases = {
            "Temp": ["Temp", "Temperature"],
            "Temperature": ["Temp", "Temperature"],
        }
        possible_names = column_aliases.get(column, [column])

        return any(name in header for name in possible_names)

    def score_fn(result: str | None = None) -> float:
        if result is None:
            logger.warning("Received None as result in check_log")
            return 0.0
        try:
            data = json.loads(result)
            log_file_path = data["log_file"]

            if "restart_file" in data:
                restart_file_path = Path(data["restart_file"])
                if not restart_file_path.is_file():
                    logger.warning(
                        f"Restart file existence check failed: {restart_file_path}"
                    )
                    return 0.0

            content = Path(log_file_path).read_text(encoding="utf-8")

            # Determine mode
            if isinstance(variable, (list | tuple)):
                var_to_check = variable[0]
                var_must_exist = variable[1]
            else:
                var_to_check = variable
                var_must_exist = None

            steps, values, header = read_log_from_text(content, var_to_check)

            # If second variable must exist, check header
            if var_must_exist is not None and not header_has_column(
                header, var_must_exist
            ):
                logger.warning(
                    f"Required column '{var_must_exist}' not found in log header"
                )
                return 0.0

            if len(values) < window:
                logger.warning(
                    f"Not enough data points ({len(values)}) for the specified window ({window})."
                )
                return 0.0

            recent_values = values[-window:]
            avg_value = np.mean(recent_values)

            tol = tolerance * abs(target)
            if (target - tol) <= avg_value <= (target + tol):
                return 1.0
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Error in check_log: {e}, result was: {result}")
            return 0.0

    return score_fn


def check_msd(target: float):
    """
    Returns a scoring function score_fn(result) -> float in {0.0, 1.0}.

    Behavior:
      - Extracts the MSD value from the result log file.
      - Compares the MSD to the target value within the given tolerance.
    """
    import numpy as np
    from sklearn.metrics import r2_score

    def read_msd_from_text(content: str):
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        if not lines:
            raise ValueError("Empty MSD file")

        def is_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        # Detect header: if any token in first line is non-numeric
        first_tokens = lines[0].split()
        has_header = not all(is_float(tok) for tok in first_tokens)

        data_lines = lines[1:] if has_header else lines

        steps = []
        msd = []

        for line in data_lines:
            tokens = line.split()
            if len(tokens) < 2:
                continue
            try:
                step = float(tokens[0])
                val = float(tokens[1])
            except ValueError:
                continue

            steps.append(step)
            msd.append(val)

        if len(msd) == 0:
            raise ValueError("No numeric data found in MSD file")

        return np.array(steps), np.array(msd), has_header

    def score_fn(result: str | None = None) -> float:
        if result is None:
            logger.warning("Received None as result in check_msd")
            return 0.0
        try:
            content = Path(result).read_text(encoding="utf-8")
            steps, msd_values, has_header = read_msd_from_text(content)
            # Convert to numpy arrays
            time_ps = np.asarray(steps, dtype=float)
            msd = np.asarray(msd_values, dtype=float)

            # Basic sanity check
            if len(time_ps) < 2:
                return 0.0

            # If you already have a mask logic, keep using it.
            # Otherwise, fit everything:
            mask = np.ones_like(time_ps, dtype=bool)

            time_fit = time_ps[mask]
            msd_fit = msd[mask]

            # Need at least 2 points after masking
            if len(time_fit) < 2:
                return 0.0

            # Linear fit
            slope, intercept = np.polyfit(time_fit, msd_fit, 1)
            msd_fit_line = slope * time_fit + intercept

            # R^2 score
            r2 = r2_score(msd_fit, msd_fit_line)

            logger.info(
                f"MSD fit results: slope={slope}, intercept={intercept}, R^2={r2}"
            )

            # Decision
            if r2 > float(target):
                return 1.0
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error in check_msd: {e}, result was: {result}")
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

                structure_path = Path(file_path)
                if structure_path.is_file():
                    logger.info(f"Relaxed structure exists: {structure_path}")
                    return 1.0
                logger.warning(f"File existence check failed for {structure_path}")
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
    """
    Create a scoring function that evaluates a local structure result against a
    reference stored in the read-only `eval_structures` Modal Volume.

    This is a higher-order function that returns `score_fn`, a callable which:
    - Accepts a single argument `result` (str or None).
    - Logs a warning and returns 0.0 if `result` is None.
    - Downloads the small reference structure and compares it to the local result.

    Args:
        target: The target identifier or path used for structure validation.
        atom_style: The LAMMPS atom style used to load both structures.

    Returns:
        Callable[[str | None], float]: A function that takes a result string (or None)
        and returns a floating-point score, or 0.0 if the result is unavailable.
    """

    def score_fn(result: str | None = None) -> float:
        if result is None:
            logger.warning("Received None as result in check_structure")
            return 0.0

        from pymatgen.analysis.structure_matcher import (
            StructureMatcher,
        )
        from pymatgen.io.lammps.data import LammpsData

        result_path = Path(result)
        if not result_path.is_file():
            logger.warning(f"Structure result was not found locally: {result_path}")
            return 0.0

        try:
            with TemporaryDirectory(prefix="corral-md-reference-") as temporary:
                target_path = Path(target)
                if target_path.is_file():
                    local_target = target_path
                elif str(target).startswith("/eval_structures/"):
                    local_target = Path(temporary) / target_path.name
                    volume = modal.Volume.from_name("eval_structures")
                    remote_path = str(target).removeprefix("/eval_structures/")
                    with local_target.open("wb") as stream:
                        for chunk in volume.read_file(remote_path):
                            stream.write(chunk)
                else:
                    logger.warning(f"Structure target was not found: {target}")
                    return 0.0

                expected = LammpsData.from_file(local_target, atom_style=atom_style)
                actual = LammpsData.from_file(result_path, atom_style=atom_style)
                return (
                    1.0
                    if StructureMatcher().fit(expected.structure, actual.structure)
                    else 0.0
                )
        except Exception as exc:
            logger.warning(f"Error in check_structure: {exc}")
            return 0.0

    return score_fn
