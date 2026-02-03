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


def check_log(variable: str, target: float, tolerance: float, window: int):
    """
    Returns a scoring function score_fn(result) -> float in {0.0, 1.0}.

    Behavior:
      - Extracts the specified variable from the lammps log file.
      - Computes the average of the last 'window' entries of that variable.
      - Compares the average to the target value within the given tolerance.
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

        for raw_line in lines:
            line = raw_line.strip()

            if line.startswith("Step"):
                header = line.split()
                if column not in header:
                    raise ValueError(f"Column '{column}' not found in header: {header}")
                col_index = header.index(column)
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

        return np.array(steps), np.array(values)

    def score_fn(result: str) -> float:
        try:
            # Read remote result file content
            data = json.loads(result)
            log_file_path = data["log_file"]
            restart_file_path = data["restart_file"]
            read_file = modal.Function.from_name("simagent", "read_file")
            content = read_file.remote(log_file_path)
            try:
                info = modal.Function.from_name("simagent", "file_info").remote(
                    restart_file_path
                )
                logger.info(f"Restart file info: {info}")
            except RuntimeError as e:
                logger.warning(f"Restart file existence check failed: {e}")
                return 0.0
            steps, values = read_log_from_text(content, variable)
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


def check_msd():
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

    def score_fn(result: str) -> float:
        try:
            # Read remote result file content
            read_file = modal.Function.from_name("simagent", "read_file")
            content = read_file.remote(result)
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

            # Decision
            if r2 > 0.9:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error in check_msd: {e}, result was: {result}")
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
