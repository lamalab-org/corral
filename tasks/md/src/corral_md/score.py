"""
MD Tutorials Scoring Module

This module provides evaluation and scoring functions for molecular dynamics simulation
tasks and benchmarks. It contains various validation methods to assess the correctness
of simulation results, including numerical comparisons, structural validations, and
analysis of simulation parameters.
"""

# import modal
import json
from pathlib import Path

from loguru import logger

# from utils import extract_lattice_coordinates


def check_potential_file(target: str):
    def score_fn(result: str) -> float:
        try:
            return 1.0 if result == target else 0.0
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Error parsing result for addition_score: {e}, result was: {result}"
            )
            return 0.0

    return score_fn


def check_numerical(target: float | None = None, tolerance: float | None = None):
    def score_fn(result: str) -> float:
        """Score an addition task with expected answer validation"""
        import logging
        import re

        logger = logging.getLogger(__name__)
        try:
            # Handle string submissions
            if isinstance(result, str):
                result = result.strip()
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict):
                        for key in ("BULK ENERGY", "SLAB ENERGY", "density"):
                            if key in parsed_result:
                                answer = float(parsed_result[key])
                                break
                        else:
                            return 0.0  # no valid key
                    else:
                        # If not a dict, treat as direct numeric
                        answer = float(parsed_result)
                except json.JSONDecodeError:
                    # If not JSON, try to extract numeric value via regex
                    match = re.search(r"[-+]?\d*\.\d+|\d+", result)
                    if match:
                        answer = float(match.group())
                    else:
                        return 0.0
            else:
                # If already parsed
                if isinstance(result, dict):
                    for key in (
                        "BULK ENERGY",
                        "SLAB ENERGY",
                        "density",
                    ):
                        if key in result:
                            answer = float(result[key])
                            break
                    else:
                        return 0.0
                else:
                    answer = float(result)

            if target is not None and tolerance is not None:
                tol = tolerance * abs(target)
                return 1.0 if (target - tol) <= answer <= (target + tol) else 0.0

            return 0.0

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Error parsing result for addition_score: {e}, result was: {result}"
            )
            return 0.0

    return score_fn


def check_structure(target, atom_style, use_modal=True):
    def score_fn(result: str) -> float:
        import logging

        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.io.lammps.data import LammpsData

        logger = logging.getLogger(__name__)

        if result is None:
            logger.warning("Received None as result in check_structure")
            return 0.0

        try:
            if use_modal:
                import modal

                read_file = modal.Function.lookup("simagent", "read_file")
                content = read_file.remote(result)
            else:
                with Path(result).open() as f:
                    content = f.read()

            with Path("temp.data").open("w") as f:
                f.write(content)

            ld1 = LammpsData.from_file(target, atom_style=atom_style)
            ld2 = LammpsData.from_file("temp.data", atom_style=atom_style)

            s1 = ld1.structure
            s2 = ld2.structure

            matcher = StructureMatcher()
            are_equal = matcher.fit(s1, s2)

            temp_file = Path("temp.data")
            if temp_file.exists():
                temp_file.unlink()

            return 1.0 if are_equal else 0.0

        except (ValueError, TypeError, KeyError, FileNotFoundError, Exception) as e:
            logger.warning(f"Error in check_structure: {e}")
            temp_file = Path("temp.data")
            if temp_file.exists():
                temp_file.unlink()
            return 0.0

    return score_fn
