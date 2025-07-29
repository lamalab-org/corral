"""
MD Tutorials Scoring Module

This module provides evaluation and scoring functions for molecular dynamics simulation
tasks and benchmarks. It contains various validation methods to assess the correctness
of simulation results, including numerical comparisons, structural validations, and
analysis of simulation parameters.
"""

import ast
import io
import modal
# import modal
import numpy as np
from loguru import logger
import json


# from utils import extract_lattice_coordinates


def check_potential_file(target):

    def score_fn(result: str) -> float:
        import os
        try:
            filename = os.path.basename(result)
            if filename == target:
                return 1.0
            else:
                return 0.0
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Error parsing result for addition_score: {e}, result was: {result}"
            )
            return 0.0
    return score_fn
        

def check_numerical(target: float | None = None, tolerance: float | None = None):
    def score_fn(result: str) -> float:
        """Score an addition task with expected answer validation"""
        import re
        import json
        import logging

        logger = logging.getLogger(__name__)
        try:
            # Handle string submissions
            if isinstance(result, str):
                result = result.strip()
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict):
                        for key in (
                            "answer", "BULK ENERGY", "SLAB ENERGY",
                            "C11", "C12", "C13", "C22", "C23", "C33"
                        ):
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
                        "answer", "BULK ENERGY", "SLAB ENERGY",
                        "C11", "C12", "C13", "C22", "C23", "C33",
                        "energy", "density"
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
        from pymatgen.io.lammps.data import LammpsData
        from pymatgen.analysis.structure_matcher import StructureMatcher
        import os
        import logging

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
                with open(result, "r") as f:
                    content = f.read()

            with open("temp.data", "w") as f:
                f.write(content)

            ld1 = LammpsData.from_file(target, atom_style=atom_style)
            ld2 = LammpsData.from_file("temp.data", atom_style=atom_style)

            s1 = ld1.structure
            s2 = ld2.structure

            matcher = StructureMatcher()
            are_equal = matcher.fit(s1, s2)

            if os.path.exists("temp.data"):
                os.remove("temp.data")

            return 1.0 if are_equal else 0.0

        except (ValueError, TypeError, KeyError, FileNotFoundError, Exception) as e:
            logger.warning(f"Error in check_structure: {e}")
            if os.path.exists("temp.data"):
                os.remove("temp.data")
            return 0.0

    return score_fn




