"""
MD Simulations Scoring Module

This module provides evaluation and scoring functions for molecular dynamics simulation
tasks and benchmarks. It contains various validation methods to assess the correctness
of simulation results, including numerical comparisons, structural validations, and
analysis of simulation parameters.
"""

import ast
import io

import modal
import numpy as np
from loguru import logger
from utils import extract_lattice_coordinates


def check_numerical(agent_answer, output):
    """Check if the agent's numerical answer is within a specified tolerance of the target value."""
    agent_answer = float(agent_answer)
    target = float(output["target"])

    tolerance = 0.10 * abs(target)  # 10% of the target
    lower_limit = target - tolerance
    upper_limit = target + tolerance

    return 1.0 if lower_limit <= agent_answer <= upper_limit else 0.0


def check_structure(answer_file, output):
    """Check if the structure in the answer file matches the target structure."""
    vol = modal.Volume.from_name("simulations")
    try:
        logger.info(f"type of answer_file {type(answer_file)}")
        if isinstance(answer_file, str) and answer_file.strip().startswith("{"):
            answer_file = ast.literal_eval(answer_file)["path_to_structure"]
        logger.info("extracted answer file", answer_file)
        final_path = answer_file.removeprefix("/results/")
        data = b""
        for chunk in vol.read_file(final_path):
            data += chunk
        structure_path = output["target"]
        coords1 = extract_lattice_coordinates(data)
        coords2 = extract_lattice_coordinates(structure_path)
        return float(np.array_equal(coords1, coords2))
    except (FileNotFoundError, ValueError):
        return 0.0


def check_minimiser(answer_directory, output):
    """Check if the minimiser used in the simulation matches the expected one."""
    vol = modal.Volume.from_name("simulations")
    list_files = modal.Function.lookup("simagent", "list_files")
    min_style_orig = output["target"]
    try:
        min_style_arg = "cg"
        files = list_files.remote(answer_directory)
        log_files = [f for f in files if f.endswith(".log")]
        for log_file in log_files:
            final_path = log_file.removeprefix("/results/")
            data = b""
            for chunk in vol.read_file(final_path):
                data += chunk
            text_data = data.decode("utf-8")
            # print("text_data", text_data)
            for line in text_data.splitlines():
                if line.startswith("min_style"):
                    min_style_arg = line.split()[1]
                    # print("min_style_arg", min_style_arg)
            if min_style_arg != min_style_orig:
                return 0.0
        return 1.0
    except (FileNotFoundError, ValueError):
        return 0.0


def energy_minimisation(answer_directory, agent_answer, output):
    """Check if the energy minimisation task is performed correctly."""
    minimiser = check_minimiser(answer_directory, output[1])
    numerical = check_numerical(agent_answer, output[0])
    return minimiser * numerical


def energy_minimisation_with_structure(answer_directory, agent_answer, output):
    """Check if the energy minimisation task is performed correctly with structure validation."""
    minimiser = check_minimiser(answer_directory, output[0])
    structure = check_structure(agent_answer, output[1])
    return minimiser * structure


def check_stress_tensor(agent_output, output):
    """Check if the stress tensor components are within a specified tolerance."""
    reference_output = output["target"]
    tolerance = float(output["threshold"])
    try:
        agent_vals = list(map(float, agent_output.strip().split(",")))
        ref_vals = list(map(float, reference_output.strip().split(",")))

        if len(agent_vals) != 6 or len(ref_vals) != 6:
            return 0
    except Exception:
        return 0

    for pred, true in zip(agent_vals, ref_vals, strict=False):
        error = abs(pred - true) if abs(true) < 1e-6 else abs((pred - true) / true)

        if error > tolerance:
            return 0  # Fail if any component is out of tolerance

    return 1  # Pass if all components are within tolerance


def check_stress_strain(agent_answer, output):
    """Check if the stress-strain curve matches the target data."""
    vol = modal.Volume.from_name("simulations")
    try:
        final_path = agent_answer.removeprefix("/results/")
        data = b""
        for chunk in vol.read_file(final_path):
            data += chunk
        text_data = data.decode("utf-8")
        agent_array = np.genfromtxt(
            io.StringIO(text_data), delimiter=",", skip_header=1
        )
        gt_data = np.loadtxt(output["target"])
        # Interpolate both datasets to common strain points
        strain_common = np.linspace(
            0, min(agent_array[:, 0].max(), gt_data[:, 0].max()), 200
        )

        agent_stress_interp = np.interp(
            strain_common, agent_array[:, 0], agent_array[:, 1]
        )
        gt_stress_interp = np.interp(strain_common, gt_data[:, 0], gt_data[:, 1])

        # Compute normalized mean absolute error
        mae = np.mean(np.abs(agent_stress_interp - gt_stress_interp))
        max_gt_stress = np.max(np.abs(gt_stress_interp))
        score = 1.0 - (mae / max_gt_stress) if max_gt_stress > 0 else 0.0
        return max(0.0, min(1.0, score))  # clamp to [0, 1]

    except (FileNotFoundError, ValueError):
        return 0.0
