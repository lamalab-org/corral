import modal 
import numpy as np
from utils import extract_lattice_coordinates
import os
import io

def check_numerical(answer_directory, agent_answer, output):
    return 1.0 if abs(float(agent_answer) - float(output['target'])) < float(output['threshold']) else 0.0

def check_structure(answer_directory, answer_file, output):
    vol = modal.Volume.from_name("simulations")
    try:
        final_path = answer_file.removeprefix("/results/")
        data = b""
        for chunk in vol.read_file(final_path):
            data += chunk
        structure_path = output['target']
        coords1 = extract_lattice_coordinates(data)
        coords2 = extract_lattice_coordinates(structure_path)
        return float(np.array_equal(coords1, coords2))
    except (FileNotFoundError, ValueError):
        return 0.0

def check_minimiser(answer_directory, answer_file, output):
    vol = modal.Volume.from_name("simulations")
    min_style_orig = output['target']
    try:
        min_style_arg = 'cg'
        final_path = answer_file.removeprefix("/results/")
        data = b""
        for chunk in vol.read_file(final_path):
            data += chunk
        text_data = data.decode('utf-8')
        # print("text_data")
        for line in text_data.splitlines():
            if line.startswith("min_style"):
                min_style_arg = line.split()[1]
        if min_style_arg == min_style_orig:
            return 1.0
        else:
            return 0.5
    except (FileNotFoundError, ValueError):
        return 0.5

def energy_minimisation(answer_directory, agent_answer, output):
    minimiser = check_minimiser(answer_directory, os.path.join(answer_directory, output[1]['input_script']), output[1]) 
    numerical = check_numerical(answer_directory, agent_answer, output[0])
    return minimiser * numerical

def energy_minimisation_with_structure(answer_directory, agent_answer, output):
    minimiser = check_minimiser(answer_directory, os.path.join(answer_directory, output[0]['input_script']), output[0])
    structure = check_structure(answer_directory, agent_answer, output[1])
    return minimiser * structure

def check_stress_tensor(answer_directory, agent_output, output):
    reference_output = output['target']
    tolerance = float(output['threshold'])
    try:

        agent_vals = list(map(float, agent_output.strip().split(",")))
        ref_vals = list(map(float, reference_output.strip().split(",")))

        if len(agent_vals) != 6 or len(ref_vals) != 6:
            return 0
    except Exception:
        return 0

    for pred, true in zip(agent_vals, ref_vals):
        if abs(true) < 1e-6:
            error = abs(pred - true)
        else:
            error = abs((pred - true) / true)

        if error > tolerance:
            return 0  # Fail if any component is out of tolerance

    return 1  # Pass if all components are within tolerance

def check_stress_strain(answer_directory, agent_answer, output):
    vol = modal.Volume.from_name("simulations")
    try:
        final_path = agent_answer.removeprefix("/results/")
        data = b""
        for chunk in vol.read_file(final_path):
            data += chunk
        text_data = data.decode('utf-8')
        agent_array = np.genfromtxt(io.StringIO(text_data), delimiter=",", skip_header=1)
        gt_data = np.loadtxt(output['target'])
        # Interpolate both datasets to common strain points
        strain_common = np.linspace(0, min(agent_array[:, 0].max(), gt_data[:, 0].max()), 200)

        agent_stress_interp = np.interp(strain_common, agent_array[:, 0], agent_array[:, 1])
        gt_stress_interp = np.interp(strain_common, gt_data[:, 0], gt_data[:, 1])

        # Compute normalized mean absolute error
        mae = np.mean(np.abs(agent_stress_interp - gt_stress_interp))
        max_gt_stress = np.max(np.abs(gt_stress_interp))
        score = 1.0 - (mae / max_gt_stress) if max_gt_stress > 0 else 0.0
        score = max(0.0, min(1.0, score))  # clamp to [0, 1]

        return score
    except (FileNotFoundError, ValueError):
        return 0.0


