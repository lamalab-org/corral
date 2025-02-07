import re
import numpy as np
from lammps_extract import extract_lattice_generation_configs, extract_lattice_coordinates

def evaluate_lattice_config(agent_input, actual_input):
    """Compares two extracted LAMMPS configurations and returns 1 if they match, otherwise 0."""
    
    agent_config = extract_lattice_generation_configs(agent_input)
    actual_config = extract_lattice_generation_configs(actual_input)

    return 1 if agent_config == actual_config else 0

def check_compilation_success(input_data):
    """Checks if the LAMMPS compilation was successful by looking for error keywords in the log.lammps file."""
    
    error_keywords = ["ERROR", "FATAL", "Segmentation fault", "Nan"]  # Add more error patterns as needed

    # If input_data is bytes, decode it to string
    if isinstance(input_data, bytes):
        content = input_data.decode('utf-8', errors='ignore')
    elif isinstance(input_data, str):
        # If it's already a string (file path)
        with open(input_data, 'r') as file:
            content = file.read()
    else:
        raise ValueError("Input data must be bytes or file path string")

    # Check for error keywords in the content
    for keyword in error_keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", content):
            return 0  # Compilation failed if any of the error keywords are found
    
    return 1  # Compilation successful if no error keywords are found

def compare_initial_lattice(file1, file2):
    """
    Compare the atomic coordinates of two LAMMPS write_data files.

    Args:
        file1 (str): Path to the first LAMMPS data file.
        file2 (str): Path to the second LAMMPS data file.

    Returns:
        int: 1 if all coordinates match exactly, 0 otherwise.
    """

    # Load coordinate data
    coords1 = extract_lattice_coordinates(file1)
    coords2 = extract_lattice_coordinates(file2)

    # Check if all coordinates match exactly
    return int(np.array_equal(coords1, coords2))

from ase.io import read
import numpy as np

def extract_lattice_parameters(file_name):
    """
    Extracts the lattice parameters (a, b, c, alpha, beta, gamma) from an ASE-readable file
    (either .xyz or .data format).
    
    Args:
    file_name (str): Path to the ASE-readable file (e.g., .xyz or .data).
    
    Returns:
    tuple: Lattice constants (a, b, c) and angles (alpha, beta, gamma) in degrees.
    """
    
    # Read the structure from the file
    atoms = read(file_name)

    print(atoms)
    
    # Get the cell (lattice vectors)
    lattice = atoms.get_cell()
    
    # Calculate the lattice constants a, b, c (the lengths of the unit cell vectors)
    a = np.linalg.norm(lattice[0])  # Length of vector a
    b = np.linalg.norm(lattice[1])  # Length of vector b
    c = np.linalg.norm(lattice[2])  # Length of vector c
    
    # Calculate the angles alpha, beta, gamma (in degrees) between the lattice vectors
    alpha = np.degrees(np.arccos(np.dot(lattice[1], lattice[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(lattice[0], lattice[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(lattice[0], lattice[1]) / (a * b)))
    
    # Return lattice constants and angles
    return a, b, c, alpha, beta, gamma


if __name__ == "__main__":
    file1 = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/data_utils/ground_truth/lattice_generation/task_5/structure.xyz"
    # file2 = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/data_utils/ground_truth/lattice_generation/task_5/structure.xyz"
    # score = compare_initial_lattice(file1, file2)
    # print(score)

    a, b, c, alpha, beta, gamma = extract_lattice_parameters(file1)

    # Print the results
    print(f"Lattice Constants: a = {a}, b = {b}, c = {c}")
    print(f"Angles: alpha = {alpha}, beta = {beta}, gamma = {gamma}")
