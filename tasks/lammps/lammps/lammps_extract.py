import re
import numpy as np

def extract_lattice_generation_configs(lammps_input):
    """Extracts lattice generation configurations from a LAMMPS input file or bytes data."""
    
    configs = {
        "units": None,
        "lattice_type": None,
        "lattice_constant": None,
        "simulation_box_size": {}
    }

    # Read data from file or bytes
    if isinstance(lammps_input, str):  # If it's a file path
        with open(lammps_input, "r") as file:
            lines = file.readlines()
    elif isinstance(lammps_input, bytes):  # If it's bytes data
        lines = lammps_input.decode("utf-8").split("\n")
    else:
        raise ValueError("Input must be a file path (str) or bytes data.")

    for line in lines:
        line = line.strip()
        if line.startswith("#") or not line:
            continue  # Skip comments and empty lines

        # Extract units
        match_units = re.match(r"^units\s+(\w+)", line)
        if match_units:
            configs["units"] = match_units.group(1)

        # Extract lattice type and lattice constant
        match_lattice = re.match(r"^lattice\s+(\w+)\s+([\d\.]+)", line)
        if match_lattice:
            configs["lattice_type"] = match_lattice.group(1)
            configs["lattice_constant"] = float(match_lattice.group(2))

        # Extract simulation box size
        match_region = re.match(r"^region\s+\w+\s+block\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)", line)
        if match_region:
            configs["simulation_box_size"] = {
                "x_min": float(match_region.group(1)),
                "x_max": float(match_region.group(2)),
                "y_min": float(match_region.group(3)),
                "y_max": float(match_region.group(4)),
                "z_min": float(match_region.group(5)),
                "z_max": float(match_region.group(6))
            }

    return configs

def extract_lattice_coordinates(lammps_data):
    """Extracts atomic coordinates from a LAMMPS write_data file or bytes data."""
    
    coordinates = []

    # Read data from file or bytes
    if isinstance(lammps_data, str):  # If it's a file path
        with open(lammps_data, "r") as file:
            lines = file.readlines()
    elif isinstance(lammps_data, bytes):  # If it's bytes data
        lines = lammps_data.decode("utf-8").split("\n")
    else:
        raise ValueError("Input must be a file path (str) or bytes data.")

    # Find the start of the "Atoms" section
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            start_idx = i + 2  # Skip the "Atoms" header line
            break

    if start_idx is None:
        raise ValueError("Could not find atomic data in the input.")

    # Extract coordinates (ignoring atom ID and type)
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        coordinates.append([x, y, z])

    return np.array(coordinates)


