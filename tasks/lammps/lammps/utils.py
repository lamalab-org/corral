import numpy as np

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

if __name__ == "__main__":

    coords1 = extract_lattice_coordinates("/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/ground_truth/npt/minimized_aluminum_structure.dat")
    coords2 = extract_lattice_coordinates("/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/ground_truth/npt/minimized_aluminum_structure.dat")

    print(int(np.array_equal(coords1, coords2)))