import re

def parse_lammps_input(file_path):
    """
    Parse a LAMMPS input file and extract key simulation parameters.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Initialize a dictionary to store extracted parameters
    config = {
        "units": None,
        "lattice_type": None,
        "lattice_constant": None,
        "simulation_box": None,
        "pbc": None,
        "dump_file": None
    }

    # Regular expressions to extract values
    units_pattern = re.compile(r"units\s+(\w+)")
    lattice_pattern = re.compile(r"lattice\s+(\w+)\s+([\d.]+)")
    region_pattern = re.compile(r"region\s+\w+\s+block\s+([\d.\s-]+)")
    boundary_pattern = re.compile(r"boundary\s+(\w)\s+(\w)\s+(\w)")
    dump_pattern = re.compile(r"dump\s+\d+\s+\w+\s+\w+\s+\d+\s+(\w+\.\w+)")

    for line in lines:
        # Extract units
        if units_pattern.search(line):
            config["units"] = units_pattern.search(line).group(1)

        # Extract lattice type and constant
        if lattice_pattern.search(line):
            config["lattice_type"] = lattice_pattern.search(line).group(1)
            config["lattice_constant"] = float(lattice_pattern.search(line).group(2))

        # Extract simulation box dimensions
        if region_pattern.search(line):
            box_values = region_pattern.search(line).group(1).split()
            # Calculate box dimensions: end - start for each axis
            x_dim = float(box_values[1]) - float(box_values[0])
            y_dim = float(box_values[3]) - float(box_values[2])
            z_dim = float(box_values[5]) - float(box_values[4])
            config["simulation_box"] = f"[{x_dim}, {y_dim}, {z_dim}]"

        # Extract periodic boundary conditions
        if boundary_pattern.search(line):
            pbc_values = boundary_pattern.search(line).groups()
            config["pbc"] = "True" if all(p == "p" for p in pbc_values) else "False"

        # Extract dump file name
        if dump_pattern.search(line):
            config["dump_file"] = dump_pattern.search(line).group(1)

    return config

def check_simulation_success(log_file_path):
    """
    Check if a LAMMPS simulation completed successfully by analyzing the log file.
    """
    with open(log_file_path, "r") as file:
        lines = file.readlines()

    # Check for errors
    errors = [line for line in lines if "ERROR:" in line]
    if errors:
        print("Simulation failed with the following errors:")
        for error in errors:
            print(error.strip())
        return False

    # Check for final timestep (successful completion)
    final_timestep = [line for line in lines if "Loop time of" in line or "Total wall time" in line]
    if not final_timestep:
        print("Simulation did not complete successfully (no final timestep found).")
        return False

    # Check for warnings
    warnings = [line for line in lines if "WARNING:" in line]
    if warnings:
        print("Simulation completed with warnings:")
        for warning in warnings:
            print(warning.strip())
    else:
        print("Simulation completed successfully with no errors or warnings.")

    return True

# # Example usage
# log_file_path = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results/lattice_generation/task_2/log.lammps"  # Replace with the path to your log.lammps file
# success = check_simulation_success(log_file_path)
# if success:
#     print("The simulation was successful.")
# else:
#     print("The simulation failed.")