import os
import json
from jinja2 import Template

name = "lattice_generation"

# Load Jinja template from file
TEMPLATE_FILE = f"templates/{name}_template.jinja"
with open(TEMPLATE_FILE, "r") as file:
    template_content = file.read()

# Jinja template
template = Template(template_content)

# List of configurations
configs = [
    {"element": "Aluminum", "lattice_type": "fcc", "lattice_constant": "4.05", "software": "lammps", "pbc": "true", "simulation_box": "5x5x5", "units": "metal", "command" : "write_data", "task" : "total energy", "output" : "-13.7", "optimizer" : "conjugate gradient"},
    {"element": "Polonium", "lattice_type": "sc", "lattice_constant": "3.345", "software": "lammps", "pbc": "true", "simulation_box": "5x5x5", "units": "metal", "command" : "write_data", "task" : "total energy", "output" : "-13.7"},
    {"element": "Iron", "lattice_type": "bcc", "lattice_constant": "2.87", "software": "lammps", "pbc": "true", "simulation_box": "5x5x5", "units": "metal", "command" : "write_data", "task" : "total energy", "output" : "-13.7"},
    {"element": "Mg", "lattice_type": "hcp", "lattice_constant": "1.633", "software": "lammps", "pbc": "true", "simulation_box": "5x5x5", "units": "metal", "command" : "write_data", "task" : "total energy", "output" : "-13.7"}
]

# Output directory
OUTPUT_DIR = f"data/{name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create directory if it doesn't exist

# Process each configuration
for idx, config in enumerate(configs, start=1):
    # Prepare input values for the template
    params = {
        "task_id": f"task_{idx}",
        "element": config["element"],
        "lattice_type": config["lattice_type"],
        "lattice_constant": config["lattice_constant"],
        "units": config["units"],
        "software": config["software"],
        "simulation_box": {k: v for k, v in zip(["x", "y", "z"], config["simulation_box"].split("x"))},
        "output_directory": f"/results/{name}/task_{idx}",
        "ground_truth_path": f"/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/data_utils/ground_truth/{name}/task_{idx}",
        "command" : config['command'],
        "task" : config['task'],
        "output" : config['output']
    }

    # Render template
    generated_json = template.render(params)
    print(generated_json)

    # Save JSON file
    output_file_path = os.path.join(OUTPUT_DIR, f"task_{idx}.json")
    with open(output_file_path, "w") as f:
        json.dump(json.loads(generated_json), f, indent=2)

    print(f"Generated: {output_file_path}")

print(f"\nJSON files successfully created in {name} directory.")




# import os
# from jinja2 import Template

# class TaskManager:
#     """Manages the generation of task JSON files for different types of tasks."""

#     def __init__(self, task_name: str, directory_root: str, result_directory: str):
#         """
#         Args:
#             task_name (str): Name of the task (e.g., "lattice_generation", "energy_minimization").
#             directory_root (str): Root directory for storing task configurations.
#             result_directory (str): Directory for storing simulation results.
#         """
#         self.task_name = task_name
#         self.directory_root = directory_root
#         self.result_directory = result_directory

#     def _load_template(self, template_path: str) -> Template:
#         """Load a Jinja2 template from a file."""
#         with open(template_path, "r") as file:
#             return Template(file.read())

#     def generate_task_configs(self, template_path: str, task_data_list: list) -> None:
#         """
#         Generate JSON configuration files for the given task.

#         Args:
#             template_path (str): Path to the Jinja2 template file.
#             task_data_list (list): List of dictionaries containing task-specific data.
#         """
#         template = self._load_template(template_path)
#         task_counter = 1

#         for task_data in task_data_list:
#             # Add task_id, directory, and result_directory to the task data
#             task_data["task_id"] = f"task_{task_counter}"
#             task_data["directory"] = f"{self.directory_root}/{self.task_name}/task_{task_counter}"
#             task_data["result_directory"] = f"{self.result_directory}/{self.task_name}/task_{task_counter}"
#             task_data["name"] = self.task_name

#             # Render the template with the task data
#             rendered_json = template.render(**task_data)

#             # Save the rendered JSON to a file
#             directory_path = os.path.join(self.directory_root, self.task_name, f"task_{task_counter}")
#             os.makedirs(directory_path, exist_ok=True)
#             filename = f"task_{task_counter}.json"
#             filepath = os.path.join(directory_path, filename)
#             with open(filepath, "w") as file:
#                 file.write(rendered_json)

#             print(f"Generated {filename}")
#             task_counter += 1

#         print(f"All JSON files for '{self.task_name}' generated successfully!")



# if __name__ == "__main__":
#     # Define task names and directories
#     directory_root = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/data"
#     result_directory = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results"

#     # Define task-specific data
#     lattice_task_data = [
#         {"element": "Aluminum", "lattice_type": "fcc", "lattice_constant": 4.05, "software": "ase", "pbc" : "true", "simulation_box" : "5.0x5.0x5.0", "units" : "metal"},
#         {"element": "Aluminum", "lattice_type": "fcc", "lattice_constant": 4.05, "software": "lammps", "pbc" : "true", "simulation_box" : "5.0x5.0x5.0", "units" : "metal"},
#         {"element": "Polonium", "lattice_type": "sc", "lattice_constant": 3.345, "software": "ase", "pbc" : "true", "simulation_box" : "5.0x5.0x5.0", "units" : "metal"},
#         {"element": "Polonium", "lattice_type": "sc", "lattice_constant": 3.345, "software": "lammps", "pbc" : "true", "simulation_box" : "5.0x5.0x5.0", "units" : "metal"},
#         {"element": "Iron", "lattice_type": "bcc", "lattice_constant": 2.87, "software": "ase", "pbc" : "true", "simulation_box" : "5x5x5", "units" : "metal"},
#         {"element": "Iron", "lattice_type": "bcc", "lattice_constant": 2.87, "software": "lammps", "pbc" : "true", "simulation_box" : "5x5x5", "units" : "metal"},
#         {"element": "Anything", "lattice_type": "edge-centered", "lattice_constant": 4.0, "software": "ase", "pbc" : "true", "simulation_box" : "5x5x5", "units" : "metal"},
#         {"element": "Anything", "lattice_type": "edge-centered", "lattice_constant": 4.0, "software": "lammps", "pbc" : "true", "simulation_box" : "5x5x5", "units" : "metal"}
#     ]

#     # energy_minimization_task_data = [
#     #     {"element": "Aluminum", "force_field": "eam", "force_tolerance": '0.01', "software": "lammps"},
#     #     {"element": "Iron", "force_field": "eam", "force_tolerance": '0.01', "software": "ase"},
#     #     {"element": "Silicon", "force_field": "tersoff", "force_tolerance": '0.001', "software": "lammps"}
#     # ]


#     energy_minimization_task_data = [

#         {
#             "task_type": "cohesive_energy",
#             "element": "Aluminum",
#             "force_field": "eam",
#             "software": "lammps",
#             "simulation_box": "5x5x5",
#             "pbc" : "true", 
#             "units" : "metal", 
#             "lattice_type" : "fcc",
#             "optimizer" : "conjugate_gradient", 
#             "lattice_constant" : "4.05", 
#             "energy_tolerance" : "1.0e-4",
#             "force_tolerance" : "1.0e-6",
#             "max_iter" : "10000", 
#             "max_eval" : "10000"
#         },

#         # {
#         #     "task_type": "elastic_constants",
#         #     "element": "Silicon",
#         #     "force_field": "tersoff",
#         #     "software": "lammps",
#         #     "simulation_box": "5x5x5",
#         # }



#     ]

#     # # Generate JSON files for lattice generation
#     # lattice_task_manager = TaskManager("lattice_generation", directory_root, result_directory)
#     # lattice_task_manager.generate_task_configs("templates/lattice_generation_template.j2", lattice_task_data)

#     # Generate JSON files for energy minimization
#     energy_minimization_task_manager = TaskManager("energy_minimization", directory_root, result_directory)
#     energy_minimization_task_manager.generate_task_configs("templates/energy_minimization_template.j2", energy_minimization_task_data)