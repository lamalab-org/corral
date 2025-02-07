import json
from jinja2 import Template
import os

task_name = "lattice_generation"


# Step 1: Define the JSON template using Jinja2 syntax
task_template = """
{
  "task_id": "{{ task_id }}",
  "task_name": "{{ element }} {{ lattice_type }} Lattice Generation using {{ software }}",
  "task_prompt": "Your task is to generate a {{ lattice_type }} lattice structure for {{ element }} with lattice constant {{ lattice_constant }} angstrom using {{ software }}. The simulation should use {{ metal }} units, define a {{ simulation_box }} simulation box, set periodic boundary conditions to {{ True }}, and dump all coordinates and lattice parameters in structure.xyz file. Save all the related files at {{ result_directory }}.",
  "simulation_config": {
    "element": "{{ element }}",
    "lattice_type": "{{ lattice_type }}",
    "lattice_constant": "{{ lattice_constant }}",
    "units": "metal",
    "simulation_box": "{{ simulation_box }}",
    "pbc": "True",
    "software": "{{ software }}"
  },
  "tools": [
    "run_{{ software }}",
    "run_bash_command"
  ],
  "subtasks": [
    {
      "subtask_id": "subtask_1",
      "name": "Directory",
      "description": "Create a directory {{ result_directory }}. If it is done successfully, print the absolute location of the directory as the final output.",
      "tools": [
        "run_bash_command"
      ],
      "submission_format": "/path/to/directory",
      "initial_input": {
        "directory": "{{ result_directory }}"
      },
      "scoring_fn": "check_directory",
      "input_from_task": null,
      "ground_truth": [
        {
          "type": "directory_path",
          "data": {
            "directory": "{{ result_directory }}"
          }
        }
      ]
    },
    {
      "subtask_id": "subtask_2",
      "name": "{{ software }} Simulation",
      "description": "Your task is to generate a {{ lattice_type }} lattice structure for {{ element }} using {{ software }} with a lattice constant of {{ lattice_constant }} angstrom. The simulation should use {{ metal }} units, define a {{ simulation_box }} simulation box, set periodic boundary conditions to {{ True }}, and dump all the coordinates in structure.xyz file. Ensure that the xyz file also contains the information about the lattice parameters of the crystal as a comment. Save all the related files to the given directory.",
      "tools": [
        "run_{{ software }}",
        "run_bash_command"
      ],
      "submission_format": null,
      "initial_input": null,
      "input_from_task": [
        "subtask_1"
      ],
      "scoring_fn": "check_{{ name }}_{{ software }}",
      "ground_truth": [
        {
          "type": "file_path",
          "data": {
            "structure.xyz": "{{ directory }}/structure.xyz",
            "input.in": "{{ directory }}/input.in",
            "log.{{ software }}": "{{ directory }}/log.{{ software }}"
          }
        },

        {
          "type" : "generated_directory",
          "data" : {
            "directory" : "{{ result_directory }}"
          }
        },

        {
          "type": "simulation_config",
          "data": {
            "element": "{{ element }}",
            "lattice_type": "{{ lattice_type }}",
            "lattice_constant": "{{ lattice_constant }}",
            "units": "metal",
            "simulation_box": "{{ simulation_box }}",
            "pbc": "True",
            "software": "{{ software }}"
          }
        }
      ]
    }
  ]
}
"""

# Step 2: Define the combinations and software options
element_lattice_combinations = [
    {"element": "Aluminum", "lattice_type": "fcc", "lattice_constant": '4.05'},
    {"element": "Polonium", "lattice_type": "sc", "lattice_constant": '3.345'},
    {"element": "Iron", "lattice_type": "bcc", "lattice_constant": '2.87'},
    {"element": "Anything", "lattice_type": "edge-centered", "lattice_constant": 4.0}
]

software_options = ["ase", "lammps"]

# Step 3: Generate JSON files for each combination
template = Template(task_template)

directory_root = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/data"
result_directory = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results"
task_counter = 1

for idx, combo in enumerate(element_lattice_combinations, start=1):
    for software in software_options:
        # Define task-specific data
        task_data = {
            "task_id": f"task_{task_counter}",
            "element": combo["element"],
            "lattice_type": combo["lattice_type"],
            "lattice_constant": combo["lattice_constant"],
            "software": software,
            "directory" : f"{directory_root}/{task_name}/task_{task_counter}",
            "result_directory" : f"{result_directory}/{task_name}/task_{task_counter}",
            "simulation_box" : [5.0, 5.0, 5.0],
            "name" : task_name
        }

        # Render the template with the task data
        rendered_json = template.render(**task_data)

        # Save the rendered JSON to a file
        directory_path = os.path.join(directory_root, "lattice_generation", f"task_{task_counter}")
        os.makedirs(directory_path, exist_ok=True)
        filename = f"task_{task_counter}.json"
        filepath = os.path.join(directory_path, filename)
        print("filepath", filepath)
        with open(filepath, "w") as file:
            file.write(rendered_json)

        print(f"Generated {filename}")
        task_counter += 1

print("All JSON files generated successfully!")
