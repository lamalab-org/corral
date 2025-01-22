import json

# Step 1: Define the element-lattice combinations
element_lattice_combinations = [
    {"element": "Aluminum", "lattice_type": "FCC", "lattice_constant": 4.05},
    {"element": "Polonium", "lattice_type": "simple cubic", "lattice_constant": 3.345},
    {"element": "Iron", "lattice_type": "bcc", "lattice_constant": 2.87},
    {"element": "Anything", "lattice_type": "edge-centered", "lattice_constant": 4.0}
]

# Step 2: Define software options
software_options = ["ase", "lammps"]

# Step 3: Define a template for the task
task_template = {
    "task_id": "{task_id}",
    "task_name": "{element} {lattice_type} Lattice Generation using {software}",
    "task_prompt": "Your task is to generate a {lattice_type} lattice structure for {element} with lattice constant {lattice_constant} angstrom using {software}. The simulation should use metal units, define a {simulation_box} simulation box, set periodic boundary conditions to True, and dump all coordinates and lattice parameters in structure.xyz file. Save all the related files at /Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results/task_{task_id}.",
    "task_properties": {
        "element": "{element}",
        "lattice_type": "{lattice_type}",
        "lattice_constant": "{lattice_constant}",
        "units": "metal",
        "simulation_box": "{simulation_box}",
        "pbc": "True",
        "software": "{software}"
    },
    "tools": ["run_lammps", "run_bash_command"],
    "subtasks": []  # Subtasks will be filled dynamically
}

# Step 4: Define a template for subtasks
subtask_template = {
    "subtask_id": "{subtask_id}",
    "name": "{name}",
    "description": "{description}",
    "tools": ["{tool}"],
    "scoring_fn": "check_structure",
    "submission_format": "{submission_format}",
    "metric": "boolean",
    "initial_input": None,  # Can be None or a dictionary
    "input_from_task": [],  # List of strings
    "ground_truth": []  # List of ground truth entries
}

# Step 5: Define subtask configurations (provided by the user)
subtask_configs = [
    {
        "name": "Directory Generation",
        "description": "Your task is to generate a directory whose path is given to you. If it is done successfully, print the absolute location of the directory as the final output.",
        "tool": "run_bash_command",
        "submission_format": "absolute/path/to/directory",
        "initial_input": {
            "directory_path": "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results/task_{task_id}"
        },
        "input_from_task": None,  # No dependencies
        "ground_truth": [
            {
                "type": "file_path",
                "properties": {
                    "expected_directory": "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results/task_{task_id}",
                }
            }
        ]
    },
    {
        "name": "{software} simulation",
        "description": "Your task is to generate a {lattice_type} lattice structure for {element} using {software} with a lattice constant of {lattice_constant} angstrom. The simulation should use metal units, define a {simulation_box} simulation box, set periodic boundary conditions to True, and dump all the coordinates and lattice parameters in structure.xyz file. Save all the related files to /Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results/task_{task_id}.",
        "tool": "run_lammps",
        "submission_format": None,
        "initial_input": None,  # No initial input
        "input_from_task": ["subtask_1"],  # Depends on subtask_1
        "ground_truth": [
            {
                "type": "file_path",
                "properties": {
                    "structure.xyz": "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results/task_{task_id}/structure.xyz"
                }
            },
            {
                "type": "file_path",
                "properties": {
                    "input.in": "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results/task_{task_id}/input.in"
                }
            },
            {
                "type": "file_path",
                "properties": {
                    "log.lammps": "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results/task_{task_id}/log.lammps"
                }
            },
    
        ]
    }
]


task_properties = task_template['task_properties']

for prop in task_properties:
    subtask_configs[1]['ground_truth'].append(
        {
            "type" : prop,
            "properties" : {
                prop : task_properties[prop]
            }
        }
        
    )


# Step 6: Function to replace placeholders in a dictionary
def fill_template(template, values):
    """
    Recursively replaces placeholders in a template dictionary with values.
    """
    if isinstance(template, dict):
        return {key: fill_template(value, values) for key, value in template.items()}
    elif isinstance(template, list):
        return [fill_template(item, values) for item in template]
    elif isinstance(template, str):
        try:
            return template.format(**values)
        except KeyError:
            return template  # Return the original string if placeholder is not found
    else:
        return template

# Step 7: Generate tasks by filling the template
tasks = []
task_id = 1
for combination in element_lattice_combinations:
    for software in software_options:
        # Define the values to replace placeholders
        values = {
            "task_id": task_id,
            "element": combination["element"],
            "lattice_type": combination["lattice_type"],
            "lattice_constant": combination["lattice_constant"],
            "simulation_box": [5, 5, 5],  # Simulation box as a list
            "software": software
        }
        # Fill the task template using the values
        task = fill_template(task_template, values)

        # Generate subtasks for this task
        task_subtasks = []
        for i, config in enumerate(subtask_configs, start=1):
            # Merge task-level values with subtask config
            subtask_values = {
                "subtask_id": f"subtask_{i}",
                **values,
                **config
            }
            # Fill the subtask template using the merged values
            filled_subtask = fill_template(config, subtask_values)
            # Handle initial_input (can be None or a dictionary)
            if config["initial_input"] is not None:
                filled_subtask["initial_input"] = fill_template(config["initial_input"], subtask_values)
            # Handle input_from_task (list of strings)
            filled_subtask["input_from_task"] = config["input_from_task"]
            # Handle ground_truth (list of objects)
            filled_subtask["ground_truth"] = [
                {
                    "type": gt["type"],
                    "properties": fill_template(gt["properties"], subtask_values)
                }
                for gt in config["ground_truth"]
            ]
            task_subtasks.append(filled_subtask)

        # Add subtasks to the task
        task["subtasks"] = task_subtasks

        # Add the task to the list of tasks
        tasks.append(task)
        task_id += 1  # Increment task ID

# Step 8: Save the generated tasks to a JSON file
output_file = "generated_tasks.json"
with open(output_file, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Generated tasks saved to {output_file}")