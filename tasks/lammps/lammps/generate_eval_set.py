import json
import os

# Define Task Skeletons
task_skeletons = {
    "lattice_generation": {
        "task_id": "{task_id}",
        "task_name": "{lattice_type} Lattice Generation for {element}",
        "task_prompt": "Your task is to generate a {lattice_type} lattice structure for {element} using {software} with lattice constant {lattice_constant} angstrom. "
                       "The simulation should use {units} units, define a {simulation_box} simulation box, "
                       "set periodic boundary conditions to {pbc}, and dump all coordinates and lattice parameters in {element}.xyz file. "
                       "Save all the related files to {directory}/{task_id}",
        "task_properties": {
            "element": "{element}",
            "lattice_type": "{lattice_type}",
            "lattice_constant": "{lattice_constant}",
            "units": "{units}",
            "simulation_box": "{simulation_box}",
            "pbc": "{pbc}",
            "software": "{software}"
        }
    },
    # Other task skeletons can go here...
}

# Define Subtask Skeletons
directory_subtask = {
    "name": "Directory Generation",
    "Description": "Your task is to generate a directory whose path is given to you. If it is done successfully print the absolute location of the directory as the final output.",
    "tools": ['run_bash_command'],
    "scoring_fn": "check_structure",
    "submission_format": "absolute/path/to/directory",
    "metric": "boolean",
    "initial_input": "{directory}/{task_id}",
    "input_from_task" : None
}

lattice_generation_subtask = {
    "name": "{software} simulation",
    "Description": "Your task is to generate a {lattice_type} lattice structure for {element} using {software} with lattice constant of {lattice_constant} angstrom. The simulation should use {units} units, define a {simulation_box} simulation box, set periodic boundary conditions to {pbc}, and dump all the coordinates and lattice parameters in {element}.xyz file. Save all the related files to at {directory}/{task_id}",
    "tools": ["run_bash_command", "run_lammps"],
    "scoring_fn": "check_structure",
    "submission_format": None,
    "metric": "boolean",
}

# Function to format a task and its subtasks dynamically
def format_task_and_subtasks(task_skeleton, properties, subtasks):
    # Format the task skeleton with task-specific properties
    task_str = json.dumps(task_skeleton)
    for key, value in properties.items():
        placeholder = f"{{{key}}}"
        task_str = task_str.replace(placeholder, str(value))
    
    # Load formatted task
    formatted_task = json.loads(task_str)
    
    # Format each subtask with the task properties and user-provided values
    formatted_subtasks = []
    
    for subtask_template in subtasks:
        subtask_str = json.dumps(subtask_template)
        
        # Replace placeholders in the subtask with task properties and manual subtask-specific values
        for key, value in properties.items():
            subtask_str = subtask_str.replace(f"{{{key}}}", str(value))
        
        # Manual input for subtask-specific fields (initial_input, input_from_task, subtask_id)
        subtask_str = subtask_str.replace("{subtask_id}", subtask_template["subtask_id"])
        if subtask_template['initial_input'] is not None:
            subtask_str = subtask_str.replace("{initial_input}", subtask_template["initial_input"])
        # subtask_str = subtask_str.replace("{input_from_task}", subtask_template["input_from_task"])
        
        # Load formatted subtask
        formatted_subtask = json.loads(subtask_str)
        formatted_subtasks.append(formatted_subtask)
    
    # Add the formatted subtasks to the task
    formatted_task['subtasks'] = formatted_subtasks
    
    return formatted_task

# Function to generate tasks with subtasks
def generate_tasks_with_subtasks(task_configs, output_directory):

    formatted_tasks = []
    
    for task_config in task_configs:
        task_type = task_config["task_type"]
        properties = task_config["properties"]
        subtasks = task_config.get("subtasks", [directory_subtask, lattice_generation_subtask])
        
        # Retrieve the correct skeleton for the task
        skeleton = task_skeletons.get(task_type)
        if not skeleton:
            print(f"Unknown task type: {task_type}")
            # continue
        
        # Handle simulation_box as a list and format its individual elements
        if "simulation_box" in properties:
            simulation_box = properties["simulation_box"]
            properties["simulation_box"] = f"{simulation_box[0]}x{simulation_box[1]}x{simulation_box[2]}"
        
        # Format the task and its subtasks
        formatted_task = format_task_and_subtasks(skeleton, properties, subtasks)   
        # Save the task with subtasks to a file
        file_path = os.path.join(output_directory, formatted_task['task_id'])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        output_file = os.path.join(file_path, f"{formatted_task['task_id']}.json")
        with open(output_file, "w") as f:
            json.dump(formatted_task, f, indent=4)

# Example Task Configurations with Manual Subtask Inputs


element_lattice_combinations = [
    {"element": "Aluminum", "lattice_type": "FCC", "lattice_constant": 4.05},
    {"element": "Polonium", "lattice_type": "simple cubic", "lattice_constant": 3.345},
    {"element": "Iron", "lattice_type": "bcc", "lattice_constant": 2.87},
    {"element": "Anything", "lattice_type": "edge-centered", "lattice_constant": 4.0}
]

# Base task properties that remain unchanged for all tasks
base_task_properties = {
    "units": "metal",
    "simulation_box": [5, 5, 5],  # List format for simulation box
    "pbc": True,
    "directory": "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results"
}

software_options = ["ase", "lammps"]

task_configs = []

task_counter = 1

for combo in element_lattice_combinations:
    for software in software_options:
        config = {
            "task_type" : "lattice_generation",
            "properties" : {
                "task_id" : f"task_{task_counter}",
                "element" : combo['element'],
                "lattice_type" : combo['lattice_type'],
                "lattice_constant" : combo['lattice_constant'],
                "units" : base_task_properties['units'],
                "simulation_box" : base_task_properties['simulation_box'],
                "pbc" : base_task_properties['pbc'],
                "software" : software, 
                "directory" : "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results"
            },
            "subtasks": [
            {
                "subtask_id": "subtask_001",  # Manual subtask ID
                # "input_from_task": "",  # Manual input from task
                **directory_subtask  # Subtask specific template
            },
            {
                "subtask_id": "subtask_002",  # Manual subtask ID
                "initial_input": None,  # Manual initial input
                "input_from_task": "subtask_001",  # Manual input from task
                **lattice_generation_subtask  # Subtask specific template
            }
        ]
        }
        task_configs.append(config)
        task_counter += 1

output_directory = "eval_data/"
generate_tasks_with_subtasks(task_configs, output_directory)
# print(task_configs[1])
# task_configs = [
#     {
#         "task_type": "lattice_generation",
#         "properties": {
#             "task_id": "task_001",
#             "element": "Aluminum",
#             "lattice_type": "FCC",
#             "lattice_constant": 4.05,
#             "units": "metal",
#             "simulation_box": [5, 5, 5],  # List to be formatted
#             "pbc": True,
#             "software": "lammps",  # Specify software here
#             "directory" : "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results"
#         },
#         "subtasks": [
#             {
#                 "subtask_id": "subtask_001",  # Manual subtask ID
#                 # "input_from_task": "",  # Manual input from task
#                 **directory_subtask  # Subtask specific template
#             },
#             {
#                 "subtask_id": "subtask_002",  # Manual subtask ID
#                 "initial_input": None,  # Manual initial input
#                 "input_from_task": "subtask_001",  # Manual input from task
#                 **lattice_generation_subtask  # Subtask specific template
#             }
#         ]
#     }
# ]

# # Element and Lattice Combinations
# element_lattice_combinations = [
#     {"element": "Aluminum", "lattice_type": "FCC", "lattice_constant": 4.05},
#     {"element": "Polonium", "lattice_type": "simple cubic", "lattice_constant": 3.345},
#     {"element": "Iron", "lattice_type": "bcc", "lattice_constant": 2.87},
#     {"element": "Anything", "lattice_type": "edge-centered", "lattice_constant": 4.0}
# ]

# # Base task properties that remain unchanged for all tasks
# base_task_properties = {
#     "units": "metal",
#     "simulation_box": [5, 5, 5],  # List format for simulation box
#     "pbc": True,
#     "directory": "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/results"
# }


# # Software options
# software_options = ["ase", "lammps"]

# # Task generation with subtasks
# def generate_task_list():
#     task_configs = []
#     task_counter = 1  # Starting task_id

#     for combo in element_lattice_combinations:
#         for software in software_options:
#             task_id = f"task_{task_counter:03d}"

#             task_config = {
#                 "task_type": "lattice_generation",
#                 "task_id": task_id,
#                 "properties": {
#                     **base_task_properties,
#                     "element": combo["element"],
#                     "lattice_type": combo["lattice_type"],
#                     "lattice_constant": combo["lattice_constant"],
#                     "software": software
#                 },
#                 "subtasks": [
#                     {
#                         "subtask_id": "subtask_001",  # Manual subtask ID
#                         **directory_subtask  # Subtask specific template
#                     },
#                     {
#                         "subtask_id": "subtask_002",  # Manual subtask ID
#                         "initial_input": None,  # Manual initial input
#                         "input_from_task": "subtask_001",  # Manual input from task
#                         **lattice_generation_subtask  # Subtask specific template
#                     }
#                 ]
#             }
#             task_configs.append(task_config)
#             task_counter += 1  # Increment task ID for next task

#     return task_configs

# Generate the task list
# output_directory = "eval_data/"

# task_configs = generate_task_list()

# generate_tasks_with_subtasks(task_configs, output_directory)

# print(task_configs[1])