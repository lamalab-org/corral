import os
import json
from env import TaskDefinition, TaskGroup, TaskEnvironment
from tools import create_tools
from corral.server import create_benchmark_server
import uvicorn
import modal 
from lammps_evaluate import evaluate_lattice_config, check_compilation_success, compare_initial_lattice

# LLM = "gpt-4"

data_path = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/data_utils/data/lattice_generation/"

def check_potential(result, ground_truth)-> float:
    if ground_truth['output'] == result['answer']:
        return 1
    return 0

def check_directory(result, ground_truth)-> float:
    gt = ground_truth['output']
    if result['answer'] == gt:
        check_directory_modal = modal.Function.lookup("simagent", "check_directory")
        output = check_directory_modal.remote(gt)
        return output

def check_numerical(result, ground_truth)-> float:
    return 0

def check_lattice_generation(result, ground_truth)-> float:
    score = 0
    # total_score = int(ground_truth['max_score'])
    agent_path = ground_truth['previous_task_output'][0]
    print("agent_path", agent_path)
    gt_path = ground_truth['ground_truth']
    print("gt_path", gt_path)
    vol = modal.Volume.from_name("simulations")

    try:
        file_path = vol.read_file(f"{agent_path.removeprefix('/results/')}/input.in")
        data = b""
        for chunk in file_path:
            data += chunk
        score += evaluate_lattice_config(data, f"{gt_path}/input.in")
    except:
        return score
    
    try: 
        file_path = vol.read_file(f"{agent_path.removeprefix('/results/')}/log.lammps")
        data = b""
        for chunk in file_path:
            data += chunk
        score += check_compilation_success(data)
    except:
        return score

    try: 
        file_path = vol.read_file(f"{agent_path.removeprefix('/results/')}/structure.xyz")
        data = b""
        for chunk in file_path:
            data += chunk
        score += compare_initial_lattice(data, f"{gt_path}/structure.xyz")
    except:
        return score
    return score

def lattice_generation(result, ground_truth):
    score = 0
    results_path = ground_truth['results_path']
    score += check_directory({'answer' : results_path}, {'output' : results_path})
    score += check_lattice_generation(None, {'previous_task_output' : [results_path], 'ground_truth' : ground_truth['ground_truth']})
    return score

environments = {}
files = os.listdir(data_path)
for file in files:
    file_path = os.path.join(data_path, file)
    with open(file_path, 'r') as f:
        task = json.load(f)
    tasks = {}
    func = locals()[task['ground_truth']['scoring_fn']]
    tasks[task['task_id']] = TaskDefinition(
        name = task['name'],
        description = task['description'],
        tools = task['tools'],
        scoring_fn = func,
        submission_format={'answer' : task['submission_format']},
        ground_truth = task['ground_truth'],
        input_from_task = None, 
        initial_input = task['initial_input']
    )
    task_group = TaskGroup(group_id = task['task_id'], tasks = tasks)

    available_tools = create_tools()
    for task_id in task_group.tasks:
        # All environments share the same task group instance
        #task_id = f"{task_group.group_id}_{_id}"
        environments[task_id] = TaskEnvironment(
            task_id=task_id,
            task_group=task_group,
            available_tools=available_tools
        )
app = create_benchmark_server(environments)
uvicorn.run(app, host="0.0.0.0", port=8000)


    