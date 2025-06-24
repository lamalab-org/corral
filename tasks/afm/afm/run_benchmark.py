import json
import os

from env import AfmEnvironment

from corral.server import run_server

task_type = "task_5"
# model = "gpt-4o"
model = "claude_37"
agent = "tool_calling"
path = "./afm_working_directory"
# path = "./test"
task_jsons_path = f"./benchmark_data/"

json_path = os.path.join(task_jsons_path, task_type)

environments = {}

with open(f"{json_path}.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    task_id = data['id']
    prompt = data['input'][0]['prompt']
    scoring_fn = data['scoring_fn']
    output = data['output'][0]['params']
    initial_params = data['output'][0]['initial_params']
    file_ = data['output'][0]['file']
    pointer = data['output'][0]['pointer']
    gt = data['output'][0]['gt']
    work_dir = f"{path}/{agent}/{model}/{task_type}"
    target_directory = data['output'][0]['target_directory']
    environments[task_id] = AfmEnvironment(task_id, prompt, output, initial_params, scoring_fn, work_dir, file = file_, pointer = pointer, gt = float(gt), target_directory = target_directory)

host = os.environ.get("CORRAL_HOST", "0.0.0.0")
port = int(os.environ.get("CORRAL_PORT", "8000"))
run_server(environments, host, port)

