import json
import os
from env import *
from corral.server import run_server


task_type = "energy_minimisation"  # "elastic_constants" or "energy_minimisation"
# model = "gpt-4o"
model = "claude_37"
agent = "react"
path = "/results/final_benchmark_data/26_June_2025_test"

task_jsons_path = f"./final_benchmark_data/{task_type}"

json_files = os.listdir(task_jsons_path)

environments = {}

for file in ["task_1.json"]:
    json_path = os.path.join(task_jsons_path, file)
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)  # Parses JSON into a Python dictionary
    task_id = data['id']
    prompt = data['input'][0]['prompt']
    answer_file = os.environ.get(
    "MODAL_BASE_IO_PATH", f"{path}/{agent}/{model}/{task_type}/{task_id}/")
    scoring_fn = data['scoring_fn']
    output = data['output']
    environments[task_id] = LammpsEnvironment(task_id, prompt, output, answer_file, scoring_fn)

host = os.environ.get("CORRAL_HOST", "0.0.0.0")
port = int(os.environ.get("CORRAL_PORT", "8000"))
run_server(environments, host, port)