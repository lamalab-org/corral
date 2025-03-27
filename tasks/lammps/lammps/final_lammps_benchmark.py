import json
import os
from env import *

task_jsons_path = "./energy_minimisation/"

json_files = os.listdir(task_jsons_path)

environments = {}

files = ['task_11.json']
for file in files:

# for file in json_files:
    json_path = os.path.join(task_jsons_path, file)
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)  # Parses JSON into a Python dictionary
    task_id = data['id']
    prompt = data['input'][0]['prompt']
    target_score = float(data['output'][0]['target'])
    threshold = float(data['output'][0]['threshold'])
    tools = data['tools'] 
    tool_functions = [globals().get(tool_name) for tool_name in tools]
    environments[task_id] = LammpsEnvironment(task_id, prompt, target_score, tool_functions, threshold, data['keywords'][-1], data['name'])
    # break

app = create_benchmark_server(environments)
uvicorn.run(app, host="0.0.0.0", port=8000)