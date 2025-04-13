import json
import os
from env import *

from corral.io import (
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    MkdirTool,
    CatFilesTool,
    FileInfoTool,
    CopyFileTool
)

def create_tools(fs_manager: FSManager) -> dict[str, Tool]:
    return {
        "list_files": ListFilesTool(fs_manager),
        "read_file": ReadFileTool(fs_manager),
        "write_file": WriteFileTool(fs_manager),
        "file_info": FileInfoTool(fs_manager),
        "copy_file": CopyFileTool(fs_manager),
        "mkdir": MkdirTool(fs_manager),
        "cat_files": CatFilesTool(fs_manager)
    }

fs_protocol = os.environ.get("CORRAL_FS_PROTOCOL", "file")
fs_kwargs = {}  # add more keyword options from config if needed
fs_manager = FSManager(protocol=fs_protocol, app="simagent")
io_tools = create_tools(fs_manager)

task_type = "uniaxial_tension"
model = "claude_35"

task_jsons_path = f"./final_benchmark_data/{task_type}"

json_files = os.listdir(task_jsons_path)

environments = {}

for file in json_files:

# for file in json_files:
    json_path = os.path.join(task_jsons_path, file)
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)  # Parses JSON into a Python dictionary
    # data = data['subtasks'][1]
    task_id = data['id']
    prompt = data['input'][0]['prompt']
    answer_file = os.environ.get(
    "MODAL_BASE_IO_PATH", f"/results/test_results/{model}/{task_type}/{task_id}/")
    scoring_fn = data['scoring_fn']
    output = data['output']
    # environments[task_id] = LammpsEnvironment(task_id, prompt, target_score, io_tools, threshold, answer_file, scoring_fn)
    environments[task_id] = LammpsEnvironment(task_id, prompt, io_tools, output, answer_file, scoring_fn)
    # break

app = create_benchmark_server(environments)
uvicorn.run(app, host="0.0.0.0", port=8000)