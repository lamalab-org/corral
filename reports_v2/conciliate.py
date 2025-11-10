import json
from pathlib import Path

root_path = Path(__file__).parent
tool_path = Path(__file__).parent / "gpt-4o/retrosynthesis/level_2/subtasks"

for json_file in root_path.glob("*.json"):
    if "gpt" not in json_file.name:
        continue
    final_data = {}
    print(f"Processing {json_file.name}")
    with json_file.open("r") as f:
        data = json.load(f)

    tool_path_d = tool_path / json_file.name
    with tool_path_d.open("r") as f:
        tool_data = json.load(f)

    final_data = data

    final_data["metrics"]["total_tool_calls"] = tool_data["metrics"]["total_tool_calls"]
    final_data["metrics"]["successful_tool_calls"] = tool_data["metrics"][
        "successful_tool_calls"
    ]
    final_data["metrics"]["failed_tool_calls"] = tool_data["metrics"][
        "failed_tool_calls"
    ]
    final_data["metrics"]["total_token_usage"] = tool_data["metrics"][
        "total_token_usage"
    ]
    final_data["metrics"]["total_tool_execution_duration"] = tool_data["metrics"][
        "total_tool_execution_duration"
    ]
    final_data["metrics"]["total_benchmark_duration"] = tool_data["metrics"][
        "total_benchmark_duration"
    ]

    for task_id, task_results in final_data["task_results"].items():
        task_results["total_token_usage"] = tool_data["task_results"][task_id][
            "total_token_usage"
        ]
        for i, trial in enumerate(task_results["trials"]):
            trial["tool_execution_duration"] = tool_data["task_results"][task_id][
                "trials"
            ][i]["tool_execution_duration"]
            trial["token_usage"] = tool_data["task_results"][task_id]["trials"][i][
                "token_usage"
            ]
            trial["tool_calls"] = tool_data["task_results"][task_id]["trials"][i][
                "tool_calls"
            ]
            trial["total_calls"] = tool_data["task_results"][task_id]["trials"][i][
                "total_calls"
            ]
            trial["successful_calls"] = tool_data["task_results"][task_id]["trials"][i][
                "successful_calls"
            ]
            trial["failed_calls"] = tool_data["task_results"][task_id]["trials"][i][
                "failed_calls"
            ]
            trial["tools_used"] = tool_data["task_results"][task_id]["trials"][i][
                "tools_used"
            ]
            trial["error_types"] = tool_data["task_results"][task_id]["trials"][i][
                "error_types"
            ]

    final_path = root_path / "gpt" / json_file.name
    with final_path.open("w") as f:
        json.dump(final_data, f, indent=4)
