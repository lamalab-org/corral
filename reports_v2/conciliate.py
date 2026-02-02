import json
from pathlib import Path

root_path = Path(__file__).parent


for json_file in root_path.glob("*.json"):
    if "gpt" not in json_file.name:
        continue
    final_data = {}
    if "lvl1" in json_file.name:
        level = "level_1"
    elif "lvl2" in json_file.name:
        level = "level_2"
    elif "lvl3" in json_file.name:
        level = "level_3"
    else:
        raise ValueError("Level not found in filename")
    tool_path = Path(__file__).parent.parent / f"gpt-4o/spectra/{level}/subtasks"
    print(f"Processing {json_file.name}")
    with json_file.open("r") as f:
        tool_data = json.load(f)

    tool_path_d = tool_path / json_file.name
    with tool_path_d.open("r") as f:
        data = json.load(f)

    final_data = data

    data["metrics"]["average_score"] = tool_data["metrics"]["Average Score"]
    data["metrics"]["overall_success_rate"] = tool_data["metrics"][
        "Overall Success Rate"
    ]
    data["metrics"]["pass@1"] = tool_data["metrics"]["Pass@1"]
    data["metrics"]["pass@2"] = tool_data["metrics"]["Pass@2"]
    data["metrics"]["pass@3"] = tool_data["metrics"]["Pass@3"]
    data["metrics"]["pass@4"] = tool_data["metrics"]["Pass@4"]
    data["metrics"]["pass@5"] = tool_data["metrics"]["Pass@5"]
    data["metrics"]["pass^1"] = tool_data["metrics"]["Pass^1"]
    data["metrics"]["pass^2"] = tool_data["metrics"]["Pass^2"]
    data["metrics"]["pass^3"] = tool_data["metrics"]["Pass^3"]
    data["metrics"]["pass^4"] = tool_data["metrics"]["Pass^4"]
    data["metrics"]["pass^5"] = tool_data["metrics"]["Pass^5"]

    for task_id, task_results in final_data["task_results"].items():
        data["task_results"][task_id]["average_score"] = tool_data["task_results"][
            task_id
        ]["Task Average Score"]
        data["task_results"][task_id]["success_rate"] = tool_data["task_results"][
            task_id
        ]["Task Success Rate"]
        data["task_results"][task_id]["pass@1"] = tool_data["task_results"][task_id][
            "Task Pass@1"
        ]
        data["task_results"][task_id]["pass@2"] = tool_data["task_results"][task_id][
            "Task Pass@2"
        ]
        data["task_results"][task_id]["pass@3"] = tool_data["task_results"][task_id][
            "Task Pass@3"
        ]
        data["task_results"][task_id]["pass@4"] = tool_data["task_results"][task_id][
            "Task Pass@4"
        ]
        data["task_results"][task_id]["pass@5"] = tool_data["task_results"][task_id][
            "Task Pass@5"
        ]
        data["task_results"][task_id]["pass^1"] = tool_data["task_results"][task_id][
            "Task Pass^1"
        ]
        data["task_results"][task_id]["pass^2"] = tool_data["task_results"][task_id][
            "Task Pass^2"
        ]
        data["task_results"][task_id]["pass^3"] = tool_data["task_results"][task_id][
            "Task Pass^3"
        ]
        data["task_results"][task_id]["pass^4"] = tool_data["task_results"][task_id][
            "Task Pass^4"
        ]
        data["task_results"][task_id]["pass^5"] = tool_data["task_results"][task_id][
            "Task Pass^5"
        ]

        for i, trial in enumerate(task_results["trials"]):
            trial["score"] = tool_data["task_results"][task_id]["trials"][i]["score"]
            trial["submitted_answer"] = tool_data["task_results"][task_id]["trials"][i][
                "submitted_answer"
            ]
            trial["success"] = tool_data["task_results"][task_id]["trials"][i][
                "success"
            ]

    final_path = root_path.parent / json_file.name
    with final_path.open("w") as f:
        json.dump(final_data, f, indent=4)
