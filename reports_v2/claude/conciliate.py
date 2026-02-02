import json
from pathlib import Path

root_path = Path(__file__).parent


for json_file in root_path.glob("*.json"):
    if "claude" not in json_file.name:
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
    tool_path = (
        Path(__file__).parent.parent
        / f"claude_sonnet_45/retrosynthesis/{level}/subtasks"
    )
    print(f"Processing {json_file.name}")
    with json_file.open("r") as f:
        data = json.load(f)

    tool_path_d = tool_path / json_file.name
    with tool_path_d.open("r") as f:
        tool_data = json.load(f)

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
        data["metrics"]["task_results"]["average_score"] = tool_data["metrics"][
            "task_results"
        ]["Task Average Score"]
        data["metrics"]["task_results"]["success_rate"] = tool_data["metrics"][
            "task_results"
        ]["Task Success Rate"]
        data["metrics"]["task_results"]["pass@1"] = tool_data["metrics"][
            "task_results"
        ]["Pass@1"]
        data["metrics"]["task_results"]["pass@2"] = tool_data["metrics"][
            "task_results"
        ]["Pass@2"]
        data["metrics"]["task_results"]["pass@3"] = tool_data["metrics"][
            "task_results"
        ]["Pass@3"]
        data["metrics"]["task_results"]["pass@4"] = tool_data["metrics"][
            "task_results"
        ]["Pass@4"]
        data["metrics"]["task_results"]["pass@5"] = tool_data["metrics"][
            "task_results"
        ]["Pass@5"]
        data["metrics"]["task_results"]["pass^1"] = tool_data["metrics"][
            "task_results"
        ]["Pass^1"]
        data["metrics"]["task_results"]["pass^2"] = tool_data["metrics"][
            "task_results"
        ]["Pass^2"]
        data["metrics"]["task_results"]["pass^3"] = tool_data["metrics"][
            "task_results"
        ]["Pass^3"]
        data["metrics"]["task_results"]["pass^4"] = tool_data["metrics"][
            "task_results"
        ]["Pass^4"]
        data["metrics"]["task_results"]["pass^5"] = tool_data["metrics"][
            "task_results"
        ]["Pass^5"]

        for i, trial in enumerate(task_results["trials"]):
            trial["score"] = tool_data["task_results"][task_id]["trials"][i]["score"]
            trial["submitted_answer"] = tool_data["task_results"][task_id]["trials"][i][
                "submitted_answer"
            ]
            trial["success"] = tool_data["task_results"][task_id]["trials"][i][
                "success"
            ]

    final_path = root_path / "claude" / json_file.name
    with final_path.open("w") as f:
        json.dump(final_data, f, indent=4)
