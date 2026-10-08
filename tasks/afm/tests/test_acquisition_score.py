"""Verify that the configured three-image sequences match the task descriptions."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    int(p.stem.split("_")[1]): json.loads(p.read_text())[0]
    for p in (ROOT / "environments/level_2/tasks_json").glob("*.json")
}


@pytest.mark.parametrize("number", range(1, 11))
def test_acquisition_sequence(number):
    task = TASKS[number]
    config = task["scoring_params"]
    sequence = config["final_params"]
    assert "acquisition_params" not in config
    assert len(sequence) == 3
    initial = task["initial_input"]["params"]
    assert all(p["pgain"] != initial["pgain"] for p in sequence)
    if number == 1:
        assert [(p["pgain"], p["igain"]) for p in sequence] == [
            (100, 50),
            (150, 50),
            (150, 100),
        ]
    elif number in (2, 3, 7):
        assert [p["setpoint"]["value"] for p in sequence] == {
            2: [0.1, 0.2, 0.5],
            3: [80, 70, 60],
            7: [0.1, 0.2, 0.3],
        }[number]
    elif number == 5:
        assert [p["image_width"] for p in sequence] == [10000, 5000, 2000]
    elif number == 6:
        assert [p["times_per_line"] for p in sequence] == [0.1, 0.0375, 0.025]
    elif number == 8:
        assert [p["times_per_line"] for p in sequence] == [0.1, 0.075, 0.05]
    elif number == 10:
        assert [(p["centre_x"], p["centre_y"]) for p in sequence] == [
            (0, 0),
            (3000, 0),
            (0, 3000),
        ]
    else:
        assert sequence[0] == sequence[1] == sequence[2]
    if number == 6:
        assert config["percent_change_reference"] == 2
        assert [
            2 * p["times_per_line"] * p["lines_per_frame"] for p in sequence
        ] == [51.2, 19.2, 12.8]
        assert "scan times of 51.2, 19.2, and 12.8 s" in task["description"]
        assert "one trace and one retrace per line" in task["description"]


def test_spatial_task_requires_roughness_and_friction_at_each_location():
    task = TASKS[10]
    config = task["scoring_params"]
    assert task["scoring_function"] == "score_roughness_and_friction"
    assert config["metrics"] == [
        "rms_roughness",
        "mean_roughness",
        "average_friction",
    ]
    assert config["friction_absolute"] is True
    assert "roughness from the height channel" in task["description"]
    assert "friction magnitude" in task["description"]
    for i in range(1, 4):
        for metric in config["metrics"]:
            assert f'"{metric}_{i}"' in task["submission_format"]
