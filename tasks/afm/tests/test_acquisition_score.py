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
    elif number in (6, 8):
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
        assert [p["times_per_line"] * p["lines_per_frame"] for p in sequence] == [
            25.6,
            19.2,
            12.8,
        ]
