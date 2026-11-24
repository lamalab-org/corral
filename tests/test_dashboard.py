"""Keep the published full-benchmark dashboard separate from mini results."""

import csv
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "corral_dashboard",
    Path(__file__).resolve().parents[1] / "reports/dashboard/build_dashboard.py",
)
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def write_trial(path, run_id, model="gpt-5.6-sol"):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "environment": "psychometrics",
        "level": "1",
        "task_id": "full_task",
        "trial": "0",
        "agent": "codex",
        "model": model,
        "success": "True",
        "error": "",
        "duration_s": "1",
        "input_tokens": "100",
        "output_tokens": "10",
        "tool_calls": "1",
        "run_id": run_id,
    }
    with path.open("w", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_full_collection_ignores_mini_versions(tmp_path):
    write_trial(
        tmp_path / "psychometrics/mini-v1/trials.csv",
        "psychometrics-mini-v1",
        "gpt-6.1-sol",
    )
    write_trial(
        tmp_path / "psychometrics/mini-v2/trials.csv",
        "psychometrics-mini-v2",
        "gpt-6.1-sol",
    )
    write_trial(tmp_path / "psychometrics/full/trials.csv", "psychometrics-full")

    rows = dashboard.read_trials(tmp_path, set())
    assert len(rows) == 1
    assert rows[0][dashboard.TRIAL_COLUMNS.index("model")] == "gpt-5.6-sol"
    assert dashboard.read_trials(tmp_path, {"psychometrics"}) == []

    # Run the dashboard calculations that failed on the deployed mini payload.
    if not shutil.which("node"):
        pytest.skip("Node.js is needed to verify dashboard calculations")
    data = {
        "trials": [
            dict(zip(dashboard.TRIAL_COLUMNS, row, strict=False)) for row in rows
        ],
        "prices": [
            {
                "model": "gpt-5.6-sol",
                "input_usd_per_m": 4,
                "output_usd_per_m": 20,
                "cache_read_usd_per_m": 0.4,
            }
        ],
    }
    script = (
        "const data = " + json.dumps(data) + ";\n"
        "const calcs = {}; const dash = {calc: (id, calc) => calcs[id] = calc};\n"
        + (dashboard.HERE / "calcs.js").read_text()
        + "\nconst get = id => data[id] ?? calcs[id].fn(...calcs[id].inputs.map(get));\n"
        "if (!get('board').length || !get('grid').length) process.exit(1);\n"
    )
    subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)


def test_mini_trials_in_full_collection_are_rejected(tmp_path):
    write_trial(tmp_path / "psychometrics/full/trials.csv", "psychometrics-mini-v1")
    with pytest.raises(ValueError, match="full-run dashboard"):
        dashboard.read_trials(tmp_path, set())


def test_full_collection_works(tmp_path):
    write_trial(tmp_path / "psychometrics/full/trials.csv", "psychometrics-full")
    assert len(dashboard.read_trials(tmp_path, set())) == 1
