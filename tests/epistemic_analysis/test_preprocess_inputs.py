"""Report preprocessing uses the current normalizer without benchmark runners."""

import gzip
import importlib
import json
from pathlib import Path

import pytest

from corral.epistemic_analysis.schema import content_hash


@pytest.mark.parametrize("error", [None, "Agent stopped"])
def test_retro_import_preserves_execution_provenance_and_source(
    tmp_path, monkeypatch, error
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts/annotations")
    )
    preprocess = importlib.import_module("preprocess_trace_inputs")
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    monkeypatch.setattr(
        preprocess,
        "task_prompt",
        lambda *_args: (
            "Public task instructions.",
            [{"path": "task.json", "sha256": "task"}],
        ),
    )
    trial = {
        "trial_id": "trial-1",
        "error_message": error,
        "score": "HIDDEN_SCORE",
        "messages": [
            {"role": "developer", "content": "Exact agent instructions.\n"},
            {"role": "user", "content": "Public task instructions."},
            {
                "role": "assistant",
                "content": "Inspect αβ.",
                "agent_run_id": "actor",
                "branch_id": "branch",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "inspect", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "content": "Measured αβ.",
                "name": "inspect",
                "tool_call_id": "call-1",
                "agent_run_id": "actor",
                "branch_id": "branch",
            },
        ],
    }
    directory = tmp_path / "reports/retrosynthesis"
    directory.mkdir(parents=True)
    (directory / "runs.csv").write_text(
        "report_path,agent,model,level\ntrial.json.gz,codex,test,1\n"
    )
    report_path = directory / "trial.json.gz"
    with gzip.open(report_path, "wt") as handle:
        json.dump(
            {"task_results": {"task-1": {"trials": [trial, {"trial_id": "missing"}]}}},
            handle,
        )
    original = report_path.read_bytes()

    job, missing = preprocess.import_retro(tmp_path / "output")

    assert missing["missing_messages"]
    document = json.loads(Path(job["input"]).read_text())
    trace = document["trace"]
    assert trace["source"]["identity"] == "trial-1"
    assert trace["source"]["locator"] == "retrosynthesis/trial.json.gz"
    assert trace["source"]["content_hash"] == content_hash(trial)
    assert trace["completion_status"] == ("failed" if error else "complete")
    assert trace["reconstruction_status"] == "complete"
    assert trace["grouping"]["agent_model"] == "codex__test"
    assert trace["messages"][:2] == trial["messages"][:2]
    assert "Inspect αβ." in trace["messages"][2]["content"]
    assert '"id": "call-1"' in trace["messages"][2]["content"]
    assert trace["message_sources"][3]["tool_call_id"] == "call-1"
    assert trace["message_sources"][3]["branch_id"] == "branch"
    assert trace["message_sources"][3]["agent_run_id"] == "actor"
    assert "HIDDEN_SCORE" not in json.dumps(document)
    assert report_path.read_bytes() == original
