"""Infrastructure scans must work with CI's partial Git LFS checkout."""

import csv
import gzip
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "corral_infra_failures",
    Path(__file__).resolve().parents[1] / "reports/find_infra_failures.py",
)
infra = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(infra)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_lfs_indexes_do_not_abort_other_environments(tmp_path, monkeypatch, capsys):
    for env, pointer in (("corral_md", "trials.csv"), ("inference_opt", "runs.csv")):
        folder = tmp_path / env / "full"
        write_csv(folder / "trials.csv", [{"execution_id": "unused"}])
        write_csv(
            folder / "runs.csv",
            [{"level": "1", "agent": "codex", "model": "gpt-5.6-sol"}],
        )
        (folder / pointer).write_bytes(
            infra.LFS_POINTER + b"\noid sha256:abc\nsize 123\n"
        )
    readable = tmp_path / "wetlab/full"
    write_csv(readable / "trials.csv", [{"execution_id": "readable"}])
    write_csv(
        readable / "runs.csv",
        [{"level": "2", "agent": "codex", "model": "gpt-5.6-sol"}],
    )
    markdown = tmp_path / "findings.md"
    monkeypatch.setattr(infra, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["find_infra_failures.py", "--markdown", str(markdown)]
    )

    assert infra.main() == 0
    assert "wetlab/full: 0 trials" in capsys.readouterr().out
    report = markdown.read_text()
    assert "| corral_md/full | unknown | — |" in report
    assert "corral_md/full/trials.csv" in report
    assert "inference_opt/full/runs.csv" in report


def test_mini_v2_scan_uses_its_own_index_and_expected_pairs(
    tmp_path, monkeypatch, capsys
):
    folder = tmp_path / "afm" / "mini-v2"
    write_csv(
        folder / "runs.csv",
        [
            {"level": str(level), "agent": agent, "model": model}
            for level in (1, 2)
            for agent, model in infra.MINI_V2_RUNS
        ],
    )
    write_csv(
        folder / "trials.csv",
        [
            {
                "environment": "afm",
                "execution_id": "mini:task:0",
                "success": "False",
                "level": "2",
                "agent": "react",
                "model": "kimi-k3.1-pro",
            }
        ],
    )
    report = folder / "level_2/react__kimi-k3.1-pro/report.json.gz"
    report.parent.mkdir(parents=True)
    with gzip.open(report, "wt") as source:
        json.dump(
            {
                "task_results": {
                    "task": {
                        "trials": [
                            {
                                "trial_id": "mini:task:0",
                                "error_message": "Too Many Requests",
                                "messages": [{"content": "request failed"}],
                            }
                        ]
                    }
                }
            },
            source,
        )
    # A full-run index must not be mixed into mini-v2 coverage.
    write_csv(tmp_path / "afm/full/trials.csv", [{"execution_id": "full:task:0"}])
    monkeypatch.setattr(infra, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["find_infra_failures.py", "afm", "--mini-v2"])

    assert infra.main() == 0
    output = capsys.readouterr().out
    assert "afm/mini-v2: 1 trials, 1 failed" in output
    assert "rate_limit" in output
    assert "missing run" not in output
    assert len(infra.MINI_V2_RUNS) == 18

    runs = list(csv.DictReader((folder / "runs.csv").open()))
    write_csv(folder / "runs.csv", runs[:-1])
    assert infra.missing_runs(folder) == ["L2 react / kimi-k2-0905"]

    monkeypatch.setattr("sys.argv", ["find_infra_failures.py", "--report", str(report)])
    assert infra.main() == 0
    output = capsys.readouterr().out
    assert "afm/mini-v2: 1 trials, 1 failed" in output
    assert "missing run" not in output


def test_terminal_provider_errors_from_report_and_harness():
    for category, error in (
        ("credits", "This request would exceed your available credits."),
        ("provider_error", "Provider returned error"),
        ("refusal", "actor provider stopped the response (finish_reason='refusal')"),
        (
            "empty_response",
            "actor returned 3 consecutive responses without public text",
        ),
    ):
        trial = {"error_message": error, "token_usage": {"llm_calls": 3}}
        assert [
            (f["category"], f["confidence"]) for f in infra.trial_findings(trial)
        ] == [(category, "high")]


def test_iteration_limit_with_refusal_and_empty_response():
    trial = {
        "error_message": "Maximum iterations reached",
        "messages": [
            {"role": "assistant", "metadata": {"finish_reason": "refusal"}},
            {"role": "assistant", "content": " "},
        ],
    }
    assert {f["category"]: f["confidence"] for f in infra.trial_findings(trial)} == {
        "agent_limit": "high",
        "refusal": "review",
        "empty_response": "review",
    }

    trial["messages"] = [
        {"role": "user", "name": "empty_response_retry"},
        {"role": "assistant", "tool_calls": [{"id": "call"}]},
    ]
    assert [f["category"] for f in infra.trial_findings(trial)] == ["agent_limit"]

    trial["messages"].pop()
    assert {f["category"] for f in infra.trial_findings(trial)} == {
        "agent_limit",
        "empty_response",
    }


def test_denario_model_request_budget_is_an_agent_limit():
    trial = {
        "error_message": (
            "Denario exhausted the model-request budget (including planning and execution)."
        ),
        "token_usage": {"llm_calls": 5},
    }

    (finding,) = infra.trial_findings(trial)
    assert finding["category"] == "agent_limit"
    assert not infra.is_rerun(finding | {"success": "False"})
    assert not infra.is_review(finding | {"success": "False"})
