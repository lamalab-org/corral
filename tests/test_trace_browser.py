"""Report browser checks independent of agent SDKs and benchmark execution."""

import csv
import gzip
import importlib.util
import io
import json
import lzma
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "corral_trace_browser",
    Path(__file__).resolve().parents[1] / "reports/serve_traces.py",
)
browser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(browser)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 16, 63, 65536])
def test_stream_handles_escapes_and_structures_across_chunks(chunk_size):
    value = {
        "brackets in strings": '}\\"[🧪é\\\\',
        "nested": [{"text": '\\"'}, None, False, True, -12.5, [], {}],
        "trailing": "\\",
    }
    raw = json.dumps({"skip": value, "keep": value}, ensure_ascii=False)
    stream = browser.JSONStream(io.StringIO(raw), chunk_size)
    found = {}
    for key in stream.object_keys():
        if key == "skip":
            stream.value(capture=False)
        else:
            found[key] = stream.value()
    assert found == {"keep": value}


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [
        (".json.gz", gzip.compress),
        (".json.xz", lzma.compress),
        (".json", lambda value: value),
    ],
)
def test_reads_only_requested_trial_and_preserves_saved_fields(
    tmp_path, suffix, compress
):
    trial = {
        "trial_id": "run:task:1",
        "messages": [{"role": "assistant", "content": 'x\\"🧪'}],
        "custom": {"value": 42},
    }
    report = {
        "metadata": {"nested": ["ignore"]},
        "task_results": {
            "before": {"trials": [{"messages": []}]},
            "task": {"metric": 0.5, "trials": [{"trial_id": "run:task:0"}, trial]},
            "after": {},
        },
    }
    path = tmp_path / f"report{suffix}"
    path.write_bytes(compress(json.dumps(report).encode()))
    assert browser.read_trial(path, "task", 1) == trial
    with pytest.raises(LookupError):
        browser.read_trial(path, "missing", 0)
    with pytest.raises(LookupError):
        browser.read_trial(path, "task", 8)


def write_index(root, report_file="level_1/react__model/report.json.gz"):
    env = root / "environment"
    env.mkdir()
    row = {
        "environment": "environment",
        "level": "1",
        "agent": "react",
        "model": "model",
        "run_id": "run",
        "task_id": "task",
        "trial": "0",
        "execution_id": "run:task:0",
        "score": "",
        "success": "",
        "error": "",
        "duration_s": "1.25",
        "report_file": report_file,
    }
    with (env / "trials.csv").open("w", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    return env / report_file


def test_index_preserves_unscored_outcomes_and_reports_lfs(tmp_path):
    path = write_index(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(browser.LFS_POINTER + b"\noid sha256:test\nsize 10\n")
    traces, _ = browser.load_index(tmp_path)
    assert traces[0]["success"] is None
    assert traces[0]["score"] is None
    assert traces[0]["duration"] == 1.25
    assert traces[0]["availability"] == "lfs"
    with pytest.raises(FileNotFoundError, match="git lfs pull"):
        browser.read_trial(path, "task", 0)


def test_index_rejects_paths_outside_reports(tmp_path):
    write_index(tmp_path, "../../outside.json")
    with pytest.raises(ValueError, match="escapes reports directory"):
        browser.load_index(tmp_path)


def test_cache_invalidates_changed_reports_and_detects_stale_index(tmp_path):
    path = write_index(tmp_path)
    path.parent.mkdir(parents=True)

    def save(trial_id, content):
        path.write_bytes(
            gzip.compress(
                json.dumps(
                    {
                        "task_results": {
                            "task": {
                                "trials": [
                                    {
                                        "trial_id": trial_id,
                                        "messages": [
                                            {"role": "assistant", "content": content}
                                        ],
                                    }
                                ]
                            }
                        }
                    }
                ).encode()
            )
        )

    save("run:task:0", "First")
    store = browser.TraceStore(tmp_path)
    trace_id = store.traces[0]["id"]
    assert (
        json.loads(gzip.decompress(store.trial(trace_id)))["messages"][0]["content"]
        == "First"
    )
    save("run:task:0", "Changed content")
    assert (
        json.loads(gzip.decompress(store.trial(trace_id)))["messages"][0]["content"]
        == "Changed content"
    )
    save("another:task:0", "Wrong trial")
    with pytest.raises(ValueError, match="index and report disagree"):
        store.trial(trace_id)


def test_stream_rejects_incomplete_json():
    with pytest.raises(ValueError, match="Incomplete"):
        browser.JSONStream(io.StringIO('{"value":"unfinished\\'), 3).value()
