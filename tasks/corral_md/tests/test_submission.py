import json
from pathlib import Path

import pytest
from corral_md.submission import resolve_local_submission, resolve_submission
from corral_md.scoring.common import Evidence


@pytest.mark.parametrize("resolver", [resolve_submission, resolve_local_submission])
@pytest.mark.parametrize(
    "wrapper",
    ["{}", "```json\n{}\n```", "```\n{}\n```", " \n``` JSON \r\n{}\r\n  ```\t\n"],
)
def test_json_resolves_artifacts_from_workspace(tmp_path, resolver, wrapper):
    output = tmp_path / "output"
    output.mkdir()
    (output / "data.json").write_text("[1, 2, 3]")
    answer = {
        "artifacts": {
            "data": "output/data.json",
            "nested": ["/workspace/output/data.json"],
        },
        "results": {
            "value": 3,
            "unit": "m2/s",
            "note": "See `output/manifest.json`; literal ```json and ``` are preserved.",
        },
    }
    resolved = resolver(wrapper.format(json.dumps(answer)), tmp_path)
    assert json.loads(resolved) == {
        **answer,
        "artifacts": {
            "data": str(output / "data.json"),
            "nested": [str(output / "data.json")],
        },
    }
    evidence = Evidence(resolved)
    plain = Evidence(resolve_submission(json.dumps(answer), tmp_path))
    assert evidence.fingerprint() == plain.fingerprint()
    assert evidence.root == evidence.base == tmp_path
    assert evidence.json("data") == [1, 2, 3]
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```"])
@pytest.mark.parametrize(
    "answer",
    [
        "/workspace/manifest.json",
        "manifest.json",
        '"/workspace/manifest.json"',
        "Completed: `/workspace/manifest.json`",
        "/workspace/manifest.json\nCompleted the thermal cycle.",
        "Manifest path: manifest.json",
        "result.csv",
        "[]",
        "null",
        "42",
        '```json\n{"artifacts": {}}',
        '{"artifacts": {}}\n```',
        '```python\n{"artifacts": {}}\n```',
        '```json\n{}\n```\n```json\n{}\n```',
        'Answer:\n```json\n{}\n```',
        '```json\n{}\n```\nFinished.',
        '{"results": {}}\nFinished.',
        '{"artifacts":{},"artifacts":{}}',
        '{"results":{"value":NaN}}',
        '{"results":{"value":1e999}}',
    ],
)
def test_final_answer_must_be_a_json_object(tmp_path, monkeypatch, answer, wrapper):
    answer = wrapper.format(answer)
    (tmp_path / "manifest.json").write_text('{"artifacts":{},"results":{}}')
    monkeypatch.setattr(
        Path, "read_text", lambda *_a, **_k: pytest.fail("must not read an answer file")
    )
    with pytest.raises(ValueError):
        resolve_submission(answer, tmp_path)
    invalid = json.loads(resolve_local_submission(answer, tmp_path))
    assert invalid["artifacts"] == invalid["results"] == {}
    assert invalid["invalid_submission"]


@pytest.mark.parametrize(
    "link", ["../secret.csv", "link.csv", "output/../../secret.csv"]
)
def test_artifact_paths_cannot_escape_workspace(tmp_path, link):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret.csv").touch()
    (workspace / "link.csv").symlink_to(tmp_path / "secret.csv")
    with pytest.raises(ValueError, match="workspace"):
        resolve_submission(json.dumps({"artifacts": {"data": link}}), workspace)


def test_missing_artifact_does_not_read_from_cwd_or_another_run(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "other").mkdir(parents=True)
    (tmp_path / "data.json").write_text("[99]")
    (workspace / "other/data.json").write_text("[99]")
    monkeypatch.chdir(tmp_path)
    evidence = Evidence(
        resolve_submission('{"artifacts":{"data":"data.json"}}', workspace)
    )
    with pytest.raises(ValueError, match="Missing artifact"):
        evidence.json("data")


def test_external_artifact_is_never_read(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "data.json"
    external.write_text("[99]")
    evidence = Evidence(
        resolve_submission(
            json.dumps({"artifacts": {"data": str(external)}}), workspace
        )
    )
    with pytest.raises(ValueError, match="escapes"):
        evidence.json("data")


def test_inline_report_prose_is_metadata_and_part_of_fingerprint(tmp_path):
    content = {
        "artifacts": {},
        "report": {"interpretation": "Scientific explanation. " * 500},
        "results": {"note": "../not a path"},
    }
    evidence = Evidence(resolve_submission(json.dumps(content), tmp_path))
    assert evidence.report == content["report"]
    original = evidence.fingerprint()
    content["report"]["interpretation"] += " An additional conclusion."
    changed = Evidence(resolve_submission(json.dumps(content), tmp_path))
    assert changed.fingerprint() != original
