import json

import pytest
from corral_md.submission import resolve_submission


def test_json_values_and_manifest_paths_are_resolved(tmp_path):
    workspace = tmp_path / "restored"
    outputs = workspace / "results"
    outputs.mkdir(parents=True)
    (outputs / "300.traj").touch()
    (outputs / "400.traj").touch()
    manifest = outputs / "manifest.json"
    content = {"300": "300.traj", "nested": ["results/400.traj", 2, "3.14"]}
    manifest.write_text(json.dumps(content))
    original = manifest.read_bytes()
    for answer in ("results/manifest.json", str(manifest)):
        resolved = json.loads(resolve_submission(answer, workspace))
        assert resolved == {
            "300": str(outputs / "300.traj"),
            "nested": [str(outputs / "400.traj"), 2, "3.14"],
        }
    assert manifest.read_bytes() == original


def test_ambiguous_basename_is_rejected(tmp_path):
    for directory in ("first", "second"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "result.csv").touch()
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_submission("result.csv", tmp_path)


def test_missing_file_does_not_read_from_cwd(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "result.csv").write_text("outside workspace")
    monkeypatch.chdir(tmp_path)
    for answer in ("result.csv", str(tmp_path / "result.csv")):
        with pytest.raises(FileNotFoundError, match="missing from the workspace"):
            resolve_submission(answer, workspace)


def test_submission_paths_cannot_escape_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "secret.csv"
    secret.touch()
    (workspace / "link.csv").symlink_to(secret)
    for answer in ("../secret.csv", "link.csv", '{"result": "link.csv"}'):
        with pytest.raises(ValueError, match="workspace"):
            resolve_submission(answer, workspace)
