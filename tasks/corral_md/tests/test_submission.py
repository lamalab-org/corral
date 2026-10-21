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


@pytest.mark.parametrize(
    "answer",
    [
        "`output/manifest.json`",
        "Completed and validated submission: `output/manifest.json`",
        "Manifest: `/workspace/output/manifest.json`.\nMeasured value: `1.7e-8 m2/s`.",
        "`output/manifest.json`\nSubmission: `output/manifest.json`",
        '"`output/manifest.json`"',
        "/workspace/output/manifest.json\n\nCompleted the thermal cycle.",
        "/workspace/output/manifest.json\nArtifacts linked from /workspace/output/manifest.json.",
        "Completed manifest:\n\n/workspace/output/manifest.json\n\nThermal cycle complete.",
        "Thermal cycle complete.\nSubmission manifest path: /workspace/output/manifest.json",
        "Deliverables: manifest.json and settings.json.\nManifest path: /workspace/output/manifest.json",
        "Completed `/workspace/output/manifest.json`; manifest.json links the evidence.",
        "Submission manifest: /workspace/output/manifest.json (validated). It links the retained data.",
        "Manifest path: /workspace/output/manifest.json. Thermal cycle complete.",
    ],
)
def test_formatted_manifest_resolves_to_identical_evidence(tmp_path, answer):
    from corral_md.workflow_scoring.common import Evidence

    output = tmp_path / "output"
    output.mkdir()
    (output / "data.json").write_text("[1, 2, 3]")
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {"data": "data.json"}}))
    original = manifest.read_bytes()
    plain = Evidence(resolve_submission("output/manifest.json", tmp_path))
    formatted = Evidence(resolve_submission(answer, tmp_path))
    assert formatted.fingerprint() == plain.fingerprint()
    assert formatted.json("data") == [1, 2, 3]
    assert manifest.read_bytes() == original


@pytest.mark.parametrize(
    "answer",
    [
        "Use `first/manifest.json` or `second/manifest.json`",
        "first/manifest.json or `second/manifest.json`",
        "first/manifest.json\nUse second/manifest.json instead.",
        "first/manifest.json\nUse `second/manifest.json` instead.",
        "/workspace/first/manifest.json\nUse `/workspace/second/manifest.json` instead.",
        "/workspace/first.json\nUse `/workspace/second/manifest.json` instead.",
        "Manifest path: first/manifest.json\nManifest path: second/manifest.json",
        "Manifest path: second/manifest.json\nAlternative: `manifest.json`",
        "Submission manifest: first/manifest.json (validated); alternative: second/manifest.json",
    ],
)
def test_formatted_manifest_ambiguity_is_not_resolved_by_existence(tmp_path, answer):
    second = tmp_path / "second"
    second.mkdir()
    (second / "manifest.json").write_text('{"results": {"value": 1}}')
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_submission(answer, tmp_path)


def test_formatted_missing_manifest_does_not_select_other_manifest(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "manifest.json").write_text('{"results": {"value": 1}}')
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_submission("Completed: `missing/manifest.json`", tmp_path)


def test_labeled_manifest_does_not_select_filename_prefix(tmp_path):
    (tmp_path / "manifest.json").write_text('{"results": {"value": 1}}')
    answer = "Manifest path: manifest.json.backup"
    with pytest.raises(FileNotFoundError):
        resolve_submission(answer, tmp_path)


@pytest.mark.parametrize("reference", ["../manifest.json", "linked/manifest.json"])
def test_formatted_manifest_cannot_escape_workspace(tmp_path, reference):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "manifest.json").write_text('{"results": {"value": 1}}')
    (workspace / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="workspace"):
        resolve_submission(f"Completed: `{reference}`", workspace)
    with pytest.raises(ValueError, match="workspace"):
        resolve_submission(f"{reference}\nCompleted the thermal cycle.", workspace)
    with pytest.raises(ValueError, match="workspace"):
        resolve_submission(f"Manifest: {reference} (validated)", workspace)


def test_manifest_prose_inside_inline_json_is_preserved(tmp_path):
    answer = {"results": {"note": "See `output/manifest.json`"}, "artifacts": {}}
    assert json.loads(resolve_submission(json.dumps(answer), tmp_path)) == answer


@pytest.mark.parametrize("filename", ["manifest.json", "results.json", "my results.json"])
def test_explicit_manifest_first_line_allows_trailing_summary(tmp_path, filename):
    output = tmp_path / "output"
    output.mkdir()
    manifest = output / filename
    manifest.write_text('{"artifacts": {}, "results": {"value": 1}}')
    original = manifest.read_bytes()
    resolved = resolve_submission(
        f"/workspace/output/{filename}\n\nCalculation summary and caveats.", tmp_path
    )
    assert json.loads(resolved)["results"] == {"value": 1}
    assert manifest.read_bytes() == original
    manifest.unlink()
    with pytest.raises(FileNotFoundError):
        resolve_submission(f"/workspace/output/{filename}\n\nSummary.", tmp_path)


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


def test_manifest_can_link_sibling_workspace_files(tmp_path):
    workspace = tmp_path / "workspace"
    output = workspace / "output"
    scripts = workspace / "scripts"
    output.mkdir(parents=True)
    scripts.mkdir()
    script = scripts / "run.py"
    script.write_text("print('ok')")
    (output / "manifest.json").write_text(
        json.dumps({"scripts": {"run": "../scripts/run.py"}})
    )

    resolved = json.loads(
        resolve_submission("/workspace/output/manifest.json", workspace)
    )

    assert resolved["scripts"]["run"] == str(script)


def test_manifest_sibling_path_cannot_escape_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    output = workspace / "output"
    output.mkdir(parents=True)
    (tmp_path / "secret.csv").write_text("secret")
    (output / "manifest.json").write_text(
        json.dumps({"artifacts": {"secret": "../../secret.csv"}})
    )

    with pytest.raises(ValueError, match="workspace"):
        resolve_submission("/workspace/output/manifest.json", workspace)


def test_manifest_sibling_symlink_is_rejected(tmp_path):
    workspace = tmp_path / "workspace"
    output = workspace / "output"
    scripts = workspace / "scripts"
    output.mkdir(parents=True)
    scripts.mkdir()
    secret = tmp_path / "secret.py"
    secret.write_text("secret")
    (scripts / "run.py").symlink_to(secret)
    (output / "manifest.json").write_text(
        json.dumps({"scripts": {"run": "../scripts/run.py"}})
    )

    with pytest.raises(ValueError, match="workspace"):
        resolve_submission("/workspace/output/manifest.json", workspace)


def test_inline_report_prose_is_metadata_and_part_of_fingerprint(tmp_path):
    from corral_md.workflow_scoring.common import Evidence

    content = {
        "artifacts": {},
        "report": {"interpretation": "Scientific explanation. " * 100},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(content))
    resolved = resolve_submission("/workspace/manifest.json", tmp_path)
    evidence = Evidence(resolved)

    assert json.loads(resolved)["report"] == content["report"]
    assert evidence.report == content["report"]
    original = evidence.fingerprint()
    content["report"]["interpretation"] += " Updated conclusion."
    manifest.write_text(json.dumps(content))
    changed = Evidence(resolve_submission("/workspace/manifest.json", tmp_path))
    assert changed.fingerprint() != original
