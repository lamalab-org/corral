import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from corral.tools.python_repl import PythonREPLTool

from corral_psychometrics import paths
from corral_psychometrics.env import load_tasks_from_json
from corral_psychometrics.generators.common import write_json
from corral_psychometrics.tools import workspace_tools

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("psychometrics_build", SOURCE / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def copy_task(root):
    task = paths.find(1, 1, root=root)
    source = task.at(SOURCE)
    shutil.copytree(source.artifacts, task.artifacts)
    for src, dst in ((source.truth, task.truth), (source.definition, task.definition)):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return task


def test_root_selection_accepts_new_directory(tmp_path, monkeypatch):
    root = tmp_path / "new"
    monkeypatch.setenv(paths.ROOT_VARIABLE, str(root))
    assert paths.task_root() == root
    assert paths.task_root(tmp_path / "explicit") == tmp_path / "explicit"
    with pytest.raises(FileNotFoundError):
        load_tasks_from_json(root / "environments", data_root=root)


@pytest.mark.parametrize(
    "field,value", [("data_sha256", "new"), ("seed", 2), ("versions", {"numpy": "new"})]
)
def test_truth_updates_meaningful_provenance(tmp_path, field, value):
    path = tmp_path / "truth.json"
    payload = {"scored": {"answer": 1}, "provenance": {"data_sha256": "old", "seed": 1}}
    write_json(path, payload)
    payload["provenance"][field] = value
    assert write_json(path, payload)
    assert json.loads(path.read_text()) == payload
    payload["provenance"].update(generated="tomorrow", git_rev="another")
    assert not write_json(path, payload)


def test_corrupt_data_rejected(tmp_path):
    task = copy_task(tmp_path)
    assert len(load_tasks_from_json(task.definition.parent, data_root=tmp_path)) == 1
    with (task.artifacts / "data.csv").open("a") as handle:
        handle.write("corrupted\n")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_tasks_from_json(task.definition.parent, data_root=tmp_path)


def test_failed_generator_preserves_previous_files(tmp_path, monkeypatch):
    task = copy_task(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def fail(*args, **kwargs):
        assert Path(kwargs["env"][paths.ROOT_VARIABLE]) != tmp_path
        return subprocess.CompletedProcess(args[0], 1, "", "simulated failure")

    monkeypatch.setattr(builder.subprocess, "run", fail)
    assert not builder.run(task)[0]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_publish_replaces_only_selected_task_and_checks_all_files(tmp_path):
    target = copy_task(tmp_path / "target")
    (target.artifacts / "obsolete.csv").write_text("obsolete")
    unrelated = target.artifacts.parent / "task_02"
    unrelated.mkdir()
    (unrelated / "keep").write_text("keep")
    staged = copy_task(tmp_path / "staged")
    builder.publish(staged, target)
    assert not (target.artifacts / "obsolete.csv").exists()
    assert (unrelated / "keep").read_text() == "keep"
    load_tasks_from_json(target.definition.parent, data_root=target.root)
    (target.artifacts / "codebook.md").write_text("corrupted")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_tasks_from_json(target.definition.parent, data_root=target.root)


def test_interrupted_publication_is_rejected(tmp_path):
    task = copy_task(tmp_path)
    task.definition.with_suffix(".building").touch()
    with pytest.raises(RuntimeError, match="unfinished task publication"):
        load_tasks_from_json(task.definition.parent, data_root=tmp_path)


def test_local_repl_uses_workspace_without_changing_parent(tmp_path):
    cwd = Path.cwd()
    (tmp_path / "data.csv").write_text("workspace data")
    repl = workspace_tools(str(tmp_path))["PythonREPL"]

    def run(code, checkpoint):
        session = repl.create_session()
        try:
            session.restore(checkpoint)
            return session.execute(code), session.snapshot()
        finally:
            session.close()

    output, checkpoint = run("value = open('data.csv').read(); value", None)
    assert "workspace data" in output
    resumed, _ = run("value", checkpoint)
    assert "workspace data" in resumed
    assert Path.cwd() == cwd


def test_repl_is_corrals_trusted_tool(tmp_path):
    repl = workspace_tools(str(tmp_path))["PythonREPL"]
    assert isinstance(repl, PythonREPLTool)
    # Docker dispatches trusted tools in the controller, which owns the checkpoint.
    assert repl.trusted


def test_scoring_checks_use_source_tests_and_selected_external_data(tmp_path, monkeypatch):
    def run(command, **kwargs):
        assert command[1] == str(SOURCE / "tests" / "test_scoring.py")
        assert command[2:] == ["--level", "2", "--tasks", "3"]
        assert kwargs["env"][paths.ROOT_VARIABLE] == str(tmp_path)
        return subprocess.CompletedProcess(command, 0, "passed", "")

    monkeypatch.setattr(builder.subprocess, "run", run)
    assert builder.scoring_tests(tmp_path, 2, [3]) == []


def test_paths_reject_symlinks_outside_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (tmp_path / "outside").write_text("private")
    (root / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes"):
        paths.resolve("link", root=root)
