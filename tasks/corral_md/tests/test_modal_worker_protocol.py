from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def worker(tmp_path: Path, monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / "modal_app/lammps_app.py"
    spec = importlib.util.spec_from_file_location("md_worker_protocol_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Volume:
        def __init__(self):
            self.commits = 0

        def reload(self):
            pass

        def commit(self):
            self.commits += 1

    volume = Volume()
    monkeypatch.setattr(module, "volume_sim", volume)
    monkeypatch.setattr(module, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(module, "RELEASES", tmp_path / "releases")
    monkeypatch.setattr(module, "RELEASE_ID", "release-1")
    monkeypatch.setattr(module.modal, "current_function_call_id", lambda: "fc-1")
    base = tmp_path / "releases/release-1/base.json"
    base.parent.mkdir(parents=True)
    base.write_text(json.dumps({
        "schema": 1, "release_id": "release-1",
        "directories": ["input", "output"],
    }))
    return module, volume


def test_worker_commits_one_action_result_and_restores_head(worker) -> None:
    module, volume = worker
    state = module._prepare("run-1", "release-1")
    assert state["head"] == "base"
    working = module.RUNS / "run-1/workspace"
    assert (working / "input").is_dir()
    assert (working / "output").is_dir()
    (working / "input/run.in").write_text("run 0\n")
    called = []

    def simulate(input_path, log_name, *_args):
        called.append(input_path)
        (input_path.parent / log_name).write_text("Step Temp\n")

    module._run_lammps = simulate
    expected = module._manifest(working)
    result = module._execute(
        "lammps", "run-1", "action-1", "input/run.in", "/local", [],
        "release-1", "base", expected,
    )
    assert result["revision"] == 1
    assert result["files"]["input/run.log"]["size"] > 0
    head = f"attempts/action-1/{result['attempt_id']}/workspace"
    assert result["workspace"].endswith(head)
    state = json.loads((module.RUNS / "run-1/run.json").read_text())
    assert state["head"] == head
    assert state["revision"] == 1
    assert volume.commits >= 3

    # A Modal retry sees the action manifest and must not run LAMMPS again.
    assert module._execute(
        "lammps", "run-1", "action-1", "input/run.in", "/local", [],
        "release-1", "base", expected,
    ) == result
    assert len(called) == 1

    (working / "stale.dat").write_text("partial upload")
    module._prepare("run-1", "release-1")
    assert not (working / "stale.dat").exists()
    assert (working / "input/run.log").is_file()


def test_worker_failure_keeps_attempt_and_does_not_advance_head(worker) -> None:
    module, _ = worker
    module._prepare("run-1", "release-1")
    working = module.RUNS / "run-1/workspace"
    (working / "run.in").write_text("bad input")

    def fail(input_path, *_args):
        (input_path.parent / "diagnostic.log").write_text("failure details")
        raise ValueError("simulation failed")

    module._run_lammps = fail
    expected = module._manifest(working)
    with pytest.raises(ValueError, match="simulation failed"):
        module._execute(
            "lammps", "run-1", "action-1", "run.in", "/local", [],
            "release-1", "base", expected,
        )
    assert json.loads((module.RUNS / "run-1/run.json").read_text())["head"] == "base"
    failure = json.loads((module.RUNS / "run-1/actions/action-1.failure.json").read_text())
    assert (Path(failure["attempt"]) / "workspace/diagnostic.log").is_file()
    assert (Path(failure["attempt"]) / "failure.json").is_file()
    assert not (module.RUNS / "run-1/actions/action-1.json").exists()


def test_worker_rejects_wrong_release(worker) -> None:
    module, _ = worker
    with pytest.raises(ValueError, match="Worker image does not match"):
        module._prepare("run-1", "release-2")


def test_worker_refuses_incomplete_delta_upload(worker) -> None:
    module, _ = worker
    module._prepare("run-1", "release-1")
    working = module.RUNS / "run-1/workspace"
    (working / "run.in").write_text("run 0\n")
    wrong = {"run.in": {"sha256": "0" * 64, "size": 6}}
    with pytest.raises(ValueError, match="failed its file manifest check"):
        module._execute(
            "lammps", "run-1", "action-1", "run.in", "/local", [],
            "release-1", "base", wrong,
        )
    assert not (module.RUNS / "run-1/actions/action-1.json").exists()


@pytest.mark.parametrize("retry_same_call", [False, True])
def test_interrupted_worker_retry_preserves_diagnostics_and_uses_clean_inputs(worker, monkeypatch, retry_same_call):
    module, _ = worker
    module._prepare("run-1", "release-1")
    working = module.RUNS / "run-1/workspace"
    (working / "run.in").write_text("run 0\n")
    expected = module._manifest(working)
    paths = []

    def simulate(input_path, *_):
        paths.append(input_path.parent)
        if len(paths) == 1:
            (input_path.parent / "partial.dump").write_text("incomplete")
            raise KeyboardInterrupt
        assert not (input_path.parent / "partial.dump").exists()
        (input_path.parent / "run.log").write_text("success")

    module._run_lammps = simulate
    args = ("lammps", "run-1", "action-1", "run.in", "/local", [], "release-1", "base", expected)
    with pytest.raises(KeyboardInterrupt):
        module._execute(*args)
    call_id = "fc-1" if retry_same_call else "fc-2"
    monkeypatch.setattr(module.modal, "current_function_call_id", lambda: call_id)
    result = module._execute(*args)
    assert paths[0] != paths[1]
    assert (paths[0] / "partial.dump").read_text() == "incomplete"
    assert result["call_id"] == call_id
    assert result["revision"] == 1
    assert module._execute(*args) == result
    assert len(paths) == 2


def test_worker_uses_versioned_assets(worker):
    module, _ = worker

    expected = module.asset_volume_names(module.ASSETS)
    assert module.volume_potential.name == expected["potentials"]
    assert module.volume_struct.name == expected["structures"]
    assert module.volume_models.name == expected["models"]


@pytest.mark.parametrize("gpu", [False, True])
def test_sandbox_gets_only_current_task_and_read_only_assets(worker, tmp_path, monkeypatch, gpu):
    from io import StringIO
    from types import SimpleNamespace

    module, _ = worker
    working = tmp_path / "attempt/workspace"
    working.mkdir(parents=True)
    (working / "script.py").write_text("print('run')")
    finished = []
    calls = []

    class TaskVolume:
        def __init__(self):
            self.files = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def batch_upload(self):
            return self

        def put_file(self, source, name):
            self.files[name] = Path(source).read_bytes()

        def iterdir(self, *_, **kwargs):
            assert finished, "Outputs cannot be collected while the sandbox is running"
            return [SimpleNamespace(path=name, type=module.FileEntryType.FILE) for name in self.files]

        def read_file(self, path):
            yield self.files[path]

    volume = TaskVolume()
    monkeypatch.setattr(module.modal.Volume, "ephemeral", lambda: volume)
    for attribute in ("volume_potential", "volume_models", "volume_struct"):
        monkeypatch.setattr(module, attribute, SimpleNamespace(
            read_only=lambda: SimpleNamespace(read_only=True),
        ))

    class Sandbox:
        object_id = "sb-test"
        stdout = StringIO("complete\n")
        stderr = StringIO("")
        returncode = 0

        def wait(self):
            volume.files["/output/result.json"] = b'{"answer":42}'

        def terminate(self, *, wait):
            assert wait
            finished.append(True)

    def create(*command, **options):
        calls.append((command, options))
        return Sandbox()

    monkeypatch.setattr(module.modal.Sandbox, "create", create)
    module._run_sandbox(["python", "/workspace/script.py"], working, gpu=gpu)
    command, options = calls[0]
    assert command[-2:] == ("python", "/workspace/script.py")
    assert options["workdir"] == "/workspace"
    assert options["gpu"] == (module.GPU_TYPE if gpu else None)
    assert options.get("secrets", ()) == ()
    assert set(options["volumes"]) == {
        "/workspace", "/assets/potentials", "/assets/models", "/assets/structures",
    }
    assert options["volumes"]["/workspace"] is volume
    for name in ("potentials", "models", "structures"):
        assert options["volumes"][f"/assets/{name}"].read_only is True
    assert json.loads((working / "output/result.json").read_text()) == {"answer": 42}
    assert (working / "output/execution.stdout.txt").read_text() == "complete\n"
    assert not (working / "sandbox.json").exists()
    assert (working.parent / "sandbox.json").is_file()


@pytest.mark.parametrize("name, entry_type", [
    ("../escape", "FILE"), ("/../escape", "FILE"),
    ("models/teacher.model", "FILE"), ("output/link", "SYMLINK"),
])
def test_sandbox_export_rejects_unsafe_files_without_replacing_attempt(worker, tmp_path, name, entry_type):
    from types import SimpleNamespace

    module, _ = worker
    working = tmp_path / "attempt/workspace"
    working.mkdir(parents=True)
    (working / "input.txt").write_text("original")
    entry = SimpleNamespace(path=name, type=getattr(module.FileEntryType, entry_type))
    volume = SimpleNamespace(iterdir=lambda *args, **kwargs: [entry])
    with pytest.raises(ValueError):
        module._collect_workspace(volume, working)
    assert (working / "input.txt").read_text() == "original"


def test_python_and_lammps_use_stable_absolute_paths_without_rewriting(worker, tmp_path, monkeypatch):
    module, _ = worker
    working = tmp_path / "workspace"
    working.mkdir()
    (working / "scripts").mkdir()
    script = working / "scripts/analyze.py"
    original = "print('/workspace/output/result.json')\n"
    script.write_text(original)
    calls = []
    monkeypatch.setattr(module, "_run_sandbox", lambda *args, **kwargs: calls.append((args, kwargs)))
    module._run_python(script, ["/workspace/input/data.json"], "/local", working)
    assert calls[0][0][0] == ["python", "/workspace/scripts/analyze.py", "/workspace/input/data.json"]
    assert script.read_text() == original
    assert calls[0][1]["gpu"] is True
    input_file = working / "run.in"
    input_file.write_text("log /workspace/custom.log\nrun 0\n")
    module._run_lammps(input_file, "run.log", "/local", working)
    assert calls[1][0][0][-4:] == ["-in", "/workspace/run.in", "-log", "/workspace/output/run.log"]
    assert input_file.read_text() == "log /workspace/custom.log\nrun 0\n"
