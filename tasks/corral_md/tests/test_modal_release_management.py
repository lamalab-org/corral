from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    app_dir = Path(__file__).resolve().parents[1] / "modal_app"
    monkeypatch.syspath_prepend(str(app_dir))
    loaded = []
    for name in ("setup_simagent", "manage_runs"):
        spec = importlib.util.spec_from_file_location(f"md_{name}_test", app_dir / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


class _Volume:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    def batch_upload(self, *, force=False):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def put_file(self, source, path):
        self.files[path] = source.read()

    def listdir(self, path):
        return [SimpleNamespace(path=f"{path}/{name}") for name in ("active", "old", "recent")]

    def remove_file(self, path, *, recursive=False):
        assert recursive
        self.deleted.append(path)


def test_build_identity_changes_with_sources_assets_and_volume(modules, tmp_path, monkeypatch):
    release, _ = modules
    source = tmp_path / "worker.py"
    source.write_text("worker v1")
    seed = tmp_path / "base.json"
    seed.write_text(json.dumps({"schema": 1, "directories": ["input", "output"]}))
    manifest_path = tmp_path / "modal_app/assets.json"
    manifest_path.parent.mkdir()
    asset_manifest = {"archives": {"potentials": {"sha256": "potentials-v1"}}, "models": {"student.model": {"sha256": "model-v1"}}}
    manifest_path.write_text(json.dumps(asset_manifest))
    monkeypatch.setattr(release, "TASK_ROOT", tmp_path)
    monkeypatch.setattr(release, "SOURCES", (source, seed, manifest_path))
    first = release.release_id()
    assert len(first) == 24
    source.write_text("worker v2")
    assert release.release_id() != first
    source.write_text("worker v1")
    manifest_path.write_text(json.dumps({**asset_manifest, "models": {"student.model": {"sha256": "model-v2"}}}))
    assert release.release_id() != first
    manifest_path.write_text(json.dumps(asset_manifest))
    monkeypatch.setenv("CORRAL_MD_MODAL_VOLUME", "other-simulations")
    assert release.release_id() != first


def test_simagent_setup_deploys_without_release_environment(modules, monkeypatch):
    release, _ = modules
    calls = []
    monkeypatch.setattr(release, "release_id", lambda: "build-1")
    monkeypatch.setattr(release, "prepare_assets", lambda source: calls.append(("prepare", source)))
    monkeypatch.setattr(release, "upload_assets", lambda source: calls.append(("upload", source)))
    monkeypatch.setattr(release.modal.Volume, "from_name", lambda *_args, **_kwargs: _Volume())
    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs: calls.append(("deploy", args, kwargs)))
    monkeypatch.setattr(release, "smoke_test", lambda *args: calls.append(("smoke", args)))
    monkeypatch.setattr(release, "smoke_verification", lambda *args: {"status": "passed"})
    monkeypatch.setattr(release, "publish_ground_truth", lambda *args: {"status": "passed"})

    assert release.deploy() == "build-1"
    deployment = next(call for call in calls if call[0] == "deploy")
    assert deployment[1][0][-4:] == [
        "deploy", "--strategy", "recreate", "modal_app/lammps_app.py"
    ]
    assert "env" not in deployment[2]
    assert any(call[0] == "smoke" for call in calls)


def test_setup_waits_for_active_simagent_build(modules, monkeypatch):
    release, _ = modules
    monkeypatch.delenv("CORRAL_MD_MODAL_VOLUME", raising=False)
    assets = json.loads((release.TASK_ROOT / "modal_app/assets.json").read_text())
    expected = {
        "schema": 1,
        "app_name": "simagent",
        "release_id": "build-1",
        "volume_name": "simulations",
        "asset_volumes": release.asset_volume_names(assets),
    }
    responses = iter([{**expected, "release_id": "previous-build"}, expected])
    monkeypatch.setattr(
        release.modal.Function,
        "from_name",
        lambda *_args: SimpleNamespace(remote=lambda: next(responses)),
    )
    sleeps = []
    monkeypatch.setattr(release.time, "sleep", sleeps.append)

    release.wait_for_active_app("build-1")
    assert sleeps == [5]


def test_closed_runs_are_retained_for_30_days(modules):
    _, manager = modules
    volume = _Volume()
    now = datetime.now(timezone.utc)
    for run_id, closed_at in (
        ("active", None),
        ("old", now - timedelta(days=31)),
        ("recent", now - timedelta(days=2)),
    ):
        state = {"schema": 1, "release_id": "release-1", "status": "running"}
        if closed_at:
            state.update(status="completed", closed_at=closed_at.isoformat())
        volume.files[f"/corral/runs/{run_id}/run.json"] = json.dumps(state).encode()
    assert manager.prune_closed_runs(volume) == ["old"]
    assert volume.deleted == []
    assert manager.prune_closed_runs(volume, execute=True) == ["old"]
    assert volume.deleted == ["/corral/runs/old"]
    manager.close_run(volume, "active", "cancelled")
    state = json.loads(volume.files["/corral/runs/active/run.json"])
    assert state["status"] == "cancelled"
    assert state["closed_at"]
    assert "active" not in manager.prune_closed_runs(volume)
