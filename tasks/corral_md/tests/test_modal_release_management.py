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
    for name in ("release", "manage_runs"):
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


def test_release_base_is_immutable_and_source_changes_change_id(modules, tmp_path, monkeypatch):
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
    monkeypatch.setattr(release, "SEED", seed)
    monkeypatch.setattr(release, "SOURCES", (source, seed, manifest_path))
    first = release.release_id()
    volume = _Volume()
    release.publish_base(volume, first)
    release.publish_base(volume, first)
    assert len(volume.files) == 1
    published = json.loads(volume.files[f"/corral/releases/{first}/base.json"])
    assert published["asset_volumes"] == release.asset_volume_names(asset_manifest)
    source.write_text("worker v2")
    assert release.release_id() != first
    source.write_text("worker v1")
    monkeypatch.setenv("CORRAL_MD_MODAL_VOLUME", "other-simulations")
    assert release.release_id() != first
    source.write_text("worker v2")
    with pytest.raises(ValueError, match="Refusing to replace"):
        release.publish_base(volume, first)


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
