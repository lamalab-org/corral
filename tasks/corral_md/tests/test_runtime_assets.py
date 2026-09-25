from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest
from corral_md import modal_workspace as bridge
from corral_md.env import create_environments

APP_DIR = Path(__file__).resolve().parents[1] / "modal_app"
spec = importlib.util.spec_from_file_location(
    "md_setup_assets", APP_DIR / "setup_assets.py"
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


@pytest.fixture
def small_model(monkeypatch):
    payload = b"test checkpoint bytes"
    manifest = {
        **assets.MANIFEST,
        "models": {
            "test.model": {
                "url": "https://example.invalid/test.model",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    monkeypatch.setattr(assets, "MANIFEST", manifest)
    monkeypatch.setattr(
        assets.urllib.request, "urlopen", lambda *_a, **_kw: io.BytesIO(payload)
    )
    return payload


def test_prepare_shipped_archives_and_cache_verified_models(
    tmp_path, small_model, monkeypatch
):
    assets.prepare_assets(tmp_path)
    assert (tmp_path / "potentials/SW/Si.sw").is_file()
    assert (tmp_path / "potentials/BKS/pot.mod").is_file()
    assert (tmp_path / "structures/melt/liq4000.dat").is_file()
    assert (tmp_path / "structures/melt/liq1300.dat").is_file()
    assert (tmp_path / "structures/cu/cu32.extxyz").is_file()
    assert not (tmp_path / "structures/si/mp-149-conventional.extxyz").exists()
    assert (tmp_path / "models/test.model").read_bytes() == small_model
    assert not (tmp_path / "__MACOSX").exists()

    def no_download(*_args, **_kwargs):
        pytest.fail("Verified cached models must not be downloaded again")

    monkeypatch.setattr(assets.urllib.request, "urlopen", no_download)
    assets.prepare_assets(tmp_path)


@pytest.mark.usefixtures("small_model")
def test_failed_model_verification_does_not_publish_partial_download(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        assets.urllib.request, "urlopen", lambda *_a, **_kw: io.BytesIO(b"wrong")
    )
    with pytest.raises(ValueError, match="Checksum mismatch for downloaded model"):
        assets.prepare_assets(tmp_path)
    assert list((tmp_path / "models").iterdir()) == []


def test_archive_verification_rejects_changed_input(tmp_path, monkeypatch):
    (tmp_path / "potentials.zip").write_bytes(b"changed archive")
    monkeypatch.setattr(assets, "TASK_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        assets.prepare_assets(tmp_path / "output")


@pytest.mark.parametrize(
    "missing_error", [FileNotFoundError, modal.exception.NotFoundError]
)
@pytest.mark.usefixtures("small_model")
def test_upload_is_repeatable_and_preserves_conflicting_assets(
    tmp_path, monkeypatch, missing_error
):
    assets.prepare_assets(tmp_path)
    files = {}
    uploaded = []

    class Volume:
        def __init__(self, name):
            self.name = name

        def read_file(self, remote):
            if (self.name, remote) not in files:
                raise missing_error(remote)
            yield files[self.name, remote]

        def batch_upload(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def put_file(self, local, remote):
            uploaded.append((self.name, remote))
            files[self.name, remote] = Path(local).read_bytes()

    monkeypatch.setattr(modal.Volume, "from_name", lambda name, **_kw: Volume(name))
    names = assets.asset_volume_names(assets.MANIFEST)
    assets.upload_assets(tmp_path)
    assert (names["potentials"], "/SW/Si.sw") in uploaded
    assert (names["models"], "/test.model") in uploaded
    uploaded.clear()
    assets.upload_assets(tmp_path)
    assert uploaded == []

    # Even missing files in an earlier volume must not be uploaded if a later
    # volume conflicts with the selected benchmark inputs.
    del files[names["potentials"], "/SW/Si.sw"]
    files[names["models"], "/test.model"] = b"another benchmark's checkpoint"
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        assets.upload_assets(tmp_path)
    assert uploaded == []
    assert files[names["models"], "/test.model"] == b"another benchmark's checkpoint"

    # A deliberate checkpoint upgrade must coexist with the old version.
    files[names["models"], "/test.model"] = (tmp_path / "models/test.model").read_bytes()
    old_files = dict(files)
    new_model = b"new checkpoint"
    (tmp_path / "models/test.model").write_bytes(new_model)
    new_manifest = json.loads(json.dumps(assets.MANIFEST))
    new_manifest["models"]["test.model"]["sha256"] = hashlib.sha256(new_model).hexdigest()
    monkeypatch.setattr(assets, "MANIFEST", new_manifest)
    new_names = assets.asset_volume_names(new_manifest)
    assert new_names["models"] != names["models"]
    assert new_names["potentials"] == names["potentials"]
    assets.upload_assets(tmp_path)
    assert files[new_names["models"], "/test.model"] == new_model
    assert all(files[key] == value for key, value in old_files.items())


def test_prompts_use_provisioned_checkpoint_names(tmp_path):
    environments = create_environments(work_dir=str(tmp_path), level=2)
    manifest = json.loads((APP_DIR / "assets.json").read_text())
    model_inputs = {"teacher_model": "teacher.model", "student_model": "student.model"}
    assert set(manifest["models"]) == set(model_inputs.values())
    for task_id in ("level_2_task_3", "level_2_task_8"):
        bound = environments[task_id].for_task(task_execution_id="asset-check")
        prompt = bound.initial_event(execution_id="asset-check").task["prompt"]
        assert bound.current_task.initial_input.items() >= model_inputs.items()
        for key, name in model_inputs.items():
            assert f"- {key}: {name}" in prompt
        assert "/workspace/models" in prompt


def test_catalog_uses_deployed_asset_volumes(monkeypatch):
    monkeypatch.delenv("CORRAL_MD_RELEASE_ID", raising=False)
    monkeypatch.setattr(
        bridge.modal.Function,
        "from_name",
        lambda app, function: SimpleNamespace(remote=lambda: {
            "schema": 1, "app_name": app, "release_id": "release-1",
            "volume_name": "internal-simulations",
            "asset_volumes": {
                "models": "corral-md-models-version1",
                "potentials": "corral-md-potentials-version1",
                "structures": "corral-md-structures-version1",
            },
        }) if (app, function) == ("simagent", "runtime_info") else None,
    )
    assert bridge.configured_asset_volume_name("potentials") == "corral-md-potentials-version1"


def test_runtime_configuration_follows_deployed_simagent(monkeypatch):
    monkeypatch.setenv("CORRAL_MD_RELEASE_ID", "stale-release")
    monkeypatch.setattr(
        bridge.modal.Function,
        "from_name",
        lambda app, function: SimpleNamespace(remote=lambda: {
            "schema": 1, "app_name": app, "release_id": "release-1",
            "volume_name": "internal-simulations",
            "asset_volumes": {
                "models": "models-v1", "potentials": "potentials-v1",
                "structures": "structures-v1",
            },
        }) if (app, function) == ("simagent", "runtime_info") else None,
    )
    assert bridge.configured_runtime() == ("release-1", "internal-simulations")


@pytest.mark.parametrize("invalid", [
    None,
    {"schema": 2, "app_name": "simagent", "release_id": "release-1", "volume_name": "simulations"},
    {"schema": 1, "app_name": "other", "release_id": "release-1", "volume_name": "simulations"},
    {"schema": 1, "app_name": "simagent", "release_id": "../bad", "volume_name": "simulations"},
    {"schema": 1, "app_name": "simagent", "release_id": "release-1", "volume_name": ""},
])
def test_runtime_configuration_rejects_invalid_worker_metadata(monkeypatch, invalid):
    monkeypatch.delenv("CORRAL_MD_RELEASE_ID", raising=False)
    monkeypatch.setattr(
        bridge.modal.Function,
        "from_name",
        lambda _app, _function: SimpleNamespace(remote=lambda: invalid),
    )
    with pytest.raises((RuntimeError, ValueError)):
        bridge.configured_runtime()


def test_asset_catalog_follows_execution_pin_after_a_new_release(monkeypatch):
    monkeypatch.setattr(
        bridge.modal.Function,
        "from_name",
        lambda _app, _function: SimpleNamespace(remote=lambda: {
            "schema": 1, "app_name": "simagent", "release_id": "current",
            "volume_name": "simulations",
            "asset_volumes": {
                "models": "models-current", "potentials": "potentials-current",
                "structures": "structures-current",
            },
        }),
    )
    with bridge.pinned_release("old"):
        with pytest.raises(RuntimeError, match="different SimAgent build"):
            bridge.configured_asset_volume_name("models")
    assert bridge.configured_asset_volume_name("models") == "models-current"
