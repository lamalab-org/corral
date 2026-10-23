from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest
from corral_md import env as md_environment
from corral_md import modal_workspace as bridge
from corral_md.env import create_environments

from corral.core.environment import Toolset
from corral.core.resources import ResourceCatalogMismatchError, declare_file_resource
from corral.core.task import TaskDefinition
from corral.core.workspace import WorkspaceState

APP_DIR = Path(__file__).resolve().parents[1] / "modal_app"
spec = importlib.util.spec_from_file_location(
    "md_setup_assets", APP_DIR / "setup_assets.py"
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def test_resource_catalog_preserves_every_pinned_asset():
    from zipfile import ZipFile

    from corral_md.resources import asset_descriptors, asset_relative_paths

    descriptors = asset_descriptors()
    paths = asset_relative_paths()
    assert len(paths) == len(set(paths.values())) == len(descriptors)
    pinned = {}
    for archive in assets.MANIFEST["archives"].values():
        with ZipFile(APP_DIR.parent / archive["file"]) as bundle:
            for entry in bundle.infolist():
                if entry.is_dir() or entry.filename.startswith("__MACOSX/"):
                    continue
                data = bundle.read(entry)
                pinned[entry.filename] = (hashlib.sha256(data).hexdigest(), len(data))
    for name, metadata in assets.MANIFEST["models"].items():
        pinned["models/" + name] = (metadata["sha256"], None)
    assert set(paths.values()) == set(pinned)
    for name, relative in paths.items():
        descriptor = descriptors[name]
        digest, size = pinned[relative]
        assert descriptor.sha256 == digest
        assert size is None or descriptor.size == size
        assert descriptor.public_path == f"/workspace/resources/{name}/{Path(relative).name}"


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
    assert not (tmp_path / "structures/melt/liq2600.dat").exists()
    assert not (tmp_path / "structures/melt/liq2600.restart").exists()
    assert not (tmp_path / "structures/melt/liq2600.preparation.json").exists()
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


def test_task2_original_melt_matches_the_scoring_reference():
    from zipfile import ZipFile

    from corral_md.scoring import task_2 as score_task_2

    with ZipFile(APP_DIR.parent / "structures.zip") as archive:
        data = archive.read("structures/melt/liq4000.dat")
    assert hashlib.sha256(data).hexdigest() == score_task_2.SOURCE_SHA256
    assert score_task_2.COOLING_STEPS == 370000
    assert score_task_2.TOTAL_STEPS == 470000


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("local", [True, False])
def test_task2_agent_can_read_the_bundled_melt_before_simulation(
    tmp_path, md_assets, monkeypatch, level, local
):
    from zipfile import ZipFile

    from corral_md.resources import asset_descriptors, asset_resource_name

    from corral.core.state import EnvironmentState, ExecutionState, TaskState
    from corral.runtime.tool_execution import ToolExecutor

    relative = "structures/melt/liq4000.dat"
    name = asset_resource_name(relative)
    descriptor = asset_descriptors()[name]
    with ZipFile(APP_DIR.parent / "structures.zip") as archive:
        original = archive.read(relative)
    # Keep unrelated assets small, but exercise the real melt and pinned catalog.
    (md_assets / relative).write_bytes(original)
    descriptors = {**md_environment.asset_descriptors(), name: descriptor}
    monkeypatch.setattr(md_environment, "asset_descriptors", lambda: descriptors)

    environment = create_environments(
        work_dir=str(tmp_path / "work"), level=level, local=local,
        assets_dir=str(md_assets),
    )[f"level_{level}_task_2"].for_task("melt-availability")
    started = environment.initial_event(execution_id="melt-availability")
    state = ExecutionState(
        through_commit_hash="0" * 64,
        execution_id="melt-availability",
        branch_id="main",
        task=TaskState(environment=started.environment_metadata),
        environment=EnvironmentState(values=started.environment),
        workspace=started.workspace,
    )
    public = environment.current_task.initial_input["path to molten state"]
    assert public == descriptor.public_path
    assert public in started.task["prompt"]
    executor = ToolExecutor(environment)
    content = executor.execute(
        state, environment.tools["read_file"], {"path": public},
        action_id="read-melt",
    )
    assert content.encode("utf-8") == original
    info = json.loads(executor.execute(
        state, environment.tools["file_info"], {"path": public},
        action_id="inspect-melt",
    ))
    assert info["size"] == len(original)
    assert info["read_only"] is True

    prepared = executor.prepare(
        state, environment.tools["run_lammps"],
        {"input_file": "/workspace/input/quench.in"}, action_id="quench",
    )
    source, mounted_path = prepared.resource_mounts[name]
    assert mounted_path == public
    assert Path(source).read_bytes() == original


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


@pytest.mark.usefixtures("md_assets")
def test_prompts_use_provisioned_checkpoint_names(tmp_path):
    environments = create_environments(work_dir=str(tmp_path), level=2)
    manifest = json.loads((APP_DIR / "assets.json").read_text())
    from corral_md.resources import asset_public_path

    model_inputs = {
        key: asset_public_path("models/" + filename)
        for key, filename in {
            "teacher_model": "teacher.model",
            "student_model": "student.model",
        }.items()
    }
    assert set(manifest["models"]) == {
        Path(path).name for path in model_inputs.values()
    }
    for task_id in ("level_2_task_3", "level_2_task_8"):
        bound = environments[task_id].for_task(task_execution_id="asset-check")
        prompt = bound.initial_event(execution_id="asset-check").task["prompt"]
        expected = model_inputs if task_id.endswith("_3") else {"teacher_model": model_inputs["teacher_model"]}
        assert bound.current_task.initial_input.items() >= expected.items()
        for key, name in expected.items():
            assert f"- {key}: {name}" in prompt
        assert "/workspace/resources" in prompt
        assert "/workspace/models" not in prompt


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


@pytest.fixture
def remote_resource_environment(tmp_path, monkeypatch):
    payload = b"pinned potential bytes"
    source = tmp_path / "Si.sw"
    source.write_bytes(payload)
    descriptor = declare_file_resource(
        "md-potentials-SW-Si", source, runtime_version="test-v1"
    )
    monkeypatch.setattr(
        md_environment,
        "asset_relative_paths",
        lambda: {descriptor.name: "potentials/SW/Si.sw"},
    )
    monkeypatch.setattr(
        md_environment, "configured_asset_volume_name", lambda _kind: "potentials-v1"
    )
    downloads = []
    remote_bytes = [payload]

    class Volume:
        def read_file(self, path):
            downloads.append(path)
            yield remote_bytes[0]

    monkeypatch.setattr(modal.Volume, "from_name", lambda _name: Volume())

    def create():
        return md_environment.MolecularDynamicsEnvironment(
            "remote-resources",
            TaskDefinition(
                name="remote-resources",
                description="Read a pinned potential.",
                tools=[],
                scoring_fn=lambda _answer: 1.0,
                submission_format={},
                resolve_answer=False,
            ),
            base_work_dir=str(tmp_path),
            workspace_path=str(tmp_path / "workspace"),
            toolset=Toolset(workspace_factory=None),
            file_resources={descriptor.name: descriptor},
        )

    (tmp_path / "workspace").mkdir()
    return create, descriptor, payload, downloads, remote_bytes


def test_remote_resources_download_once_and_stay_outside_snapshots(
    remote_resource_environment,
):
    create, descriptor, payload, downloads, _ = remote_resource_environment
    environment = create()
    assert downloads == []

    handles = environment.materialize_file_resources((descriptor.name,))
    assert downloads == ["/SW/Si.sw"]
    assert Path(handles[descriptor.name].controller_path).read_bytes() == payload
    assert handles[descriptor.name].public_path == descriptor.public_path
    environment.ensure_file_resources_durable()
    # Another runtime with no source bindings reuses the durable blob cache.
    restored = create()
    assert restored.file_resource_sources == {}
    restored_handles = restored.materialize_file_resources((descriptor.name,))
    assert (
        Path(restored_handles[descriptor.name].controller_path).read_bytes() == payload
    )
    assert downloads == ["/SW/Si.sw"]

    workspace = Path(environment.workspace_path)
    (workspace / "result.txt").write_text("mutable output")
    snapshot = environment.capture_workspace(WorkspaceState(id="resource-snapshot"))
    assert set(snapshot.files) == {"result.txt"}
    assert not (workspace / "resources").exists()


def test_remote_resource_hash_mismatch_is_not_cached(remote_resource_environment):
    create, descriptor, payload, downloads, remote_bytes = remote_resource_environment
    remote_bytes[0] = b"x" + payload[1:]
    environment = create()

    with pytest.raises(ResourceCatalogMismatchError, match="pinned descriptor"):
        environment.materialize_file_resources((descriptor.name,))
    assert environment.file_resource_sources == {}
    assert not list(
        (Path(environment.base_work_dir) / ".corral" / "md-assets").rglob(
            descriptor.filename
        )
    )

    remote_bytes[0] = payload
    handles = environment.materialize_file_resources((descriptor.name,))
    assert Path(handles[descriptor.name].controller_path).read_bytes() == payload
    assert downloads == ["/SW/Si.sw", "/SW/Si.sw"]
