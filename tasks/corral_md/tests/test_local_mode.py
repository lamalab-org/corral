"""Default MD runs use local simulation and scoring; Modal requires opt-in."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from corral_md import env, modal_workspace, tools
from corral_md.scoring import evaluator as score
from corral_md.scoring.local_verifier import (
    Level1LocalVerifier,
    LocalCalculationTransport,
    LocalVerifier,
)
from corral_md.workspace import MDWorkspaceFilesystem


def _assets(tmp_path: Path) -> Path:
    root = tmp_path / "assets"
    for kind in ("models", "potentials", "structures"):
        (root / kind).mkdir(parents=True)
    (root / "potentials/SW").mkdir()
    (root / "potentials/SW/Si.sw").write_text("potential")
    return root


def test_local_image_assets_use_private_fallback_and_preserve_explicit_paths(
    tmp_path, monkeypatch
):
    from corral_md import local_runtime

    image_assets = _assets(tmp_path / "image")
    explicit_assets = _assets(tmp_path / "explicit")
    monkeypatch.setattr(
        local_runtime,
        "Path",
        lambda value: image_assets
        if str(value) == "/opt/corral/assets/corral_md"
        else Path(value),
    )
    assert local_runtime.assets_directory(explicit_assets) == explicit_assets
    assert local_runtime.assets_directory(tmp_path / "host-only-assets") == image_assets
    monkeypatch.delenv("CORRAL_MD_ASSETS_DIR", raising=False)
    assert local_runtime.assets_directory() == image_assets


@pytest.mark.parametrize("local", [False, True])
def test_sandbox_rejects_legacy_public_image_assets(tmp_path, monkeypatch, local):
    from corral.runtime import permissions

    legacy_assets = _assets(tmp_path / "image")
    monkeypatch.setattr(permissions, "enabled", lambda: True)
    monkeypatch.setattr(
        env,
        "Path",
        lambda value: legacy_assets
        if str(value) == "/usr/local/share/corral-md-assets"
        else Path(value),
    )
    with pytest.raises(ValueError, match="Rebuild the MD image"):
        env.create_environments(
            work_dir=str(tmp_path / "work"),
            local=local,
            assets_dir=str(_assets(tmp_path / "private")) if local else None,
        )


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("backend_options", [{}, {"local": True}])
def test_local_factory_uses_local_scoring_and_local_assets(
    tmp_path, monkeypatch, level, backend_options
):
    assets = _assets(tmp_path)
    from corral.core.resources import declare_file_resource
    descriptor = declare_file_resource(
        "md-potentials-SW-Si", assets / "potentials/SW/Si.sw", runtime_version="test",
    )
    monkeypatch.setattr(env, "asset_descriptors", lambda: {descriptor.name: descriptor})
    monkeypatch.setattr(env, "asset_relative_paths", lambda: {descriptor.name: "potentials/SW/Si.sw"})

    def unexpected_modal(*_args, **_kwargs):
        pytest.fail("Local runs must not contact Modal")

    monkeypatch.setattr(modal_workspace.modal.Function, "from_name", unexpected_modal)
    monkeypatch.setattr(modal_workspace.modal.Volume, "from_name", unexpected_modal)
    environments = env.create_environments(
        work_dir=str(tmp_path / "work"), level=level,
        assets_dir=str(assets), **backend_options,
    )
    selected = environments[f"level_{level}_task_4"]
    assert selected.local_mode is True
    assert isinstance(
        selected.current_task.scoring_fn.verifier,
        Level1LocalVerifier if level == 1 else LocalVerifier,
    )
    assert selected.for_task("local-prompt-test").local_mode is True
    assert "run_verified_md" not in selected.tools
    assert not selected.tools["run_lammps"].trusted
    handles = selected.materialize_file_resources((descriptor.name,))
    assert Path(handles[descriptor.name].controller_path).read_text() == "potential"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert selected.file_resource_sources[descriptor.name] == assets / "potentials/SW/Si.sw"
    fs = MDWorkspaceFilesystem(workspace, resource_paths={
        descriptor.public_path: assets / "potentials/SW/Si.sw",
    })
    assert fs.read_file(descriptor.public_path) == "potential"
    fs.copy_file(descriptor.public_path, "/workspace/input/sample.sw")
    assert (workspace / "input/sample.sw").read_text() == "potential"
    with pytest.raises(ValueError, match="reserved"):
        fs.write_file(descriptor.public_path, "changed")


@pytest.mark.parametrize("level", [1, 2])
def test_modal_factory_remains_an_explicit_option(tmp_path, monkeypatch, level):
    monkeypatch.delenv("CORRAL_MD_ASSETS_DIR", raising=False)
    monkeypatch.delenv("CORRAL_MD_VERIFICATION", raising=False)
    monkeypatch.setattr(
        env, "assets_directory",
        lambda *_: pytest.fail("Explicit Modal runs must not require local assets"),
    )
    selected = env.create_environments(
        work_dir=str(tmp_path / "work"), level=level, local=False,
    )[f"level_{level}_task_4"]
    assert selected.local_mode is False
    assert selected.for_task("modal-prompt-test").local_mode is False
    assert "run_verified_md" in selected.tools
    assert selected.tools["run_lammps"].trusted
    assert not isinstance(
        selected.current_task.scoring_fn.verifier,
        (Level1LocalVerifier, LocalVerifier),
    )
    assert selected.current_task.scoring_fn.verifier is not None
    assert selected.file_resource_sources == {}


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("verification", [None, "modal", "offline"])
def test_scoring_cli_defaults_to_local_with_explicit_backend_overrides(
    tmp_path, monkeypatch, level, verification,
):
    assets = _assets(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    output = tmp_path / "report.json"
    scorers = []
    factory_name = f"check_level{level}_workflow"
    factory = getattr(score, factory_name)

    def capture_factory(*args, **kwargs):
        scorer = factory(*args, **kwargs)
        scorers.append(scorer)
        return scorer

    monkeypatch.setattr(score, factory_name, capture_factory)
    # A legacy remote setting must not silently override the CLI's local default.
    monkeypatch.setenv("CORRAL_MD_VERIFICATION", "modal")
    arguments = [
        "corral_md.scoring", "4", str(manifest), "--level", str(level),
        "--assets-dir", str(assets), "--output", str(output),
    ]
    if verification is not None:
        arguments.extend(["--verification", verification])
    monkeypatch.setattr("sys.argv", arguments)
    assert score.main() == 0
    assert json.loads(output.read_text())["score"] == 0
    verifier = scorers[0].verifier
    if verification is None:
        assert isinstance(verifier, Level1LocalVerifier if level == 1 else LocalVerifier)
    elif verification == "offline":
        assert verifier is None
    else:
        assert verifier is not None
        assert not isinstance(verifier, (Level1LocalVerifier, LocalVerifier))


def test_local_lammps_runs_host_executable_and_preserves_input(tmp_path, monkeypatch):
    assets = _assets(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.in").write_text(
        "pair_coeff * * /workspace/resources/md-potentials-SW-Si/Si.sw Si\n"
        "write_data /workspace/output/final.dat\n"
    )
    executable = tmp_path / "fake-lmp"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "source = pathlib.Path(sys.argv[sys.argv.index('-in') + 1])\n"
        "log = pathlib.Path(sys.argv[sys.argv.index('-log') + 1])\n"
        "log.write_text(source.read_text())\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("CORRAL_MD_LAMMPS_BIN", str(executable))
    tool = tools.build_run_lammps_tool(workspace, local=True, assets_dir=str(assets))
    result = tool.execute(input_file="/workspace/input.in")
    assert "completed locally" in result
    log = (workspace / "output/input.log").read_text()
    assert str(assets / "potentials/SW/Si.sw") in log
    assert str(workspace / "output/final.dat") in log
    assert "/workspace/resources/md-potentials-SW-Si/Si.sw" in (workspace / "input.in").read_text()
    assert not list(workspace.glob(".corral-md-*.in"))


def test_local_gpu_dispatch_uses_worker_not_modal(tmp_path, monkeypatch):
    assets = _assets(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "script.py").write_text("print('ok')\n")
    captured = {}

    def worker(*args, **kwargs):
        captured.update(kwargs)
        return json.dumps({"backend": "local_gpu"})

    monkeypatch.setattr(tools, "_run_python_in_local_worker", worker)
    monkeypatch.setattr(
        tools, "run_python_gpu_in_modal",
        lambda *args, **kwargs: pytest.fail("Modal GPU dispatch was used"),
    )
    tool = tools.build_execute_python_script_tool(
        workspace, local=True, assets_dir=str(assets)
    )
    result = json.loads(tool.execute(script_path="/workspace/script.py", use_gpu=True))
    assert result["backend"] == "local_gpu"
    assert captured == {"gpu": True}


def test_local_terminal_preserves_cuda_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    terminal = tools.build_md_terminal_tool(tmp_path, local=True)
    result = json.loads(
        terminal.execute(command="printf '%s' \"$CUDA_VISIBLE_DEVICES\"")
    )
    assert result["exit_code"] == 0
    assert result["output"] == "2"


def test_local_verifier_routes_mace_to_configured_host_gpu(tmp_path, monkeypatch):
    assets = _assets(tmp_path)
    monkeypatch.setenv("CORRAL_MD_VERIFIER_GPUS", "2,3")
    transport = LocalCalculationTransport(assets)
    calls = []

    monkeypatch.setattr(
        "corral_md.scoring.local_verifier.subprocess.check_output",
        lambda *args, **kwargs: b"sha256:" + b"a" * 64,
    )

    def run(command, **kwargs):
        calls.append(command)
        output_mount = next(
            item for item in command if item.startswith("type=bind,") and ",dst=/output" in item
        )
        output = Path(output_mount.split(",", 2)[1].removeprefix("src="))
        (output / "result.json").write_text('{"jobs": []}')
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        "corral_md.scoring.local_verifier.subprocess.run", run
    )
    for task_number in (1, 2):
        plan = SimpleNamespace(
            task_number=task_number,
            jobs=({"operation": "mace"},),
            files={},
            challenge="test",
        )
        transport(plan, "fingerprint")
    assert [command[command.index("--gpus") + 1] for command in calls] == [
        '"device=2"',
        '"device=3"',
    ]


def test_local_verifier_rejects_invalid_gpu_selector(tmp_path, monkeypatch):
    monkeypatch.setenv("CORRAL_MD_VERIFIER_GPUS", "2,2")
    with pytest.raises(ValueError, match="unique GPU indices"):
        LocalCalculationTransport(_assets(tmp_path))


def test_local_assets_do_not_follow_catalog_symlinks(tmp_path):
    assets = _assets(tmp_path)
    (tmp_path / "workspace").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (assets / "models/escape").symlink_to(outside)
    public = "/workspace/resources/escape/outside.txt"
    fs = MDWorkspaceFilesystem(tmp_path / "workspace", resource_paths={
        public: assets / "models/escape",
    })
    with pytest.raises(ValueError, match="symbolic links"):
        fs.read_file(public)


def test_sandboxed_local_lammps_preserves_source_and_workspace_cwd(tmp_path, monkeypatch):
    from corral_md import local_runtime

    (tmp_path / "input").mkdir()
    source = tmp_path / "input/run.in"
    original = "log /workspace/output/custom.log\nprint '/workspace/resources/fixture/file'\nrun 0\n"
    source.write_text(original)
    monkeypatch.setattr(local_runtime, "local_path", lambda _root, public, **_kw: tmp_path / public.removeprefix("/workspace/"))
    monkeypatch.setattr(local_runtime.shutil, "which", lambda _name: "/usr/bin/lmp")
    monkeypatch.setattr(local_runtime, "assets_directory", lambda *_: pytest.fail("sandbox must use authorized mounts"))

    def run(command, **options):
        assert options["cwd"] == Path("/workspace")
        assert command[command.index("-in") + 1] == str(source)
        assert source.read_text() == original
        Path(command[command.index("-log") + 1]).write_text("simulation log")
        return SimpleNamespace(wait=lambda **_kwargs: None, pid=123, returncode=0)

    monkeypatch.setattr(local_runtime.subprocess, "Popen", run)
    monkeypatch.setattr(local_runtime.os, "killpg", lambda *_: None)
    result = local_runtime.run_lammps_locally("/workspace", "/workspace/input/run.in", assets_dir=None)
    assert result.read_text() == "simulation log"
    assert source.read_text() == original
