"""The explicit local backend must not dispatch LAMMPS or GPU Python to Modal."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from corral_md import env, tools
from corral_md.workspace import MDWorkspaceFilesystem
from corral_md.workflow_scoring.local_verifier import LocalCalculationTransport


def _assets(tmp_path: Path) -> Path:
    root = tmp_path / "assets"
    for kind in ("models", "potentials", "structures"):
        (root / kind).mkdir(parents=True)
    (root / "potentials" / "sample.sw").write_text("potential")
    return root


def test_local_factory_uses_local_scoring_and_local_assets(tmp_path):
    assets = _assets(tmp_path)
    environments = env.create_environments(
        work_dir=str(tmp_path / "work"), local=True, assets_dir=str(assets)
    )
    selected = environments["level_1_task_3"]
    assert selected.local_mode is True
    assert selected.current_task.scoring_fn.verifier is not None
    assert selected.for_task("local-prompt-test").local_mode is True
    assert "run_verified_md" not in selected.tools
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bound = env._md_file_tools(str(workspace), local=True, assets_dir=str(assets))
    assert (
        bound["read_file"].execute(path="/workspace/potentials/sample.sw")
        == "potential"
    )
    bound["copy_file"].execute(
        source="/workspace/potentials/sample.sw",
        destination="/workspace/input/sample.sw",
    )
    assert (workspace / "input/sample.sw").read_text() == "potential"
    with pytest.raises(PermissionError):
        bound["write_file"].execute(
            path="/workspace/potentials/sample.sw", content="changed"
        )


def test_local_lammps_runs_host_executable_and_preserves_input(tmp_path, monkeypatch):
    assets = _assets(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.in").write_text(
        "pair_coeff * * /workspace/potentials/sample.sw Si\n"
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
    assert str(assets / "potentials/sample.sw") in log
    assert str(workspace / "output/final.dat") in log
    assert "/workspace/potentials/sample.sw" in (workspace / "input.in").read_text()
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
    assert captured == {"gpu": True, "assets_dir": str(assets)}


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
        "corral_md.workflow_scoring.local_verifier.subprocess.check_output",
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
        "corral_md.workflow_scoring.local_verifier.subprocess.run", run
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
    fs = MDWorkspaceFilesystem(tmp_path / "workspace", assets_dir=assets)
    with pytest.raises(ValueError, match="escapes"):
        fs.read_file("/workspace/models/escape")
