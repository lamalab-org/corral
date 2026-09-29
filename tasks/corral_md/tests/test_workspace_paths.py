"""The public MD namespace is absolute, isolated and read-only for assets."""

import json
from types import SimpleNamespace

import pytest
from corral_md import tools as md_tools
from corral_md.env import MolecularDynamicsEnvironment, _md_file_tools
from corral_md.submission import resolve_submission
from corral_md.workspace import MDWorkspaceFilesystem
from modal.volume import FileEntryType

from corral.core.environment import Toolset
from corral.core.task import TaskDefinition
from corral.runtime import permissions
from corral.workspace import workspace_relative_path


@pytest.mark.parametrize(
    "path",
    [
        "output/result.json",
        "./output/result.json",
        "",
        ".",
        "/tmp/result.json",
        "/workspace-other/result.json",
        "/workspace/../result.json",
        "/workspace/output/../../result.json",
        "/workspace/./result.json",
        "/workspace//result.json",
        "//workspace/result.json",
        "/workspace/a\\b",
        "/workspace/a\x00b",
        "/workspace/output/",
    ],
)
def test_noncanonical_paths_are_rejected_before_resolution(path, tmp_path):
    tools = _md_file_tools(str(tmp_path))
    for name, args in [
        ("read_file", {"path": path}),
        ("write_file", {"path": path, "content": "bad"}),
        ("list_files", {"path": path}),
        ("file_info", {"path": path}),
        ("cat_files", {"paths": [path]}),
        ("copy_file", {"source": path, "destination": "/workspace/result.txt"}),
        ("copy_file", {"source": "/workspace/input.txt", "destination": path}),
        ("grep", {"pattern": "test", "path": path}),
        ("run_lammps", {"input_file": path}),
        ("execute_python_script", {"script_path": path}),
    ]:
        with pytest.raises(ValueError, match="workspace"):
            tools[name].execute(**args)
    assert list(tmp_path.iterdir()) == []


def test_absolute_paths_roundtrip_without_exposing_storage(tmp_path):
    fs = MDWorkspaceFilesystem(tmp_path)
    fs.write_file("/workspace/output/result.json", '{"value": 42}')
    fs.copy_file("/workspace/output/result.json", "/workspace/copy.json")
    assert fs.read_file("/workspace/copy.json") == '{"value": 42}'
    assert fs.list_files() == ["/workspace/copy.json"]
    assert fs.file_info("/workspace")["path"] == "/workspace"
    assert (
        fs.grep("42", "/workspace/output", recursive=True)[0]["file"]
        == "/workspace/output/result.json"
    )
    tools = _md_file_tools(str(tmp_path))
    assert json.loads(tools["list_files"].execute())["files"] == [
        "/workspace/copy.json"
    ]
    with pytest.raises(ValueError):
        fs.read_file(str(tmp_path / "copy.json"))


def test_symlinks_cannot_alias_files_inside_or_outside_workspace(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    (tmp_path / "secret").write_text("private")
    (root / "real").mkdir()
    (root / "inside").symlink_to(root / "real", target_is_directory=True)
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    fs = MDWorkspaceFilesystem(root)
    for alias in ("inside", "outside"):
        with pytest.raises(ValueError):
            fs.write_file(f"/workspace/{alias}/secret", "changed")
    assert (tmp_path / "secret").read_text() == "private"


@pytest.mark.parametrize("kind", ["models", "potentials", "structures"])
def test_assets_cannot_be_written_or_shadowed(kind, tmp_path):
    fs = MDWorkspaceFilesystem(tmp_path)
    fs.write_file("/workspace/input.txt", "input")
    for operation in (
        lambda: fs.write_file(f"/workspace/{kind}/file", "bad"),
        lambda: fs.mkdir(f"/workspace/{kind}"),
        lambda: fs.copy_file("/workspace/input.txt", f"/workspace/{kind}/file"),
        lambda: fs.move_file(f"/workspace/{kind}/file", "/workspace/file"),
    ):
        with pytest.raises(PermissionError, match="read-only"):
            operation()
    assert not (tmp_path / kind).exists()


def test_assets_are_read_remotely_without_materializing_catalog(tmp_path, monkeypatch):
    from corral_md import modal_workspace, workspace

    entry = SimpleNamespace(path="SW/Si.sw", type=FileEntryType.FILE, size=9)
    volume = SimpleNamespace(
        read_file=lambda path: iter([b"potential"]),
        listdir=lambda path, **kwargs: [entry],
    )
    monkeypatch.setattr(
        modal_workspace, "configured_asset_volume_name", lambda name: name
    )
    monkeypatch.setattr(workspace.modal.Volume, "from_name", lambda name: volume)
    fs = MDWorkspaceFilesystem(tmp_path)
    assert fs.list_files("/workspace/potentials", recursive=True) == [
        "/workspace/potentials/SW/Si.sw"
    ]
    assert fs.read_file("/workspace/potentials/SW/Si.sw") == "potential"
    assert fs.file_info("/workspace/potentials/SW/Si.sw")["read_only"] is True
    assert (
        fs.grep("potential", "/workspace/potentials/SW/Si.sw")[0]["file"]
        == "/workspace/potentials/SW/Si.sw"
    )
    fs.copy_file("/workspace/potentials/SW/Si.sw", "/workspace/input/copied.sw")
    assert (tmp_path / "input/copied.sw").read_text() == "potential"
    assert not (tmp_path / "potentials").exists()


def test_cpu_python_runs_in_local_workspace_and_checks_working_dir(tmp_path):
    (tmp_path / "script.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text(str(Path.cwd()))\n"
        "print('local cpu')\n"
    )
    (tmp_path / "output").mkdir()
    tool = md_tools.build_execute_python_script_tool(tmp_path)
    with pytest.raises(ValueError):
        tool.execute(script_path="/workspace/script.py", working_dir=".")
    result = json.loads(
        tool.execute(
            script_path="/workspace/script.py",
            args=["/workspace/output/result.txt"],
        )
    )
    assert result["success"]
    assert result["backend"] == "local_cpu"
    assert (tmp_path / "output/result.txt").read_text() == str(tmp_path.resolve())
    assert (tmp_path / "output/script.stdout.txt").read_text() == "local cpu\n"
    assert "use_gpu" in tool.params_json_schema["properties"]
    assert tool.trusted is False
    assert tool.controller_dispatch is True


def test_cpu_python_dispatches_through_restricted_local_worker(tmp_path, monkeypatch):
    (tmp_path / "script.py").write_text("print('run')")
    calls = []

    def run_worker(operation, payload, workspace, **policy):
        calls.append((operation, payload, workspace, policy))
        return {"content": json.dumps({"success": True, "backend": "local_cpu"})}

    monkeypatch.setattr(permissions, "enabled", lambda: True)
    monkeypatch.setattr(permissions, "run_worker", run_worker)
    result = json.loads(
        md_tools.build_execute_python_script_tool(tmp_path).execute(
            script_path="/workspace/script.py"
        )
    )

    assert result["backend"] == "local_cpu"
    assert calls[0][0] == "tool"
    assert calls[0][1][0].name == "_corral_md_local_python"
    assert calls[0][1][1]["script_relative"] == "script.py"
    assert calls[0][1][1]["workspace"] == "/workspace"
    assert "corral_action_id" not in calls[0][1][1]
    assert calls[0][2] == str(tmp_path)
    assert calls[0][3]["workspace_access"] == "read_write"
    assert calls[0][3]["resource_mounts"] == {}


def test_gpu_python_uses_only_modal_backend_and_checks_working_dir(
    tmp_path, monkeypatch
):
    (tmp_path / "script.py").write_text("print('run')")
    calls = []
    monkeypatch.setattr(
        md_tools,
        "run_python_gpu_in_modal",
        lambda *args, **kwargs: calls.append((args, kwargs)) or 1,
    )
    tool = md_tools.build_execute_python_script_tool(tmp_path)
    with pytest.raises(ValueError):
        tool.execute(script_path="/workspace/script.py", working_dir=".")
    assert not calls
    result = json.loads(
        tool.execute(script_path="/workspace/script.py", use_gpu=True)
    )
    assert result["success"]
    assert result["backend"] == "modal_a100"
    assert calls[0][1]["working_dir"] == "/workspace"


def test_domain_paths_require_public_absolute_names(tmp_path):
    task = TaskDefinition(
        name="md",
        description="md",
        tools=[],
        scoring_fn=lambda _: 1.0,
        submission_format={},
    )
    env = MolecularDynamicsEnvironment(
        "md",
        task,
        base_work_dir=str(tmp_path),
        task_execution_id="test",
        toolset=Toolset(workspace_factory=None),
    )
    for tool, arguments in env._PATH_ARGUMENTS.items():
        for argument in arguments:
            with pytest.raises(ValueError):
                env.preprocess_arguments(tool, {argument: "input/result.txt"})
            resolved = env.preprocess_arguments(
                tool, {argument: "/workspace/input/result.txt"}
            )
            assert resolved[argument] == "/workspace/input/result.txt"


def test_potential_metadata_cannot_accept_existing_host_file(tmp_path):
    host_file = tmp_path / "Si.sw"
    host_file.write_text("host file")
    for path in (str(host_file), "Si.sw", "/workspace/potentials/../Si.sw"):
        with pytest.raises(ValueError):
            md_tools.get_potential_metadata.execute(file_path=path)
    assert md_tools.get_potential_metadata.execute(
        file_path="/workspace/potentials/SW/Si.sw"
    )


def test_public_submission_paths_map_exactly_to_restored_files(tmp_path):
    from corral_md.workflow_scoring.common import Evidence

    output = tmp_path / "output"
    output.mkdir()
    (output / "result.csv").write_text("1,2")
    manifest = {"artifacts": {"data": "/workspace/output/result.csv"}}
    (output / "manifest.json").write_text(json.dumps(manifest))
    resolved = json.loads(
        resolve_submission("/workspace/output/manifest.json", tmp_path)
    )
    assert resolved["artifacts"]["data"] == str(output / "result.csv")
    evidence = Evidence(resolve_submission("/workspace/output/manifest.json", tmp_path))
    assert evidence._path("/workspace/output/result.csv") == output / "result.csv"
    with pytest.raises(ValueError):
        evidence._path("/workspace/../output/result.csv")
    with pytest.raises(FileNotFoundError):
        resolve_submission("/workspace/wrong/result.csv", tmp_path)
    with pytest.raises(ValueError):
        workspace_relative_path("/workspace/../output/result.csv")


def test_terminal_runs_in_local_workspace(tmp_path):
    tools = _md_file_tools(str(tmp_path))
    assert "execute_python_script_gpu" not in tools
    terminal = tools["terminal"]
    result = json.loads(terminal.execute(command="pwd; printf 42 > result.txt"))
    assert result["exit_code"] == 0
    assert str(tmp_path.resolve()) in result["output"]
    assert (tmp_path / "result.txt").read_text() == "42"
    assert terminal.worker_operation == "terminal"
