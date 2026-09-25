"""MD tools must preserve their namespace and policy through the shared executor."""

import json
import os
import tempfile
from pathlib import Path

import pytest
from corral_md import modal_workspace, tools
from corral_md.env import MolecularDynamicsEnvironment, _md_file_tools

from corral.core import Action
from corral.core.environment import Toolset
from corral.core.state import (
    ActionState,
    EnvironmentState,
    ExecutionState,
    RuntimeState,
    TaskState,
)
from corral.core.task import TaskDefinition
from corral.core.transition import ToolRecoveryPending, execute_action
from corral.runtime import permissions
from corral.runtime.tool_execution import ToolArgumentError, ToolExecutor


def _execution(root):
    environment = MolecularDynamicsEnvironment(
        "md",
        TaskDefinition(
            name="md",
            description="md",
            submission_format={},
            scoring_fn=lambda _: 1.0,
            resolve_answer=False,
            tools=["keyword_log_extractor", "get_structure_from_mp_text"],
        ),
        base_work_dir=str(root.parent),
        workspace_path=str(root),
        toolset=Toolset(
            pool={
                item.name: item
                for item in (
                    tools.keyword_log_extractor,
                    tools.get_structure_from_mp_text,
                )
            },
            workspace_factory=_md_file_tools,
        ),
    )
    started = environment.initial_event(execution_id="md-routing")
    state = ExecutionState(
        through_commit_hash="0" * 64,
        execution_id="md-routing",
        branch_id="main",
        task=TaskState(environment=started.environment_metadata),
        environment=EnvironmentState(values=started.environment),
        workspace=started.workspace,
        runtime=RuntimeState(
            metadata={
                "corral_md_release_id": "pinned-build",
                "corral_md_volume_name": "pinned-volume",
            }
        ),
    )
    return environment, state


def _call(environment, state, name, **arguments):
    return ToolExecutor(environment).execute(
        state, environment.tools[name], arguments, action_id="md-action"
    )


@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("kind", ["models", "structures", "potentials", "resources"])
def test_write_cannot_bypass_reserved_namespaces(
    tmp_path, monkeypatch, restricted, kind
):
    environment, state = _execution(tmp_path / "task")
    monkeypatch.setattr(permissions, "enabled", lambda: restricted)
    monkeypatch.setattr(
        permissions,
        "run_worker",
        lambda *_a, **_k: pytest.fail(
            "MD file operations must retain their asset checks"
        ),
    )
    with pytest.raises((PermissionError, ValueError)):
        _call(
            environment,
            state,
            "write_file",
            path=f"/workspace/{kind}/file",
            content="bad",
        )
    assert not (Path(environment.workspace_path) / kind).exists()


def test_cpu_execution_captures_outputs_and_public_paths(tmp_path):
    root = tmp_path / "task"
    environment, state = _execution(root)
    (root / "scripts/analyze.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "Path(sys.argv[1]).write_text('42')\nprint('done')\n"
    )
    action = Action(
        id="cpu-action",
        name="execute_python_script",
        arguments={
            "script_path": "/workspace/scripts/analyze.py",
            "args": ["/workspace/output/result.txt"],
        },
    )
    state = ExecutionState.model_validate(
        {
            **state.model_dump(mode="python"),
            "actions": {
                action.id: ActionState(action=action, requested_by_run_id="agent")
            },
        }
    )
    effects = execute_action(environment, state, action)
    assert effects.success, effects.observation
    assert json.loads(effects.observation)["backend"] == "local_cpu"
    assert str(root) not in effects.observation
    assert (root / "output/result.txt").read_text() == "42"
    assert effects.workspace_delta is not None


def test_cpu_worker_receives_only_public_paths_and_declared_access(
    tmp_path, monkeypatch
):
    environment, state = _execution(tmp_path / "task")
    Path(environment.workspace_path, "scripts/analyze.py").write_text("print(42)")
    monkeypatch.setattr(permissions, "enabled", lambda: True)

    def run_worker(kind, payload, workspace, **policy):
        selected, arguments = payload
        assert kind == "tool"
        assert workspace == environment.workspace_path
        assert not selected.trusted
        assert arguments["workspace"] == "/workspace"
        assert arguments["script_relative"] == "scripts/analyze.py"
        assert arguments["directory_relative"] == "."
        assert arguments["args"] == ["/workspace/output/result.txt"]
        assert "corral_action_id" not in arguments
        assert policy["workspace_access"] == "read_write"
        assert policy["resource_mounts"] == {}
        return {"content": "restricted CPU"}

    monkeypatch.setattr(permissions, "run_worker", run_worker)
    assert (
        _call(
            environment,
            state,
            "execute_python_script",
            script_path="/workspace/scripts/analyze.py",
            args=["/workspace/output/result.txt"],
        )
        == "restricted CPU"
    )


@pytest.mark.parametrize(
    ("name", "backend", "arguments"),
    [
        (
            "run_lammps",
            "_run_lammps_for_workspace",
            {"input_file": "/workspace/input/run.in"},
        ),
        (
            "execute_python_script",
            "run_python_gpu_in_modal",
            {"script_path": "/workspace/scripts/run.py", "use_gpu": True},
        ),
    ],
)
def test_remote_dispatch_keeps_release_recovery_and_action_identity(
    tmp_path, monkeypatch, name, backend, arguments
):
    environment, state = _execution(tmp_path / "task")
    root = Path(environment.workspace_path)
    (root / "input/run.in").write_text("run 0")
    (root / "scripts/run.py").write_text("print(42)")
    monkeypatch.setattr(permissions, "enabled", lambda: True)
    monkeypatch.setattr(
        permissions,
        "run_worker",
        lambda *_a, **_k: pytest.fail("remote operation entered local worker"),
    )

    def remote(workspace, path, *_args, action_id, **_kwargs):
        assert workspace == str(root)
        assert Path(path).is_relative_to(root)
        assert action_id == "md-action"
        assert modal_workspace._PINNED_RELEASE.get() == "pinned-build"
        assert modal_workspace._PINNED_VOLUME.get() == "pinned-volume"
        assert modal_workspace._RECOVERY_SNAPSHOT.get() == (
            state.workspace,
            environment.workspace_manager,
        )
        raise ToolRecoveryPending("remote still running")

    monkeypatch.setattr(tools, backend, remote)
    with pytest.raises(ToolRecoveryPending, match="still running"):
        _call(environment, state, name, **arguments)
    assert modal_workspace._PINNED_RELEASE.get() is None
    assert modal_workspace._RECOVERY_SNAPSHOT.get() is None
    with pytest.raises(ToolArgumentError):
        _call(environment, state, name, **arguments, corral_action_id="forged")


@pytest.mark.parametrize("restricted", [False, True])
def test_domain_paths_stay_public_until_execution(tmp_path, monkeypatch, restricted):
    environment, state = _execution(tmp_path / "task")
    root = Path(environment.workspace_path)
    (root / "output/log.lammps").write_text("fix thermostat all nvt\n")
    monkeypatch.setattr(permissions, "enabled", lambda: restricted)

    def run_worker(kind, payload, workspace, **policy):
        assert kind == "tool"
        assert workspace == environment.workspace_path
        assert payload[1]["path"] == "/workspace/output/log.lammps"
        assert policy["workspace_access"] == "read"
        return {"content": "fix thermostat all nvt"}

    monkeypatch.setattr(permissions, "run_worker", run_worker)
    executor = ToolExecutor(environment)
    prepared = executor.prepare(
        state,
        environment.tools["keyword_log_extractor"],
        {"path": "/workspace/output/log.lammps", "keyword": "fix"},
        action_id="domain",
    )
    assert prepared.visible_arguments["path"] == "/workspace/output/log.lammps"
    assert "thermostat" in executor.execute_prepared(state, prepared)
    with pytest.raises(ValueError):
        _call(
            environment,
            state,
            "keyword_log_extractor",
            path=str(root / "output/log.lammps"),
            keyword="fix",
        )


def test_credentialed_domain_tool_translates_only_during_dispatch(
    tmp_path, monkeypatch
):
    environment, state = _execution(tmp_path / "task")
    root = Path(environment.workspace_path)
    monkeypatch.setattr(permissions, "enabled", lambda: True)

    def fetch(mp_id, file_path):
        assert mp_id == "mp-149"
        assert file_path == str(root / "input/structure.cif")
        return {"path": file_path}

    monkeypatch.setattr(
        environment.tools["get_structure_from_mp_text"], "execute", fetch
    )
    assert _call(
        environment,
        state,
        "get_structure_from_mp_text",
        mp_id="mp-149",
        file_path="/workspace/input/structure.cif",
    ) == {"path": "/workspace/input/structure.cif"}


@pytest.mark.skipif(
    os.environ.get("CORRAL_PERMISSION_TESTS") != "1",
    reason="requires the real Docker trial permission boundary",
)
def test_cpu_script_and_terminal_are_confined_by_linux(tmp_path, monkeypatch):
    root = Path(tempfile.mkdtemp(prefix="md-permissions-", dir="/workspace"))
    environment, state = _execution(root)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(mode=0o700)
    secret = tmp_path / "controller-secret"
    secret.write_text("private")
    monkeypatch.setattr(permissions, "_enabled", False)
    monkeypatch.setenv("MP_API_KEY", "controller-only")
    permissions.configure(checkpoints)
    (root / "scripts/probe.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "assert os.getuid() != 0\n"
        "assert 'MP_API_KEY' not in os.environ\n"
        f"try:\n    Path({str(secret)!r}).read_text()\n"
        "except (PermissionError, FileNotFoundError):\n    pass\n"
        "else:\n    raise AssertionError('controller file was readable')\n"
        "Path('/workspace/output/result.txt').write_text('isolated')\n"
    )
    result = _call(
        environment,
        state,
        "execute_python_script",
        script_path="/workspace/scripts/probe.py",
    )
    assert json.loads(result)["success"]
    result = _call(
        environment, state, "terminal", command="cat /workspace/output/result.txt"
    )
    assert json.loads(result)["output"] == "isolated"
    (root / "output/log.lammps").write_text("fix thermostat all nvt\n")
    assert "thermostat" in _call(
        environment,
        state,
        "keyword_log_extractor",
        path="/workspace/output/log.lammps",
        keyword="fix",
    )
    assert secret.read_text() == "private"
