"""Portable dispatch checks and opt-in tests of the real Docker REPL boundary."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
from dataclasses import asdict

import cloudpickle
import pytest
from stargazer.docker import execute_analysis
from stargazer.env import create_environments

from corral.core.state import EnvironmentState, ExecutionState, TaskState
from corral.runtime import permissions
from corral.runtime.tool_execution import ToolExecutor


def test_docker_dispatch_sends_only_public_data_and_opaque_checkpoint(
    tmp_path, monkeypatch
):
    from stargazer.env import PROTOCOL_VERSION
    from stargazer.fit import validate_fit
    from stargazer.protocol import public_resources

    environment = create_environments(
        level=1, work_dir=tmp_path, development_mode=True
    )["seed15_diff5"].for_task("docker-dispatch")
    task = environment.current_task.scoring_inputs["benchmark_task"]
    history = [
        validate_fit(task.public_fit_context(), {"planets": [], "noise_jitter_ms": 0.1})
    ]
    hidden = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_session": "untrusted-opaque-checkpoint",
        "analysis": {"history": history, "revision": 1},
        "analysis_history_revision": 0,
    }
    started = environment.initial_event(execution_id="docker-dispatch")
    state = ExecutionState(
        through_commit_hash="0" * 64,
        execution_id="docker-dispatch",
        branch_id="main",
        task=TaskState(metadata=started.task, environment=started.environment_metadata),
        environment=EnvironmentState(
            values={**dict(started.environment), "hidden_arguments": hidden}
        ),
        workspace=started.workspace,
    )
    requests = []

    def run(kind, payload, workspace, **kwargs):
        assert kind == "tool"
        assert workspace == environment.workspace_path
        assert kwargs["cancel"] is None
        assert kwargs["workspace_access"] == "read_write"
        step, arguments = payload
        assert step.network_access == "none"
        assert not step.trusted
        assert not step.hidden_args
        arguments["public_data"] = json.loads(arguments["public_data"])
        requests.append(arguments)
        return {
            "content": json.dumps(
                {
                    "output": "42",
                    "checkpoint": "new-opaque-checkpoint",
                    "protocol_ack": True,
                }
            )
        }

    monkeypatch.setattr(permissions, "enabled", lambda: True)
    monkeypatch.setattr(permissions, "run_worker", run)
    result = ToolExecutor(environment).execute(
        state,
        environment.tools["PythonREPL"],
        {"input_code": "print(6 * 7)"},
        action_id="repl-action",
    )
    assert requests == [
        {
            "code": "print(6 * 7)",
            "checkpoint": hidden["analysis_session"],
            "public_data": json.loads(
                json.dumps(
                    {
                        **asdict(task.observations),
                        "star_mass_sun": task.star_mass_sun,
                        "history": history,
                        "analysis_assistance": True,
                        "maximum_rms_factor": 1.5,
                        "los_axis": task.public_fit_context().los_axis,
                        "integrator_preference": task.public_fit_context().integrator_preference,
                        "public_resources": public_resources(),
                    }
                )
            ),
        }
    ]
    assert result.content == "42"
    assert result.environment["hidden_arguments"] == {
        **hidden,
        "analysis_session": "new-opaque-checkpoint",
        "analysis_history_revision": 1,
    }
    assert hidden["analysis_session"] == "untrusted-opaque-checkpoint"


@pytest.fixture
def docker_workspace(tmp_path):
    if os.environ.get("CORRAL_PERMISSION_TESTS") != "1":
        pytest.skip("requires the real Docker trial permission boundary")
    assert os.geteuid() == 0
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(mode=0o700)
    permissions.configure(checkpoints)
    with tempfile.TemporaryDirectory(
        prefix="stargazer-test-", dir="/workspace"
    ) as workspace:
        yield workspace
    permissions._enabled = False


def test_docker_repl_persists_functions_arrays_and_blocks_source_reads(
    docker_workspace, simple_task
):
    data = {
        **asdict(simple_task.observations),
        "star_mass_sun": simple_task.star_mass_sun,
    }
    first = execute_analysis(
        code="values = np.arange(3.0)\ndef shifted():\n    return values + 2\nprint(shifted().tolist())",
        public_data=data,
        checkpoint=None,
        workspace=docker_workspace,
    )
    assert json.loads(first["output"]) == [2.0, 3.0, 4.0]
    restored = execute_analysis(
        code="values[0] = 40.0\nprint(shifted().tolist())",
        public_data=data,
        checkpoint=first["checkpoint"],
        workspace=docker_workspace,
    )
    assert json.loads(restored["output"]) == [42.0, 3.0, 4.0]
    allowed = execute_analysis(
        code="from pathlib import Path\nnp.save('fit.npy', values)\nprint(np.load('fit.npy').tolist())",
        public_data=data,
        checkpoint=restored["checkpoint"],
        workspace=docker_workspace,
    )
    assert json.loads(allowed["output"]) == [40.0, 1.0, 2.0]
    for code, error in (
        (
            "open('/opt/corral/tasks/stargazer/data/synthetic/seed15_diff5.json').read()",
            "PermissionError",
        ),
        (
            # NumPy reports inaccessible paths as missing files.
            "np.loadtxt('/opt/corral/tasks/stargazer/data/synthetic/seed15_diff5.json')",
            "FileNotFoundError",
        ),
        ("Path('/corral-state').iterdir().__next__()", "PermissionError"),
        (
            "import subprocess\nimport sys\nprint(subprocess.run([sys.executable, '-c', \"open('/opt/corral/tasks/stargazer/data/synthetic/seed15_diff5.json').read()\"], capture_output=True, text=True).stderr)",
            "PermissionError",
        ),
    ):
        blocked = execute_analysis(
            code=code,
            public_data=data,
            checkpoint=allowed["checkpoint"],
            workspace=docker_workspace,
        )
        assert error in blocked["output"]
        assert "truth_planets" not in blocked["output"]


@pytest.mark.parametrize("assisted", [False, True])
def test_docker_functions_see_globals_across_cells_and_restore(
    docker_workspace, simple_task, assisted
):
    data = {
        **asdict(simple_task.observations),
        "star_mass_sun": simple_task.star_mass_sun,
        "analysis_assistance": assisted,
    }
    checkpoint = None
    for code, expected in (
        ("def total():\n    return t.sum()\nprint(total())", "NameError"),
        ("t = np.arange(3)\nprint(total())", "3\n"),
        ("t = np.array([8, 9])\nprint(total())", "17\n"),
        ("t = np.array([42])\nprint(total())", "42\n"),
    ):
        result = execute_analysis(
            code=code,
            public_data=data,
            checkpoint=checkpoint,
            workspace=docker_workspace,
        )
        assert expected in result["output"]
        checkpoint = result["checkpoint"]


def test_docker_repl_has_no_execution_deadline(docker_workspace, simple_task):
    data = {
        **asdict(simple_task.observations),
        "star_mass_sun": simple_task.star_mass_sun,
    }
    # Exceed the former forty-second deadline, including worker startup.
    result = execute_analysis(
        code="import time\nretained = 42\ntime.sleep(40.2)\nprint(retained)",
        public_data=data,
        checkpoint=None,
        workspace=docker_workspace,
    )
    assert result["output"].strip() == "42"
    restored = execute_analysis(
        code="retained",
        public_data=data,
        checkpoint=result["checkpoint"],
        workspace=docker_workspace,
    )
    assert restored["output"].strip() == "42"


def test_docker_repl_remains_cancellable(docker_workspace, simple_task):
    data = {
        **asdict(simple_task.observations),
        "star_mass_sun": simple_task.star_mass_sun,
    }
    cancelled = threading.Event()
    timer = threading.Timer(5, cancelled.set)
    timer.start()
    try:
        with pytest.raises(RuntimeError, match="restricted worker cancelled"):
            execute_analysis(
                code="import time\nwhile True:\n    time.sleep(1)",
                public_data=data,
                checkpoint=None,
                workspace=docker_workspace,
                cancel=cancelled,
            )
    finally:
        timer.cancel()


def test_docker_checkpoint_decoding_is_unprivileged_and_has_no_credentials(
    docker_workspace,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-credential")

    class CheckpointProbe:
        def __reduce__(self):
            # A pickle can execute arbitrary code before namespace validation.
            # This must only run after chroot, privilege drop, and env clearing.
            expression = "(_ for _ in ()).throw(RuntimeError('checkpoint uid=' + str(__import__('os').getuid()) + ',key=' + str('OPENAI_API_KEY' in __import__('os').environ)))"
            return eval, (expression,)

    checkpoint = base64.b64encode(cloudpickle.dumps(CheckpointProbe())).decode()
    with pytest.raises(RuntimeError, match=r"checkpoint uid=[1-9][0-9]+,key=False"):
        execute_analysis(
            code="1", public_data={}, checkpoint=checkpoint, workspace=docker_workspace
        )


def test_docker_protocol_and_history_updates_survive_checkpoint(
    docker_workspace, simple_task
):
    from stargazer.tools import STARGAZER_SUBMISSION_GUIDE

    data = {
        **asdict(simple_task.observations),
        "star_mass_sun": simple_task.star_mass_sun,
    }
    first = execute_analysis(
        code="print(STARGAZER_SUBMISSION_GUIDE)\n_protocol_guide_ack = True\nprint(_protocol_guide_ack)",
        public_data=data,
        checkpoint=None,
        workspace=docker_workspace,
    )
    assert first["protocol_ack"] is True
    assert first["output"] == STARGAZER_SUBMISSION_GUIDE + "\nTrue\n"
    history = [{"valid": True, "candidate": {"planets": []}, "residuals": {"rms": 1.0}}]
    second = execute_analysis(
        code="import json\nprint(json.dumps(history))",
        public_data={**data, "history": history},
        checkpoint=first["checkpoint"],
        workspace=docker_workspace,
    )
    assert second["protocol_ack"] is True
    assert json.loads(second["output"]) == history


def test_docker_network_disabled_before_checkpoint_decoding(
    docker_workspace, simple_task
):
    import socket

    # Even a reachable controller-local service must be inaccessible in the worker.
    server = socket.socket()
    server.bind(("0.0.0.0", 0))
    server.listen()
    port = server.getsockname()[1]
    try:
        code = f"""import socket, subprocess, sys
s=socket.socket()
s.settimeout(.2)
print(s.connect_ex(('127.0.0.1',{port})) != 0)
print(subprocess.check_output([sys.executable,'-c',"import socket;s=socket.socket();s.settimeout(.2);print(s.connect_ex(('127.0.0.1',{port})) != 0)"],text=True).strip())
s.close()
del s"""
        result = execute_analysis(
            code=code,
            public_data={
                **asdict(simple_task.observations),
                "star_mass_sun": simple_task.star_mass_sun,
            },
            checkpoint=None,
            workspace=docker_workspace,
        )
        assert result["output"].strip() == "True\nTrue"

        class NetworkCheckpoint:
            def __reduce__(self):
                expression = f"(_ for _ in ()).throw(RuntimeError('isolated=' + str(__import__('socket').socket().connect_ex(('127.0.0.1',{port})) != 0)))"
                return eval, (expression,)

        checkpoint = base64.b64encode(cloudpickle.dumps(NetworkCheckpoint())).decode()
        with pytest.raises(RuntimeError, match="isolated=True"):
            execute_analysis(
                code="1",
                public_data={},
                checkpoint=checkpoint,
                workspace=docker_workspace,
            )
    finally:
        server.close()


def test_docker_post_validation_introspection_and_tampering(
    docker_workspace, simple_task, exact_submission
):
    from stargazer.fit import validate_fit
    from stargazer.score import make_stargazer_scorer

    context = simple_task.public_fit_context()
    diagnostics = validate_fit(context, exact_submission)
    data = {
        **asdict(context.observations),
        "star_mass_sun": context.star_mass_sun,
        "history": [diagnostics],
    }
    code = """import gc, sys, json
leaks=[]
for obj in gc.get_objects():
    if type(obj).__name__ in ('StargazerTask', 'StargazerScorer'):
        leaks.append(type(obj).__name__)
frame=sys._getframe()
while frame is not None:
    for value in list(frame.f_locals.values()):
        if type(value).__name__ in ('StargazerTask', 'StargazerScorer'):
            leaks.append(type(value).__name__)
    frame=frame.f_back
print(json.dumps(leaks))
history[0]['residuals']['rms']=999
rvs_ms[:]=999
try:
    import stargazer.score as private_score
except ModuleNotFoundError:
    pass
else:
    private_score._match_planets=lambda *args: (1.0, ())
import stargazer.public_rv as public_rv
public_rv._log_likelihood=lambda *args: 999
del frame, value, obj
"""
    first = execute_analysis(
        code=code, public_data=data, checkpoint=None, workspace=docker_workspace
    )
    assert first["output"].strip() == "[]"
    second = execute_analysis(
        code='print(history[0]["residuals"]["rms"])',
        public_data={k: v for k, v in data.items() if k != "history"},
        checkpoint=first["checkpoint"],
        workspace=docker_workspace,
    )
    assert second["output"].strip() == "999"
    assert validate_fit(context, exact_submission) == diagnostics
    assert make_stargazer_scorer(simple_task)(json.dumps(exact_submission)) == 1
    for path in (
        "/corral-private/0/private-manifest.json",
        "/opt/corral/tasks/stargazer/data/reference_audit.json",
        "/corral-state",
    ):
        result = execute_analysis(
            code=f"print(open({path!r}).read())",
            public_data=data,
            checkpoint=second["checkpoint"],
            workspace=docker_workspace,
        )
        assert (
            "PermissionError" in result["output"]
            or "FileNotFoundError" in result["output"]
            or "IsADirectoryError" in result["output"]
        )


def test_private_mount_is_excluded_from_worker(docker_workspace, simple_task):
    from pathlib import Path

    private = Path("/corral-private/0/private-manifest.json")
    if not private.exists():
        pytest.skip("mount a private bank at /corral-private/0 for this packaging test")
    assert json.loads(private.read_text())["frozen"]
    result = execute_analysis(
        code=f"print(open({str(private)!r}).read())",
        public_data={
            **asdict(simple_task.observations),
            "star_mass_sun": simple_task.star_mass_sun,
        },
        checkpoint=None,
        workspace=docker_workspace,
    )
    assert (
        "PermissionError" in result["output"] or "FileNotFoundError" in result["output"]
    )
    assert "bank_hash" not in result["output"]


def test_docker_public_helpers_source_checkpoint_and_tampering(
    docker_workspace, simple_task, exact_submission
):
    from stargazer.fit import validate_fit

    data = {
        **asdict(simple_task.observations),
        "star_mass_sun": simple_task.star_mass_sun,
    }
    first = execute_analysis(
        code=f"""from pathlib import Path
import subprocess, sys
planets = {exact_submission["planets"]!r}
d = stargazer_diagnostics(planets)
curve = stargazer_predict(planets)
print(d['residuals']['rms'])
print(stargazer_predict.__func__.__closure__)
print('load_task' in stargazer_predict.__func__.__globals__)
Path('public_rv.py').write_text(STARGAZER_PUBLIC_RESOURCES['public_rv.py'])
print(subprocess.check_output([sys.executable, '-c', 'import public_rv; print(hasattr(public_rv, "PublicRV"))'], text=True).strip())
""",
        public_data=data,
        checkpoint=None,
        workspace=docker_workspace,
    )
    assert first["output"].strip() == "0.0\nNone\nFalse\nTrue"
    second = execute_analysis(
        code="""print(np.array_equal(curve, stargazer_predict(planets)))
print(stargazer_diagnostics(planets)['residuals']['rms'])
d['model_ms'][:]=999
rvs_ms[:]=999
STARGAZER_PUBLIC_RESOURCES['public_rv.py']='tampered'
print(stargazer_diagnostics(planets)['residuals']['rms'])
""",
        public_data=data,
        checkpoint=first["checkpoint"],
        workspace=docker_workspace,
    )
    assert second["output"].strip() == "True\n0.0\n0.0"
    assert (
        validate_fit(simple_task.public_fit_context(), exact_submission)["residuals"][
            "rms"
        ]
        == 0
    )
    blocked = execute_analysis(
        code="print(open('/opt/corral/tasks/stargazer/data/synthetic/seed15_diff5.json').read())",
        public_data=data,
        checkpoint=second["checkpoint"],
        workspace=docker_workspace,
    )
    assert "PermissionError" in blocked["output"]


def test_docker_blind_arm_does_not_preload_assistance(docker_workspace, simple_task):
    result = execute_analysis(
        code="""import sys
print('stargazer_predict' in globals())
print('STARGAZER_PUBLIC_RESOURCES' in globals())
print('stargazer.public_rv' in sys.modules)
try:
    import stargazer.public_rv
except ModuleNotFoundError:
    print('source isolated')
""",
        public_data={
            **asdict(simple_task.observations),
            "star_mass_sun": simple_task.star_mass_sun,
            "analysis_assistance": False,
        },
        checkpoint=None,
        workspace=docker_workspace,
    )
    assert result["output"].strip() == "False\nFalse\nFalse\nsource isolated"
