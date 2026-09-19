"""Portable dispatch checks and opt-in tests of the real Docker REPL boundary."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
from dataclasses import asdict
from types import SimpleNamespace

import cloudpickle
import pytest
from stargazer.docker import execute_analysis
from stargazer.env import create_environments

from corral.runtime import permissions


def test_docker_dispatch_sends_only_public_data_and_opaque_checkpoint(
    tmp_path, monkeypatch
):
    environment = create_environments(level=1, work_dir=tmp_path)["seed15_diff5"]
    task = environment.current_task.scoring_inputs["benchmark_task"]
    hidden = {
        "benchmark_task": task.task_id,
        "analysis_session": "untrusted-opaque-checkpoint",
        "submission_session": {
            "history": [],
            "steps": 0,
            "done": False,
            "protocol_ack": False,
            "force_submit": False,
        },
        "analysis_history_revision": 0,
    }
    state = SimpleNamespace(
        environment=SimpleNamespace(values={"hidden_arguments": hidden})
    )
    requests = []

    def run(kind, payload, workspace, **kwargs):
        assert kind == "tool"
        assert workspace == environment.workspace_path
        assert kwargs["cancel"] is None
        step, arguments = payload
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
    result = permissions.execute_tool(
        environment,
        state,
        environment.tools["PythonREPL"],
        {"input_code": "print(6 * 7)"},
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
                    }
                )
            ),
        }
    ]
    assert result.content == "42"
    assert result.environment["hidden_arguments"] == {
        **hidden,
        "analysis_session": "new-opaque-checkpoint",
        "submission_session": {**hidden["submission_session"], "protocol_ack": True},
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
    history = [{"step": 1, "reward": -1, "done": False, "success": False}]
    second = execute_analysis(
        code="import json\nprint(json.dumps(history))",
        public_data={**data, "history": history},
        checkpoint=first["checkpoint"],
        workspace=docker_workspace,
    )
    assert second["protocol_ack"] is True
    assert json.loads(second["output"]) == history
