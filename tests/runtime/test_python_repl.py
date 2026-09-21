from __future__ import annotations

import json

import numpy as np

from corral.core.tool import ToolConcurrency, WorkspaceAccess
from corral.runtime import permissions
from corral.runtime.python_repl import (
    PythonREPLSession,
    create_python_namespace,
    execute_in_namespace,
    execute_python_repl,
    restore_namespace,
    snapshot_namespace,
)
from corral.tools import create_python_repl_tool


def _scientific_namespace(initial_data):
    return {
        "__builtins__": __builtins__,
        "__name__": "__test_repl__",
        "np": np,
        "values": np.asarray(initial_data["values"]),
        "status": initial_data.get("status", "initial"),
    }


def test_namespace_checkpoint_preserves_functions_closures_and_rng():
    namespace = create_python_namespace({})
    namespace["np"] = np
    execute_in_namespace(
        """np.random.seed(123)
def counter():
    count = 0
    def advance():
        nonlocal count
        count += 1
        return count
    def read():
        return count
    return advance, read
advance, read = counter()""",
        namespace,
    )
    checkpoint = snapshot_namespace(namespace)
    expected_random = execute_in_namespace(
        "print(np.random.random(3).tolist())", namespace
    )

    restored = restore_namespace(checkpoint)
    assert (
        execute_in_namespace("print((advance(), read()))", restored).strip() == "(1, 1)"
    )
    assert (
        execute_in_namespace("print(np.random.random(3).tolist())", restored)
        == expected_random
    )


def test_local_session_supports_initial_state_updates_and_exports():
    with PythonREPLSession(
        {"values": [1, 2, 3], "status": "initial"},
        export_names=("status",),
    ) as session:
        assert session.execute("print(sum(values))") == "6\n"
        session.execute("total = sum(values)", {"status": "updated"})
        checkpoint = session.snapshot()
        assert session.exports() == {"status": "updated"}
        session.restore(checkpoint)
        assert session.execute("print((total, status))") == "(6, 'updated')\n"


def test_repl_tool_factory_is_serial_and_controller_managed():
    repl = create_python_repl_tool(argument_name="code")
    assert repl.trusted is True
    assert repl.concurrency == ToolConcurrency.SERIAL
    assert repl.workspace_access == WorkspaceAccess.NONE
    assert repl.hidden_args == {}
    assert repl.params_json_schema["required"] == ["code"]


def test_restricted_dispatch_sends_only_public_json_and_opaque_checkpoint(
    monkeypatch, tmp_path
):
    requests = []

    def run_worker(kind, payload, workspace, **kwargs):
        worker_tool, arguments = payload
        requests.append((kind, worker_tool, arguments, workspace, kwargs))
        return {
            "content": json.dumps(
                {
                    "output": "6\n",
                    "checkpoint": "replacement",
                    "status_export": "updated",
                }
            )
        }

    monkeypatch.setattr(permissions, "run_worker", run_worker)
    result = execute_python_repl(
        code="values.sum()",
        initial_data={"values": [1, 2, 3]},
        namespace_updates={"status": "updated"},
        synchronized_names=("status",),
        checkpoint="opaque-untrusted-checkpoint",
        workspace=str(tmp_path),
        namespace_factory=_scientific_namespace,
        export_names=("status",),
        export_result_names={"status": "status_export"},
    )

    assert result.output == "6\n"
    assert result.checkpoint == "replacement"
    assert result.exports == {"status": "updated"}
    kind, worker_tool, arguments, workspace, kwargs = requests[0]
    assert kind == "tool"
    assert not worker_tool.trusted
    assert not worker_tool.hidden_args
    assert workspace == str(tmp_path)
    assert kwargs["cancel"] is None
    assert kwargs["workspace_access"] == "none"
    assert arguments == {
        "code": "values.sum()",
        "public_data": json.dumps({"values": [1, 2, 3], "status": "updated"}),
        "checkpoint": "opaque-untrusted-checkpoint",
    }
