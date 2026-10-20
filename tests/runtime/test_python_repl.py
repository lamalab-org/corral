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


def test_checkpoint_preserves_regex_matches_closed_handles_and_recursive_functions(
    tmp_path,
):
    namespace = create_python_namespace({"path": str(tmp_path / "sample.txt")})
    execute_in_namespace(
        """import re
matches = list(re.finditer(r'(?P<letter>[ab])', 'aba'))
with open(path, 'w') as handle:
    handle.write('saved')
def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)
""",
        namespace,
    )
    for _ in range(3):
        namespace = restore_namespace(snapshot_namespace(namespace))
        assert (
            execute_in_namespace(
                "print([(m.span(), m.groupdict(), m.pos) for m in matches])", namespace
            ).strip()
            == "[((0, 1), {'letter': 'a'}, 0), ((1, 2), {'letter': 'b'}, 0), ((2, 3), {'letter': 'a'}, 0)]"
        )
        assert namespace["handle"].closed
        assert namespace["handle"].name == str(tmp_path / "sample.txt")
        assert execute_in_namespace("print(factorial(6))", namespace).strip() == "720"


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


def test_checkpoint_preserves_incomplete_dynamic_pydantic_models():
    namespace = create_python_namespace({})
    execute_in_namespace(
        """import types
module = types.ModuleType('unregistered_models')
exec('from __future__ import annotations\\n'
     'from pydantic import BaseModel\\n'
     'class Record(BaseModel):\\n'
     '    value: MissingType\\n', module.__dict__)
counter = 41
""",
        namespace,
    )
    from pydantic._internal._mock_val_ser import MockValSer

    assert isinstance(namespace["module"].Record.__pydantic_validator__, MockValSer)
    for _ in range(3):
        namespace = restore_namespace(snapshot_namespace(namespace))
        assert execute_in_namespace("print(counter + 1)", namespace) == "42\n"
    # Both deferred helpers must remain usable after the referenced type is
    # supplied; preserving a broken placeholder would silently lose state.
    model = namespace["module"].Record
    model.model_rebuild(_types_namespace={"MissingType": int})
    instance = model.model_validate({"value": "7"})
    assert instance.model_dump() == {"value": 7}


def test_repl_retains_blas_limits_without_exposing_credentials(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "private-test-value")
    with PythonREPLSession({}, address_space_bytes=8 * 1024**3) as session:
        output = session.execute(
            "import os\nfrom scipy import linalg, show_config\n"
            "from threadpoolctl import threadpool_info\n"
            "print({'threads': os.environ.get('OPENBLAS_NUM_THREADS'), "
            "'omp_threads': os.environ.get('OMP_NUM_THREADS'), "
            "'veclib_threads': os.environ.get('VECLIB_MAXIMUM_THREADS'), "
            "'secret': 'ANTHROPIC_API_KEY' in os.environ, "
            "'blas': show_config(mode='dicts')['Build Dependencies']['blas']['name'], "
            "'pools': [p['num_threads'] for p in threadpool_info() "
            "if p['internal_api'] == 'openblas']})"
        )
    import ast

    result = ast.literal_eval(output)
    assert result["threads"] == "1"
    assert result["omp_threads"] == "1"
    assert result["veclib_threads"] == "1"
    assert not result["secret"]
    # macOS wheels can use Accelerate, which threadpoolctl does not report.
    if "openblas" in result["blas"].lower():
        assert result["pools"]
    assert all(count == 1 for count in result["pools"])


def test_repl_tool_factory_is_serial_and_controller_managed():
    repl = create_python_repl_tool(argument_name="code")
    assert repl.trusted is False
    assert repl.controller_dispatch is True
    assert repl.concurrency == ToolConcurrency.SERIAL
    assert repl.workspace_access == WorkspaceAccess.NONE
    assert repl.hidden_args == {}
    assert repl.params_json_schema["required"] == ["code"]


def test_restricted_dispatch_sends_only_public_json_and_opaque_checkpoint(
    monkeypatch, tmp_path
):
    requests = []

    def run_worker_with_bulk(kind, payload, workspace, **kwargs):
        worker_tool, arguments = payload
        requests.append((kind, worker_tool, arguments, workspace, kwargs))
        # The checkpoint arrives on the bulk channel, not inside the JSON reply.
        return (
            {"content": json.dumps({"output": "6\n", "status_export": "updated"})},
            {"checkpoint": b"replacement"},
        )

    monkeypatch.setattr(permissions, "run_worker_with_bulk", run_worker_with_bulk)
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
