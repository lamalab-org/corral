"""Compatibility adapter for Stargazer's former task-local REPL executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from corral.runtime.python_repl import execute_python_repl
from stargazer.protocol import public_resources
from stargazer.tools import _execute_persistent, worker_namespace_factory

if TYPE_CHECKING:
    import threading


def execute_analysis(
    *,
    code: str,
    public_data: dict[str, Any],
    checkpoint: str | None,
    workspace: str,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Execute through Corral's generic restricted, checkpointed REPL."""
    history = public_data.get("history")
    initial_data = {
        key: value for key, value in public_data.items() if key != "history"
    }
    if initial_data.get("analysis_assistance", True):
        initial_data.setdefault("public_resources", public_resources())
    result = execute_python_repl(
        code=code,
        initial_data=initial_data,
        namespace_updates={"history": history} if history is not None else None,
        synchronized_names=("history",),
        checkpoint=checkpoint,
        workspace=workspace,
        namespace_factory=worker_namespace_factory(
            initial_data.get("analysis_assistance", True)
        ),
        code_executor=_execute_persistent,
        export_names=("_protocol_guide_ack",),
        export_result_names={"_protocol_guide_ack": "protocol_ack"},
        workspace_access="read_write",
        network_access="none",
        cancel=cancel,
    )
    return {
        "output": result.output,
        "checkpoint": result.checkpoint,
        "protocol_ack": bool(result.exports["_protocol_guide_ack"]),
    }
