"""Agent-facing tools for the psychometrics environment.

The analysis session persists between calls and runs in Corral's restricted
worker when the environment is containerized.
"""

from __future__ import annotations

import ast
import base64
import builtins
import io
import multiprocessing as mp
import os
import traceback
from contextlib import redirect_stdout
from typing import Any

import cloudpickle
import numpy as np

from corral.core.tool import Tool, tool
from corral.workspace import WorkspaceFilesystem, build_workspace_tools

MAX_CODE_CHARS = 50_000
MAX_OUTPUT_CHARS = 10_000
MAX_LOCAL_SECONDS = 600


def _initial_namespace() -> dict[str, Any]:
    """Create the public namespace for a new analysis session."""
    return {
        "__builtins__": vars(builtins).copy(),
        "__name__": "__psychometrics_session__",
        "np": np,
    }


def _snapshot(namespace: dict[str, Any]) -> str:
    """Serialize session state as an opaque value for the controller."""
    return base64.b64encode(cloudpickle.dumps(namespace)).decode("ascii")


def _restore(checkpoint: str | None) -> dict[str, Any]:
    if checkpoint is None:
        return _initial_namespace()
    if not isinstance(checkpoint, str):
        raise ValueError("analysis checkpoint must be a string or null")
    namespace = cloudpickle.loads(base64.b64decode(checkpoint, validate=True))
    if not isinstance(namespace, dict) or "__builtins__" not in namespace:
        raise ValueError("invalid analysis checkpoint")
    return namespace


def _execute_code(code: str, checkpoint: str | None) -> dict[str, Any]:
    """Execute one REPL cell and return text plus the next checkpoint."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("input_code must be non-empty Python code")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError(f"input_code is limited to {MAX_CODE_CHARS:,} characters")

    namespace = _restore(checkpoint)
    output = io.StringIO()
    result: Any = None
    try:
        tree = ast.parse(code, mode="exec")
        last = tree.body[-1] if tree.body else None
        if isinstance(last, ast.Expr):
            tree.body = tree.body[:-1]
            with redirect_stdout(output):
                if tree.body:
                    exec(compile(tree, "<psychometrics-repl>", "exec"), namespace)
                result = eval(
                    compile(ast.Expression(last.value), "<psychometrics-repl>", "eval"),
                    namespace,
                )
            if result is not None:
                print(repr(result), file=output)
        else:
            with redirect_stdout(output):
                exec(compile(tree, "<psychometrics-repl>", "exec"), namespace)
    except BaseException:
        traceback.print_exc(file=output)

    text = output.getvalue()
    truncated = len(text) > MAX_OUTPUT_CHARS
    if truncated:
        text = text[:MAX_OUTPUT_CHARS] + "\n... output truncated"
    return {
        "output": text or "No output.",
        "checkpoint": _snapshot(namespace),
        "truncated": truncated,
    }


@tool
def _analysis_step(code: str, checkpoint: str | None = None) -> dict[str, Any]:
    """Execute one public analysis cell inside a restricted worker."""
    return _execute_code(code, checkpoint)


def _local_worker(connection: Any, code: str, checkpoint: str | None) -> None:
    try:
        connection.send({"ok": True, "result": _execute_code(code, checkpoint)})
    except BaseException:
        connection.send({"ok": False, "error": traceback.format_exc(limit=8)})
    finally:
        connection.close()


def local_analysis_step(code: str, checkpoint: str | None) -> dict[str, Any]:
    """Development fallback when Docker permission workers are unavailable."""
    context = mp.get_context("fork" if os.name == "posix" else "spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_local_worker,
        args=(child, code, checkpoint),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(MAX_LOCAL_SECONDS):
            process.kill()
            raise TimeoutError(f"analysis timed out after {MAX_LOCAL_SECONDS} seconds")
        response = parent.recv()
    finally:
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "analysis worker failed"))
    return response["result"]


@tool
def PythonREPL(input_code: str) -> str:  # noqa: N802 - public tool name follows Stargazer
    """Execute Python code in the task's persistent public-data-only session."""
    raise RuntimeError("PythonREPL must be dispatched by PsychometricsEnvironment")


@tool
def validate_model_syntax(syntax: str) -> dict[str, Any]:
    """Check semopy model syntax without fitting a model or reading task data."""
    if not isinstance(syntax, str) or not syntax.strip():
        return {"valid": False, "error": "syntax must be a non-empty string"}
    try:
        from semopy import Model

        Model(syntax)
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"valid": True, "error": None}


def workspace_tools(workspace: str) -> dict[str, Tool]:
    """Return the file tools and persistent REPL available to every task."""
    tools = build_workspace_tools(WorkspaceFilesystem(workspace))
    return {
        name: tools[name]
        for name in (
            "list_files",
            "read_file",
            "write_file",
            "file_info",
            "cat_files",
            "grep",
        )
    } | {
        "PythonREPL": PythonREPL,
        "validate_model_syntax": validate_model_syntax,
    }
