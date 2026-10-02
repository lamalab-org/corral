"""Workspace, Python, and model-syntax tools for agents."""

from __future__ import annotations

import ast
import io
import os
import traceback
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

import numpy as np

from corral.core.tool import Tool, tool
from corral.runtime.python_repl import Namespace, create_python_namespace
from corral.tools.python_repl import create_python_repl_tool
from corral.workspace import WorkspaceFilesystem, build_workspace_tools

MAX_CODE_CHARS = 50_000
MAX_OUTPUT_CHARS = 10_000

#: How long one PythonREPL call may run. Override per run with the
#: `repl_timeout_minutes` environment argument.
DEFAULT_REPL_TIMEOUT_MINUTES = 15.0


def repl_description(timeout_minutes: float | None) -> str:
    """The REPL's tool description, which states its time limit, if any."""
    text = (
        "Execute Python code in the task's persistent public-data-only session. "
        "The working directory is the task workspace, so open files by relative "
        "path, e.g. 'data.csv'."
    )
    if timeout_minutes is None:
        return text
    return (
        f"{text} Each call may run for at most {timeout_minutes:g} minutes; a call "
        "that runs longer is stopped and the session returns to its state before it."
    )


def timeout_message(timeout_minutes: float) -> str:
    """What the agent sees when a call is stopped at the time limit."""
    return (
        f"Stopped: this call ran longer than the {timeout_minutes:g}-minute limit and "
        "produced no result. The session is back to its state before the call, so "
        "variables defined earlier are still available; anything this call created "
        "or changed is not."
    )


def _namespace(initial_data: Mapping[str, Any]) -> Namespace:
    """Start a session with NumPy already imported."""
    namespace = create_python_namespace(initial_data)
    namespace["__name__"] = "__psychometrics_session__"
    namespace["np"] = np
    return namespace


def _run_cell(code: str, namespace: Namespace) -> str:
    """Execute one cell, echoing a trailing expression the way a notebook does."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("input_code must be non-empty Python code")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError(f"input_code is limited to {MAX_CODE_CHARS:,} characters")

    output = io.StringIO()
    try:
        tree = ast.parse(code, mode="exec")
        last = tree.body[-1] if tree.body else None
        with redirect_stdout(output):
            if isinstance(last, ast.Expr):
                tree.body = tree.body[:-1]
                if tree.body:
                    exec(compile(tree, "<psychometrics-repl>", "exec"), namespace)
                value = eval(
                    compile(ast.Expression(last.value), "<psychometrics-repl>", "eval"),
                    namespace,
                )
                if value is not None:
                    print(repr(value), file=output)
            else:
                exec(compile(tree, "<psychometrics-repl>", "exec"), namespace)
    except BaseException:
        traceback.print_exc(file=output)

    text = output.getvalue()
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n... output truncated"
    return text or "No output."


@dataclass(frozen=True)
class _CellExecutor:
    """Runs cells with the task's workspace as the working directory.

    Must stay picklable: the local session sends it to a spawned process.
    """

    workspace: str

    def __call__(self, code: str, namespace: Namespace) -> str:
        # A restricted worker already starts in its /workspace mount, where the
        # controller's path does not exist; a local session starts elsewhere.
        if os.path.isdir(self.workspace):
            os.chdir(self.workspace)
        return _run_cell(code, namespace)


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


def workspace_tools(
    workspace: str, repl_timeout_minutes: float | None = DEFAULT_REPL_TIMEOUT_MINUTES
) -> dict[str, Tool]:
    """Every tool a psychometrics task exposes, bound to one workspace.

    `repl_timeout_minutes` limits each PythonREPL call; None removes the limit.
    """
    files = build_workspace_tools(WorkspaceFilesystem(workspace))
    repl = create_python_repl_tool(
        name="PythonREPL",
        description=repl_description(repl_timeout_minutes),
        argument_name="input_code",
        argument_description="A valid Python command.",
        namespace_factory=_namespace,
        code_executor=_CellExecutor(workspace),
        max_code_chars=MAX_CODE_CHARS,
        max_output_chars=MAX_OUTPUT_CHARS,
        # Cells read the task data and may save intermediate files; nothing
        # they need is online.
        workspace_access="read_write",
        network_access="none",
        timeout_seconds=None if repl_timeout_minutes is None else repl_timeout_minutes * 60,
    )
    return {
        name: files[name]
        for name in (
            "list_files",
            "read_file",
            "write_file",
            "file_info",
            "cat_files",
            "grep",
        )
    } | {
        "PythonREPL": repl,
        "validate_model_syntax": validate_model_syntax,
    }
