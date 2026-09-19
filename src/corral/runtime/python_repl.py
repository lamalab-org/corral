"""Checkpointed Python REPL execution for stateful Corral environments.

The controller treats checkpoints as opaque strings.  In Docker trials they
are decoded only by an unprivileged worker, after Corral has entered the trial
workspace and dropped its OS identity.
"""

from __future__ import annotations

import ast
import base64
import builtins
import io
import json
import multiprocessing as mp
import os
import re
import tempfile
import threading
import traceback
import types
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np

from corral.core.tool import Tool
from corral.runtime import permissions

if TYPE_CHECKING:
    from typing_extensions import Self

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None  # type: ignore[assignment]

DEFAULT_MAX_OUTPUT_CHARS = 5_000 + len("...(output truncated)")
DEFAULT_MAX_CODE_CHARS = 50_000
DEFAULT_WORKER_ADDRESS_SPACE_BYTES = 2 * 1024**3
_WORKER_ADDRESS_SPACE_HEADROOM_BYTES = 512 * 1024**2
_NO_OUTPUT = (
    "No output. You likely forgot to print the result. "
    "Please use `print(...)` to see any output."
)

Namespace = dict[str, Any]
NamespaceFactory = Callable[[Mapping[str, Any]], Namespace]
CodeExecutor = Callable[[str, Namespace], str]


def _session_builtins() -> dict[str, Any]:
    """Return a detached builtin namespace for one REPL."""
    return dict(vars(builtins))


def create_python_namespace(initial_data: Mapping[str, Any]) -> Namespace:
    """Create the default namespace, injecting optional JSON-safe bindings."""
    namespace = {
        "__builtins__": _session_builtins(),
        "__name__": "__corral_repl__",
    }
    namespace.update(
        {
            name: value
            for name, value in initial_data.items()
            if name not in {"__builtins__", "__name__"}
        }
    )
    return namespace


def _materialize_namespace(
    initial_data: Mapping[str, Any], namespace_factory: NamespaceFactory | None
) -> Namespace:
    factory = namespace_factory or create_python_namespace
    namespace = factory(initial_data)
    if not isinstance(namespace, dict):
        raise TypeError("Python REPL namespace factory must return a dict")
    namespace.setdefault("__name__", "__corral_repl__")
    session_builtins = namespace.get("__builtins__")
    if isinstance(session_builtins, types.ModuleType):
        session_builtins = dict(vars(session_builtins))
    elif isinstance(session_builtins, Mapping):
        session_builtins = dict(session_builtins)
    else:
        session_builtins = _session_builtins()
    for name, value in vars(builtins).items():
        session_builtins.setdefault(name, value)
    namespace["__builtins__"] = session_builtins
    return namespace


def _restore_session_function(
    code: types.CodeType,
    namespace: Namespace,
    session_builtins: dict[str, Any],
    closure: tuple[types.CellType, ...] | None,
) -> types.FunctionType:
    """Restore a function while retaining its shared REPL globals."""
    for name, value in vars(builtins).items():
        session_builtins.setdefault(name, value)
    namespace["__builtins__"] = session_builtins
    return types.FunctionType(code, namespace, closure=closure)


def _restore_function_state(function: types.FunctionType, state: tuple) -> None:
    attributes, closure = state
    for name, value in attributes.items():
        setattr(function, name, value)
    if closure is not None:
        for target, source in zip(function.__closure__, closure, strict=True):
            with suppress(ValueError):  # A closure cell can be empty.
                target.cell_contents = source.cell_contents


class _SessionPickler(cloudpickle.CloudPickler):
    """Keep functions attached to the restored REPL namespace."""

    def __init__(self, file: io.BytesIO, namespace: Namespace):
        super().__init__(file)
        self.namespace = namespace
        self.closure_cells: dict[int, types.CellType] = {}

    def reducer_override(self, value: Any) -> Any:
        if isinstance(value, types.FunctionType) and (
            value.__globals__ is self.namespace
            # Checkpoints created by the original Stargazer implementation
            # used a copied globals dict and identified it by this module name.
            or value.__globals__.get("__name__") == "__stargazer_session__"
        ):
            attributes = {
                name: getattr(value, name)
                for name in (
                    "__name__",
                    "__qualname__",
                    "__defaults__",
                    "__kwdefaults__",
                    "__annotations__",
                    "__dict__",
                    "__module__",
                    "__doc__",
                )
            }
            # Shared nonlocal variables need shared cells. Empty placeholders
            # allow a closure to refer recursively to its own function.
            closure = (
                tuple(
                    self.closure_cells.setdefault(id(cell), types.CellType())
                    for cell in value.__closure__
                )
                if value.__closure__ is not None
                else None
            )
            return (
                _restore_session_function,
                (
                    value.__code__,
                    value.__globals__,
                    value.__globals__["__builtins__"],
                    closure,
                ),
                (attributes, value.__closure__),
                None,
                None,
                _restore_function_state,
            )
        return super().reducer_override(value)


def snapshot_namespace(namespace: Namespace) -> str:
    """Serialize a REPL namespace and NumPy's legacy module RNG state."""
    output = io.BytesIO()
    random_state = np.random.get_state()  # noqa: NPY002
    _SessionPickler(output, namespace).dump((namespace, random_state))
    return base64.b64encode(output.getvalue()).decode("ascii")


def restore_namespace(checkpoint: str) -> Namespace:
    """Restore a namespace from an opaque checkpoint.

    Checkpoints are pickle payloads and therefore untrusted.  Controller code
    must not call this function for model-influenced state; use
    `execute_python_repl` so restoration happens after privilege drop.
    """
    restored = cloudpickle.loads(base64.b64decode(checkpoint, validate=True))
    if not isinstance(restored, tuple) or len(restored) != 2:
        raise ValueError("Invalid Python REPL checkpoint")
    namespace, random_state = restored
    if not isinstance(namespace, dict):
        raise ValueError("Invalid Python REPL namespace")
    np.random.set_state(random_state)  # noqa: NPY002
    return namespace


def sanitize_input(query: str) -> str:
    """Remove common Markdown fencing and expand unquoted `\\n` text."""
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    query = re.sub(r"(\s|`)*$", "", query)
    result: list[str] = []
    index = 0
    string_char: str | None = None
    while index < len(query):
        character = query[index]
        if string_char is None:
            if character in "\"'":
                string_char = character
                result.append(character)
            elif query[index : index + 2] == "\\n":
                result.append("\n")
                index += 1
            else:
                result.append(character)
        elif character == "\\" and index + 1 < len(query):
            result.append(query[index : index + 2])
            index += 1
        elif character == string_char:
            string_char = None
            result.append(character)
        else:
            result.append(character)
        index += 1
    return "".join(result)


def _is_bare_expression(source: str) -> bool:
    try:
        ast.parse(source, mode="eval")
    except SyntaxError:
        return False
    return True


def wrap_last_line_with_print(code: str) -> str:
    """Echo a final simple expression while leaving statements untouched."""
    lines = code.strip().split("\n")
    last_line = lines[-1].strip()
    if re.match(r"^[^\s,()]+$", last_line) and _is_bare_expression(last_line):
        indent = lines[-1][: len(lines[-1]) - len(lines[-1].lstrip())]
        lines[-1] = f"{indent}print({last_line})"
    return "\n".join(lines)


def _format_execution_error(error: Exception, code: str) -> str:
    result = "Error Traceback:\n"
    lines = code.split("\n")
    for frame in traceback.extract_tb(error.__traceback__):
        if frame.filename == "<string>":
            result += f"  line {frame.lineno}:\n"
            if 0 < frame.lineno <= len(lines):
                result += f"    {lines[frame.lineno - 1].strip()}\n"
    return result + f"{type(error).__name__}: {error}"


class _CappedOutput(io.StringIO):
    def __init__(self, maximum: int):
        super().__init__()
        self.maximum = maximum
        self.truncated = False

    def write(self, value: str) -> int:
        remaining = self.maximum - self.tell()
        if remaining > 0:
            super().write(value[:remaining])
        self.truncated |= len(value) > remaining
        return len(value)


def execute_in_namespace(
    code: str,
    namespace: Namespace,
    *,
    preloaded_names: Sequence[str] = (),
    max_output_chars: int = 5_000,
) -> str:
    """Execute one cell and retain all resulting namespace mutations."""
    cleaned = wrap_last_line_with_print(sanitize_input(code))
    output = _CappedOutput(max_output_chars)
    session_builtins = namespace.get("__builtins__")
    if not isinstance(session_builtins, dict):
        session_builtins = _session_builtins()
        namespace["__builtins__"] = session_builtins
    else:
        for name, value in vars(builtins).items():
            session_builtins.setdefault(name, value)
    result: str | None = None
    try:
        with redirect_stdout(output):
            exec(cleaned, namespace)
    except ModuleNotFoundError as exc:
        if exc.name in preloaded_names:
            name = exc.name
            result = (
                f"ModuleNotFoundError: No module named '{name}'\n\n"
                f"HINT: `{name}` is a PRE-LOADED VARIABLE, not a module.\n"
                "Do NOT import it. Just use it directly:\n"
                f"  CORRECT: print({name})\n"
                f"  WRONG:   from {name} import {name}"
            )
        else:
            result = _format_execution_error(exc, cleaned)
    except Exception as exc:
        result = _format_execution_error(exc, cleaned)
    if result is None:
        result = output.getvalue()
        if output.truncated:
            result += "...(output truncated)"
    elif len(result) > max_output_chars:
        result = result[:max_output_chars] + "...(output truncated)"
    return result or _NO_OUTPUT


def apply_worker_resource_limits(
    address_space_bytes: int = DEFAULT_WORKER_ADDRESS_SPACE_BYTES,
) -> None:
    """Bound pathological allocations where POSIX address-space limits exist."""
    if resource is None or address_space_bytes <= 0:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = address_space_bytes
        # BLAS-backed wheels reserve several GiB of virtual address space while
        # using far less resident memory. Never install a limit below the
        # already-mapped runtime or ordinary SciPy imports will fail.
        try:
            pages = int(Path("/proc/self/statm").read_text().split()[0])
            current = pages * int(os.sysconf("SC_PAGE_SIZE"))
            limit = max(limit, current + _WORKER_ADDRESS_SPACE_HEADROOM_BYTES)
        except (FileNotFoundError, OSError, ValueError):
            pass
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        if soft == resource.RLIM_INFINITY or soft > limit:
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (OSError, ValueError):
        return


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Python REPL {label} must be JSON-serializable") from exc


def _export_namespace(namespace: Namespace, names: Sequence[str]) -> dict[str, Any]:
    exports = {name: namespace.get(name) for name in names}
    return _json_copy(exports, label="exports")


class PythonREPLSession:
    """Local handle for an isolated, checkpointable Python worker process.

    This path is intended for local development.  Docker environments should
    use `execute_python_repl`, which restores checkpoints only after the
    worker has dropped privileges.
    """

    def __init__(
        self,
        initial_data: Mapping[str, Any] | None = None,
        *,
        namespace_factory: NamespaceFactory | None = None,
        code_executor: CodeExecutor | None = None,
        export_names: Sequence[str] = (),
        max_code_chars: int = DEFAULT_MAX_CODE_CHARS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        address_space_bytes: int = DEFAULT_WORKER_ADDRESS_SPACE_BYTES,
    ):
        self._initial_data = _json_copy(initial_data or {}, label="initial data")
        self._namespace_factory = namespace_factory
        self._code_executor = code_executor or execute_in_namespace
        self._export_names = tuple(export_names)
        self._max_code_chars = max_code_chars
        self._max_output_chars = max_output_chars
        self._address_space_bytes = address_space_bytes
        self._connection: Any = None
        self._process: Any = None
        self._worker_directory: tempfile.TemporaryDirectory[str] | None = None
        self._lock = threading.RLock()

    def _stop_worker(self) -> None:
        connection, process = self._connection, self._process
        worker_directory = self._worker_directory
        self._connection = None
        self._process = None
        self._worker_directory = None
        if connection is not None:
            with suppress(OSError, ValueError):
                connection.close()
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1.0)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=1.0)
                process.close()
            except (AssertionError, OSError, ValueError):
                pass
        if worker_directory is not None:
            with suppress(OSError, PermissionError):
                worker_directory.cleanup()

    def _start_worker(self) -> None:
        self._stop_worker()
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        worker_directory = tempfile.TemporaryDirectory(prefix="corral-python-repl-")
        process = context.Process(
            target=_python_repl_worker,
            args=(
                child_connection,
                self._initial_data,
                self._namespace_factory,
                self._code_executor,
                self._export_names,
                worker_directory.name,
                self._max_output_chars,
                self._address_space_bytes,
            ),
            daemon=True,
            name="corral-python-repl",
        )
        try:
            process.start()
        except BaseException:
            parent_connection.close()
            child_connection.close()
            worker_directory.cleanup()
            raise
        child_connection.close()
        self._connection = parent_connection
        self._process = process
        self._worker_directory = worker_directory
        try:
            ready = parent_connection.recv()
        except EOFError as exc:
            self._stop_worker()
            raise RuntimeError("The Python REPL worker failed to start") from exc
        if ready != {"status": "ready"}:
            self._stop_worker()
            raise RuntimeError(f"The Python REPL worker failed: {ready!r}")

    def execute(
        self, code: str, namespace_updates: Mapping[str, Any] | None = None
    ) -> str:
        """Execute code, optionally synchronizing public bindings first."""
        if len(code) > self._max_code_chars:
            raise ValueError(
                f"Python REPL code is limited to {self._max_code_chars:,} characters"
            )
        updates = _json_copy(namespace_updates or {}, label="namespace updates")
        return str(
            self._request({"command": "execute", "code": code, "updates": updates})
        )

    def exports(self) -> dict[str, Any]:
        """Read configured small, JSON-compatible values from the worker."""
        if self._process is None:
            return {}
        return dict(self._request({"command": "exports"}))

    def snapshot(self) -> str | None:
        """Return an opaque checkpoint; a stopped worker has no state."""
        with self._lock:
            if self._process is None or not self._process.is_alive():
                return None
            return self._request({"command": "snapshot"})

    def restore(self, checkpoint: str | None) -> None:
        """Restore only inside the worker, never in the calling controller."""
        if checkpoint is not None and not isinstance(checkpoint, str):
            raise ValueError("Python REPL checkpoint must be an opaque string")
        with self._lock:
            self._stop_worker()
            if checkpoint is not None:
                self._request({"command": "restore", "checkpoint": checkpoint})

    def _request(self, request: dict[str, Any]) -> Any:
        with self._lock:
            process = self._process
            if process is None or not process.is_alive():
                self._start_worker()
            connection = self._connection
            assert connection is not None
            try:
                connection.send(request)
                response = connection.recv()
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._stop_worker()
                raise RuntimeError(
                    "The Python REPL worker stopped unexpectedly; "
                    "the persistent session was reset"
                ) from exc
            except BaseException:
                self._stop_worker()
                raise
            if isinstance(response, dict) and "error" in response:
                self._stop_worker()
                raise RuntimeError(response["error"])
            if not isinstance(response, dict) or "result" not in response:
                self._stop_worker()
                raise RuntimeError(
                    "The Python REPL worker returned an invalid response; "
                    "the persistent session was reset"
                )
            return response["result"]

    def close(self) -> None:
        """Release the worker and its private IPC channel."""
        with self._lock:
            connection = self._connection
            process = self._process
            if connection is not None and process is not None and process.is_alive():
                try:
                    connection.send({"command": "close"})
                    process.join(timeout=1.0)
                except (BrokenPipeError, OSError):
                    pass
            self._stop_worker()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self._stop_worker()


def _python_repl_worker(
    connection: Any,
    initial_data: Mapping[str, Any],
    namespace_factory: NamespaceFactory | None,
    code_executor: CodeExecutor,
    export_names: tuple[str, ...],
    working_directory: str,
    max_output_chars: int,
    address_space_bytes: int,
) -> None:
    try:
        os.chdir(working_directory)
        os.environ.clear()
        namespace = _materialize_namespace(initial_data, namespace_factory)
        # NumPy loads the legacy RNG extension lazily. Map it before applying
        # RLIMIT_AS so snapshotting does not fail merely by importing it.
        np.random.get_state()  # noqa: NPY002
        apply_worker_resource_limits(address_space_bytes)
        connection.send({"status": "ready"})
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            if not isinstance(request, dict):
                connection.send({"result": "Invalid Python REPL worker request."})
                continue
            command = request.get("command")
            if command == "close":
                return
            if command in {"snapshot", "restore", "exports"}:
                try:
                    if command == "snapshot":
                        result: Any = snapshot_namespace(namespace)
                    elif command == "restore":
                        namespace = restore_namespace(request["checkpoint"])
                        result = None
                    else:
                        result = _export_namespace(namespace, export_names)
                    connection.send({"result": result})
                except BaseException:
                    connection.send({"error": traceback.format_exc(limit=8)})
                continue
            if command != "execute" or not isinstance(request.get("code"), str):
                connection.send({"result": "Invalid Python REPL worker request."})
                continue
            try:
                namespace.update(request.get("updates") or {})
                result = code_executor(request["code"], namespace)
            except BaseException:
                result = traceback.format_exc(limit=8)
            if len(result) > max_output_chars:
                result = result[:max_output_chars] + "\n... output truncated"
            connection.send({"result": result})
    except BaseException:
        with suppress(BaseException):
            connection.send({"status": "error", "error": traceback.format_exc(limit=8)})
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class PythonREPLResult:
    """One transactional REPL execution and its replacement checkpoint."""

    output: str
    checkpoint: str
    exports: Mapping[str, Any]


class _RestrictedPythonREPLTool(Tool):
    """Internal tool whose `execute` method runs only after privilege drop."""

    def __init__(
        self,
        *,
        namespace_factory: NamespaceFactory | None,
        code_executor: CodeExecutor,
        synchronized_names: tuple[str, ...],
        export_names: tuple[str, ...],
        export_result_names: Mapping[str, str],
        max_output_chars: int,
        address_space_bytes: int,
    ):
        super().__init__(
            name="_corral_python_repl_step",
            description="Internal restricted Python REPL execution step.",
            params_json_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "public_data": {"type": "string"},
                    "checkpoint": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["code", "public_data", "checkpoint"],
            },
        )
        self.namespace_factory = namespace_factory
        self.code_executor = code_executor
        self.synchronized_names = synchronized_names
        self.export_names = export_names
        self.export_result_names = dict(export_result_names)
        self.max_output_chars = max_output_chars
        self.address_space_bytes = address_space_bytes

    def execute(self, *, code: str, public_data: str, checkpoint: str | None) -> str:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            raise RuntimeError("Docker Python REPL must run as an unprivileged worker")
        os.environ.clear()
        np.random.get_state()  # noqa: NPY002 - preload before RLIMIT_AS
        apply_worker_resource_limits(self.address_space_bytes)
        data = json.loads(public_data)
        if not isinstance(data, dict):
            raise ValueError("Python REPL public data must be a JSON object")
        namespace = (
            _materialize_namespace(data, self.namespace_factory)
            if checkpoint is None
            else restore_namespace(checkpoint)
        )
        if checkpoint is not None:
            namespace.update(
                {name: data[name] for name in self.synchronized_names if name in data}
            )
        try:
            output = self.code_executor(code, namespace)
        except BaseException:
            output = traceback.format_exc(limit=8)
        exports = _export_namespace(namespace, self.export_names)
        result: dict[str, Any] = {
            "output": output[: self.max_output_chars],
            "checkpoint": snapshot_namespace(namespace),
        }
        if self.export_result_names:
            result.update(
                {
                    result_name: exports[namespace_name]
                    for namespace_name, result_name in self.export_result_names.items()
                }
            )
        else:
            result["exports"] = exports
        return json.dumps(result)


def execute_python_repl(
    *,
    code: str,
    workspace: str,
    initial_data: Mapping[str, Any] | None = None,
    checkpoint: str | None = None,
    namespace_factory: NamespaceFactory | None = None,
    code_executor: CodeExecutor | None = None,
    namespace_updates: Mapping[str, Any] | None = None,
    synchronized_names: Sequence[str] = (),
    export_names: Sequence[str] = (),
    export_result_names: Mapping[str, str] | None = None,
    cancel: threading.Event | None = None,
    max_code_chars: int = DEFAULT_MAX_CODE_CHARS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    address_space_bytes: int = DEFAULT_WORKER_ADDRESS_SPACE_BYTES,
) -> PythonREPLResult:
    """Execute and checkpoint Python in Corral's restricted worker.

    `initial_data` and `namespace_updates` must contain public JSON values.
    Initial data is materialized only when no checkpoint exists; on restoration,
    only bindings named by `synchronized_names` are refreshed.
    """
    if len(code) > max_code_chars:
        raise ValueError(
            f"Python REPL code is limited to {max_code_chars:,} characters"
        )
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise ValueError("Python REPL checkpoint must be an opaque string")
    unexpected_updates = set(namespace_updates or {}) - set(synchronized_names)
    if unexpected_updates:
        raise ValueError(
            "Python REPL namespace updates are not configured for synchronization: "
            f"{sorted(unexpected_updates)!r}"
        )
    public_data = dict(initial_data or {})
    public_data.update(namespace_updates or {})
    public_data = _json_copy(public_data, label="public data")
    result_names = dict(export_result_names or {})
    if len(set(export_names)) != len(tuple(export_names)):
        raise ValueError("Python REPL export names must be unique")
    if result_names and set(result_names) != set(export_names):
        raise ValueError("export_result_names must map every configured export name")
    if len(set(result_names.values())) != len(result_names) or {
        "output",
        "checkpoint",
        "exports",
    } & set(result_names.values()):
        raise ValueError(
            "Python REPL export result names must be unique and unreserved"
        )
    worker_tool = _RestrictedPythonREPLTool(
        namespace_factory=namespace_factory,
        code_executor=code_executor or execute_in_namespace,
        synchronized_names=tuple(synchronized_names),
        export_names=tuple(export_names),
        export_result_names=result_names,
        max_output_chars=max_output_chars,
        address_space_bytes=address_space_bytes,
    )
    response = permissions.run_worker(
        "tool",
        (
            worker_tool,
            {
                "code": code,
                "public_data": json.dumps(public_data),
                "checkpoint": checkpoint,
            },
        ),
        workspace,
        cancel=cancel,
    )["content"]
    result = json.loads(response)
    expected = (
        {"output", "checkpoint", *result_names.values()}
        if result_names
        else {"output", "checkpoint", "exports"}
    )
    if (
        not isinstance(result, dict)
        or set(result) != expected
        or not isinstance(result.get("output"), str)
        or not isinstance(result.get("checkpoint"), str)
    ):
        raise RuntimeError("Invalid Python REPL worker result")
    exports = (
        {
            namespace_name: result[result_name]
            for namespace_name, result_name in result_names.items()
        }
        if result_names
        else result["exports"]
    )
    if not isinstance(exports, dict) or set(exports) != set(export_names):
        raise RuntimeError("Invalid Python REPL worker exports")
    return PythonREPLResult(
        output=result["output"], checkpoint=result["checkpoint"], exports=exports
    )


__all__ = [
    "DEFAULT_MAX_CODE_CHARS",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_WORKER_ADDRESS_SPACE_BYTES",
    "CodeExecutor",
    "Namespace",
    "NamespaceFactory",
    "PythonREPLResult",
    "PythonREPLSession",
    "apply_worker_resource_limits",
    "create_python_namespace",
    "execute_in_namespace",
    "execute_python_repl",
    "restore_namespace",
    "sanitize_input",
    "snapshot_namespace",
    "wrap_last_line_with_print",
]
