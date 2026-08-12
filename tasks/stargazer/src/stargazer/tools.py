"""Agent tools for iterative Stargazer analysis and model submission."""

from __future__ import annotations

import ast
import builtins
import io
import json
import multiprocessing as mp
import os
import sys
import sysconfig
import tempfile
import threading
import traceback
import types
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from scipy import optimize, signal

from corral.backend.tool import Tool, tool
from stargazer.evaluator import SubmissionError, evaluate_submission
from stargazer.models import (
    CandidatePlanet,
    CandidateSubmission,
    StargazerTask,
    mass_from_semi_amplitude,
)

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Callable

_MAX_OUTPUT_CHARS = 12_000
_MAX_CODE_CHARS = 50_000
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 10.0
_WORKER_START_TIMEOUT_SECONDS = 30.0
_WORKER_ADDRESS_SPACE_BYTES = 2 * 1024**3
_ALLOWED_IMPORTS = {
    "collections",
    "functools",
    "itertools",
    "json",
    "math",
    "numpy",
    "scipy",
    "statistics",
}
_SAFE_BUILTIN_NAMES = {
    "ArithmeticError",
    "AssertionError",
    "Exception",
    "IndexError",
    "KeyError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "abs",
    "all",
    "any",
    "bool",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hex",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_BLOCKED_MODULE_ATTRIBUTES = {
    "DataSource",
    "ctypes",
    "ctypeslib",
    "distutils",
    "dump",
    "dumps",
    "f2py",
    "fromfile",
    "genfromtxt",
    "io",
    "load",
    "load_library",
    "loads",
    "loadtxt",
    "memmap",
    "open_memmap",
    "popen",
    "save",
    "savetxt",
    "source",
    "system",
    "tofile",
}


class UnsafeAnalysisCode(ValueError):
    """Raised when REPL code attempts unsupported introspection or I/O."""


class AnalysisTimeoutError(TimeoutError):
    """Raised when an analysis worker exceeds its per-call time budget."""


class _SafeModuleProxy:
    """Expose numerical module attributes while denying I/O escape hatches."""

    __slots__ = ("__wrapped_module",)

    def __init__(self, module: types.ModuleType):
        object.__setattr__(self, "_SafeModuleProxy__wrapped_module", module)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_") or name in _BLOCKED_MODULE_ATTRIBUTES:
            raise AttributeError(f"Module attribute {name!r} is unavailable")
        module = object.__getattribute__(self, "_SafeModuleProxy__wrapped_module")
        value = getattr(module, name)
        return _safe_module(value)

    def __repr__(self) -> str:
        module = object.__getattribute__(self, "_SafeModuleProxy__wrapped_module")
        return f"<restricted module {module.__name__!r}>"


_MODULE_PROXIES: dict[str, _SafeModuleProxy] = {}


def _safe_module(value: Any) -> Any:
    if not isinstance(value, types.ModuleType):
        return value
    proxy = _MODULE_PROXIES.get(value.__name__)
    if proxy is None:
        proxy = _SafeModuleProxy(value)
        _MODULE_PROXIES[value.__name__] = proxy
    return proxy


class AnalysisSession:
    """Handle for a persistent, public-data-only analysis worker process."""

    def __init__(
        self,
        public_data: dict[str, Any],
        *,
        execution_timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    ):
        if execution_timeout_seconds <= 0.0:
            raise ValueError("execution_timeout_seconds must be positive")
        self._public_data = public_data
        self._execution_timeout_seconds = float(execution_timeout_seconds)
        self._connection: Any = None
        self._process: Any = None
        self._worker_directory: tempfile.TemporaryDirectory[str] | None = None
        self._lock = threading.RLock()

    def __deepcopy__(self, memo: dict[int, Any]) -> AnalysisSession:
        # Corral drops hidden arguments from snapshots immediately after its
        # dataclass traversal. Keeping this opaque avoids trying to pickle
        # imported scientific modules during that traversal.
        return self

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
        worker_directory = tempfile.TemporaryDirectory(
            prefix="corral-stargazer-analysis-"
        )
        process = context.Process(
            target=_analysis_worker,
            args=(child_connection, self._public_data, worker_directory.name),
            daemon=True,
            name="stargazer-analysis",
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
        if not parent_connection.poll(_WORKER_START_TIMEOUT_SECONDS):
            self._stop_worker()
            raise RuntimeError("The Stargazer analysis worker did not start")
        try:
            ready = parent_connection.recv()
        except EOFError as exc:
            self._stop_worker()
            raise RuntimeError("The Stargazer analysis worker failed to start") from exc
        if ready != {"status": "ready"}:
            self._stop_worker()
            raise RuntimeError(f"The Stargazer analysis worker failed: {ready!r}")

    def execute(self, code: str) -> str:
        """Run code in the persistent worker, terminating it on timeout."""
        if len(code) > _MAX_CODE_CHARS:
            raise UnsafeAnalysisCode(
                f"Analysis code is limited to {_MAX_CODE_CHARS:,} characters"
            )
        with self._lock:
            process = self._process
            if process is None or not process.is_alive():
                self._start_worker()
            connection = self._connection
            assert connection is not None
            try:
                connection.send({"command": "execute", "code": code})
                if not connection.poll(self._execution_timeout_seconds):
                    timeout = self._execution_timeout_seconds
                    self._stop_worker()
                    raise AnalysisTimeoutError(
                        f"Analysis exceeded {timeout:g} seconds; "
                        "the persistent session was reset"
                    )
                response = connection.recv()
            except AnalysisTimeoutError:
                raise
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._stop_worker()
                raise RuntimeError(
                    "The analysis worker stopped unexpectedly; "
                    "the persistent session was reset"
                ) from exc
            if not isinstance(response, dict) or "result" not in response:
                self._stop_worker()
                raise RuntimeError(
                    "The analysis worker returned an invalid response; "
                    "the persistent session was reset"
                )
            return str(response["result"])

    def close(self) -> None:
        """Release the worker process and its private IPC channel."""
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

    def __del__(self) -> None:
        with suppress(Exception):
            self._stop_worker()


def _restricted_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
    root = name.split(".", 1)[0]
    if root not in _ALLOWED_IMPORTS:
        raise ImportError(
            f"Import of {root!r} is disabled in the Stargazer analysis session"
        )
    if any(part in _BLOCKED_MODULE_ATTRIBUTES for part in name.split(".")):
        raise ImportError(f"Import of {name!r} is disabled in the analysis session")
    return _safe_module(builtins.__import__(name, globals_, locals_, fromlist, level))


def _safe_builtins() -> dict[str, Any]:
    values = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
    values["__import__"] = _restricted_import
    return values


def create_analysis_session(
    *,
    times_days: tuple[float, ...],
    rvs_ms: tuple[float, ...],
    sigmas_ms: tuple[float, ...],
    instruments: tuple[str, ...],
    star_mass_sun: float,
    execution_timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
) -> AnalysisSession:
    """Create a persistent worker containing only public trial data."""
    times = tuple(float(value) for value in times_days)
    if not times:
        raise ValueError("At least one observation is required")
    public_data: dict[str, Any] = {
        "times_days": times,
        "rvs_ms": tuple(float(value) for value in rvs_ms),
        "sigmas_ms": tuple(float(value) for value in sigmas_ms),
        "instruments": tuple(str(value) for value in instruments),
        "star_mass_sun": float(star_mass_sun),
    }
    lengths = {
        len(public_data["times_days"]),
        len(public_data["rvs_ms"]),
        len(public_data["sigmas_ms"]),
        len(public_data["instruments"]),
    }
    if len(lengths) != 1:
        raise ValueError("Observation arrays must have equal lengths")
    return AnalysisSession(
        public_data,
        execution_timeout_seconds=execution_timeout_seconds,
    )


def _create_worker_namespace(public_data: dict[str, Any]) -> dict[str, Any]:
    """Materialize a numerical namespace inside the isolated worker."""
    times = np.asarray(public_data["times_days"], dtype=float).copy()
    return {
        "__builtins__": _safe_builtins(),
        "__name__": "__stargazer_session__",
        "np": _safe_module(np),
        "scipy": _safe_module(scipy),
        "optimize": _safe_module(optimize),
        "signal": _safe_module(signal),
        "times_days": times,
        "rvs_ms": np.asarray(public_data["rvs_ms"], dtype=float).copy(),
        "sigmas_ms": np.asarray(public_data["sigmas_ms"], dtype=float).copy(),
        "instruments": np.asarray(public_data["instruments"], dtype=str).copy(),
        "star_mass_sun": float(public_data["star_mass_sun"]),
        "t_ref_days": float(times[0]),
    }


def _runtime_read_roots() -> tuple[str, ...]:
    """Return interpreter/package roots needed for allow-listed lazy imports."""
    paths = sysconfig.get_paths()
    roots = {
        str(Path(path).resolve())
        for name in ("stdlib", "platstdlib", "purelib", "platlib")
        if (path := paths.get(name))
    }
    roots.add(str(Path(__file__).resolve().parents[1]))
    return tuple(sorted(roots))


def _path_is_within(path: str, roots: tuple[str, ...]) -> bool:
    resolved = str(Path(path).resolve())
    for root in roots:
        try:
            if os.path.commonpath((resolved, root)) == root:
                return True
        except ValueError:
            continue
    return False


def _install_worker_audit_hook() -> Callable[[], None]:
    """Deny worker filesystem, network, and process access outside runtime code."""
    runtime_roots = _runtime_read_roots()
    enabled = [True]
    blocked_prefixes = (
        "ctypes.",
        "ftplib.",
        "http.client.",
        "pty.",
        "socket.",
        "subprocess.",
        "urllib.Request",
    )
    blocked_events = {
        "os.chdir",
        "os.chmod",
        "os.chown",
        "os.exec",
        "os.fork",
        "os.kill",
        "os.posix_spawn",
        "os.putenv",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.spawn",
        "os.startfile",
        "os.system",
        "os.truncate",
        "os.unsetenv",
        "os.utime",
    }

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if not enabled[0]:
            return
        if event == "open":
            target = args[0] if args else None
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            write_flags = (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            )
            if (
                isinstance(mode, str)
                and any(marker in mode for marker in ("w", "a", "x", "+"))
            ) or (isinstance(flags, int) and flags & write_flags):
                raise PermissionError("Filesystem writes are unavailable")
            if isinstance(target, int):
                return
            try:
                path = os.fsdecode(target)
            except TypeError as exc:
                raise PermissionError("Filesystem access is unavailable") from exc
            if path == os.devnull or _path_is_within(path, runtime_roots):
                return
            raise PermissionError("Filesystem access is unavailable")
        if event in {"os.listdir", "os.scandir"}:
            target = args[0] if args else None
            if isinstance(target, str | bytes | os.PathLike) and _path_is_within(
                os.fsdecode(target), runtime_roots
            ):
                return
            raise PermissionError("Directory access is unavailable")
        if event in blocked_events or event.startswith(blocked_prefixes):
            raise PermissionError(f"Operation {event!r} is unavailable")

    sys.addaudithook(audit)

    def disable() -> None:
        enabled[0] = False

    return disable


def _apply_worker_resource_limits() -> None:
    """Bound pathological allocations where POSIX address-space limits exist."""
    if resource is None:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = _WORKER_ADDRESS_SPACE_BYTES
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        if soft == resource.RLIM_INFINITY or soft > limit:
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (OSError, ValueError):
        # Windows lacks ``resource``; some macOS/container policies reject
        # RLIMIT_AS changes. The process timeout remains independently killable.
        return


def _analysis_worker(
    connection: Any,
    public_data: dict[str, Any],
    working_directory: str,
) -> None:
    """Serve persistent executions in a process that never receives task truth."""

    def disable_audit() -> None:
        pass

    try:
        os.chdir(working_directory)
        os.environ.clear()
        namespace = _create_worker_namespace(public_data)
        _apply_worker_resource_limits()
        disable_audit = _install_worker_audit_hook()
        connection.send({"status": "ready"})
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            if not isinstance(request, dict):
                connection.send({"result": "Invalid analysis-worker request."})
                continue
            command = request.get("command")
            if command == "close":
                return
            if command != "execute" or not isinstance(request.get("code"), str):
                connection.send({"result": "Invalid analysis-worker request."})
                continue
            try:
                result = _execute_persistent(request["code"], namespace)
            except BaseException:
                result = traceback.format_exc(limit=8)
            if len(result) > _MAX_OUTPUT_CHARS:
                result = result[:_MAX_OUTPUT_CHARS] + "\n... output truncated"
            connection.send({"result": result})
    except BaseException:
        with suppress(BaseException):
            connection.send({"status": "error", "error": traceback.format_exc(limit=8)})
    finally:
        disable_audit()
        connection.close()


def _validate_repl_tree(tree: ast.AST) -> None:
    """Reject introspection and known file/process entry points before exec."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeAnalysisCode("Dunder names are unavailable")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr in _BLOCKED_MODULE_ATTRIBUTES
        ):
            raise UnsafeAnalysisCode(
                f"Dunder or I/O attribute {node.attr!r} is unavailable "
                "in the analysis session"
            )


def _execute_persistent(code: str, namespace: dict[str, Any]) -> str:
    class CappedOutput(io.StringIO):
        truncated = False

        def write(self, value: str) -> int:
            remaining = _MAX_OUTPUT_CHARS - self.tell()
            if remaining > 0:
                super().write(value[:remaining])
            if len(value) > remaining:
                self.truncated = True
            return len(value)

    output = CappedOutput()
    real_print = builtins.print

    def captured_print(
        *values: Any,
        sep: str = " ",
        end: str = "\n",
        file: Any = None,
        flush: bool = False,
    ) -> None:
        if file is not None:
            raise ValueError("Writing to file handles is disabled")
        real_print(*values, sep=sep, end=end, file=output, flush=flush)

    namespace["__builtins__"]["print"] = captured_print
    tree = ast.parse(code, mode="exec")
    _validate_repl_tree(tree)
    final_expression = None
    statements = tree.body
    if statements and isinstance(statements[-1], ast.Expr):
        final_expression = ast.Expression(statements[-1].value)
        statements = statements[:-1]
    if statements:
        module = ast.Module(body=statements, type_ignores=[])
        exec(compile(module, "<stargazer-repl>", "exec"), namespace)
    if final_expression is not None:
        value = eval(compile(final_expression, "<stargazer-repl>", "eval"), namespace)
        if value is not None:
            captured_print(repr(value))
    result = output.getvalue()
    if output.truncated:
        result += "\n... output truncated"
    return result or "Code executed successfully (no output)."


@tool(hidden_args=["analysis_session"])
def python_repl(code: str, analysis_session: Any) -> str:
    """[BRIEF] Execute scientific Python in a persistent RV-analysis session. [/BRIEF]

    [DETAILED] Runs Python against the current trial's preloaded radial-velocity
    arrays. The namespace contains ``times_days``, ``rvs_ms``, ``sigmas_ms``,
    ``instruments``, NumPy as ``np``, SciPy helpers, ``star_mass_sun``, and
    ``t_ref_days``. Variables and functions persist between calls in the same
    trial. [/DETAILED]

    [PROCEDURAL] Use this tool for numerical exploration, period searches,
    optimization, and residual analysis. Print important intermediate values
    or leave an expression as the last line to inspect it. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION]
    1. [PREREQUISITE] Inspect the preloaded measurements and uncertainties. [/PREREQUISITE]
    2. [CURRENT] Execute one bounded analysis or fitting step and retain useful variables. [/CURRENT]
    3. [FOLLOW_UP] Refine the model or pass fitted planet parameters to ``evaluate_candidate``. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] The session uses an isolated namespace per Corral trial.
    Scientific imports are allow-listed, direct built-in file access is
    removed, output is captured, and the final expression is displayed.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    - ``python_repl("print(len(times_days), np.median(sigmas_ms))")``
    - ``python_repl("from scipy.signal import lombscargle")``
    - ``python_repl("candidate_periods[:5]")``
    [/SYNTACTICAL]

    Args:
        code: [ARGS_BRIEF] Python code to execute. [/ARGS_BRIEF]
              [ARGS_DETAILED] One or more Python statements or an expression
              evaluated in the persistent trial namespace. [/ARGS_DETAILED]
              [ARGS_SYNTACTICAL] A valid Python source string. [/ARGS_SYNTACTICAL]
              [ARGS_EXAMPLES] ``"print(np.std(rvs_ms))"``,
              ``"best_period = 12.3"`` [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] Captured analysis output. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] Printed text, the representation of the final
             expression, a traceback, or a no-output confirmation. Long output
             is truncated. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] ``"120 0.94\n"``, ``"array([1., 2.])\n"`` [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
            [ERROR_WHEN] If the hidden per-trial session was not configured. [/ERROR_WHEN]
            [ERROR_DETAILS] Corral setup did not inject the analysis namespace. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Start a configured Stargazer trial before calling the tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] This is a numerical notebook-like helper, not a hardened
    Python security boundary. Direct file/network/process imports are blocked,
    plotting packages are not installed, and output is capped at 12,000
    characters. [/LIMITATIONS]
    """
    if not isinstance(analysis_session, AnalysisSession):
        raise RuntimeError("The persistent analysis session is not configured")
    try:
        result = analysis_session.execute(code)
    except (AnalysisTimeoutError, UnsafeAnalysisCode, RuntimeError) as exc:
        result = f"{type(exc).__name__}: {exc}"
    except Exception:
        result = traceback.format_exc(limit=8)
    if len(result) > _MAX_OUTPUT_CHARS:
        return result[:_MAX_OUTPUT_CHARS] + "\n... output truncated"
    return result


@tool(hidden_args=["star_mass_sun"])
def planet_from_fit(
    period_days: float,
    semi_amplitude_ms: float,
    star_mass_sun: Any,
    eccentricity: float = 0.0,
    omega_rad: float = 0.0,
    mean_anomaly_rad: float = 0.0,
) -> str:
    """[BRIEF] Convert fitted RV parameters into native Stargazer fields. [/BRIEF]

    [DETAILED] Converts period, velocity semi-amplitude, eccentricity,
    argument of periapsis, and mean anomaly into the minimum-mass and mean
    longitude representation accepted by ``evaluate_candidate``. The current
    task's stellar mass is injected privately by Corral. [/DETAILED]

    [PROCEDURAL] Use this after fitting a Keplerian semi-amplitude in the
    analysis session. Copy the returned planet object into a candidate system
    and combine it with any other recovered signals. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION]
    1. [PREREQUISITE] Fit a period, semi-amplitude, eccentricity, and phase. [/PREREQUISITE]
    2. [CURRENT] Convert those values to native submission parameters. [/CURRENT]
    3. [FOLLOW_UP] Pass the returned object to ``evaluate_candidate``. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] Angles use radians at the public reference epoch. Mean
    longitude is ``omega_rad + mean_anomaly_rad`` modulo one orbit. [/CONTEXTUAL]

    [SYNTACTICAL] Usage example:
    ``planet_from_fit(period_days=23.5, semi_amplitude_ms=4.2,
    eccentricity=0.1, omega_rad=1.2, mean_anomaly_rad=2.2)``
    [/SYNTACTICAL]

    Args:
        period_days: [ARGS_BRIEF] Orbital period in days. [/ARGS_BRIEF]
                     [ARGS_DETAILED] A finite period greater than 0.5 days. [/ARGS_DETAILED]
                     [ARGS_SYNTACTICAL] A positive number. [/ARGS_SYNTACTICAL]
                     [ARGS_EXAMPLES] ``23.5`` [/ARGS_EXAMPLES]
        semi_amplitude_ms: [ARGS_BRIEF] RV semi-amplitude in m/s. [/ARGS_BRIEF]
                           [ARGS_DETAILED] A finite, non-negative fitted stellar velocity amplitude. [/ARGS_DETAILED]
                           [ARGS_SYNTACTICAL] A non-negative number. [/ARGS_SYNTACTICAL]
                           [ARGS_EXAMPLES] ``4.2`` [/ARGS_EXAMPLES]
        eccentricity: [ARGS_BRIEF] Orbital eccentricity. [/ARGS_BRIEF]
                      [ARGS_DETAILED] A finite value from zero through 0.8. [/ARGS_DETAILED]
                      [ARGS_SYNTACTICAL] A number in ``[0, 0.8]``. [/ARGS_SYNTACTICAL]
                      [ARGS_EXAMPLES] ``0.1`` [/ARGS_EXAMPLES]
        omega_rad: [ARGS_BRIEF] Argument of periapsis in radians. [/ARGS_BRIEF]
                   [ARGS_DETAILED] A finite angle normalized modulo two pi. [/ARGS_DETAILED]
                   [ARGS_SYNTACTICAL] A number in radians. [/ARGS_SYNTACTICAL]
                   [ARGS_EXAMPLES] ``1.2`` [/ARGS_EXAMPLES]
        mean_anomaly_rad: [ARGS_BRIEF] Mean anomaly at the reference epoch. [/ARGS_BRIEF]
                          [ARGS_DETAILED] A finite phase angle used to form mean longitude. [/ARGS_DETAILED]
                          [ARGS_SYNTACTICAL] A number in radians. [/ARGS_SYNTACTICAL]
                          [ARGS_EXAMPLES] ``2.2`` [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] A JSON planet object. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] Native period, minimum mass, eccentricity,
             periapsis, and mean-longitude fields ready for submission. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] ``{"P_days": 23.5, "m_sin_i_mjup": 0.1,
             "e": 0.1, "omega_rad": 1.2, "l_rad": 3.4}`` [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If a numeric parameter is non-finite or out of range. [/ERROR_WHEN]
            [ERROR_DETAILS] Invalid orbital values cannot be converted safely. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Correct the fitted units or bounds and retry. [/ERROR_RECOVERY]
        RuntimeError:
            [ERROR_WHEN] If the stellar mass was not configured. [/ERROR_WHEN]
            [ERROR_DETAILS] Corral setup did not inject the public stellar mass. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Start a configured Stargazer trial before calling the tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] This approximation assumes planet mass is small compared
    with stellar mass and reports minimum mass rather than true mass. [/LIMITATIONS]
    """
    values = {
        "period_days": period_days,
        "semi_amplitude_ms": semi_amplitude_ms,
        "star_mass_sun": star_mass_sun,
        "eccentricity": eccentricity,
        "omega_rad": omega_rad,
        "mean_anomaly_rad": mean_anomaly_rad,
    }
    try:
        numeric = {name: float(value) for name, value in values.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("All fitted parameters must be numeric") from exc
    if not all(np.isfinite(value) for value in numeric.values()):
        raise ValueError("All fitted parameters must be finite")
    if numeric["period_days"] <= 0.5:
        raise ValueError("period_days must be greater than 0.5")
    if numeric["semi_amplitude_ms"] < 0.0:
        raise ValueError("semi_amplitude_ms must be non-negative")
    if not 0.0 <= numeric["eccentricity"] <= 0.8:
        raise ValueError("eccentricity must be between 0 and 0.8")
    if numeric["star_mass_sun"] <= 0.0:
        raise RuntimeError("The trial stellar mass is not configured")

    omega = numeric["omega_rad"] % (2.0 * np.pi)
    result = {
        "P_days": numeric["period_days"],
        "m_sin_i_mjup": mass_from_semi_amplitude(
            numeric["semi_amplitude_ms"],
            numeric["period_days"],
            numeric["eccentricity"],
            numeric["star_mass_sun"],
        ),
        "e": numeric["eccentricity"],
        "omega_rad": omega,
        "l_rad": (omega + numeric["mean_anomaly_rad"]) % (2.0 * np.pi),
    }
    return json.dumps(result, allow_nan=False)


@tool(hidden_args=["benchmark_task", "evaluation_session"])
def evaluate_candidate(
    planets: list[CandidatePlanet],
    benchmark_task: Any,
    evaluation_session: Any,
    noise_jitter_ms: float = 0.0,
) -> str:
    """[BRIEF] Evaluate a candidate planetary system and return diagnostic feedback. [/BRIEF]

    [DETAILED] Forward-models a proposed set of Keplerian planets and checks
    statistical fit, residual quality, physical parameter recovery, and planet
    count against evaluator-only truth. This iterative call does not finish the
    Corral task. Valid calls consume the task's diagnostic-evaluation budget.
    [/DETAILED]

    [PROCEDURAL] Call after fitting a plausible system. Use the returned
    criterion-level diagnostics to revise the candidate. Once satisfied,
    repeat the best candidate as the final task answer. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION]
    1. [PREREQUISITE] Estimate one or more Keplerian planet models from the RV data. [/PREREQUISITE]
    2. [CURRENT] Submit native Stargazer parameters for evaluator feedback. [/CURRENT]
    3. [FOLLOW_UP] Refine failed criteria or return the best candidate as final JSON. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] The interactive tool uses exactly the native fields
    ``P_days``, ``m_sin_i_mjup``, ``e``, ``omega_rad``, and ``l_rad``. Use the
    separate ``planet_from_fit`` tool to convert semi-amplitude to mass. Hidden
    truth and planet assignments are never included in the tool result. [/CONTEXTUAL]

    [SYNTACTICAL] Usage example:
    ``evaluate_candidate(planets=[{"P_days": 23.5,
    "m_sin_i_mjup": 0.12, "e": 0.1, "omega_rad": 1.2,
    "l_rad": 3.4}], noise_jitter_ms=0.2)``
    [/SYNTACTICAL]

    Args:
        planets: [ARGS_BRIEF] Candidate Keplerian planets. [/ARGS_BRIEF]
                 [ARGS_DETAILED] A list of typed planet objects using the five
                 native Stargazer fields. [/ARGS_DETAILED]
                 [ARGS_SYNTACTICAL] A JSON array of planet objects. [/ARGS_SYNTACTICAL]
                 [ARGS_EXAMPLES] ``[{"P_days": 10.0, "m_sin_i_mjup": 0.2,
                 "e": 0.0, "omega_rad": 0.0, "l_rad": 1.0}]`` [/ARGS_EXAMPLES]
        noise_jitter_ms: [ARGS_BRIEF] Additional white-noise jitter in m/s. [/ARGS_BRIEF]
                         [ARGS_DETAILED] A finite, non-negative scalar added in
                         quadrature to measurement uncertainties. [/ARGS_DETAILED]
                         [ARGS_SYNTACTICAL] A non-negative number. [/ARGS_SYNTACTICAL]
                         [ARGS_EXAMPLES] ``0.0``, ``0.5`` [/ARGS_EXAMPLES]
    Returns:
        str: [RETURNS_BRIEF] JSON evaluator feedback. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] Reports whether the candidate was accepted,
             remaining evaluations, overall success, and redacted diagnostics
             for all four criteria. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] ``{"accepted": true, "success": false,
             "remaining_evaluations": 2, "criteria": {...}}`` [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
            [ERROR_WHEN] If hidden task or evaluation state is not configured. [/ERROR_WHEN]
            [ERROR_DETAILS] The tool cannot evaluate without trial-specific hidden inputs. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Start a configured Stargazer trial before calling the tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Feedback is intentionally redacted and does not reveal true
    planet parameters. Invalid candidates do not consume budget; accepted
    candidates do. A successful tool call still requires a final Corral answer.
    [/LIMITATIONS]
    """
    if not isinstance(benchmark_task, StargazerTask):
        raise RuntimeError("The hidden Stargazer task is not configured")
    if not isinstance(evaluation_session, dict):
        raise RuntimeError("The evaluation session is not configured")

    evaluations = evaluation_session.setdefault("evaluations", [])
    maximum = int(evaluation_session.get("max_evaluations", 1))
    remaining = max(0, maximum - len(evaluations))
    if evaluation_session.get("locked", False):
        return json.dumps(
            {
                "accepted": False,
                "error": (
                    "A candidate already passed all gates. Return that candidate "
                    "as the final answer; further diagnostic evaluations are locked."
                ),
                "evaluation_number": len(evaluations),
                "remaining_evaluations": remaining,
            },
            indent=2,
        )
    if not remaining:
        return json.dumps(
            {
                "accepted": False,
                "error": "Diagnostic evaluation budget exhausted",
                "evaluation_number": len(evaluations),
                "remaining_evaluations": 0,
            },
            indent=2,
        )

    planet_payloads = [
        (
            planet
            if isinstance(planet, CandidatePlanet)
            else CandidatePlanet.model_validate(planet)
        ).model_dump()
        for planet in planets
    ]
    candidate = CandidateSubmission(
        planets=planet_payloads,
        noise_jitter_ms=noise_jitter_ms,
    )
    try:
        result = evaluate_submission(benchmark_task, candidate)
    except (SubmissionError, ValueError, TypeError, OverflowError) as exc:
        return json.dumps(
            {
                "accepted": False,
                "error": str(exc),
                "evaluation_number": len(evaluations),
                "remaining_evaluations": maximum - len(evaluations),
            },
            indent=2,
        )

    public_result = result.agent_feedback()
    record = {
        "candidate": candidate.canonical_payload(),
        "feedback": public_result,
    }
    evaluations.append(record)
    if result.success:
        evaluation_session["locked"] = True
    feedback = {
        "accepted": True,
        "evaluation_number": len(evaluations),
        "remaining_evaluations": maximum - len(evaluations),
        **public_result,
    }
    if result.success:
        feedback["message"] = (
            "Candidate passed all gates. Return these exact arguments as the final "
            "answer; further diagnostic evaluations are locked."
        )
    return json.dumps(feedback, indent=2, allow_nan=False)


def create_tools() -> dict[str, Tool]:
    """Return the tool pool used by all Stargazer task definitions."""
    return {
        python_repl.name: python_repl,
        planet_from_fit.name: planet_from_fit,
        evaluate_candidate.name: evaluate_candidate,
    }
