"""Agent tools for iterative Stargazer analysis and model submission."""

from __future__ import annotations

import ast
import builtins
import importlib
import types
from typing import Any

import numpy as np

from corral.core.tool import Tool
from corral.runtime import python_repl as _python_repl
from corral.tools.python_repl import create_python_repl_tool

# Compatibility aliases keep old analysis checkpoints and task imports usable.
_MAX_CODE_CHARS = _python_repl.DEFAULT_MAX_CODE_CHARS
_MAX_OUTPUT_CHARS = _python_repl.DEFAULT_MAX_OUTPUT_CHARS
PythonREPLSession = _python_repl.PythonREPLSession
_restore_function_state = _python_repl._restore_function_state
_restore_session_function = _python_repl._restore_session_function
_apply_worker_resource_limits = _python_repl.apply_worker_resource_limits
execute_in_namespace = _python_repl.execute_in_namespace
sanitize_input = _python_repl.sanitize_input
_snapshot_namespace = _python_repl.snapshot_namespace
wrap_last_line_with_print = _python_repl.wrap_last_line_with_print


def _restore_module_proxy(name: str) -> types.ModuleType:
    """Restore legacy checkpoints as ordinary modules, without restrictions."""
    return baselines if name == "baselines" else importlib.import_module(name)


def _session_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
    """Support the in-memory baseline module alongside normal Python imports."""
    if name == "baselines" and level == 0:
        return baselines
    return builtins.__import__(name, globals_, locals_, fromlist, level)


# Old checkpoints refer to this import adapter by name.
_restricted_import = _session_import


def _session_builtins() -> dict[str, Any]:
    return {**vars(builtins), "__import__": _session_import}


def create_analysis_session(
    *,
    times_days: tuple[float, ...],
    rvs_ms: tuple[float, ...],
    sigmas_ms: tuple[float, ...],
    instruments: tuple[str, ...],
    star_mass_sun: float,
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
    return AnalysisSession(public_data)


def _create_worker_namespace(public_data: dict[str, Any]) -> dict[str, Any]:
    """Materialize a numerical namespace inside the isolated worker."""
    times = np.asarray(public_data["times_days"], dtype=float).copy()
    return {
        "__builtins__": _session_builtins(),
        "__name__": "__stargazer_session__",
        "np": np,
        "times_days": times,
        "rvs_ms": np.asarray(public_data["rvs_ms"], dtype=float).copy(),
        "sigmas_ms": np.asarray(public_data["sigmas_ms"], dtype=float).copy(),
        "star_mass_sun": float(public_data["star_mass_sun"]),
        "t_ref_days": float(times[0]),
        "baselines": baselines,
        "history": public_data.get("history") or [],
        "stargazer_planet_from_fit": make_planet_from_fit(
            float(public_data["star_mass_sun"])
        ),
        "STARGAZER_SUBMISSION_GUIDE": STARGAZER_SUBMISSION_GUIDE,
        "_protocol_guide_ack": False,
    }


def _execute_persistent(code: str, namespace: dict[str, Any]) -> str:
    """Apply Stargazer policy around Corral's generic namespace executor."""
    if "matplotlib" in code:
        return "No plotting is allowed. Code was not executed since it contained 'matplotlib'."
    namespace["__builtins__"].update(_session_builtins())
    cleaned = wrap_last_line_with_print(sanitize_input(code))
    # Preserve Stargazer's historical cell-global behavior: functions retain
    # the bindings from their defining cell, while mutable values remain shared.
    execution_namespace = dict(namespace)
    conflict = detect_shadowing_callable_conflict(cleaned, execution_namespace)
    if conflict:
        return conflict
    result = execute_in_namespace(
        cleaned, execution_namespace, preloaded_names=tuple(PRELOADED_VARS)
    )
    namespace.update(execution_namespace)
    return result


class AnalysisSession(PythonREPLSession):
    """Backward-compatible Stargazer adapter over Corral's generic REPL."""

    def __init__(self, public_data: dict[str, Any]):
        super().__init__(
            public_data,
            namespace_factory=_create_worker_namespace,
            code_executor=_execute_persistent,
            export_names=("_protocol_guide_ack",),
        )

    def execute(self, code: str, history: list[dict[str, Any]] | None = None) -> str:
        updates = {"history": history} if history is not None else None
        return super().execute(code, updates)

    def protocol_acknowledged(self) -> bool:
        return bool(self.exports().get("_protocol_guide_ack", False))


class StargazerTool(Tool):
    """Original tool schema dispatched through the trusted Corral environment."""

    def execute(self, **_kwargs: Any) -> Any:
        raise RuntimeError("Stargazer tools must execute through StargazerEnvironment")


# The pinned implementation uses this fallback because no guide file is shipped.
STARGAZER_SUBMISSION_GUIDE = (
    "Stargazer Submission Guide\n"
    "1) Preferred fields: P_days, m_sin_i_mjup, e, omega_rad, l_rad\n"
    "2) Reference epoch: t_ref = times_days[0]\n"
    "3) Convert M0 to l_rad via l_rad = (Omega_rad + omega_rad + M0) mod 2pi\n"
    "4) Avoid mixing phase aliases; if using l_rad, treat it as canonical\n"
    "5) Before submit: verify converted action rv_model residual RMS is near sigma\n"
)
PRELOADED_VARS = {
    "times_days",
    "rvs_ms",
    "sigmas_ms",
    "np",
    "baselines",
    "history",
    "star_mass_sun",
    "t_ref_days",
    "stargazer_planet_from_fit",
    "STARGAZER_SUBMISSION_GUIDE",
}
REQUIRED_OBS_KEYS = ("times_days", "rvs_ms", "sigmas_ms")


def make_planet_from_fit(star_mass_sun):
    def stargazer_planet_from_fit(
        P_days: float,
        K_ms: float,
        e: float = 0.0,
        omega_rad: float = 0.0,
        M0_rad: float = 0.0,
        inc_rad: float = float(np.pi / 2.0),
        Omega_rad: float = 0.0,
        m_sin_i_mjup: float | None = None,
    ) -> dict[str, float]:
        """Convert fitted Keplerian params into canonical Stargazer planet fields."""
        P_days_f = float(P_days)
        K_ms_f = float(max(0.0, K_ms))
        e_f = float(np.clip(e, 0.0, 0.8))
        omega_f = float(omega_rad % (2.0 * np.pi))
        M0_f = float(M0_rad % (2.0 * np.pi))
        inc_f = float(np.clip(inc_rad, 0.0, np.pi))
        Omega_f = float(Omega_rad % (2.0 * np.pi))
        l_rad = float((Omega_f + omega_f + M0_f) % (2.0 * np.pi))

        if m_sin_i_mjup is None:
            P_years = P_days_f / 365.25
            denom = (
                28.4329
                * (star_mass_sun ** (-2.0 / 3.0))
                * (P_years ** (-1.0 / 3.0))
                / np.sqrt(max(1e-12, 1.0 - e_f * e_f))
            )
            msi = float(np.clip(K_ms_f / denom, 1e-3, 30.0))
        else:
            msi = float(np.clip(m_sin_i_mjup, 1e-3, 30.0))

        return {
            "P_days": P_days_f,
            "m_sin_i_mjup": msi,
            "e": e_f,
            "inc_rad": inc_f,
            "Omega_rad": Omega_f,
            "omega_rad": omega_f,
            "l_rad": l_rad,
        }

    return stargazer_planet_from_fit


def _validate_observation(observation: dict[str, Any], fn_name: str) -> None:
    if not isinstance(observation, dict):
        raise TypeError(
            f"{fn_name} expects a dict observation with keys {REQUIRED_OBS_KEYS}, "
            f"got {type(observation).__name__}"
        )
    missing = [k for k in REQUIRED_OBS_KEYS if k not in observation]
    if missing:
        raise KeyError(f"{fn_name} missing required keys: {missing}")


def baseline_null_model(observation: dict[str, Any]) -> dict[str, Any]:
    _validate_observation(observation, "baseline_null_model")
    np.array(
        observation["times_days"], dtype=float
    )  # Validate the original input contract.
    y = np.array(observation["rvs_ms"], dtype=float)
    s_arr = np.array(observation["sigmas_ms"], dtype=float)
    if y.size == 0:
        raise ValueError("Empty observations")
    if np.any(~np.isfinite(s_arr)) or np.any(s_arr <= 0):
        raise ValueError("All per-point uncertainties must be positive and finite")
    w = 1.0 / (s_arr**2)
    mu = float(np.sum(w * y) / np.sum(w))
    rv_model = np.full_like(y, mu)
    return {"rv_model": rv_model.tolist(), "planets": []}


def baseline_one_sine(observation: dict[str, Any]) -> dict[str, Any]:
    _validate_observation(observation, "baseline_one_sine")
    t = np.array(observation["times_days"], dtype=float)
    y = np.array(observation["rvs_ms"], dtype=float)
    s = np.array(observation["sigmas_ms"], dtype=float)
    if y.size == 0:
        raise ValueError("Empty observations")
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("All per-point uncertainties must be positive and finite")

    weights = 1.0 / (s**2 + 1e-6)
    y_mean = float(np.average(y, weights=weights))
    y0 = y - y_mean
    freqs = np.linspace(1 / 300.0, 1 / 2.0, 2000)
    best = None
    for f in freqs:
        omega = 2 * np.pi * f
        X = np.vstack([np.sin(omega * t), np.cos(omega * t), np.ones_like(t)]).T
        WX = X * weights[:, None]
        beta = np.linalg.pinv(X.T @ WX) @ (X.T @ (weights * y0))
        model = X @ beta + y_mean
        rss = np.sum(((y - model) / s) ** 2)
        if best is None or rss < best[0]:
            best = (rss, f, model, beta)
    rv_model = best[2]
    P_days = 1.0 / best[1]
    beta_best = best[3]
    amp = float(np.hypot(beta_best[0], beta_best[1]))
    omega = 2.0 * np.pi * best[1]
    # Model is y = gamma + A*sin(w t) + B*cos(w t).
    # Convert to y = gamma + K*cos(M), where M = M0 + w*(t-t_ref), e=0.
    # Using sin(wt+phi) form, M0 = phi - pi/2 + w*t_ref.
    phase_sine = float(np.arctan2(beta_best[1], beta_best[0]))
    t_ref = float(t[0])
    M0 = float((phase_sine - np.pi / 2.0 + omega * t_ref) % (2.0 * np.pi))
    rv_offset_ms = float(y_mean + beta_best[2])
    resid = y - rv_model
    rms_ms = float(np.sqrt(np.mean(resid**2)))
    wrms_ms = float(np.sqrt(np.average(resid**2, weights=weights)))
    P_years = P_days / 365.25
    m_sin_i_estimate = amp / (28.4329 * (P_years ** (-1.0 / 3.0)))
    m_sin_i_estimate = float(np.clip(m_sin_i_estimate, 0.001, 10.0))
    guess = {
        "P_days": float(P_days),
        "m_sin_i_mjup": m_sin_i_estimate,
        "e": 0.0,
        "inc_rad": float(np.pi / 2),
        "Omega_rad": 0.0,
        "omega_rad": 0.0,
        "l_rad": M0,
    }
    return {
        "rv_model": rv_model.tolist(),
        "planets": [guess],
        "period_days": float(P_days),
        "semi_amplitude_ms": amp,
        "phase_rad": M0,
        "rv_offset_ms": rv_offset_ms,
        "rms_ms": rms_ms,
        "wrms_ms": wrms_ms,
    }


baselines = types.ModuleType("baselines")
baselines.baseline_null_model = baseline_null_model
baselines.baseline_one_sine = baseline_one_sine


def detect_shadowing_callable_conflict(code: str, namespace: dict) -> str | None:
    """Detect assigning to a callable name and then calling it in the same snippet."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    callable_names = {k for k, v in namespace.items() if callable(v)}
    assigned_names = set()
    called_names = set()
    defined_functions = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            defined_functions.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign) and isinstance(
            node.target, ast.Name
        ):
            assigned_names.add(node.target.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
            if (
                node.func.id in {"least_squares", "minimize", "root", "curve_fit"}
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                callback_name = node.args[0].id
                if callback_name in namespace and not callable(
                    namespace[callback_name]
                ):
                    return (
                        f"Invalid optimizer callback: `{callback_name}` is not callable. "
                        f"It may have been overwritten by a variable. Rename that variable "
                        f"(e.g., `{callback_name}_arr`) or redefine `def {callback_name}(...):`."
                    )

    conflict_names = sorted(
        name
        for name in assigned_names & called_names
        if name in callable_names or name in defined_functions
    )
    if not conflict_names:
        invalid_calls = sorted(
            name
            for name in called_names
            if name not in defined_functions
            and name in namespace
            and not callable(namespace[name])
        )
        if not invalid_calls:
            return None
        bad_name = invalid_calls[0]
        return (
            f"Invalid call detected: `{bad_name}` is not callable in the current session. "
            f"It may have been overwritten by a variable. Rename the variable (e.g., `{bad_name}_arr`) "
            f"or redefine `def {bad_name}(...):` before calling it."
        )

    conflict = conflict_names[0]
    return (
        f"Name shadowing detected: `{conflict}` is assigned and called in the same code block. "
        f"Use a different variable name like `{conflict}_arr` or `{conflict}_vec`."
    )


def create_tools() -> dict[str, Tool]:
    """Original two-tool interface, dispatched by the Corral environment."""
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "PythonREPL",
                "description": "A persistent Python REPL with no per-call execution timeout. Use print(...) to see results. Normal Python builtins and installed packages are available; file, process, and network access follow the runtime's permissions. You cannot use matplotlib. No plotting is allowed.\nPreloaded variables: np, times_days, rvs_ms, sigmas_ms, baselines, history, star_mass_sun, t_ref_days, stargazer_planet_from_fit, STARGAZER_SUBMISSION_GUIDE.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_code": {
                            "type": "string",
                            "description": "A valid python command.",
                        }
                    },
                    "required": ["input_code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_action",
                "description": "Submit a candidate RV model to the environment. Preferred protocol: Stargazer-native planet fields (P_days, m_sin_i_mjup, e, omega_rad, l_rad). Provide up to 7 planets with P_days > 0.5 and eccentricity between 0 and 0.8. This task currently expects submission_mode='params_and_model'. The environment will evaluate your submission and return a reward with detailed metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "planets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "P_days": {
                                        "type": "number",
                                        "description": "Stargazer planet format: orbital period in days (alternative to period_days).",
                                    },
                                    "period_days": {
                                        "type": "number",
                                        "description": "Orbital period in days (must be > 0.5)",
                                    },
                                    "m_sin_i_mjup": {
                                        "type": "number",
                                        "description": "Stargazer planet format: minimum mass in Jupiter masses (optional; if provided, K is approximated assuming M_star=1Msun).",
                                    },
                                    "semi_amplitude_ms": {
                                        "type": "number",
                                        "description": "Semi-amplitude K in m/s (must be >= 0)",
                                    },
                                    "phase_deg": {
                                        "type": "number",
                                        "description": "Phase in degrees (optional, provide either phase_deg or phase_rad)",
                                    },
                                    "phase_rad": {
                                        "type": "number",
                                        "description": "Phase in radians (optional, provide either phase_deg or phase_rad)",
                                    },
                                    "phase": {
                                        "type": "number",
                                        "description": "Phase alias (optional). Interpreted as radians if |phase|<=2π, else degrees if |phase|<=360.",
                                    },
                                    "phase_frac": {
                                        "type": "number",
                                        "description": "Phase as fraction of an orbit in [0,1) (optional). Converted to radians via 2π*phase_frac.",
                                    },
                                    "l_rad": {
                                        "type": "number",
                                        "description": "Stargazer planet format: mean longitude at t_ref=times_days[0] in radians.",
                                    },
                                    "eccentricity": {
                                        "type": "number",
                                        "description": "Eccentricity (between 0 and 0.8)",
                                    },
                                    "e": {
                                        "type": "number",
                                        "description": "Stargazer planet format: eccentricity (alias for eccentricity).",
                                    },
                                    "omega_rad": {
                                        "type": "number",
                                        "description": "Stargazer planet format: argument of periapsis in radians (optional; matching uses l_rad).",
                                    },
                                    "inc_rad": {
                                        "type": "number",
                                        "description": "Inclination in radians for REBOUND forward model (optional; default pi/2).",
                                    },
                                    "Omega_rad": {
                                        "type": "number",
                                        "description": "Longitude of ascending node in radians for REBOUND forward model (optional; default 0).",
                                    },
                                },
                                "description": "Recommended: use Stargazer native fields (P_days, m_sin_i_mjup, e, omega_rad, l_rad). Legacy aliases are accepted.",
                            },
                            "description": "List of planet hypotheses, sorted by confidence (most confident first).",
                        },
                        "rv_offset_ms": {
                            "type": "number",
                            "description": "Constant RV offset in m/s (optional, will be estimated if not provided).",
                        },
                        "noise_jitter_ms": {
                            "type": "number",
                            "description": "Optional white-noise jitter term in m/s.",
                        },
                        "notes": {
                            "type": "string",
                            "description": "Brief rationale for this submission (optional but recommended).",
                        },
                    },
                    "required": ["planets"],
                },
            },
        },
    ]
    repl, submission = (entry["function"] for entry in definitions)
    repl_tool = create_python_repl_tool(
        name=repl["name"],
        description=repl["description"],
        argument_name="input_code",
        argument_description=repl["parameters"]["properties"]["input_code"][
            "description"
        ],
        namespace_factory=_create_worker_namespace,
        code_executor=_execute_persistent,
        synchronized_names=("history",),
        export_names=("_protocol_guide_ack",),
        # Preserve the Stargazer worker wire format during migration.
        export_result_names={"_protocol_guide_ack": "protocol_ack"},
    )
    return {
        repl_tool.name: repl_tool,
        submission["name"]: StargazerTool(
            name=submission["name"],
            description=submission["description"],
            params_json_schema=submission["parameters"],
            trusted=True,
        ),
    }
