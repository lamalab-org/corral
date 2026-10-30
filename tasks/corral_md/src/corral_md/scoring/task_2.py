"""Score sodium-silicate cooling and hold directly from final LAMMPS logs.

Both levels submit only results and artifacts.raw_logs. Logs retain echoed
commands and thermo output; no submitted analysis code is executed. Continuity
is checked within the recorded execution, not attested against log fabrication.
Numerical screens below are evaluator policy, not a calibrated reference Tg.
"""

from __future__ import annotations

import hashlib
import re
import shlex

import numpy as np

from corral.evaluation import SubmissionScore

from .common import Evidence, EvidenceError, Rubric
from .sampling import mean_standard_error

SOURCE_PATH = "/workspace/resources/md-structures-melt-liq4000/liq4000.dat"
SOURCE_SHA256 = "1674c66496b0bab9eda0ce79b439ad90f9ce2097c5d5450a86965442f5cdfa63"
COOLING_STEPS = 370_000
TOTAL_STEPS = 470_000
_COEFFICIENTS = {
    (1, 1): (0, 1, 0),
    (1, 2): (0, 1, 0),
    (2, 2): (0, 1, 0),
    (1, 3): (101093.3472, 0.243838, 707.96963),
    (2, 3): (316001.3219145, 0.193817, 1260.9930729),
    (3, 3): (42541.49842, 0.343645, 4441.068122),
}
_COLUMNS = ("Step", "Temp", "Press", "Density")
_STATE_CHANGES = {
    "clear",
    "read_restart",
    "read_dump",
    "reset_timestep",
    "velocity",
    "set",
    "create_atoms",
    "delete_atoms",
    "displace_atoms",
    "change_box",
    "replicate",
    "mass",
    "minimize",
    "temper",
    "rerun",
    "run_style",
    "fix_modify",
    "pair_modify",
    "kspace_modify",
    "special_bonds",
    "python",
    "shell",
    "if",
    "jump",
    "next",
}


def _require(condition, detail):
    if not condition:
        raise EvidenceError(detail)


def read_logs(e: Evidence) -> np.ndarray:
    """Return Step/Temp/Press/Density at 1 ps spacing after protocol checks.

    Read logs in submitted order. Each advancing run must have a completed
    thermo table, and adjacent runs must share the same boundary sample.
    Split runs with start/stop or upto are supported.
    """
    step, loaded, timestep = 0, False, 1.0
    settings = {"boundary": ["p", "p", "p"]}
    coefficients, fixes = {}, {}
    pending, header, rows = None, None, []
    trace = []
    paths = e.artifacts("raw_logs")
    _require(
        bool(paths) and len(set(paths)) == len(paths),
        "Provide distinct final raw_logs in execution order",
    )
    for path in paths:
        for raw in path.read_text().replace("&\n", " ").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if re.match(r"^(?:read_data|read_restart) CPU\s*=", line):
                # Native LAMMPS timing output is not a second executed command.
                continue
            _require(
                not line.startswith("ERROR") and "Lost atoms" not in line,
                "LAMMPS reported an execution failure",
            )
            if line.split()[0] == "Step":
                _require(
                    pending is not None and header is None,
                    "Unexpected or repeated thermo table",
                )
                header = line.split()
                _require(
                    all(c in header for c in _COLUMNS),
                    "Thermo output must contain Step Temp Press Density",
                )
                continue
            if line.startswith("Loop time"):
                _require(
                    pending is not None and rows, "Run completion has no thermo samples"
                )
                start, end = pending
                completed = re.search(r"for (\d+) steps with (\d+) atoms", line)
                _require(
                    completed is not None
                    and tuple(map(int, completed.groups())) == (end - start, 192),
                    "Incomplete run or changed atom count",
                )
                block = np.asarray(rows)
                gaps = np.diff(block[:, 0])
                _require(
                    block[0, 0] == start and block[-1, 0] == end,
                    "Thermo samples must include run endpoints",
                )
                _require(
                    np.all((gaps > 0) & (gaps <= 1000)),
                    "Retain chronological thermo samples at intervals of at most 1000 steps",
                )
                if trace:
                    _require(
                        # PPPM may rebuild its mesh between runs, slightly
                        # changing pressure without changing positions/velocities.
                        np.allclose(
                            np.asarray(trace[-1])[[0, 1, 3]],
                            block[0, [0, 1, 3]],
                            rtol=1e-7,
                            atol=1e-8,
                        ),
                        "Discontinuous thermo values between runs",
                    )
                    trace.extend(rows[1:])
                else:
                    trace.extend(rows)
                step, pending, header, rows = end, None, None, []
                continue
            if header is not None:
                if line.startswith("WARNING"):
                    continue
                try:
                    values = np.asarray(line.split(), dtype=float)
                except ValueError as exc:
                    raise EvidenceError("Malformed or unfinished thermo table") from exc
                _require(
                    len(values) == len(header) and np.isfinite(values).all(),
                    "Nonfinite or malformed thermo values",
                )
                row = [values[header.index(c)] for c in _COLUMNS]
                _require(
                    row[0] == round(row[0]) and row[1] > 0 and row[3] > 0,
                    "Invalid step, temperature, or density",
                )
                if "Time" in header:
                    _require(
                        np.isclose(
                            values[header.index("Time")], row[0], atol=1e-6, rtol=0
                        ),
                        "Time must be in fs at the required 1 fs timestep",
                    )
                if "Atoms" in header:
                    _require(values[header.index("Atoms")] == 192, "Atom count changed")
                rows.append(row)
                continue
            command = shlex.split(line, comments=True)
            if not command:
                continue
            name, *args = command
            _require(
                name not in _STATE_CHANGES,
                f"Unsupported state-changing command: {name}",
            )
            # LAMMPS logs may echo both original and variable-expanded commands.
            # Only the expanded line describes the executed numerical command.
            if "$" in line:
                continue
            if name == "echo":
                _require(
                    args in (["log"], ["both"]), "Keep command echo enabled in the log"
                )
            elif name == "read_data":
                _require(
                    not loaded and step == 0 and len(args) == 1,
                    "Load the supplied melt exactly once",
                )
                if args[0] != SOURCE_PATH:
                    source = e._path(path.parent / args[0])
                    _require(
                        hashlib.sha256(source.read_bytes()).hexdigest()
                        == SOURCE_SHA256,
                        "Incorrect supplied molten state",
                    )
                loaded = True
            elif name in {
                "units",
                "atom_style",
                "boundary",
                "pair_style",
                "kspace_style",
            }:
                settings[name] = args
                if name == "pair_style":
                    coefficients.clear()
            elif name == "pair_coeff":
                values = args[2:]
                if values[:1] == ["buck/coul/long"]:
                    values = values[1:]
                for pair in _COEFFICIENTS:
                    if all(
                        token == "*" or token == str(index)
                        for token, index in zip(args[:2], pair, strict=True)
                    ):
                        coefficients[pair] = list(map(float, values))
            elif name == "timestep":
                timestep = float(args[0])
            elif name == "fix":
                _require(
                    args[2] in {"npt", "ave/time", "ave/atom", "print"},
                    "Only NPT integration and passive recording fixes are allowed",
                )
                fixes[args[0]] = args
            elif name == "unfix":
                fixes.pop(args[0], None)
            elif name == "thermo_modify":
                _require(
                    not {"temp", "press", "colname", "lost"}.intersection(args),
                    "Use the standard thermo temperature, pressure, and column names",
                )
            elif name == "run":
                _require(pending is None, "Previous run did not finish")
                end = int(args[0]) + (0 if "upto" in args else step)
                _require(end >= step, "Run steps go backwards")
                options = {}
                index = 1
                while index < len(args):
                    key = args[index]
                    _require(
                        key in {"upto", "start", "stop", "pre", "post"},
                        "Unsupported run options",
                    )
                    if key == "upto":
                        index += 1
                    else:
                        options[key] = args[index + 1]
                        index += 2
                _require(
                    loaded and timestep == 1,
                    "Start from the supplied melt with a 1 fs timestep",
                )
                _require(
                    settings.get("units", ["real"]) == ["real"]
                    and settings.get("atom_style", ["charge"]) in (["charge"], ["full"])
                    and settings["boundary"] == ["p", "p", "p"],
                    "Use the task's real units, charged atoms, and periodic boundaries",
                )
                style = settings.get("pair_style", [])
                if style[:1] in (["hybrid"], ["hybrid/overlay"]):
                    style = style[1:]
                pppm = settings.get("kspace_style", [])
                _require(
                    len(style) == 3
                    and style[0] == "buck/coul/long"
                    and list(map(float, style[1:])) == [8, 12],
                    "Use the supplied BKS potential and cutoffs",
                )
                _require(
                    len(pppm) == 2 and pppm[0] == "pppm" and 0 < float(pppm[1]) <= 1e-4,
                    "Use PPPM accuracy of 1e-4 or better",
                )
                _require(
                    all(
                        pair in coefficients
                        and len(coefficients[pair]) == 3
                        and np.allclose(
                            coefficients[pair], expected, rtol=1e-8, atol=1e-8
                        )
                        for pair, expected in _COEFFICIENTS.items()
                    ),
                    "Incorrect BKS pair coefficients",
                )
                if end > step:
                    integrators = [f for f in fixes.values() if f[2] == "npt"]
                    _require(len(integrators) == 1, "Use exactly one NPT integrator")
                    fix = integrators[0]
                    _require(
                        len(fix) == 11
                        and fix[1:4] == ["all", "npt", "temp"]
                        and fix[7] == "iso",
                        "Use isotropic NPT for all atoms",
                    )
                    first, last, damping = map(float, fix[4:7])
                    pressure = list(map(float, fix[8:11]))
                    cooling = step < COOLING_STEPS
                    _require(
                        end <= (COOLING_STEPS if cooling else TOTAL_STEPS),
                        "Run crosses a stage boundary or continues after the hold",
                    )
                    _require(
                        damping == 100
                        and pressure == [0, 0, 10000 if cooling else 5000],
                        "Incorrect temperature/pressure damping or target pressure",
                    )
                    start, stop = (
                        int(options.get("start", step)),
                        int(options.get("stop", end)),
                    )
                    _require(start <= step < end <= stop, "Invalid ramp start/stop")
                    target = first + (last - first) * (
                        np.array([step, end]) - start
                    ) / (stop - start)
                    expected = (
                        4000 - np.array([step, end]) / 100 if cooling else [300, 300]
                    )
                    _require(
                        np.allclose(target, expected, rtol=0, atol=1e-6),
                        "Incorrect cooling or hold temperature schedule",
                    )
                pending = (step, end)
    _require(
        pending is None and step == TOTAL_STEPS,
        "Logs must cover the completed 370 ps cooling and 100 ps hold",
    )
    trace = np.asarray(trace)
    grid = np.arange(0, TOTAL_STEPS + 1, 1000)
    sampled = np.column_stack(
        [grid, *(np.interp(grid, trace[:, 0], trace[:, i]) for i in range(1, 4))]
    )
    # Check tracking, never stationarity, during the nonstationary cooling ramp.
    for block in np.array_split(sampled[:370], 37):
        target = 4000 - block[:, 0].mean() / 100
        _require(
            abs(block[:, 1].mean() - target) <= max(150, 0.1 * target),
            "Measured cooling temperature does not follow the prescribed ramp",
        )
    return sampled


def hold_statistics(trace: np.ndarray) -> dict:
    """Use the final 80 ps; assess settling without requiring glass equilibrium."""
    hold = trace[trace[:, 0] >= 390_000]
    time = hold[:, 0] / 1000
    temperature, density = hold[:, 1], hold[:, 3]
    drift = float(np.polyfit(time - time.mean(), density, 1)[0])
    # 10 ps blocks reduce sensitivity to rapid thermostat/barostat oscillations.
    blocks = np.array(
        [part[:, [1, 3]].mean(axis=0) for part in np.array_split(hold, 8)]
    )
    temperature_se, density_se = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
    _require(
        abs(temperature.mean() - 300) <= 15 and temperature_se <= 5,
        "Hold temperature is not adequately controlled near 300 K",
    )
    _require(
        density_se <= 0.005 * density.mean(),
        "Hold density is too noisy for a representative mean",
    )
    _require(
        abs(drift) * 80 <= 0.02 * density.mean(),
        "Hold density has not settled within 2% over the final 80 ps",
    )
    # LAMMPS real units report pressure in atmospheres. Use the same pressure
    # precision scale as the NPT reference task, converted from GPa explicitly.
    pressure = hold[:, 2]
    pressure_se = mean_standard_error(pressure)
    pressure_floor = 0.05 / 1.01325e-4
    pressure_limit = 0.1 / 1.01325e-4
    _require(
        pressure_se <= pressure_limit
        and abs(pressure.mean()) <= 3 * pressure_se + pressure_floor,
        "Hold pressure is too uncertain or misses the zero-pressure NPT target",
    )
    pressure_drift = float(np.polyfit(time - time.mean(), pressure, 1)[0])
    halves = [part.mean() for part in np.array_split(pressure, 2)]
    _require(
        abs(halves[1] - halves[0]) <= 4 * pressure_se + pressure_floor
        and abs(pressure_drift) * np.ptp(time) <= 4 * pressure_se + 2 * pressure_floor,
        "Hold pressure has not settled",
    )
    for start in (len(pressure) // 4, len(pressure) // 2):
        tail = pressure[start:]
        tail_se = mean_standard_error(tail)
        _require(
            tail_se <= pressure_limit * np.sqrt(len(pressure) / len(tail))
            and abs(tail.mean()) <= 3 * tail_se + pressure_floor,
            "Later hold pressure is too uncertain or misses the NPT target",
        )
    return {
        "hold_temperature_K": float(temperature.mean()),
        "hold_density_g_cm3": float(density.mean()),
        "hold_density_drift_g_cm3_ps": drift,
        "temperature_se_K": float(temperature_se),
        "density_se_g_cm3": float(density_se),
        "hold_pressure_atm": float(pressure.mean()),
        "pressure_se_atm": pressure_se,
        "hold_pressure_drift_atm_ps": pressure_drift,
    }


def _fit_transition(x, y):
    """Compare a continuous two-slope fit with a straight line on identical data."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    _require(
        len(x) >= 24 and np.ptp(x) >= 1600,
        "Insufficient temperature coverage for transition analysis",
    )
    linear = np.column_stack([np.ones(len(x)), x - x.mean()])
    residual = y - linear @ np.linalg.lstsq(linear, y, rcond=None)[0]
    single_sse = float(residual @ residual)
    fits = []
    for split in range(6, len(x) - 6):
        knot = (x[split - 1] + x[split]) / 2
        design = np.column_stack([linear, np.maximum(x - knot, 0)])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ coef
        fits.append((float(residual @ residual), float(knot), float(coef[2])))
    sse, knot, slope_change = min(fits)
    floor = np.finfo(float).eps * float(y @ y)
    gain = len(x) * np.log(max(single_sse, floor) / max(sse, floor)) - 2 * np.log(
        len(x)
    )
    # Profile-likelihood width is a diagnostic, not permission for arbitrarily
    # broad answer tolerances. Window stability is assessed separately below.
    candidates = [
        k for error, k, _ in fits if error <= max(sse, floor) * (1 + 4 / (len(x) - 4))
    ]
    width = max(candidates) - min(candidates)
    return {
        "estimate_K": knot,
        "bic_gain": float(gain),
        "width_K": width,
        "residual_rms_g_cm3": float(np.sqrt(sse / len(x))),
        "slope_change_g_cm3_K": slope_change,
    }


def cooling_transition(trace: np.ndarray) -> dict:
    """Fit measured T <= 2600 K, with 5 ps blocks and three fitting windows."""
    cooling = trace[(trace[:, 0] <= COOLING_STEPS) & (trace[:, 1] <= 2600)]
    _require(len(cooling) >= 180, "Too few cooling samples below 2600 K")
    groups = cooling[:, 0].astype(int) // 5000
    blocks = np.array(
        [
            cooling[groups == group][:, [1, 3]].mean(axis=0)
            for group in np.unique(groups)
            if np.count_nonzero(groups == group) >= 3
        ]
    )
    fits = []
    for low, high in ((300, 2600), (400, 2500), (500, 2400)):
        selected = blocks[(blocks[:, 0] >= low) & (blocks[:, 0] <= high)]
        fits.append(_fit_transition(selected[:, 0], selected[:, 1]))
    density_resolved = max(f["residual_rms_g_cm3"] for f in fits) <= 0.005 * np.mean(
        blocks[:, 1]
    )
    estimates = [f["estimate_K"] for f in fits]
    spread = max(estimates) - min(estimates)
    resolved = (
        density_resolved
        and all(f["bic_gain"] >= 6 and f["width_K"] <= 200 for f in fits)
        and spread <= 150
    )
    return {
        "transition": "resolved" if resolved else "unresolved",
        "cooling_tg_K": float(np.median(estimates)) if resolved else None,
        "window_spread_K": spread,
        "fits": fits,
        "density_resolved": bool(density_resolved),
    }


class Task2Scorer:
    """Binary log-based scorer shared by Task 2 preparation and analysis."""

    task_number = 2
    verifier = None

    def __init__(self, *, level=2):
        if type(level) is not int or level not in (1, 2):
            raise ValueError("level must be 1 or 2")
        self.level = level

    def __call__(self, submission):
        return self.score_submission(submission)

    def score_submission(self, submission):
        return self.evaluate_submission(submission).score

    def evaluate_submission(self, submission):
        report = self.evaluate(submission)
        failures = [
            f"{c['name']}: {c['detail']}"
            for c in report["checks"]
            if c["status"] == "failed"
        ]
        return SubmissionScore(
            score=report["score"],
            feedback="; ".join(failures) or None,
            metadata={"workflow_evaluation": report},
        )

    def evaluate(self, submission):
        rubric = Rubric(2, binary=True)
        computed = {}

        def check():
            evidence = Evidence(submission, task_number=2, level=self.level)
            trace = read_logs(evidence)
            computed.update(hold_statistics(trace))
            results = evidence.results
            tolerances = {
                "hold_temperature_K": 1.0,
                "hold_density_g_cm3": 0.001,
                "hold_density_drift_g_cm3_ps": 1e-5,
                "hold_pressure_atm": 1.0,
                "pressure_se_atm": 0.01,
                "hold_pressure_drift_atm_ps": 0.01,
            }
            for key, atol in tolerances.items():
                if key not in results:
                    continue
                actual = results.get(key)
                _require(
                    type(actual) in (float, int) and np.isfinite(actual),
                    f"Report a finite {key}",
                )
                _require(
                    abs(actual - computed[key]) <= atol + 0.01 * abs(computed[key]),
                    f"{key}: reported {actual:g}, recomputed {computed[key]:g}",
                )
            if self.level == 2:
                computed.update(cooling_transition(trace))
                _require(
                    computed["transition"] == "resolved",
                    "Level 2 requires a resolved Tg; the cooling transition is "
                    "unresolved in the retained data",
                )
                _require(
                    results.get("transition", "resolved") == "resolved",
                    "Report transition='resolved' with a numerical cooling_tg_K",
                )
                actual = results.get("cooling_tg_K")
                _require(
                    type(actual) in (int, float) and np.isfinite(actual),
                    "Report a finite numerical cooling_tg_K in kelvin",
                )
                _require(
                    abs(actual - computed["cooling_tg_K"]) <= 100,
                    "Cooling Tg differs from the independently fitted "
                    f"{computed['cooling_tg_K']:g} K",
                )
            return True

        rubric.check("cooling_and_hold", 100, check)
        return {
            **rubric.as_dict(),
            "level": self.level,
            "mode": "saved_logs",
            "computed": computed,
        }
