"""Recompute glass-transition analysis from saved evidence; never run MD."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from .common import UnsupportedEvidence, result_close

# From potentials.zip/potentials/BKS/pot.mod and the supplied liq4000.dat.
_CHARGES = {"Na": 0.6, "Si": 2.4, "O": -1.2}
_PAIR_COEFFICIENTS = {
    (1, 1): (0, 1, 0),
    (1, 2): (0, 1, 0),
    (2, 2): (0, 1, 0),
    (1, 3): (101093.3472, 0.243838, 707.96963),
    (2, 3): (316001.3219145, 0.193817, 1260.9930729),
    (3, 3): (42541.49842, 0.343645, 4441.068122),
}


def _close(a, b, atol=0.5, rtol=1e-3):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(
        np.all(np.isfinite(a))
        and np.all(np.isfinite(b))
        and np.allclose(a, b, atol=atol, rtol=rtol)
    )


def _verified(function):
    """Unsupported free-choice methods are unverified, not scientific failures."""
    try:
        return function()
    except UnsupportedEvidence as exc:
        return None, str(exc)


def _trace(e):
    table = e.table("thermal_trace", "thermal_traces", "trace").copy()
    table = table.rename(columns={"measured_temperature_K": "temperature_K"})
    columns = [
        "time_ps",
        "target_temperature_K",
        "temperature_K",
        "pressure_atm",
        "density_g_cm3",
    ]
    table[columns] = table[columns].astype(float)
    if not np.isfinite(table[columns].to_numpy()).all():
        raise ValueError("Thermal trace contains nonfinite values")
    if not (table.temperature_K.gt(0).all() and table.density_g_cm3.gt(0).all()):
        raise ValueError("Measured temperature and density must be positive")
    table["stage"] = table.stage.astype(str).str.lower()
    if set(table.stage) != {"cooling", "hold", "reheating"}:
        raise ValueError("Trace must contain cooling, hold and reheating stages")
    return table


def _schedule(e):
    t = _trace(e)
    expected = [
        ("cooling", 0, 370, 4000, 300),
        ("hold", 370, 470, 300, 300),
        ("reheating", 470, 700, 300, 2600),
    ]
    origin = float(t.time_ps.iloc[0])
    if np.any(np.diff(t.time_ps) < 0):
        return False, "Trace elapsed times are not chronological"
    for stage, start, end, first, last in expected:
        part = t[t.stage == stage]
        if len(part) < 2 or np.any(np.diff(part.time_ps) <= 0):
            return False, f"{stage} needs chronological samples including endpoints"
        times = part.time_ps.to_numpy() - origin
        if not _close([times[0], times[-1]], [start, end], atol=0.5, rtol=0.01):
            return False, f"{stage} duration or boundary timing is incorrect"
        target = first + (times - start) * (last - first) / (end - start)
        if not _close(part.target_temperature_K, target, atol=5, rtol=0.01):
            return (
                False,
                f"{stage} target schedule does not match the requested thermal cycle",
            )
    return True


def _text(e, *aliases):
    return "\n".join(p.read_text(errors="replace") for p in e.artifacts(*aliases))


def _commands(e):
    """Read simple saved commands, expanding linked includes without execution."""
    paths = e.artifacts("lammps_inputs", "lammps_input", "inputs")
    linked = paths + (e.artifacts("potential") if "potential" in e._artifacts else [])
    variables, included = {}, set()

    def read(path, parents=()):
        if path in parents:
            raise ValueError("Recursive LAMMPS include")
        for raw in path.read_text().replace("&\n", " ").splitlines():
            line = raw.split("#", 1)[0]
            for name, value in variables.items():
                line = line.replace("${" + name + "}", value)
                if len(name) == 1:
                    line = line.replace("$" + name, value)
            c = shlex.split(line)
            if not c:
                continue
            if "$" in line or c[0] in {"jump", "if", "next", "python"}:
                raise UnsupportedEvidence(
                    "LAMMPS expressions/control flow are unverified"
                )
            if (
                c[0] == "variable"
                and len(c) == 4
                and c[2] in {"equal", "index", "string"}
            ):
                if c[2] == "equal":
                    try:
                        float(c[3])
                    except ValueError as exc:
                        raise UnsupportedEvidence(
                            "Nonliteral LAMMPS expressions are unverified"
                        ) from exc
                variables[c[1]] = c[3]
            elif c[0] == "include":
                candidates = [p for p in linked if p.name == Path(c[1]).name]
                if len(candidates) != 1:
                    raise ValueError(
                        "Link each included input/potential file unambiguously"
                    )
                included.add(candidates[0])
                yield from read(candidates[0], (*parents, path))
            else:
                yield c

    for path in paths:
        if path not in included:
            yield from read(path)


def _input_cycle(e):
    timestep, step, elapsed = 1.0, 0, 0.0
    active = {}
    for c in _commands(e):
        if c[0] == "timestep":
            timestep = float(c[1])
            if not np.isfinite(timestep) or timestep <= 0:
                return False, "Integration timestep must be positive"
        elif c[0] == "reset_timestep":
            step = int(c[1])
        elif c[0] == "fix":
            active[c[1]] = c
        elif c[0] == "unfix":
            active.pop(c[1], None)
        elif c[0] == "run":
            if "every" in c:
                raise UnsupportedEvidence(
                    "Commands executed during a run are unverified"
                )
            count = int(c[1]) - (step if "upto" in c else 0)
            if count < 0:
                return False, "Negative run duration"
            if count == 0:
                continue
            fixes = [f for f in active.values() if f[3] in {"npt", "nvt", "nve"}]
            if len(fixes) != 1 or fixes[0][2:4] != ["all", "npt"]:
                return False, "Each run must advance all atoms under NPT"
            fix = fixes[0]
            temp, iso = fix.index("temp"), fix.index("iso")
            first, last, damping = map(float, fix[temp + 1 : temp + 4])
            pressure = list(map(float, fix[iso + 1 : iso + 4]))
            if (
                not np.isfinite([first, last, damping, *pressure]).all()
                or pressure[:2] != [0, 0]
                or pressure[2] <= 0
                or damping <= 0
            ):
                return False, "Each run requires zero-pressure isotropic NPT"
            start = int(c[c.index("start") + 1]) if "start" in c else step
            stop = int(c[c.index("stop") + 1]) if "stop" in c else step + count
            if not start <= step < step + count <= stop:
                return False, "Run lies outside its temperature ramp"
            actual = first + (last - first) * (
                np.array([step, step + count]) - start
            ) / (stop - start)
            duration = count * timestep / 1000
            if elapsed < 370 - 1e-7:
                end, expected = 370, 4000 - 10 * np.array([elapsed, elapsed + duration])
            elif elapsed < 470 - 1e-7:
                end, expected = 470, [300, 300]
            else:
                end, expected = (
                    700,
                    300 + 10 * (np.array([elapsed, elapsed + duration]) - 470),
                )
            if elapsed + duration > end + 1e-7 or not _close(
                actual, expected, atol=1e-5, rtol=0
            ):
                return False, "Run commands contradict the requested thermal cycle"
            elapsed += duration
            step += count
    return _close(
        elapsed, 700, atol=1e-7, rtol=0
    ), "Run commands must cover the full 700 ps cycle"


def _potential(e):
    coefficients, cutoffs = {}, None
    pppm = False
    runs = 0
    for c in _commands(e):
        if c[0] == "pair_style":
            args = c[1:]
            if args[0] in {"hybrid", "hybrid/overlay"}:
                args = args[1:]
            if args[0] != "buck/coul/long":
                return False, "Expected the supplied Buckingham/Coulomb potential"
            cutoffs = list(map(float, args[1:]))
            coefficients = {}
        elif c[0] == "pair_coeff":
            args = c[3:]
            if args[0] == "buck/coul/long":
                args = args[1:]
            values = list(map(float, args))
            for pair in _PAIR_COEFFICIENTS:
                if all(
                    token == "*" or int(token) == index
                    for token, index in zip(c[1:3], pair, strict=True)
                ):
                    coefficients[pair] = values
        elif c[0] == "kspace_style":
            pppm = c[1] == "pppm" and 0 < float(c[2]) < 1
        elif c[0] == "run" and int(c[1]) > 0:
            runs += 1
            if (
                not pppm
                or cutoffs != [8.0, 12.0]
                or any(
                    pair not in coefficients
                    or not _close(coefficients[pair], values, atol=1e-7, rtol=1e-7)
                    for pair, values in _PAIR_COEFFICIENTS.items()
                )
            ):
                return (
                    False,
                    "Run commands do not use the supplied BKS coefficients and cutoffs",
                )
    return runs > 0


def _physics(e):
    commands = list(_commands(e))
    inputs = "\n".join(" ".join(c) for c in commands)
    logs = _text(e, "raw_logs", "logs", "lammps_logs")
    # Comments alone are not evidence of an input command.
    clean = "\n".join(line.split("#", 1)[0] for line in inputs.splitlines()).lower()
    required = [
        r"(?m)^\s*units\s+real\b",
        r"(?m)^\s*atom_style\s+charge\b",
        r"(?m)^\s*pair_style\s+[^\n]*buck[^\n]*coul",
        r"(?m)^\s*kspace_style\s+pppm\b",
        r"(?m)^\s*read_(?:data|restart)\s+\S+",
    ]
    if not all(re.search(pattern, clean) for pattern in required):
        return False, "Saved inputs do not establish the requested LAMMPS physics"
    boundary = ["p", "p", "p"]  # LAMMPS default before an explicit override.
    for command in commands:
        if command[0] == "boundary":
            boundary = command[1:]
        elif (
            command[0] == "run" and int(command[1]) > 0 and boundary != ["p", "p", "p"]
        ):
            return False, "Every dynamics stage requires periodic boundaries"
    fixes = re.findall(r"(?m)^\s*fix\s+[^\n]+\bnpt\b[^\n]*", clean)
    if not fixes or any(
        not re.search(r"\biso\s+0(?:\.0*)?\s+0(?:\.0*)?(?:\s|$)", fix) for fix in fixes
    ):
        return (
            False,
            "Isotropic zero-pressure NPT commands are missing or contradictory",
        )
    if re.search(
        r"\b(?:nvt|nve)\b", "\n".join(re.findall(r"(?m)^\s*fix\s+[^\n]+", clean))
    ):
        return False, "Another saved integrator contradicts NPT throughout"
    if (
        not re.search(r"\bStep\b", logs)
        or not re.search(r"\bTemp\b", logs)
        or not re.search(r"\bDensity\b", logs)
    ):
        return False, "Raw thermo logs must include step, temperature and density"
    return _potential(e)


def _logged_trace(e):
    t = _trace(e)
    rows = {}
    header = None
    for line in _text(e, "raw_logs", "logs", "lammps_logs").splitlines():
        tokens = line.split()
        if {"Step", "Temp", "Press", "Density"} <= set(tokens):
            header = tokens
            continue
        if header is None or len(tokens) != len(header):
            continue
        try:
            values = dict(zip(header, map(float, tokens), strict=False))
        except ValueError:
            continue
        rows.setdefault(values["Step"], []).append(values)
    if not rows:
        return False, "No numeric thermo rows corroborate the trace"
    if "step" in t:
        steps = t.step.to_numpy(dtype=float)
    else:
        settings = [c[1] for c in _commands(e) if c[0] == "timestep"]
        try:
            dt = [float(value) for value in settings] or [1.0]
        except ValueError as exc:
            raise UnsupportedEvidence(
                "Include trace step values when timestep expressions cannot be resolved"
            ) from exc
        if len(set(dt)) != 1 or dt[0] <= 0:
            raise UnsupportedEvidence(
                "Include trace step values when changing the integration timestep"
            )
        steps = (t.time_ps.to_numpy() - t.time_ps.iloc[0]) * 1000 / dt[0] + min(rows)
    for position, step in enumerate(steps):
        if not _close(step, round(step), atol=1e-4, rtol=0):
            return False, "Trace sample time does not correspond to an integration step"
        sample = t.iloc[position]
        candidates = rows.get(round(step), [])
        if not any(
            _close(row["Temp"], sample.temperature_K, atol=0.01)
            and _close(row["Density"], sample.density_g_cm3, atol=1e-6)
            and _close(row["Press"], sample.pressure_atm, atol=0.01)
            for row in candidates
        ):
            return (
                False,
                f"Trace row {position} is not corroborated by saved thermo output",
            )
    return True


def _boundary_density(e):
    t = _trace(e)
    states = e.json("boundary_states", "stage_boundaries")
    for name, stage, location in [
        ("initial", "cooling", 0),
        ("cooling_end", "cooling", -1),
        ("hold_start", "hold", 0),
        ("hold_end", "hold", -1),
        ("reheating_start", "reheating", 0),
        ("reheating_end", "reheating", -1),
    ]:
        state = _state(states[name])
        mass = sum(
            atomic_masses[atomic_numbers[str(symbol)]] for symbol in state["species"]
        )
        density = mass * 1.66053906660 / np.linalg.det(state["cell"])
        sample = t[t.stage == stage].iloc[location]
        if not _close(density, sample.density_g_cm3, atol=1e-5):
            return (
                False,
                f"{name} cell and atomic masses contradict the endpoint density",
            )
        if not _close(state["time_ps"], sample.time_ps, atol=0.5, rtol=0.001):
            return False, f"{name} timing contradicts the corresponding trace endpoint"
    return True


def _state(s):
    ids = np.asarray(s["ids"])
    order = np.argsort(ids)
    if len(ids) < 3 or len(np.unique(ids)) != len(ids):
        raise ValueError("Boundary atom IDs must be unique")
    state = {
        key: np.asarray(s[key])[order]
        for key in ["positions", "velocities", "charges", "species"]
    }
    state["ids"] = ids[order]
    state["cell"] = np.asarray(s["cell"], float)
    if (
        state["cell"].shape != (3, 3)
        or not np.isfinite(state["cell"]).all()
        or np.linalg.det(state["cell"]) <= 0
    ):
        raise ValueError("Boundary cell is not a finite positive-volume cell")
    for key in ["positions", "velocities"]:
        if (
            state[key].shape != (len(ids), 3)
            or not np.isfinite(state[key].astype(float)).all()
        ):
            raise ValueError(f"Invalid boundary {key}")
    if (
        state["charges"].shape != (len(ids),)
        or not np.isfinite(state["charges"].astype(float)).all()
    ):
        raise ValueError("Invalid charges")
    if set(state["species"]) != {"Na", "Si", "O"}:
        raise ValueError("States must contain the sodium silicate species")
    expected = [_CHARGES[symbol] for symbol in state["species"]]
    if not _close(state["charges"], expected, atol=1e-7, rtol=0):
        raise ValueError("Boundary charges do not match the supplied Na/Si/O potential")
    state["time_ps"] = float(s["time_ps"])
    return state


def _boundaries(e):
    states = e.json("boundary_states", "stage_boundaries")
    names = [
        "initial",
        "cooling_end",
        "hold_start",
        "hold_end",
        "reheating_start",
        "reheating_end",
    ]
    data = {name: _state(states[name]) for name in names}
    initial = data["initial"]
    for s in data.values():
        if not np.array_equal(initial["ids"], s["ids"]) or not np.array_equal(
            initial["species"], s["species"]
        ):
            return False, "Atom identities change between saved states"
        if not _close(initial["charges"], s["charges"], atol=1e-7, rtol=0):
            return False, "Atomic charges change between saved states"
    times = np.asarray([data[name]["time_ps"] for name in names]) - initial["time_ps"]
    if not _close(times, [0, 370, 370, 470, 470, 700], atol=0.5, rtol=0.01):
        return False, "Boundary times contradict the thermal cycle"
    for left, right in [("cooling_end", "hold_start"), ("hold_end", "reheating_start")]:
        a, b = data[left], data[right]
        if not _close(a["cell"], b["cell"], atol=1e-7, rtol=1e-6):
            return False, "Cell changed at a stage handoff"
        if not _close(a["velocities"], b["velocities"], atol=1e-7, rtol=1e-6):
            return False, "Velocities changed at a stage handoff"
        fractional = (
            a["positions"].astype(float) - b["positions"].astype(float)
        ) @ np.linalg.inv(a["cell"])
        difference = (fractional - np.rint(fractional)) @ a["cell"]
        if not _close(difference, 0, atol=2e-5, rtol=0):
            return False, "Positions changed at a stage handoff"
    return True


def _indices(record):
    if record["method"] == "line_intersection":
        return record["low_indices"] + record["high_indices"]
    return record["indices"]


def _selected(t, stage, indices):
    index = np.asarray(indices)
    if (
        index.ndim != 1
        or not len(index)
        or index.dtype.kind not in "iu"
        or np.any(index < 0)
        or np.any(index >= len(t))
    ):
        raise ValueError(
            "Selection must contain valid zero-based integer trace indices"
        )
    selected = t.iloc[index]
    if not (selected.stage == stage).all():
        raise ValueError("Selection contains samples from a different thermal stage")
    return selected


def _estimate(t, stage, rec):
    if rec.get("temperature_coordinate") != "temperature_K":
        raise ValueError("Transition estimation must use measured temperature")
    selected = _selected(t, stage, _indices(rec))
    if "temperature_values_K" in rec and not _close(
        rec["temperature_values_K"], selected.temperature_K, atol=1e-5, rtol=1e-7
    ):
        raise ValueError(
            "Supplied temperature coordinate does not match the measured trace"
        )
    if rec["method"] == "line_intersection":
        fits = []
        parts = []
        for key in ["low", "high"]:
            part = _selected(t, stage, rec[f"{key}_indices"])
            x, y = part.temperature_K.to_numpy(), part.density_g_cm3.to_numpy()
            if len(np.unique(x)) < 2:
                raise ValueError(
                    "Each line requires two distinct measured temperatures"
                )
            weights = np.asarray(
                rec.get("fit_weights", {}).get(key, np.ones(len(x))), float
            )
            if not (
                weights.shape == x.shape
                and np.isfinite(weights).all()
                and np.all(weights > 0)
            ):
                raise ValueError(
                    "Line-fit weights must be finite, positive, and match selected data"
                )
            fits.append(np.polyfit(x, y, 1, w=np.sqrt(weights)))
            parts.append(x)
            supplied = rec.get("fit_coefficients", {}).get(key)
            if supplied is not None and not result_close(supplied, fits[-1], atol=1e-8):
                raise ValueError("Reported line coefficients do not reproduce the data")
        if np.max(parts[0]) >= np.min(parts[1]):
            raise ValueError(
                "Low and high fitting regions must be ordered and nonoverlapping"
            )
        if abs(fits[0][0] - fits[1][0]) < 1e-12:
            raise ValueError("Parallel density lines do not identify a transition")
        value = (fits[1][1] - fits[0][1]) / (fits[0][0] - fits[1][0])
    elif rec["method"] == "change_point":
        x, y = selected.temperature_K.to_numpy(), selected.density_g_cm3.to_numpy()
        candidates = np.asarray(rec["candidates_K"], float)
        if (
            candidates.ndim != 1
            or len(np.unique(candidates)) < 1
            or not np.isfinite(candidates).all()
        ):
            raise ValueError("Change point candidates must contain finite temperatures")
        losses = []
        for candidate in candidates:
            if (
                len(np.unique(x[x < candidate])) < 2
                or len(np.unique(x[x > candidate])) < 2
            ):
                losses.append(np.inf)
                continue
            design = np.column_stack([np.ones(len(x)), x, np.maximum(x - candidate, 0)])
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            losses.append(np.sum((design @ coefficients - y) ** 2))
        if not np.isfinite(losses).any():
            raise ValueError("No change point has sufficient samples on both sides")
        value = candidates[np.argmin(losses)]
    else:
        if not isinstance(rec.get("calculations"), dict) or not rec["calculations"]:
            raise ValueError("Retain calculations for the chosen transition estimator")
        raise UnsupportedEvidence(
            f"Estimator {rec['method']} needs a dedicated numerical verifier"
        )
    x = selected.temperature_K.to_numpy()
    if (
        not np.isfinite(value)
        or len(np.unique(x[x < value])) < 2
        or len(np.unique(x[x > value])) < 2
    ):
        raise ValueError("Transition is not supported on both sides by measured data")
    if not result_close(rec["estimate_K"], value, atol=0.5):
        raise ValueError("Reported estimate does not reproduce the declared estimator")
    return float(value)


def _analysis(e):
    if "analysis" in e._artifacts:
        return e.json("analysis")
    return e.report


def _transitions(e):
    return _analysis(e)["transitions"]


def _selection(e):
    t = _trace(e)
    for stage, rec in _transitions(e).items():
        if stage not in {"cooling", "reheating"}:
            continue
        part = _selected(t, stage, _indices(rec))
        if len(part) < 2 or np.ptp(part.temperature_K) <= 0:
            return False
    return {"cooling", "reheating"} <= set(_transitions(e))


def _coordinate(e):
    t = _trace(e)
    for stage in ["cooling", "reheating"]:
        rec = _transitions(e)[stage]
        if rec.get("temperature_coordinate") != "temperature_K":
            return False
        selected = _selected(t, stage, _indices(rec))
        if "temperature_values_K" in rec and not _close(
            rec["temperature_values_K"], selected.temperature_K, atol=1e-5, rtol=1e-7
        ):
            return False
    return True


def _hold(e, drift=False):
    t = _trace(e)
    rec = _analysis(e)["hold"]
    part = _selected(t, "hold", rec["drift_indices" if drift else "indices"])
    minimum = 2 if drift else 1
    if len(part) < minimum or len(np.unique(part.time_ps)) < minimum:
        raise ValueError(
            "Hold selection has insufficient distinct times for its calculation"
        )
    return part


def evaluate(e, r):
    """Add task checks to the shared rubric (exactly 90 available points)."""

    trace_ok = False

    def thermal_cycle():
        nonlocal trace_ok
        trace = _schedule(e)
        if isinstance(trace, tuple) and not trace[0]:
            return trace
        trace_ok = bool(trace)
        return _input_cycle(e)

    r.check("thermal_cycle", 12, lambda: _verified(thermal_cycle))
    r.check("saved_physics_and_logs", 8, lambda: _verified(lambda: _physics(e)))
    r.check("boundary_state_continuity", 10, lambda: _boundaries(e))
    r.check("thermal_observables", 4, lambda: _verified(lambda: _logged_trace(e)))
    r.check("boundary_density_consistency", 3, lambda: _boundary_density(e))
    selections = r.check("transition_selections", 8, lambda: _selection(e))
    coordinate = r.check(
        "measured_temperature_coordinate", 8, lambda: selections and _coordinate(e)
    )
    for stage in ["cooling", "reheating"]:
        r.check(
            f"{stage}_transition_reproduction",
            6,
            lambda stage=stage: _verified(
                lambda: (
                    coordinate
                    and trace_ok
                    and result_close(
                        e.results[f"{stage}_tg_K"],
                        _estimate(_trace(e), stage, _transitions(e)[stage]),
                        atol=0.5,
                    )
                )
            ),
        )
        try:
            rec = _transitions(e)[stage]
            if rec.get("method") not in {"line_intersection", "change_point"}:
                r.unverified(
                    f"{stage}_custom_estimator",
                    "Saved custom estimator calculations require independent review before a final score is available.",
                )
        except (KeyError, TypeError):
            pass
    r.check(
        "signed_transition_difference",
        4,
        lambda: coordinate
        and result_close(
            e.results["delta_tg_K"],
            float(e.results["reheating_tg_K"]) - float(e.results["cooling_tg_K"]),
            atol=0.5,
        ),
    )
    r.check(
        "reported_estimates_and_units",
        4,
        lambda: all(
            e.results["units"].get(key) == value
            for key, value in [
                ("temperature", "K"),
                ("density", "g/cm3"),
                ("density_drift", "g/cm3/ps"),
            ]
        )
        and all(
            result_close(
                e.results[f"{stage}_tg_K"],
                _transitions(e)[stage]["estimate_K"],
                atol=0.5,
            )
            for stage in ["cooling", "reheating"]
        ),
    )
    r.check(
        "hold_windows",
        5,
        lambda: len(_hold(e)) >= 1 and len(_hold(e, True)) >= 2,
    )
    r.check(
        "hold_means",
        6,
        lambda: trace_ok
        and result_close(
            e.results["hold_temperature_K"], _hold(e).temperature_K.mean(), atol=1e-05
        )
        and result_close(
            e.results["hold_density_g_cm3"], _hold(e).density_g_cm3.mean(), atol=1e-07
        ),
    )

    def drift():
        part = _hold(e, True)
        record = _analysis(e)["hold"]
        method = record.get("drift_method", "linear_slope")
        x, y = part.time_ps.to_numpy(), part.density_g_cm3.to_numpy()
        if method == "linear_slope":
            slope = np.polyfit(x, y, 1)[0]
        elif method == "endpoint_slope":
            first, last = np.argmin(x), np.argmax(x)
            slope = (y[last] - y[first]) / (x[last] - x[first])
        elif method == "theil_sen":
            i, j = np.triu_indices(len(x), 1)
            distinct = x[i] != x[j]
            slope = np.median(
                (y[j[distinct]] - y[i[distinct]]) / (x[j[distinct]] - x[i[distinct]])
            )
        else:
            if (
                not isinstance(record.get("calculations"), dict)
                or not record["calculations"]
            ):
                raise ValueError("Retain calculations for the chosen drift estimator")
            raise UnsupportedEvidence("Saved custom drift calculations require review")
        return trace_ok and result_close(
            e.results["hold_density_drift_g_cm3_ps"], slope, atol=1e-09
        )

    r.check("hold_density_drift", 6, lambda: _verified(drift))
    r.unverified(
        "execution_provenance",
        "Saved inputs, logs and states can establish consistency; they do not prove execution, supplied-potential identity, or absence of unrecorded velocity resets.",
    )
