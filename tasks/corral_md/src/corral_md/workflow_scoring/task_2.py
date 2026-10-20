"""Recompute glass-transition analysis from saved evidence; never run MD."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import numpy as np
from ase import units
from ase.data import atomic_masses, atomic_numbers
from ase.io import read

from .common import EvidenceError, UnsupportedEvidence, aliased_value, result_close

# From potentials.zip/potentials/BKS/pot.mod and the supplied 4000 K melt.
_CHARGES = {"Na": 0.6, "Si": 2.4, "O": -1.2}
PROTOCOL_VERSION = "task2-4000K-v3"
_INITIAL_STEP = 0
_COOLING_DURATION_PS = 370
_HOLD_DURATION_PS = 100
_REHEATING_DURATION_PS = 370
_START_TEMPERATURE_K = 4000
_MAX_FIT_TEMPERATURE_K = 2600
_END_TEMPERATURE_K = 300
_RAMP_RATE_K_PS = 10
_COOLING_STEPS = 370_000
_PREPARATION_STEPS = 470_000
_CYCLE_STEPS = 840_000
_MIN_TRANSITION_BIC_GAIN = 6.0
_HOLD_END_PS = _COOLING_DURATION_PS + _HOLD_DURATION_PS
_CYCLE_END_PS = _HOLD_END_PS + _REHEATING_DURATION_PS
_SUPPLIED_BKS_PATH = "/workspace/potentials/BKS/pot.mod"
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


def _trace(e, required_stages=None):
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
    expected = (
        {"cooling", "hold", "reheating"}
        if required_stages is None
        else set(required_stages)
    )
    if "stage" not in table:
        if expected != {"cooling"}:
            raise ValueError("Trace must identify each thermal stage")
        table["stage"] = "cooling"
    else:
        table["stage"] = table.stage.astype(str).str.lower().replace(
            {"cool": "cooling", "reheat": "reheating"}
        )
    stages = set(table.stage)
    if stages == expected | {"initial"} and "cooling" in expected:
        initial = table[table.stage == "initial"]
        cooling_start = table[table.stage == "cooling"].iloc[0]
        if (
            len(initial) != 1
            or initial.index[0] != table.index[0]
            or initial.iloc[0].time_ps != 0
            or ("step" in table and initial.iloc[0].step != 0)
            or not _close(
                initial.iloc[0][columns].to_numpy(dtype=float),
                cooling_start[columns].to_numpy(dtype=float),
                atol=1e-8,
                rtol=1e-7,
            )
        ):
            raise ValueError("Initial trace sample must reproduce the cooling start")
        # Keep the row so all submitted positional analysis indices stay valid.
        stages.remove("initial")
    if stages != expected:
        names = ", ".join(sorted(expected))
        raise ValueError(f"Trace must contain exactly these stages: {names}")
    return table


def _schedule(e, required_stages=None):
    t = _trace(e, required_stages)
    expected = [
        ("cooling", 0, _COOLING_DURATION_PS, _START_TEMPERATURE_K, _END_TEMPERATURE_K),
        (
            "hold",
            _COOLING_DURATION_PS,
            _HOLD_END_PS,
            _END_TEMPERATURE_K,
            _END_TEMPERATURE_K,
        ),
        (
            "reheating",
            _HOLD_END_PS,
            _CYCLE_END_PS,
            _END_TEMPERATURE_K,
            _START_TEMPERATURE_K,
        ),
    ]
    origin = float(t.time_ps.iloc[0])
    if not _close(origin, 0, atol=1e-7, rtol=0):
        return False, "The supplied melt starts at simulation time zero"
    if np.any(np.diff(t.time_ps) < 0):
        return False, "Trace elapsed times are not chronological"
    for stage, start, end, first, last in expected:
        if required_stages is not None and stage not in required_stages:
            continue
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
    linked = list(dict.fromkeys(
        path for name in e._artifacts for path in e.artifacts(name)
    ))
    variables, included = {}, set()

    def read(path, parents=()):
        if path in parents:
            raise ValueError("Recursive LAMMPS include")
        for raw in path.read_text().replace("&\n", " ").splitlines():
            line = raw.split("#", 1)[0]
            preview = shlex.split(line)
            if preview[:1] == ["print"] and len(preview) == 2:
                # A screen-only status message with literal step/time arithmetic
                # cannot alter the integration protocol. Keep other expressions
                # and print-to-file commands subject to the normal checks.
                status = re.sub(
                    r"\$\(((?:step|time|dt|[0-9.\s+*/%\-])+)\)",
                    "",
                    preview[1],
                )
                if "$" not in status:
                    continue
            if len(preview) > 3 and preview[:1] == ["fix"] and preview[3] == "print":
                # Output-only expressions may reference step, time, or thermo
                # variables without changing the NPT protocol being checked.
                continue
            for name, value in variables.items():
                if value is None and (
                    "${" + name + "}" in line or (len(name) == 1 and "$" + name in line)
                ):
                    raise UnsupportedEvidence(
                        "LAMMPS expressions used by simulation commands are unverified"
                    )
                if value is None:
                    continue
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
                    except ValueError:
                        variables[c[1]] = None
                        continue
                variables[c[1]] = c[3]
            elif c[0] == "include":
                candidates = [p for p in linked if p.name == Path(c[1]).name]
                if not candidates and c[1] == _SUPPLIED_BKS_PATH:
                    # This read-only asset is mounted in every MD workspace.
                    # Its exact coefficients are fixed by the benchmark.
                    yield [
                        "pair_style",
                        "hybrid/overlay",
                        "buck/coul/long",
                        "8.0",
                        "12.0",
                    ]
                    yield [
                        "pair_coeff",
                        "*",
                        "*",
                        "buck/coul/long",
                        "0.0",
                        "1.0",
                        "0.0",
                    ]
                    for (first, last), values in _PAIR_COEFFICIENTS.items():
                        if values != (0, 1, 0):
                            yield [
                                "pair_coeff",
                                str(first),
                                str(last),
                                "buck/coul/long",
                                *map(str, values),
                            ]
                    yield ["kspace_style", "pppm", "1.0e-4"]
                    continue
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


def _input_cycle(
    e, end_time_ps=_CYCLE_END_PS, *, required_timestep_fs=None, required_steps=None
):
    timestep, step, elapsed = 1.0, 0, 0.0
    advanced_steps = 0
    active = {}
    for c in _commands(e):
        if c[0] == "read_restart":
            return (
                False,
                "Start the full ramp with read_data from the supplied 4000 K melt",
            )
        elif c[0] == "read_data":
            if advanced_steps:
                return False, "Do not reload the system during the thermal cycle"
            step = _INITIAL_STEP
        elif c[0] == "timestep":
            timestep = float(c[1])
            if not np.isfinite(timestep) or timestep <= 0:
                return False, "Integration timestep must be positive"
        elif c[0] == "reset_timestep":
            return False, "Preserve the simulation step count and elapsed time"
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
            if required_timestep_fs is not None and (
                not _close(timestep, required_timestep_fs, atol=1e-12, rtol=0)
                or any(f[3] == "dt/reset" for f in active.values())
            ):
                return (
                    False,
                    f"Every advancing run requires a {required_timestep_fs:g} fs timestep",
                )
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
            if elapsed < _COOLING_DURATION_PS - 1e-7:
                end = _COOLING_DURATION_PS
                expected = _START_TEMPERATURE_K - _RAMP_RATE_K_PS * np.array(
                    [elapsed, elapsed + duration]
                )
            elif elapsed < _HOLD_END_PS - 1e-7:
                end, expected = _HOLD_END_PS, [_END_TEMPERATURE_K, _END_TEMPERATURE_K]
            else:
                end, expected = (
                    _CYCLE_END_PS,
                    _END_TEMPERATURE_K
                    + _RAMP_RATE_K_PS
                    * (np.array([elapsed, elapsed + duration]) - _HOLD_END_PS),
                )
            required_pdamp = (
                5000 if _COOLING_DURATION_PS <= elapsed < _HOLD_END_PS else 10000
            )
            if damping != 100 or pressure[2] != required_pdamp:
                return (
                    False,
                    "Use 100 fs thermostat damping and 10000/5000/10000 fs stage pressure damping",
                )
            if elapsed + duration > end + 1e-7 or not _close(
                actual, expected, atol=1e-5, rtol=0
            ):
                return False, "Run commands contradict the requested thermal cycle"
            elapsed += duration
            step += count
            advanced_steps += count
    if required_steps is not None and advanced_steps != required_steps:
        return False, f"Protocol requires exactly {required_steps:,} advancing steps"
    return _close(
        elapsed, end_time_ps, atol=1e-7, rtol=0
    ), f"Run commands must cover the full {end_time_ps:g} ps requested cycle"


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


def _logged_trace(e, required_stages=None, *, required_timestep_fs=None):
    t = _trace(e, required_stages)
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
    if required_timestep_fs is not None and steps[0] != _INITIAL_STEP:
        return False, "Thermo steps must start at step zero for the supplied melt"
    if required_timestep_fs is not None and not _close(
        steps - steps[0],
        (t.time_ps.to_numpy() - t.time_ps.iloc[0]) * 1000 / required_timestep_fs,
        atol=1e-5,
        rtol=0,
    ):
        return (
            False,
            "Trace step values disagree with the required timestep and elapsed time",
        )
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


def _boundary_density(e, required_stages=None):
    t = _trace(e, required_stages)
    states = _boundary_states(e)
    for name, stage, location in [
        ("initial", "cooling", 0),
        ("cooling_end", "cooling", -1),
        ("hold_start", "hold", 0),
        ("hold_end", "hold", -1),
        ("reheating_start", "reheating", 0),
        ("reheating_end", "reheating", -1),
    ]:
        if required_stages is not None and stage not in required_stages:
            continue
        state = states[name]
        mass = sum(
            atomic_masses[atomic_numbers[str(symbol)]] for symbol in state["species"]
        )
        density = mass * 1.66053906660 / np.linalg.det(state["cell"])
        # A shared handoff can be recorded only under the preceding stage.
        matching = t[np.isclose(t.time_ps, state["time_ps"], rtol=0, atol=1e-6)]
        if matching.empty:
            return False, f"{name} has no thermal sample at its boundary time"
        previous_stage = {"hold_start": "cooling", "reheating_start": "hold"}.get(name)
        if previous_stage and (matching.stage == previous_stage).any():
            sample = t[t.stage == previous_stage].iloc[-1]
        else:
            sample = t[t.stage == stage].iloc[location]
        if not _close(state["time_ps"], sample.time_ps, atol=0.5, rtol=0.001):
            return False, f"{name} timing contradicts the corresponding trace endpoint"
        if not _close(density, matching.density_g_cm3, atol=1e-5):
            return (
                False,
                f"{name} cell and atomic masses contradict the endpoint density",
            )
    return True


def _state(s):
    s = {
        **s,
        **{
            key: aliased_value(s, key, alias)
            for key, alias in (
                ("ids", "atom_ids"),
                ("cell", "cell_A"),
                ("positions", "positions_A"),
                ("velocities", "velocities_A_fs"),
                ("charges", "charges_e"),
            )
        },
    }
    if "positions_lammps" in s:
        # An explicit pair may describe zero-origin positions alongside the
        # original LAMMPS coordinates. Verify the declared translation before
        # using the original frame for source and handoff comparisons.
        origin = np.asarray(s["cell_origin"], dtype=float)
        absolute = np.asarray(s["positions_lammps"], dtype=float)
        relative = np.asarray(s["positions"], dtype=float)
        if (
            origin.shape != (3,)
            or absolute.shape != relative.shape
            or not _close(absolute, relative + origin, atol=1e-7, rtol=1e-7)
        ):
            raise EvidenceError("Conflicting boundary coordinate origins")
        s["positions"] = absolute
    elif {"positions_wrapped", "images", "cell_origin"} <= s.keys():
        origin = np.asarray(s["cell_origin"], dtype=float)
        wrapped = np.asarray(s["positions_wrapped"], dtype=float)
        positions = np.asarray(s["positions"], dtype=float)
        images = np.asarray(s["images"], dtype=float)
        cell = np.asarray(s["cell"], dtype=float)
        if cell.shape == (3,):
            cell = np.diag(cell)
        if (
            origin.shape != (3,)
            or wrapped.shape != positions.shape
            or images.shape != positions.shape
            or not np.array_equal(images, np.rint(images))
        ):
            raise EvidenceError("Invalid wrapped boundary coordinate metadata")
        absolute = wrapped + images @ cell
        if not (
            _close(positions, absolute, atol=1e-7, rtol=1e-7)
            or _close(positions + origin, absolute, atol=1e-7, rtol=1e-7)
        ):
            raise EvidenceError("Conflicting wrapped boundary coordinate metadata")
        s["positions"] = absolute
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
    if state["cell"].shape == (3,) and np.all(state["cell"] > 0):
        # Three positive lengths explicitly describe an orthorhombic cell.
        state["cell"] = np.diag(state["cell"])
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


def _boundary_states(e):
    """Read inline states or explicitly linked, restartable LAMMPS data files."""
    document = e.artifact("boundary_states", "stage_boundaries")
    data = e.json("boundary_states", "stage_boundaries")
    result = {}
    for name, record in data.items():
        if name in ("description", "atom_style", "atom_type_species"):
            continue
        linked = record.get("lammps_data")
        if "data_path" in record and record.get("data_format") == "lammps-data":
            linked = aliased_value(record, "lammps_data", "data_path")
        if linked is None:
            result[name] = _state(record)
            continue
        path = e.linked_path(linked, document)
        species = record.get("atom_type_species", data.get("atom_type_species"))
        mapping = (
            {int(key): atomic_numbers[value] for key, value in species.items()}
            if species is not None
            else None
        )
        atoms = read(
            path,
            format="lammps-data",
            units="real",
            atom_style=record.get("atom_style", data.get("atom_style", "charge")),
            Z_of_type=mapping,
        )
        if not {"id", "initial_charges", "momenta"} <= atoms.arrays.keys():
            raise EvidenceError(
                f"Boundary {name} must save atom IDs, charges and velocities"
            )
        exported = {
            "time_ps": record["time_ps"],
            "ids": atoms.arrays["id"],
            "species": atoms.get_chemical_symbols(),
            "positions": atoms.positions,
            "cell": atoms.cell.array,
            "charges": atoms.get_initial_charges(),
            "velocities": atoms.get_velocities() * units.fs,
        }
        state = _state(exported)
        if "ids" in record or "atom_ids" in record:
            inline = _state(record)
            for key in state:
                same = (
                    np.array_equal(inline[key], state[key])
                    if key in ("ids", "species")
                    else _close(inline[key], state[key], atol=1e-7, rtol=1e-7)
                )
                if key == "positions" and not same:
                    # LAMMPS image flags are unwrapped by ASE, whereas inline
                    # snapshots may retain wrapped coordinates. Compare the
                    # same periodic images with the existing numeric tolerance.
                    displacement = inline[key].astype(float) - state[key]
                    images = np.rint(displacement @ np.linalg.inv(state["cell"]))
                    aligned = inline[key] - images @ state["cell"]
                    same = _close(aligned, state[key], atol=1e-7, rtol=1e-7)
                if not same:
                    raise EvidenceError(
                        f"Inline and LAMMPS boundary evidence disagree on {key}"
                    )
        result[name] = state
    return result


def _boundaries(e, required_stages=None):
    states = _boundary_states(e)
    names = [
        "initial",
        "cooling_end",
        "hold_start",
        "hold_end",
        "reheating_start",
        "reheating_end",
    ]
    expected_times = [
        0,
        _COOLING_DURATION_PS,
        _COOLING_DURATION_PS,
        _HOLD_END_PS,
        _HOLD_END_PS,
        _CYCLE_END_PS,
    ]
    if required_stages == {"cooling", "hold"}:
        names, expected_times = names[:4], expected_times[:4]
    data = {name: states[name] for name in names}
    initial = data["initial"]
    for s in data.values():
        if not np.array_equal(initial["ids"], s["ids"]) or not np.array_equal(
            initial["species"], s["species"]
        ):
            return False, "Atom identities change between saved states"
        if not _close(initial["charges"], s["charges"], atol=1e-7, rtol=0):
            return False, "Atomic charges change between saved states"
    times = np.asarray([data[name]["time_ps"] for name in names]) - initial["time_ps"]
    if not _close(
        times,
        expected_times,
        atol=0.5,
        rtol=0.01,
    ):
        return False, "Boundary times contradict the thermal cycle"
    for left, right in [("cooling_end", "hold_start"), ("hold_end", "reheating_start")]:
        if right not in data:
            continue
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
        or len(np.unique(index)) != len(index)
    ):
        raise ValueError(
            "Selection must contain unique valid zero-based integer trace indices"
        )
    selected = t.iloc[index]
    if not (selected.stage == stage).all():
        # Deduplicated traces may retain a handoff once, labeled with the stage
        # that ended. Accept only that exact shared endpoint, without shifting
        # indices or substituting any submitted samples.
        handoff = {
            "hold": ("cooling", _COOLING_DURATION_PS, _COOLING_STEPS),
            "reheating": ("hold", _HOLD_END_PS, _PREPARATION_STEPS),
        }.get(stage)
        other = selected[selected.stage != stage]
        shared = False
        if handoff is not None and len(other) == 1:
            previous, time_ps, step = handoff
            row = other.iloc[0]
            current = t[t.stage == stage]
            preceding = t[t.stage == previous]
            shared = (
                row.stage == previous
                and _close(row.time_ps, time_ps, atol=1e-9, rtol=0)
                and ("step" not in t or _close(row.step, step, atol=1e-7, rtol=0))
                and len(current) > 0
                and (current.time_ps > time_ps).all()
                and len(preceding) > 0
                and preceding.index[-1] == other.index[0]
                and np.isclose(t.time_ps, time_ps, atol=1e-9, rtol=0).sum() == 1
            )
        if not shared:
            raise ValueError("Selection contains samples from a different thermal stage")
    return selected


def _estimate(t, stage, rec):
    if rec.get("temperature_coordinate") != "temperature_K":
        raise ValueError("Transition estimation must use measured temperature")
    selected = _selected(t, stage, _indices(rec))
    if (selected.temperature_K > _MAX_FIT_TEMPERATURE_K).any():
        raise ValueError(
            "Exclude measured temperatures above 2600 K from transition fitting"
        )
    if rec["method"] == "unresolved":
        if (
            rec["estimate_K"] is not None
            or not isinstance(rec.get("reason"), str)
            or not rec["reason"].strip()
        ):
            raise ValueError(
                "An unresolved transition needs a null estimate and an explanation"
            )
        x, y = selected.temperature_K.to_numpy(), selected.density_g_cm3.to_numpy()
        if len(np.unique(x)) < 4:
            raise ValueError(
                "Retain the measured temperature range assessed for an unresolved transition"
            )
        coefficients = np.polyfit(x, y, 1)
        sse = np.sum((y - np.polyval(coefficients, x)) ** 2)
        if not result_close(rec["diagnostics"]["single_line_sse"], sse, atol=1e-10):
            raise ValueError(
                "Unresolved-transition diagnostic does not reproduce the selected data"
            )
        return None
    if "temperature_values_K" in rec and not _close(
        rec["temperature_values_K"], selected.temperature_K, atol=1e-5, rtol=1e-7
    ):
        raise ValueError(
            "Supplied temperature coordinate does not match the measured trace"
        )
    if rec["method"] == "line_intersection":
        fits = []
        parts = []
        fit_weights = []
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
            fit_weights.append(weights)
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
        predictions = np.concatenate(
            [np.polyval(fit, x) for fit, x in zip(fits, parts, strict=True)]
        )
        weights = np.concatenate(fit_weights)
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
        design = np.column_stack([np.ones(len(x)), x, np.maximum(x - value, 0)])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        if abs(coefficients[2]) < 1e-12:
            raise ValueError("A straight density line does not identify a transition")
        predictions = design @ coefficients
        weights = np.ones(len(x))
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
    # A reproducible optimum also exists in noisy, nearly linear data. This is
    # an empirical ambiguity screen, not a significance test for correlated MD
    # samples. Four fitted parameters account for either two independent lines
    # or three hinge coefficients plus the selected breakpoint. Weak evidence
    # goes to review, rather than becoming an incorrect Tg or a scientific fail.
    y = selected.density_g_cm3.to_numpy()
    if len(np.unique(x)) < 6:
        raise UnsupportedEvidence(
            "Too few distinct temperatures to assess two density regimes"
        )
    single = np.polyfit(x, y, 1, w=np.sqrt(weights))
    single_sse = float(np.sum(weights * (y - np.polyval(single, x)) ** 2))
    segmented_sse = float(np.sum(weights * (y - predictions) ** 2))
    tiny = np.finfo(float).tiny
    bic_gain = len(x) * np.log(
        max(single_sse, tiny) / max(segmented_sse, tiny)
    ) - 2 * np.log(len(x))
    if bic_gain < _MIN_TRANSITION_BIC_GAIN:
        raise UnsupportedEvidence(
            f"The two-regime fit has weak support over one line (BIC gain {bic_gain:.3g} < {_MIN_TRANSITION_BIC_GAIN:g}). "
            "Retain diagnostics and fitting-window sensitivity for scientific review, "
            "or report an unresolved transition."
        )
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


def _transition_result_matches(e, stage):
    value = _estimate(_trace(e), stage, _transitions(e)[stage])
    reported = e.results[f"{stage}_tg_K"]
    if value is None:
        if reported is not None:
            return False
        raise UnsupportedEvidence(
            "The unresolved-transition diagnostics reproduce the selected data. "
            "Independent scientific review must assess the explanation and data coverage; "
            "a numerical Tg is not mandatory."
        )
    return result_close(reported, value, atol=0.5)


def _difference_matches(e):
    values = [e.results[f"{stage}_tg_K"] for stage in ("cooling", "reheating")]
    if any(value is None for value in values):
        return e.results["delta_tg_K"] is None
    return result_close(
        e.results["delta_tg_K"], float(values[1]) - float(values[0]), atol=0.5
    )


def _reported_estimate_matches(e, stage):
    rec = _transitions(e)[stage]
    value = e.results[f"{stage}_tg_K"]
    if rec.get("method") == "unresolved":
        return value is None and rec.get("estimate_K") is None
    return result_close(value, rec["estimate_K"], atol=0.5)


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
    from . import lammps_checks

    trace_ok = False

    def thermal_cycle():
        nonlocal trace_ok
        trace = _schedule(e)
        if isinstance(trace, tuple) and not trace[0]:
            return trace
        trace_ok = bool(trace)
        return _input_cycle(e, required_timestep_fs=1.0, required_steps=_CYCLE_STEPS)

    r.check("thermal_cycle", 12, lambda: _verified(thermal_cycle))
    r.check("saved_physics_and_logs", 8, lambda: _verified(lambda: _physics(e)))

    def boundary_check():
        continuity = _boundaries(e)
        if continuity is not True:
            return continuity
        return (
            lammps_checks.supplied_silicate_initial_state(e)
            and lammps_checks.no_silicate_state_resets(e)
            and lammps_checks.cooled_endpoint_temperature(e)
        )

    r.check("boundary_state_continuity", 10, boundary_check)
    r.check(
        "thermal_observables",
        4,
        lambda: _verified(lambda: _logged_trace(e, required_timestep_fs=1.0)),
    )
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
                lambda: coordinate and trace_ok and _transition_result_matches(e, stage)
            ),
        )
        try:
            rec = _transitions(e)[stage]
            if rec.get("method") == "unresolved":
                r.unverified(
                    f"{stage}_transition_interpretation",
                    "No resolved Tg is claimed. The saved diagnostics and explanation require independent scientific review; a numerical Tg is not mandatory.",
                )
            elif rec.get("method") not in {"line_intersection", "change_point"}:
                r.unverified(
                    f"{stage}_custom_estimator",
                    "Saved custom estimator calculations require independent review before a final score is available.",
                )
        except (KeyError, TypeError):
            pass
    r.check(
        "signed_transition_difference",
        4,
        lambda: coordinate and _difference_matches(e),
    )
    r.check(
        "reported_estimates_and_units",
        4,
        lambda: (
            all(
                e.results["units"].get(key) == value
                for key, value in [
                    ("temperature", "K"),
                    ("density", "g/cm3"),
                    ("density_drift", "g/cm3/ps"),
                ]
            )
            and all(
                _reported_estimate_matches(e, stage)
                for stage in ["cooling", "reheating"]
            )
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
        lambda: (
            trace_ok
            and result_close(
                e.results["hold_temperature_K"],
                _hold(e).temperature_K.mean(),
                atol=1e-05,
            )
            and result_close(
                e.results["hold_density_g_cm3"],
                _hold(e).density_g_cm3.mean(),
                atol=1e-07,
            )
        ),
    )

    def drift():
        part = _hold(e, True)
        record = _analysis(e)["hold"]
        method = record.get("drift_method", "linear_slope")
        if isinstance(method, str):
            method = re.sub(r"\s+", "_", method.strip().lower())
        x, y = part.time_ps.to_numpy(), part.density_g_cm3.to_numpy()
        if method in {
            "linear_slope",
            "ordinary_least_squares_linear_slope",
            "ordinary_least_squares_density_vs_time",
            "unweighted_linear_slope",
            "unweighted_linear_slope_of_density_versus_elapsed_time",
            "unweighted_linear_slope_vs_time",
            "unweighted_linear_slope_density_vs_time",
            "unweighted_ols_density_versus_elapsed_time",
            "unweighted_linear_slope_of_density_versus_elapsed_time_over_the_final_80_ps,"
            "_excluding_the_first_20_ps_as_initial_hold_relaxation",
        }:
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
