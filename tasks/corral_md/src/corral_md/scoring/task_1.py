"""Task 1: reconstruct preparation and diffusion from a LAMMPS log and dump.

No submitted analysis or executable code is used. Diffusion is reconstructed
from the prescribed 500 ps interval at 2500 K and compared with the answer.
Both coefficients must also agree with the fixed benchmark reference.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

import numpy as np
from ase import units
from ase.build import bulk

from .common import EvidenceError, UnsupportedEvidence

SILICON_MASS = 28.0855
THERMO_COLUMNS = ("Step", "Time", "Temp", "Press", "Density")
DUMP_COLUMNS = ("id", "type", "xu", "yu", "zu", "vx", "vy", "vz")

# Evaluator-owned target and relative tolerance from Level 2 Task 1 in commit
# 9880925185b52e006212c41878fac70cfed31e57. Submission uncertainty must not
# widen this range; the trajectory comparison below has its own tolerance.
REFERENCE_DIFFUSION_M2_S = 1.74e-8
REFERENCE_DIFFUSION_RTOL = 0.1


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _close(actual, expected):
    return np.allclose(actual, expected, rtol=2e-5, atol=1e-6)


def recording_input(source: str, trajectory: str) -> str:
    """Add native recording to a copy of a Task 1 input, without changing MD."""
    lines = source.splitlines()
    first = next(
        (i for i, line in enumerate(lines) if line.split()[:1] == ["run"]), None
    )
    _require(first is not None, "Task 1 recording needs a run command in the input")
    lines.insert(
        first,
        f'dump corral_task1 all custom $(ceil(1.0/dt):%.0f) "{trajectory}" '
        + " ".join(DUMP_COLUMNS)
        + "\ndump_modify corral_task1 sort id format float %.16g",
    )
    # Reapply thermo output before each run because thermo_style can be reset.
    result = ["echo both"]
    for line in lines:
        if line.split()[:1] == ["run"]:
            result.extend(
                [
                    "thermo_style custom step time temp press density",
                    "thermo_modify format float %.16g",
                    "thermo $(ceil(1.0/dt):%.0f)",
                ]
            )
        result.append(line)
    return "\n".join(result) + "\n"


@dataclass
class Trajectory:
    steps: np.ndarray
    cells: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray | None
    wrapped: bool = False


def read_trajectory(path) -> Trajectory:
    """Read orthogonal native dumps, preserving images and sorting persistent IDs."""
    steps, cells, positions, velocities = [], [], [], []
    identities = None
    wrapped = None
    with path.open() as stream:
        while line := stream.readline():
            if not line.strip():
                continue
            _require(line.strip() == "ITEM: TIMESTEP", "Expected a LAMMPS dump frame")
            step = int(stream.readline())
            _require(
                stream.readline().strip() == "ITEM: NUMBER OF ATOMS",
                "Missing atom count",
            )
            _require(int(stream.readline()) == 216, "Task 1 requires 216 atoms")
            box = stream.readline().split()
            triclinic = box == [
                "ITEM:",
                "BOX",
                "BOUNDS",
                "xy",
                "xz",
                "yz",
                "pp",
                "pp",
                "pp",
            ]
            _require(
                triclinic or box == ["ITEM:", "BOX", "BOUNDS", "pp", "pp", "pp"],
                "Retain orthogonal, fully periodic cells for isotropic NPT",
            )
            bounds = np.array([stream.readline().split() for _ in range(3)], float)
            _require(bounds.shape == (3, 3 if triclinic else 2), "Invalid box bounds")
            if triclinic:
                if not np.allclose(bounds[:, 2], 0, rtol=0, atol=1e-10):
                    raise UnsupportedEvidence(
                        "Nonorthogonal dump cells require a triclinic trajectory adapter"
                    )
                bounds = bounds[:, :2]
            lengths = bounds[:, 1] - bounds[:, 0]
            _require(np.isfinite(bounds).all() and (lengths > 0).all(), "Invalid cell")
            header = stream.readline().split()
            _require(header[:2] == ["ITEM:", "ATOMS"], "Missing atom columns")
            columns = header[2:]
            _require(
                {"id", "type"} <= set(columns), "Retain persistent atom identities"
            )
            has_velocities = {"vx", "vy", "vz"} <= set(columns)
            unwrapped = {"xu", "yu", "zu"} <= set(columns)
            scaled = {"xs", "ys", "zs"} <= set(columns)
            imaged = {"ix", "iy", "iz"} <= set(columns)
            _require(
                unwrapped or scaled or {"x", "y", "z"} <= set(columns),
                "Retain atomic coordinates",
            )
            is_wrapped = not unwrapped and not imaged
            _require(
                wrapped is None or wrapped == is_wrapped,
                "Coordinate convention changed between frames",
            )
            wrapped = is_wrapped
            rows = np.array([stream.readline().split() for _ in range(216)], float)
            _require(
                rows.shape == (216, len(columns)) and np.isfinite(rows).all(),
                "Truncated or nonfinite trajectory frame",
            )
            rows = rows[np.argsort(rows[:, columns.index("id")])]
            ids = rows[:, [columns.index("id"), columns.index("type")]]
            _require(
                (ids == ids.astype(int)).all()
                and (ids > 0).all()
                and len(np.unique(ids[:, 0])) == 216
                and len(np.unique(ids[:, 1])) == 1,
                "Invalid atom identities",
            )
            if identities is None:
                identities = ids
            _require(np.array_equal(ids, identities), "Atom identities changed")
            cell = np.diag(lengths)
            coordinates = (
                ("xu", "yu", "zu")
                if unwrapped
                else (("xs", "ys", "zs") if scaled else ("x", "y", "z"))
            )
            pos = rows[:, [columns.index(c) for c in coordinates]]
            pos = pos * lengths if scaled and not unwrapped else pos - bounds[:, 0]
            if imaged and not unwrapped:
                images = rows[:, [columns.index(c) for c in ("ix", "iy", "iz")]]
                _require(
                    np.array_equal(images, np.rint(images)),
                    "Image counters must be integers",
                )
                pos += images * lengths
            vel = (
                rows[:, [columns.index(c) for c in ("vx", "vy", "vz")]]
                if has_velocities
                else None
            )
            if steps and step == steps[-1]:
                _require(
                    all(
                        (a is None and b is None)
                        or (a is not None and b is not None and _close(a, b))
                        for a, b in zip(
                            (cell, pos, vel),
                            (cells[-1], positions[-1], velocities[-1]),
                            strict=True,
                        )
                    ),
                    "Discontinuous repeated trajectory boundary",
                )
                continue
            _require(not steps or step > steps[-1], "Trajectory steps must increase")
            steps.append(step)
            cells.append(cell)
            positions.append(pos)
            velocities.append(vel)
    _require(len(steps) >= 2, "Retain a nonempty trajectory including endpoints")
    return Trajectory(
        np.array(steps),
        np.array(cells),
        np.array(positions),
        np.array(velocities) if all(v is not None for v in velocities) else None,
        bool(wrapped),
    )


def read_log(path):
    """Extract echoed commands and native thermo rows, without executing input."""
    commands, samples, header = [], [], None
    keywords = {
        "units",
        "atom_style",
        "pair_style",
        "pair_coeff",
        "velocity",
        "timestep",
        "reset_timestep",
        "fix",
        "unfix",
        "run",
        "read_data",
        "read_restart",
        "clear",
        "create_atoms",
        "delete_atoms",
        "change_box",
        "set",
        "displace_atoms",
        "replicate",
        "mass",
        "minimize",
    }
    for line in path.read_text().splitlines():
        _require(not line.startswith("ERROR"), "LAMMPS reported an error")
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "Step":
            _require(
                set(THERMO_COLUMNS) <= set(fields),
                "Log must contain Step Time Temp Press Density",
            )
            header = fields
        elif fields[0] in keywords and "$" not in line:
            commands.append(shlex.split(line, comments=True))
        elif header and len(fields) == len(header):
            try:
                row = np.array([float(fields[header.index(c)]) for c in THERMO_COLUMNS])
            except ValueError:
                continue
            _require(np.isfinite(row).all(), "Nonfinite thermal log")
            if samples and row[0] == samples[-1][0]:
                _require(_close(row, samples[-1]), "Discontinuous thermal boundary")
            else:
                samples.append(row)
    _require(len(samples) >= 2, "No usable thermal records")
    return commands, np.array(samples)


def _protocol(commands, thermal, end):
    dt, step, elapsed, initialized = 0.001, 0, 0.0, False
    active, settings, segments = {}, {}, []
    for command in commands:
        name, *args = command
        if name in {"units", "atom_style", "pair_style", "pair_coeff"}:
            settings[name] = args
        elif name == "timestep":
            dt = float(args[0])
            _require(np.isfinite(dt) and dt > 0, "Invalid integration timestep")
        elif name == "fix":
            active[args[0]] = args[1:]
        elif name == "unfix":
            active.pop(args[0], None)
        elif name == "velocity":
            _require(
                not segments
                and not initialized
                and args[:2] == ["all", "create"]
                and float(args[2]) == 300
                and int(args[3]) > 0,
                "Initialize velocities once at 300 K before heating",
            )
            initialized = True
        elif name == "reset_timestep":
            _require(
                not segments and args == ["0"], "Do not reset time during the cycle"
            )
        elif segments and name in {
            "read_data",
            "read_restart",
            "clear",
            "create_atoms",
            "delete_atoms",
            "change_box",
            "set",
            "displace_atoms",
            "replicate",
            "mass",
            "minimize",
        }:
            raise EvidenceError("State was modified during the thermal cycle")
        elif name == "run":
            _require(
                "every" not in args, "Retain separate echoed runs instead of run every"
            )
            count = int(args[0]) - (step if "upto" in args else 0)
            _require(count >= 0, "Invalid run length")
            if not count:
                continue
            _require(
                settings.get("units") == ["metal"]
                and settings.get("atom_style") == ["full"]
                and settings.get("pair_style") == ["sw"],
                "Use metal/full and the SW model",
            )
            coeff = settings.get("pair_coeff", [])
            _require(
                len(coeff) == 4
                and coeff[:2] == ["*", "*"]
                and coeff[-1] == "Si"
                and (
                    coeff[2] == "/workspace/resources/md-potentials-SW-Si/Si.sw"
                    or coeff[2].endswith("/potentials/SW/Si.sw")
                ),
                "Use the supplied silicon SW potential",
            )
            _require(initialized, "Log the 300 K velocity initialization and seed")
            integrators = [f for f in active.values() if f[1] in {"npt", "nvt", "nve"}]
            _require(
                len(integrators) == 1 and integrators[0][:2] == ["all", "npt"],
                "Each stage requires one isotropic NPT integrator",
            )
            _require(
                all(
                    f is integrators[0]
                    or f[1] in {"ave/time", "ave/correlate", "print", "vector"}
                    for f in active.values()
                ),
                "Additional dynamics fixes require independent review",
            )
            fix = integrators[0]
            ti, pi = fix.index("temp"), fix.index("iso")
            _require(
                _close([float(v) for v in fix[pi + 1 : pi + 3]], [0, 0])
                and float(fix[ti + 3]) > 0
                and float(fix[pi + 3]) > 0,
                "Use zero target pressure and positive damping",
            )
            start = int(args[args.index("start") + 1]) if "start" in args else step
            stop = int(args[args.index("stop") + 1]) if "stop" in args else step + count
            _require(
                start <= step < step + count <= stop, "Invalid temperature ramp bounds"
            )
            first, last = map(float, fix[ti + 1 : ti + 3])
            targets = first + (last - first) * (
                np.array([step, step + count]) - start
            ) / (stop - start)
            finish = elapsed + count * dt
            expected = np.minimum(2500, 300 + 2.2 * np.array([elapsed, finish]))
            _require(
                finish <= (1000 if elapsed < 1000 - 1e-6 else end) + 1e-6
                and _close(targets, expected),
                "Incorrect heating or production schedule",
            )
            segments.append((step, step + count, elapsed, dt))
            elapsed, step = finish, step + count
    _require(_close(elapsed, end), f"Complete the requested {end:g} ps cycle")
    _require(
        _close(thermal[[0, -1], 1], [0, end])
        and (np.diff(thermal[:, 0]) > 0).all()
        and (np.diff(thermal[:, 1]) > 0).all()
        and (thermal[:, [2, 4]] > 0).all(),
        "Incomplete or unordered thermal history",
    )
    _require(np.max(np.diff(thermal[:, 1])) <= 5, "Thermal sampling is too sparse")
    expected_times = _times(thermal[:, 0], segments)
    _require(
        _close(expected_times, thermal[:, 1]), "Thermal times contradict run commands"
    )
    # Heating is a ramp, not a stationary state. Compare block means to its target.
    for block in np.array_split(thermal[thermal[:, 1] <= 1000], 10):
        _require(
            len(block) >= 2
            and abs(np.mean(block[:, 2] - (300 + 2.2 * block[:, 1]))) < 200,
            "Measured temperature does not follow heating",
        )
    return segments


def _times(steps, segments):
    times = np.full(len(steps), np.nan)
    for first, last, start, dt in segments:
        selected = (steps >= first) & (steps <= last)
        times[selected] = start + (steps[selected] - first) * dt
    _require(
        np.isfinite(times).all() and (steps == steps.astype(int)).all(),
        "Steps outside the recorded runs",
    )
    return times


def _states(data, thermal, segments, end):
    times = _times(data.steps, segments)
    valid_start = _close(times[0], 0) if end == 1000 else times[0] <= 1000
    _require(
        _close(times[-1], end) and valid_start and np.any(np.isclose(times, 1000)),
        "Retain the final state and the full interval needed for the requested calculation",
    )
    lengths = np.diagonal(data.cells, axis1=1, axis2=2)
    _require(
        np.allclose(lengths, lengths[:, :1], rtol=0.002),
        "Expected isotropic cubic cells",
    )
    lattice = lengths[0, 0] / 3
    _require(5.3 < lattice < 5.6, "Initial cell is inconsistent with mp-149 silicon")
    expected = bulk("Si", "diamond", a=lattice, cubic=True).repeat((3, 3, 3))
    actual = data.positions[0]
    matches = False
    for anchor in expected.positions[:8]:
        delta = actual[:, None] - actual[0] - (expected.positions[None] - anchor)
        delta -= np.rint(delta / lengths[0]) * lengths[0]
        distances = np.linalg.norm(delta, axis=-1)
        if (
            np.max(np.min(distances, axis=1)) < 0.02
            and len(np.unique(np.argmin(distances, axis=1))) == 216
        ):
            matches = True
            break
    if _close(times[0], 0):
        _require(matches, "Initial state is not the 3 x 3 x 3 diamond repeat")
    density = 216 * SILICON_MASS * 1.66053906660 / np.linalg.det(data.cells)
    indices = np.searchsorted(thermal[:, 0], data.steps)
    _require(
        (indices < len(thermal)).all()
        and np.array_equal(thermal[indices, 0], data.steps),
        "Retain a thermal row for each trajectory frame",
    )
    temperature = thermal[indices, 2]
    if data.velocities is not None:
        measured = (
            SILICON_MASS
            * np.sum((data.velocities / (1000 * units.fs)) ** 2, axis=(1, 2))
            / (645 * units.kB)
        )
        _require(
            np.allclose(measured, temperature, rtol=0.02, atol=1),
            "Trajectory velocities contradict temperature logs",
        )
    _require(
        np.allclose(density, thermal[indices, 4], rtol=0.002),
        "Trajectory cells contradict density logs",
    )
    if _close(times[0], 0):
        _require(abs(temperature[0] - 300) < 60, "Initial temperature is not at 300 K")
    _require(
        abs(temperature[-1] - 2500) < 500,
        "Final state is not near the requested 2500 K",
    )
    # Fractional increments remove homogeneous barostat motion, then COM drift.
    fractional = np.einsum("faj,fjk->fak", data.positions, np.linalg.inv(data.cells))
    increments_fractional = np.diff(fractional, axis=0)
    if data.wrapped:
        _require(
            end == 1000, "Diffusion requires unwrapped positions or image counters"
        )
        # Minimum-image motion is sufficient to establish mobility, not to
        # estimate diffusion. Level 1 asks only for the heated state.
        increments_fractional -= np.rint(increments_fractional)
    motion = np.einsum(
        "faj,fjk->fak",
        increments_fractional,
        (data.cells[1:] + data.cells[:-1]) / 2,
    )
    motion -= motion.mean(axis=1, keepdims=True)
    high_temperature = times[1:] >= (1000 if end > 1000 else times[-2])
    increments = np.mean(np.sum(motion[high_temperature] ** 2, axis=-1), axis=1)
    _require(
        increments.max() <= 10 * max(float(np.median(increments)), 1e-6),
        "Trajectory contains a discontinuous jump or an unwrapping error",
    )
    positions = np.concatenate([np.zeros((1, 216, 3)), np.cumsum(motion, axis=0)])
    return times, positions


def _phase(data, times, positions, level):
    start = 900 if level == 1 else 1100
    indices = np.flatnonzero(times >= start)
    _require(len(indices) >= 2, "Retain multiple high-temperature states")
    adjacency = []
    coordination = []
    for i in indices[:: max(1, len(indices) // 10)]:
        delta = data.positions[i, :, None] - data.positions[i, None]
        lengths = np.diag(data.cells[i])
        delta -= np.rint(delta / lengths) * lengths
        neighbors = (np.linalg.norm(delta, axis=-1) < 3.1) & ~np.eye(216, dtype=bool)
        adjacency.append(neighbors)
        coordination.append(float(neighbors.sum(axis=1).mean()))
    survival = np.count_nonzero(adjacency[0] & adjacency[-1]) / max(
        1, np.count_nonzero(adjacency[0])
    )
    displacement = float(
        np.mean(np.sum((positions[-1] - positions[indices[0]]) ** 2, axis=1))
    )
    return {
        "neighbor_survival": float(survival),
        "mean_coordination": float(np.mean(coordination)),
        "displacement_A2": displacement,
        "liquid_criteria_met": bool(
            3 < np.mean(coordination) < 12 and survival < 0.5 and displacement > 9
        ),
    }


def _production(thermal):
    selected = thermal[thermal[:, 1] >= 1100]
    _require(len(selected) >= 100, "Insufficient production thermal samples")
    temp, density = selected[:, 2], selected[:, 4]
    blocks = np.array_split(selected, 4)
    _require(
        abs(temp.mean() - 2500) < 125
        and temp.std() < 250
        and (np.abs(temp - 2500) < 750).all(),
        "Production temperature is not stable near 2500 K",
    )
    _require(
        np.ptp([b[:, 2].mean() for b in blocks]) < 125
        and np.ptp([b[:, 4].mean() for b in blocks]) / density.mean() < 0.03
        and density.std() / density.mean() < 0.03,
        "Production temperature or density is still drifting",
    )
    return {"temperature_K": float(temp.mean()), "density_g_cm3": float(density.mean())}


def _msd_fit(times, positions, low=0.1, high=0.3):
    _require(len(times) >= 25, "Insufficient samples for an MSD fit")
    dt = np.diff(times)
    _require(
        np.allclose(dt, dt[0], rtol=1e-5, atol=1e-6),
        "Sample the analysis trajectory regularly",
    )
    lags = np.unique(
        np.linspace(max(1, int(low * len(times))), int(high * len(times)), 16).astype(
            int
        )
    )
    x = lags * dt[0]
    y = np.array(
        [
            np.mean(np.sum((positions[lag:] - positions[:-lag]) ** 2, axis=-1))
            for lag in lags
        ]
    )
    _require(len(x) >= 5 and (y > 0).all(), "MSD is unresolved")
    slope, intercept = np.polyfit(x, y, 1)
    exponent = np.polyfit(np.log(x), np.log(y), 1)[0]
    residual = np.sqrt(np.mean((y - slope * x - intercept) ** 2)) / y.mean()
    _require(
        slope > 0 and 0.8 <= exponent <= 1.2 and residual < 0.1,
        "No resolved diffusive MSD regime",
    )
    return float(slope / 6 * 1e-8)


def diffusion(times, positions):
    """Use the full 2500 K interval, checking shorter windows and time blocks."""
    estimates = []
    for start in (1000, 1100, 1200):
        selected = times >= start
        for low, high in ((0.1, 0.3), (0.15, 0.4), (0.2, 0.5)):
            estimates.append(_msd_fit(times[selected], positions[selected], low, high))
    # Heating ends at 1000 ps. The prompt supplies a further 500 ps at 2500 K;
    # use that full interval for the answer, with later starts as diagnostics.
    selected = times >= 1000
    t, p = times[selected], positions[selected]
    value = _msd_fit(t, p)
    increments = np.mean(np.sum(np.diff(p, axis=0) ** 2, axis=-1), axis=1)
    increments -= increments.mean()
    variance = np.dot(increments, increments)
    correlation = 0.5
    if variance > 0:
        for lag in range(1, len(increments) // 2):
            rho = np.dot(increments[:-lag], increments[lag:]) / variance
            if rho <= 0:
                break
            correlation += rho
    correlation_time = float(correlation * np.mean(np.diff(t)))
    if len(t) >= 100:
        blocks = np.array_split(np.arange(len(t)), 4)
        block_values = [_msd_fit(t[b], p[b]) for b in blocks]
        uncertainty_basis = "time_blocks"
    else:
        # Saved coordinates can be sparse while each frame contains all 216
        # particle displacements. Do not require an unpublished
        # dump interval: retain the full fitting interval for atom-group estimates.
        groups = np.array_split(np.arange(p.shape[1]), 4)
        block_values = [_msd_fit(t, p[:, group]) for group in groups]
        uncertainty_basis = "atom_groups"
    sem = float(np.std(block_values, ddof=1) / 2)
    # Student-t 95% interval from the four estimates (3 degrees of freedom).
    uncertainty = 3.182 * sem
    spread = float(np.max(np.abs(np.array(estimates) - value)))
    _require(
        uncertainty <= 0.2 * value and spread <= 0.2 * value,
        "Diffusion is not stable across blocks and fitting/production windows",
    )
    return {
        "diffusion_m2_s": value,
        "analysis_interval_ps": [float(t[0]), float(t[-1])],
        "uncertainty_95_m2_s": uncertainty,
        "window_spread_m2_s": spread,
        "block_diffusion_m2_s": block_values,
        "correlation_time_ps": correlation_time,
        "uncertainty_basis": uncertainty_basis,
    }


def evaluate(e, r, *, level=2):
    """Populate a binary rubric and return independently calculated diagnostics."""
    diagnostics = {}
    end = 1000 if level == 1 else 1500
    if not r.check(
        "recorded_log_and_trajectory",
        10,
        lambda: all(
            e.artifact(role).stat().st_size > 0 for role in ("lammps_log", "trajectory")
        ),
    ):
        r.check("remaining_evidence", 90, False)
        return diagnostics
    commands, thermal = read_log(e.artifact("lammps_log"))
    data = read_trajectory(e.artifact("trajectory"))
    segments = _protocol(commands, thermal, end)
    r.check("heating_and_production_protocol", 25, True)
    times, positions = _states(data, thermal, segments, end)
    phase = _phase(data, times, positions, level)
    r.check("state_and_thermal_continuity", 65 if level == 1 else 25, True)
    diagnostics["phase" if level == 1 else "liquid"] = phase
    if level == 1:
        return diagnostics
    _require(
        phase["liquid_criteria_met"],
        "Structure and neighbor exchange do not support a mobile liquid",
    )
    r.check("liquid_state", 10, True)
    diagnostics["production"] = _production(thermal)
    r.check("stable_production", 5, True)
    diagnostics.update(diffusion(times, positions))
    r.check("resolved_stable_diffusion", 10, True)
    value = diagnostics["diffusion_m2_s"]
    reported = e.results.get("diffusion_m2_s")
    valid_reported = (
        type(reported) in (int, float) and np.isfinite(reported) and reported > 0
    )
    tolerance = max(
        0.01 * value,
        diagnostics["uncertainty_95_m2_s"],
        diagnostics["window_spread_m2_s"],
    )
    r.check(
        "reported_diffusion",
        5,
        valid_reported and abs(reported - value) <= tolerance,
        f"Independent MSD: {value:.8g} m2/s; comparison tolerance: {tolerance:.3g} m2/s",
    )
    reference_tolerance = REFERENCE_DIFFUSION_RTOL * REFERENCE_DIFFUSION_M2_S
    lower = REFERENCE_DIFFUSION_M2_S - reference_tolerance
    upper = REFERENCE_DIFFUSION_M2_S + reference_tolerance
    diagnostics["reference_diffusion"] = {
        "diffusion_m2_s": REFERENCE_DIFFUSION_M2_S,
        "relative_tolerance": REFERENCE_DIFFUSION_RTOL,
        "accepted_range_m2_s": [lower, upper],
    }
    r.check(
        "reference_diffusion",
        10,
        valid_reported and lower <= reported <= upper and lower <= value <= upper,
        f"Reference: {REFERENCE_DIFFUSION_M2_S:.8g} m2/s; "
        f"accepted range: [{lower:.8g}, {upper:.8g}] m2/s; "
        f"reported: {reported!r}; independent MSD: {value:.8g} m2/s. "
        "Both coefficients must lie within the fixed reference range.",
    )
    return diagnostics
