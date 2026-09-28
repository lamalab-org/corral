"""Read-only, evidence-based checks for liquid-silicon self diffusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import units

from .common import (
    EvidenceError,
    UnsupportedEvidence,
    aliased_value,
    optional_results_match,
    result_close,
    scientific_screen,
)
from .task_2 import _commands as _saved_commands

if TYPE_CHECKING:
    from .common import Evidence, Rubric


def _close(a, b, rtol=1e-3, atol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (
        a.shape == b.shape
        and np.all(np.isfinite(a))
        and np.all(np.isfinite(b))
        and np.allclose(a, b, rtol=rtol, atol=atol)
    )


def _minimum_image(delta, cell):
    frac = np.asarray(delta) @ np.linalg.inv(cell)
    return (frac - np.round(frac)) @ cell


def _ids(atoms):
    return np.asarray(
        atoms.arrays.get(
            "atom_ids", atoms.arrays.get("id", np.arange(1, len(atoms) + 1))
        )
    )


def _geometry(e):
    reference = e.trajectory("reference_cell", "starting_cell")[0]
    initial = e.trajectory("prepared_state", "initial_state")[0]
    if not (len(reference) == 8 and len(initial) == 216):
        raise ValueError("Expected 8-atom conventional cell and 216-atom preparation")
    for atoms in (reference, initial):
        if not (set(atoms.get_chemical_symbols()) == {"Si"} and np.all(atoms.pbc)):
            raise ValueError(
                "Evidence check failed: set(atoms.get_chemical_symbols()) == {'Si'} and np.all(atoms.pbc)"
            )
        if not (np.linalg.det(atoms.cell.array) > 0):
            raise ValueError(
                "Evidence check failed: np.linalg.det(atoms.cell.array) > 0"
            )
    cell = reference.cell.array
    metric = cell @ cell.T
    if not (_close(metric, np.eye(3) * np.trace(metric) / 3, rtol=0.01, atol=0.01)):
        raise ValueError("Reference is not a conventional cubic cell")
    # Translation- and atom-order-invariant diamond basis validation.
    basis = np.array(
        [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ]
    )
    frac = reference.get_scaled_positions()
    if not (
        any(
            np.max(
                np.min(
                    np.linalg.norm(
                        _minimum_image(
                            (frac[:, None] - shift - basis[None]) @ cell, cell
                        ),
                        axis=-1,
                    ),
                    axis=1,
                )
            )
            < 0.02
            for shift in frac
        )
    ):
        raise ValueError("Reference is not diamond silicon")
    expected = reference.repeat((3, 3, 3))
    if not (_close(initial.cell.array, expected.cell.array, rtol=1e-4, atol=1e-3)):
        raise ValueError(
            "Evidence check failed: _close(initial.cell.array, expected.cell.array, rtol=0.0001, atol=0.001)"
        )
    matches = False
    for anchor in expected.positions[:8]:
        shift = initial.positions[0] - anchor
        dist = np.linalg.norm(
            _minimum_image(
                initial.positions[:, None] - shift - expected.positions[None],
                initial.cell.array,
            ),
            axis=-1,
        )
        if (
            len(np.unique(np.argmin(dist, axis=1))) == 216
            and np.max(np.min(dist, axis=1)) < 0.02
        ):
            matches = True
            break
    if not (matches):
        raise ValueError("Prepared coordinates are not the specified repeat")
    if not (len(np.unique(_ids(initial))) == 216):
        raise ValueError("Evidence check failed: len(np.unique(_ids(initial))) == 216")
    velocities = initial.get_velocities()
    if not (velocities is not None and np.all(np.isfinite(velocities))):
        raise ValueError(
            "Evidence check failed: velocities is not None and np.all(np.isfinite(velocities))"
        )
    if not (
        abs(initial.get_temperature() - 300)
        <= 300 * 6 * np.sqrt(2 / (3 * len(initial)))
    ):
        raise ValueError("Prepared velocities are inconsistent with 300 K")
    return True


def _texts(e, alias):
    return "\n".join(p.read_text() for p in e.artifacts(alias))


class Unsupported(UnsupportedEvidence):
    pass


def _thermo(e, end_time_ps=1500):
    table = e.table("thermo", "thermodynamics")
    cols = [
        "step",
        "time_ps",
        "target_temperature_K",
        "temperature_K",
        "pressure_bar",
        "density_g_cm3",
    ]
    values = table[cols].to_numpy(dtype=float)
    if not (len(values) >= 2 and np.all(np.isfinite(values))):
        raise ValueError(
            "Retain finite thermodynamic samples including stage endpoints"
        )
    if not (
        np.all(np.diff(values[:, 1]) >= 0)
        and values[0, 1] == 0
        and _close(values[-1, 1], end_time_ps)
    ):
        raise ValueError(
            "Evidence check failed: np.all(np.diff(values[:, 1]) >= 0) and values[0, 1] == 0 and _close(values[-1, 1], 1500)"
        )
    if not (np.all(values[:, 3] > 0) and np.all(values[:, 5] > 0)):
        raise ValueError(
            "Evidence check failed: np.all(values[:, 3] > 0) and np.all(values[:, 5] > 0)"
        )
    if not (_close(values[:, 2], np.minimum(2500.0, 300.0 + 2.2 * values[:, 1]))):
        raise ValueError(
            "Evidence check failed: _close(values[:, 2], np.minimum(2500.0, 300.0 + 2.2 * values[:, 1]))"
        )
    if not np.all(values[:, 0] == values[:, 0].astype(int)):
        raise ValueError("Thermo step indices must be integers")
    return table


def _logged_thermo(e, end_time_ps=1500):
    table = _thermo(e, end_time_ps)
    # Parse numeric thermo sections only; never evaluate log content as code.
    logged, header = {}, None
    for line in _texts(e, "lammps_log").splitlines():
        fields = line.split()
        if {"Step", "Temp", "Press", "Density"}.issubset(fields):
            header = fields
            continue
        if header and len(fields) == len(header):
            try:
                values = [float(x) for x in fields]
            except ValueError:
                continue
            row = dict(zip(header, values, strict=False))
            logged.setdefault(row["Step"], []).append(
                [row["Temp"], row["Press"], row["Density"]]
            )
    for row in table.itertuples(index=False):
        expected = [row.temperature_K, row.pressure_bar, row.density_g_cm3]
        if not (
            any(
                _close(actual, expected, rtol=0.002, atol=0.002)
                for actual in logged.get(row.step, [])
            )
        ):
            raise ValueError("Thermo table does not match raw LAMMPS log")
    return table


def _protocol(e, end_time_ps=1500):
    table = _logged_thermo(e, end_time_ps)
    commands = list(_saved_commands(e))
    if not (["units", "metal"] in commands and ["atom_style", "full"] in commands):
        raise ValueError(
            "Evidence check failed: ['units', 'metal'] in commands and ['atom_style', 'full'] in commands"
        )
    if not (any(c[:2] == ["pair_style", "sw"] for c in commands)):
        raise ValueError(
            "Evidence check failed: any((c[:2] == ['pair_style', 'sw'] for c in commands))"
        )
    if not (any(c[0] == "pair_coeff" and "Si" in c and len(c) >= 5 for c in commands)):
        raise ValueError(
            "Evidence check failed: any((c[0] == 'pair_coeff' and 'Si' in c and (len(c) >= 5) for c in commands))"
        )
    recorded_timestep = float(e.settings["timestep_ps"])
    if not np.isfinite(recorded_timestep) or recorded_timestep <= 0:
        raise ValueError("Record a finite positive initial integration timestep")
    timestep = 0.001  # LAMMPS metal-unit default when no command overrides it.
    creates = [
        i for i, c in enumerate(commands) if c[:3] == ["velocity", "all", "create"]
    ]
    seed = int(e.settings["velocity_seed"])
    if seed <= 0:
        raise ValueError("Record a positive initialization seed")
    if len(creates) > 1:
        raise ValueError("Velocities must not be reinitialized")
    if creates:
        create = commands[creates[0]]
        if not (_close(float(create[3]), 300) and int(create[4]) == seed):
            raise ValueError("Velocity initialization contradicts the saved settings")
    elif not any(c[0] in {"read_data", "read_restart"} for c in commands):
        raise ValueError("Retain the command that loads the prepared state")
    active, elapsed, step, segments = {}, 0.0, 0, []
    first_run = None
    for index, c in enumerate(commands):
        if c[0] == "timestep":
            timestep = float(c[1])
            if not np.isfinite(timestep) or timestep <= 0:
                raise ValueError("Integration timestep must be finite and positive")
        elif c[0] == "reset_timestep":
            if len(c) != 2:
                raise Unsupported("Additional reset_timestep options require review")
            step = int(c[1])
        elif c[0] == "fix" and len(c) > 3:
            active[c[1]] = c
        elif c[0] == "unfix":
            active.pop(c[1], None)
        elif c[0] == "run":
            if "every" in c:
                raise Unsupported("Commands executed during a run require review")
            count = int(c[1]) - (step if "upto" in c else 0)
            if count < 0:
                raise ValueError("Negative run duration")
            if count == 0:
                continue
            if first_run is None and not _close(timestep, recorded_timestep):
                raise ValueError(
                    "Recorded initial timestep contradicts the effective input"
                )
            first_run = index if first_run is None else first_run
            npt = [f for f in active.values() if f[3] in {"npt", "nvt", "nve"}]
            if not (len(npt) == 1 and npt[0][2:4] == ["all", "npt"]):
                raise ValueError("Each dynamics stage must have one NPT integrator")
            f = npt[0]
            ti, pi = f.index("temp"), f.index("iso")
            if not (_close([float(f[pi + 1]), float(f[pi + 2])], [0, 0])):
                raise ValueError(
                    "Evidence check failed: _close([float(f[pi + 1]), float(f[pi + 2])], [0, 0])"
                )
            if not (float(f[ti + 3]) > 0 and float(f[pi + 3]) > 0):
                raise ValueError(
                    "Evidence check failed: float(f[ti + 3]) > 0 and float(f[pi + 3]) > 0"
                )
            start = int(c[c.index("start") + 1]) if "start" in c else step
            stop = int(c[c.index("stop") + 1]) if "stop" in c else step + count
            if not start <= step < step + count <= stop:
                raise ValueError("Run lies outside its declared temperature ramp")
            first, last = float(f[ti + 1]), float(f[ti + 2])
            actual = first + (last - first) * (
                np.array([step, step + count]) - start
            ) / (stop - start)
            duration = count * timestep
            end = 1000 if elapsed < 1000 - 1e-7 else 1500
            expected = np.minimum(
                2500, 300 + 2.2 * np.array([elapsed, elapsed + duration])
            )
            if elapsed + duration > end + 1e-7 or not _close(actual, expected):
                raise ValueError("Run commands contradict the requested thermal cycle")
            segments.append((step, step + count, elapsed, timestep))
            elapsed += duration
            step += count
        if first_run is not None and index > first_run:
            if c[0] in {"create_atoms", "change_box"}:
                raise ValueError(
                    "Detected atom replacement or manual cell manipulation"
                )
            if c[0] in {"read_data", "read_restart", "clear"}:
                raise Unsupported(
                    "Loaded restart continuity needs independent verification"
                )
            if not (c[0] != "velocity"):
                raise ValueError("Detected velocity modification after dynamics began")
    if first_run is None or (creates and creates[0] >= first_run):
        raise ValueError("Velocity preparation must precede dynamics")
    if not _close(elapsed, end_time_ps):
        raise ValueError(
            f"Run commands must cover the full {end_time_ps:g} ps requested stage"
        )
    for row in table.itertuples(index=False):
        if not any(
            first <= row.step <= last
            and _close(start + (row.step - first) * dt, row.time_ps, atol=1e-6)
            for first, last, start, dt in segments
        ):
            raise ValueError(
                "Thermo step/time mapping contradicts the integration schedule"
            )
    return True


def _data(e):
    data = dict(e.json("diffusion_data", "production_data"))
    if "arrays_file" in data:
        path = e.linked_path(
            data["arrays_file"], e.artifact("diffusion_data", "production_data")
        )
        if path.suffix.lower() != ".npz":
            raise UnsupportedEvidence(
                "Diffusion arrays_file must be a numeric NPZ archive"
            )
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                if name in data and not np.array_equal(data[name], archive[name]):
                    raise EvidenceError(f"Diffusion JSON and NPZ disagree on {name}")
                data[name] = archive[name]
        for target, selector in (
            ("unwrapped_positions_A", "positions_key"),
            ("velocities_A_ps", "velocities_key"),
        ):
            if selector in data:
                value = aliased_value(data, data[selector])
                if target in data and not np.array_equal(data[target], value):
                    raise EvidenceError(f"Conflicting diffusion array: {target}")
                data[target] = value
    if "cells_A" not in data and "cell_lengths_A" in data:
        lengths = np.asarray(data["cell_lengths_A"], float)
        if lengths.ndim != 2 or lengths.shape[1] != 3:
            raise EvidenceError("cell_lengths_A must have shape [frames, 3]")
        if "box_bounds_A" in data:
            bounds = np.asarray(data["box_bounds_A"], float)
            if bounds.shape not in ((len(lengths), 3, 2), (len(lengths), 3, 3)):
                raise EvidenceError(
                    "box_bounds_A must contain each frame's LAMMPS box bounds"
                )
            if bounds.shape[-1] == 3 and np.any(bounds[:, :, 2] != 0):
                raise UnsupportedEvidence(
                    "Triclinic diffusion cells require explicit cells_A matrices"
                )
            if not _close(bounds[:, :, 1] - bounds[:, :, 0], lengths):
                raise EvidenceError("cell_lengths_A contradicts the saved box bounds")
        data["cells_A"] = lengths[:, :, None] * np.eye(3)
    t = np.asarray(data["times_ps"], float)
    p = np.asarray(
        aliased_value(data, "unwrapped_positions_A", "positions_unwrapped_A"), float
    )
    cell = np.asarray(data["cells_A"], float)
    velocity = np.asarray(data["velocities_A_ps"], float)
    if not (t.ndim == 1 and len(t) >= 2 and np.all(np.diff(t) > 0)):
        raise ValueError(
            "Retain chronological production times including both endpoints"
        )
    if not (_close(t[[0, -1]], [1000.0, 1500.0])):
        raise ValueError("Evidence check failed: _close(t[[0, -1]], [1000.0, 1500.0])")
    if not (
        p.shape == velocity.shape == (len(t), 216, 3) and cell.shape == (len(t), 3, 3)
    ):
        raise ValueError(
            "Evidence check failed: p.shape == velocity.shape == (len(t), 216, 3) and cell.shape == (len(t), 3, 3)"
        )
    if not (all(np.all(np.isfinite(a)) for a in (t, p, cell, velocity))):
        raise ValueError(
            "Evidence check failed: all((np.all(np.isfinite(a)) for a in (t, p, cell, velocity)))"
        )
    if not (
        np.all(np.linalg.det(cell) > 0)
        and np.asarray(data["pbc"]).shape == (3,)
        and all(data["pbc"])
    ):
        raise ValueError(
            "Evidence check failed: np.all(np.linalg.det(cell) > 0) and np.asarray(data['pbc']).shape == (3,) and all(data['pbc'])"
        )
    if not (data["species"] == ["Si"] * 216):
        raise ValueError("Evidence check failed: data['species'] == ['Si'] * 216")
    ids = np.asarray(data["atom_ids"])
    if ids.ndim == 2:
        if not (ids.shape == (len(t), 216) and np.all(ids == ids[0])):
            raise ValueError(
                "Evidence check failed: ids.shape == (len(t), 216) and np.all(ids == ids[0])"
            )
        ids = ids[0]
    if not (ids.shape == (216,) and len(np.unique(ids)) == 216):
        raise ValueError(
            "Evidence check failed: ids.shape == (216,) and len(np.unique(ids)) == 216"
        )
    if "types" in data and not (
        np.asarray(data["types"]).shape == (216,) and len(set(data["types"])) == 1
    ):
        raise ValueError(
            "Evidence check failed: np.asarray(data['types']).shape == (216,) and len(set(data['types'])) == 1"
        )
    return t, p, cell, velocity, ids


def _continuity(e):
    t, p, cells, velocities, ids = _data(e)
    for key, index in (("heating_end_state", 0), ("final_state", -1)):
        atoms = e.trajectory(key)[0]
        if not (
            len(atoms) == 216
            and set(atoms.get_chemical_symbols()) == {"Si"}
            and np.all(atoms.pbc)
        ):
            raise ValueError(
                "Evidence check failed: len(atoms) == 216 and set(atoms.get_chemical_symbols()) == {'Si'} and np.all(atoms.pbc)"
            )
        order = {value: i for i, value in enumerate(_ids(atoms))}
        if not (set(order) == set(ids)):
            raise ValueError("Evidence check failed: set(order) == set(ids)")
        ii = [order[value] for value in ids]
        if not (_close(atoms.cell.array, cells[index], rtol=1e-5, atol=1e-5)):
            raise ValueError(
                "Evidence check failed: _close(atoms.cell.array, cells[index], rtol=1e-05, atol=1e-05)"
            )
        if not (
            np.max(np.abs(_minimum_image(atoms.positions[ii] - p[index], cells[index])))
            < 1e-4
        ):
            raise ValueError(
                "Evidence check failed: np.max(np.abs(_minimum_image(atoms.positions[ii] - p[index], cells[index]))) < 0.0001"
            )
        v = atoms.get_velocities()
        if not (
            v is not None
            and _close(
                v[ii] * units.fs * 1000, velocities[index], rtol=0.002, atol=1e-4
            )
        ):
            raise ValueError(
                "Evidence check failed: v is not None and _close(v[ii] * units.fs * 1000, velocities[index], rtol=0.002, atol=0.0001)"
            )
    initial = e.trajectory("prepared_state", "initial_state")[0]
    if not (set(_ids(initial)) == set(ids)):
        raise ValueError("Evidence check failed: set(_ids(initial)) == set(ids)")
    # Data and thermo must describe the same cells and kinetic temperatures.
    table = _thermo(e)
    initial_mass = initial.get_masses().sum()
    density = initial_mass * 1.66053906660 / np.linalg.det(cells)
    temp = np.sum(
        initial.get_masses()[None, :, None] * (velocities / (units.fs * 1000)) ** 2,
        axis=(1, 2),
    ) / (3 * 216 * units.kB)
    for frame, time in enumerate(t):
        rows = table[np.isclose(table.time_ps, time, rtol=0, atol=1e-6)]
        if len(rows):
            if not (
                _close(float(rows.iloc[-1].density_g_cm3), density[frame], rtol=0.002)
            ):
                raise ValueError(
                    "Evidence check failed: _close(float(rows.iloc[-1].density_g_cm3), density[frame], rtol=0.002)"
                )
            # LAMMPS removes three translational degrees of freedom by default.
            measured = float(rows.iloc[-1].temperature_K)
            if not (abs(measured - temp[frame]) <= 0.01 * max(measured, temp[frame])):
                raise ValueError(
                    "Evidence check failed: abs(measured - temp[frame]) <= 0.01 * max(measured, temp[frame])"
                )
    return True


def _msd(t, positions, cells, spec):
    if spec.get("estimator", "einstein_msd") != "einstein_msd":
        raise Unsupported(
            "Only saved-data Einstein-MSD reconstruction is currently supported"
        )
    p = positions.copy()
    if spec["cell_motion"] == "reference_cell":
        p = np.einsum("faj,fjk->fak", p, np.linalg.inv(cells)) @ cells[0]
    elif spec["cell_motion"] != "lab":
        raise Unsupported("Unsupported cell-motion convention")
    if not (
        isinstance(spec["remove_com_drift"], bool)
        and isinstance(spec["fit_intercept"], bool)
    ):
        raise ValueError(
            "Evidence check failed: isinstance(spec['remove_com_drift'], bool) and isinstance(spec['fit_intercept'], bool)"
        )
    lags = np.asarray(spec["lag_steps"])
    if not (
        lags.ndim == 1
        and len(lags) >= 3
        and np.all(lags == lags.astype(int))
        and np.all(lags > 0)
        and np.all(np.diff(lags) > 0)
    ):
        raise ValueError(
            "Evidence check failed: lags.ndim == 1 and len(lags) >= 3 and np.all(lags == lags.astype(int)) and np.all(lags > 0) and np.all(np.diff(lags) > 0)"
        )
    means, lagtimes, spread = [], [], []
    for lag in lags.astype(int):
        origins = (
            np.arange(len(t) - lag)
            if spec["origin_indices"] == "all"
            else np.asarray(spec["origin_indices"])
        )
        if not (
            origins.ndim == 1
            and np.all(origins == origins.astype(int))
            and np.all(origins >= 0)
        ):
            raise ValueError(
                "Evidence check failed: origins.ndim == 1 and np.all(origins == origins.astype(int)) and np.all(origins >= 0)"
            )
        if not (np.all(origins < len(t))):
            raise ValueError("Evidence check failed: np.all(origins < len(t))")
        origins = origins[origins + lag < len(t)].astype(int)
        if not (len(origins) > 0 and len(np.unique(origins)) == len(origins)):
            raise ValueError(
                "Evidence check failed: len(origins) > 0 and len(np.unique(origins)) == len(origins)"
            )
        displacement = p[origins + lag] - p[origins]
        if spec["remove_com_drift"]:
            displacement -= displacement.mean(axis=1, keepdims=True)
        means.append(np.mean(np.sum(displacement**2, axis=-1)))
        delta_t = t[origins + lag] - t[origins]
        lagtimes.append(np.mean(delta_t))
        spread.append(np.std(delta_t))
    x, y = np.asarray(lagtimes), np.asarray(means)
    lo, hi = np.asarray(spec["fit_interval_ps"], float)
    selected = (x >= lo) & (x <= hi)
    if not (np.count_nonzero(selected) >= 3 and 0 <= lo < hi):
        raise ValueError(
            "Evidence check failed: np.count_nonzero(selected) >= 3 and 0 <= lo < hi"
        )
    design = (
        np.column_stack([x[selected], np.ones(sum(selected))])
        if spec["fit_intercept"]
        else x[selected, None]
    )
    weights = np.asarray(spec.get("fit_weights", np.ones(len(x))), float)
    if not (
        weights.shape == x.shape
        and np.all(np.isfinite(weights))
        and np.all(weights > 0)
    ):
        raise ValueError(
            "Evidence check failed: weights.shape == x.shape and np.all(np.isfinite(weights)) and np.all(weights > 0)"
        )
    coeff = np.linalg.lstsq(
        design * np.sqrt(weights[selected, None]),
        y[selected] * np.sqrt(weights[selected]),
        rcond=None,
    )[0]
    predicted = design @ coeff
    return {
        "x": x,
        "y": y,
        "spread": np.asarray(spread),
        "selected": selected,
        "d": coeff[0] / 6 * 1e-8,
        "predicted": predicted,
    }


def _analysis(e):
    return e.json("analysis", "diffusion_analysis")


def _spec(e):
    analysis = _analysis(e)
    return analysis["diffusion"] if "diffusion" in analysis else analysis["msd"]


def _custom(record, subject):
    """A method name alone is not evidence for an alternative calculation."""
    if not isinstance(record.get("calculations"), dict) or not record["calculations"]:
        raise ValueError(f"Retain numerical calculations for the chosen {subject}")
    raise Unsupported(f"The saved {subject} requires an additional numerical verifier")


def _green_kubo(t, velocities, spec):
    """Recompute the dot-product VACF and its trapezoidal Green-Kubo integral."""
    if not isinstance(spec["remove_com_drift"], bool):
        raise ValueError("remove_com_drift must be boolean")
    if spec.get("integration", "trapezoid") != "trapezoid":
        _custom(spec, "VACF integration convention")
    lags = np.asarray(spec["lag_steps"])
    if not (
        lags.ndim == 1
        and len(lags) >= 3
        and lags[0] == 0
        and np.all(lags == lags.astype(int))
        and np.all(np.diff(lags) > 0)
    ):
        raise ValueError(
            "VACF lags must start at zero and increase as integer frame offsets"
        )
    v = velocities.copy()
    if spec["remove_com_drift"]:
        v -= v.mean(axis=1, keepdims=True)
    values, times, spreads = [], [], []
    for lag in lags.astype(int):
        origins = (
            np.arange(len(t) - lag)
            if spec["origin_indices"] == "all"
            else np.asarray(spec["origin_indices"])
        )
        if not (
            origins.ndim == 1
            and np.all(origins == origins.astype(int))
            and np.all(origins >= 0)
            and np.all(origins < len(t))
            and len(np.unique(origins)) == len(origins)
        ):
            raise ValueError("Invalid VACF origin indices")
        origins = origins[origins + lag < len(t)].astype(int)
        if not len(origins):
            raise ValueError("VACF lag has no valid time origins")
        values.append(np.mean(np.sum(v[origins] * v[origins + lag], axis=-1)))
        delta = t[origins + lag] - t[origins]
        times.append(np.mean(delta))
        spreads.append(np.std(delta))
    x, y = np.asarray(times), np.asarray(values)
    if np.any(np.diff(x) <= 0):
        raise ValueError("VACF elapsed lags must increase")
    integral = np.r_[0, np.cumsum(np.diff(x) * (y[:-1] + y[1:]) / 2)] / 3 * 1e-8
    cutoff = float(spec["integration_cutoff_ps"])
    matches = np.flatnonzero(np.isclose(x, cutoff, rtol=1e-8, atol=1e-12))
    if len(matches) != 1 or matches[0] == 0:
        raise ValueError("Choose a positive retained lag as the integration cutoff")
    return {
        "x": x,
        "y": y,
        "spread": np.asarray(spreads),
        "d": integral[matches[0]],
        "integral": integral,
        "selected": x <= cutoff,
    }


def _reconstruct(e, spec=None):
    t, p, cells, velocity, _ = _data(e)
    spec = spec or _spec(e)
    if "sample_window_ps" in spec:
        lo, hi = spec["sample_window_ps"]
        if not t[0] <= lo < hi <= t[-1]:
            raise ValueError("Estimator window lies outside retained production data")
        mask = (t >= lo) & (t <= hi)
        t, p, cells, velocity = t[mask], p[mask], cells[mask], velocity[mask]
    method = spec.get("estimator", "einstein_msd")
    if method == "einstein_msd":
        return _msd(t, p, cells, spec)
    if method == "green_kubo":
        return _green_kubo(t, velocity, spec)
    return _custom(spec, "diffusion estimator")


def _msd_check(e):
    spec = _spec(e)
    result = _reconstruct(e, spec)
    if not (result_close(result["x"], spec["lag_times_ps"], atol=1e-12)):
        raise ValueError(
            "Evidence check failed: _close(result['x'], spec['lag_times_ps'], rtol=0.002)"
        )
    value_key = "vacf_A2_ps2" if spec.get("estimator") == "green_kubo" else "msd_A2"
    if not (result_close(result["y"], spec[value_key], atol=1e-12)):
        raise ValueError(
            "Evidence check failed: _close(result['y'], spec['msd_A2'], rtol=0.002)"
        )
    if np.any(result["spread"] > 1e-6) and not result_close(
        result["spread"], spec["lag_time_spread_ps"], atol=1e-12
    ):
        raise ValueError(
            "Evidence check failed: _close(result['spread'], spec['lag_time_spread_ps'], rtol=0.002)"
        )
    return True


def _estimate(e):
    result = _reconstruct(e)
    if not (np.isfinite(result["d"]) and result["d"] > 0):
        raise ValueError(
            "Evidence check failed: np.isfinite(result['d']) and result['d'] > 0"
        )
    if e.results["diffusion_units"] not in {"m2/s", "m^2/s", "m²/s"}:
        raise ValueError(
            "Evidence check failed: e.results['diffusion_units'] in {'m2/s', 'm^2/s', 'm²/s'}"
        )
    return True


def _reported(e):
    calculated = _reconstruct(e)["d"]
    if not (result_close(calculated, e.result("diffusion_m2_s"), atol=1e-15)):
        raise ValueError(
            "Evidence check failed: _close(calculated, e.result('diffusion_m2_s'), atol=1e-15)"
        )
    return True


def _used_window(t, spec):
    if "sample_window_ps" in spec:
        return spec["sample_window_ps"]
    if spec.get("estimator", "einstein_msd") not in {"einstein_msd", "green_kubo"}:
        # With no narrower declared selection, validate the entire saved interval.
        return [t[0], t[-1]]
    used = []
    for value in spec["lag_steps"]:
        if not isinstance(value, int) or value < 0:
            raise ValueError("Lag indices must be nonnegative integers")
        origins = (
            np.arange(len(t) - value)
            if spec["origin_indices"] == "all"
            else np.asarray(spec["origin_indices"])
        )
        if not (
            origins.ndim == 1
            and np.all(origins == origins.astype(int))
            and np.all(origins >= 0)
            and np.all(origins < len(t))
        ):
            raise ValueError("Invalid time origins")
        origins = origins[origins + value < len(t)].astype(int)
        if not len(origins):
            raise ValueError("Lag has no valid time origins")
        lagtime = np.mean(t[origins + value] - t[origins])
        interval = spec.get(
            "fit_interval_ps", [0, spec.get("integration_cutoff_ps", np.inf)]
        )
        if interval[0] <= lagtime <= interval[1]:
            used.extend([t[origins].min(), t[origins + value].max()])
    if not used:
        raise ValueError("The declared analysis selects no retained data")
    return [min(used), max(used)]


def _equilibrium(e):
    validation = _analysis(e)["validation"]
    lo, hi = validation["equilibrium_window_ps"]
    if not (1000 <= lo < hi <= 1500):
        raise ValueError("Evidence check failed: 1000 <= lo < hi <= 1500")
    table = _thermo(e)
    selected = table[(table.time_ps >= lo) & (table.time_ps <= hi)]
    if not (len(selected) >= 2):
        raise ValueError("Retain observations across the equilibrium window")
    t = _data(e)[0]
    spec = _spec(e)
    window = _used_window(t, spec)
    if not lo <= window[0] < window[1] <= hi:
        raise ValueError(
            "Main analysis uses data outside its declared equilibrium window"
        )
    if validation.get("method", "half_means") != "half_means":
        _custom(validation, "equilibration diagnostic")
    compatible = []
    for column, mean_key, diff_key in (
        ("temperature_K", "temperature_mean_K", "temperature_half_difference_K"),
        ("density_g_cm3", "density_mean_g_cm3", "density_half_difference_g_cm3"),
        ("pressure_bar", "pressure_mean_bar", "pressure_half_difference_bar"),
    ):
        values = selected[column].to_numpy(float)
        halves = np.array_split(values, 2)
        difference = halves[1].mean() - halves[0].mean()
        if not optional_results_match(
            validation, {mean_key: values.mean()}, atol=1e-12
        ) or not optional_results_match(validation, {diff_key: difference}, atol=1e-6):
            raise ValueError(
                "Evidence check failed: _close(values.mean(), validation[mean_key]) and _close(difference, validation[diff_key], atol=1e-06)"
            )
        block_means = np.array(
            [b.mean() for b in np.array_split(values, min(4, len(values)))]
        )
        sem = block_means.std(ddof=1) / np.sqrt(len(block_means))
        floor = 500 if column == "pressure_bar" else 0.05 * abs(values.mean())
        compatible.append(abs(difference) <= max(floor, 3 * sem))
        if column == "temperature_K":
            compatible.append(abs(values.mean() - 2500) <= max(125, 3 * sem))
        elif column == "pressure_bar":
            compatible.append(abs(values.mean()) <= max(500, 3 * sem))
    return scientific_screen(
        all(compatible),
        "The retained equilibrium and target diagnostics are inconclusive",
    )


def _diffusive(e):
    analysis = _analysis(e)
    diagnostic = analysis["diffusive"]
    method = diagnostic.get("method", "msd_scaling")
    if method == "vacf_plateau":
        if _spec(e).get("estimator") != "green_kubo":
            raise ValueError("VACF plateau evidence needs a retained VACF estimator")
        result = _reconstruct(e)
        lo, hi = diagnostic["interval_ps"]
        values = result["integral"][(result["x"] >= lo) & (result["x"] <= hi)]
        if not (
            0 < lo < hi <= float(_spec(e)["integration_cutoff_ps"]) and len(values) >= 3
        ):
            raise ValueError(
                "Retain at least three running integral values in the plateau interval"
            )
        mean = values.mean()
        if mean <= 0:
            raise ValueError("The diffusion plateau must be positive")
        spread = np.ptp(values) / mean
        if not (
            result_close(mean, diagnostic["integral_mean_m2_s"], atol=1e-15)
            and result_close(spread, diagnostic["integral_relative_spread"], atol=1e-12)
        ):
            raise ValueError(
                "Reported VACF plateau statistics do not reproduce the data"
            )
        if spread > 0.2:
            raise Unsupported(
                "The running VACF integral is inconclusive for a diffusion plateau"
            )
        return True
    if method != "msd_scaling":
        _custom(diagnostic, "diffusive-regime diagnostic")
    for key in ("log_slope", "residual_fraction"):
        if not np.isfinite(float(diagnostic[key])):
            raise ValueError("Retain finite MSD-scaling diagnostic values")
    if (
        "msd" not in diagnostic
        and _spec(e).get("estimator", "einstein_msd") != "einstein_msd"
    ):
        raise ValueError(
            "Retain a separate MSD specification for the chosen scaling diagnostic"
        )
    result = _reconstruct(e, diagnostic.get("msd", _spec(e)))
    if "predicted" not in result:
        raise ValueError(
            "Retain an MSD specification for the chosen MSD-scaling diagnostic"
        )
    x, y = result["x"][result["selected"]], result["y"][result["selected"]]
    if not (np.all(y > 0)):
        raise ValueError("Evidence check failed: np.all(y > 0)")
    log_slope = np.polyfit(np.log(x), np.log(y), 1)[0]
    residual = np.sqrt(np.mean((y - result["predicted"]) ** 2)) / np.mean(y)
    if not (result_close(log_slope, analysis["diffusive"]["log_slope"], atol=1e-12)):
        raise ValueError(
            "Evidence check failed: _close(log_slope, analysis['diffusive']['log_slope'])"
        )
    if not (
        result_close(residual, analysis["diffusive"]["residual_fraction"], atol=1e-08)
    ):
        raise ValueError(
            "Evidence check failed: _close(residual, analysis['diffusive']['residual_fraction'], atol=1e-08)"
        )
    if not (0.7 <= log_slope <= 1.3 and residual <= 0.2):
        raise Unsupported(
            "The retained MSD diagnostic is inconclusive for a diffusive regime"
        )
    return True


def _liquid(e):
    liquid = _analysis(e)["liquid"]
    if liquid.get("method", "neighbor_survival") != "neighbor_survival":
        _custom(liquid, "liquid-state diagnostic")
    t, p, cells, _, _ = _data(e)
    i, j = liquid["frame_indices"]
    if not (isinstance(i, int) and isinstance(j, int) and 0 <= i < j < len(t)):
        raise ValueError(
            "Evidence check failed: isinstance(i, int) and isinstance(j, int) and (0 <= i < j < len(t))"
        )
    cutoff = float(liquid["neighbor_cutoff_A"])
    initial = e.trajectory("prepared_state", "initial_state")[0]
    distances = np.sort(initial.get_all_distances(mic=True)[0])[1:]
    if not (distances[0] < cutoff < distances[4]):
        raise ValueError(
            "Liquid neighbor cutoff must separate initial first and second shells"
        )
    adj = []
    for k in (i, j):
        d = np.linalg.norm(
            _minimum_image(p[k, :, None] - p[k, None, :], cells[k]), axis=-1
        )
        adj.append((d < cutoff) & ~np.eye(216, dtype=bool))
    if not (np.count_nonzero(adj[0]) > 0):
        raise ValueError("Evidence check failed: np.count_nonzero(adj[0]) > 0")
    survival = np.count_nonzero(adj[0] & adj[1]) / np.count_nonzero(adj[0])
    if not (result_close(survival, liquid["neighbor_survival_fraction"], atol=1e-12)):
        raise ValueError(
            "Evidence check failed: _close(survival, liquid['neighbor_survival_fraction'])"
        )
    displacement = p[j] - p[i]
    displacement -= displacement.mean(axis=0)
    if not (
        survival < 0.5
        and np.sqrt(np.mean(np.sum(displacement**2, axis=1))) > distances[0]
    ):
        raise Unsupported(
            "Neighbor persistence and atomic motion are inconclusive for a liquid state"
        )
    return True


def evaluate(e: Evidence, r: Rubric) -> None:
    """Award 90 task-specific points; common scoring supplies reproducibility."""
    geometry = r.check("diamond_preparation", 8, lambda: _geometry(e))
    try:
        protocol = _protocol(e)
    except UnsupportedEvidence as exc:
        protocol = None
        detail = (
            str(exc)
            or "The saved input/log/thermo evidence does not meet the thermal-protocol requirements"
        )
    except Exception as exc:
        protocol = False
        detail = str(exc)
    else:
        detail = "Input, thermo and original log describe the required thermal cycle"
    r.check("thermal_protocol_and_log", 12, protocol, detail)
    raw = r.check("production_data_integrity", 10, lambda: _data(e) is not None)
    continuous = r.check(
        "state_and_thermo_continuity", 12, lambda: raw and geometry and _continuity(e)
    )
    # Valid upstream physical evidence is required for scientific analysis credit.
    valid = geometry and raw and continuous
    for name, points, check in (
        ("msd_reconstruction", 12, _msd_check),
        ("diffusion_estimator_and_units", 10, _estimate),
        ("reported_diffusion", 8, _reported),
        ("equilibrated_production", 9, _equilibrium),
        ("liquid_and_diffusive_evidence", 5, _liquid),
        ("diffusive_regime", 4, _diffusive),
    ):
        if not valid:
            r.check(
                name,
                points,
                False,
                "Requires valid geometry, production data and boundary/thermo consistency",
            )
            continue
        try:
            outcome = check(e)
        except UnsupportedEvidence as exc:
            r.check(name, points, None, str(exc))
        except Exception as exc:
            r.check(
                name,
                points,
                False,
                str(exc)
                or "The saved evidence fails the published numerical or physical consistency check",
            )
        else:
            r.check(name, points, outcome)
    r.unverified(
        "execution_provenance",
        "Saved inputs, logs and arrays establish consistency but cannot prove execution, checkpoint identity, or absence of an unrecorded state reset.",
    )
