"""Read-only checks for saved Level 1 molecular-dynamics states and traces.

These checks establish agreement among retained artifacts. They do not establish
that an unrecorded dynamics step or velocity reset never occurred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase import units
from ase.build import bulk

from .common import (
    Evidence,
    EvidenceError,
    UnsupportedEvidence,
    close,
    finite_array,
    result_close,
    scientific_screen,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _time_fs(atoms) -> float | None:
    if "time_fs" in atoms.info:
        return float(finite_array(atoms.info["time_fs"], shape=()))
    if "time_ps" in atoms.info:
        return 1000 * float(finite_array(atoms.info["time_ps"], shape=()))
    return None


def _restartable(atoms) -> bool:
    """Positions, cell, identities, and actual stored momenta suffice to resume MD."""
    if "momenta" not in atoms.arrays or not len(atoms) or not bool(np.all(atoms.pbc)):
        return False
    cell = finite_array(atoms.cell.array, shape=(3, 3))
    return bool(
        abs(np.linalg.det(cell)) > 1e-8
        and finite_array(atoms.positions, shape=(len(atoms), 3)).shape
        == (len(atoms), 3)
        and finite_array(atoms.arrays["momenta"], shape=(len(atoms), 3)).shape
        == (len(atoms), 3)
        and finite_array(atoms.get_masses(), shape=(len(atoms),)).min() > 0
    )


def same_restartable_state(last_frame, saved_state, *, check_time: bool = True) -> bool:
    """Compare a continuation state with the last trajectory frame.

    Cell-equivalent wrapped positions are accepted. A text export may round
    coordinates and momenta by a few parts in 1e9, so absolute allowances are
    smaller than normal MD motion but larger than that round-trip error.
    """
    if not (_restartable(last_frame) and _restartable(saved_state)):
        return False
    if len(last_frame) != len(saved_state):
        return False
    if not (
        np.array_equal(last_frame.numbers, saved_state.numbers)
        and np.array_equal(last_frame.pbc, saved_state.pbc)
        and close(last_frame.get_masses(), saved_state.get_masses(), rtol=0, atol=1e-8)
        and close(last_frame.cell.array, saved_state.cell.array, rtol=0, atol=1e-6)
        and close(
            last_frame.arrays["momenta"],
            saved_state.arrays["momenta"],
            rtol=0,
            atol=1e-7,
        )
    ):
        return False
    fractional = (saved_state.positions - last_frame.positions) @ np.linalg.inv(
        last_frame.cell.array
    )
    fractional -= np.rint(fractional)
    if not close(
        fractional @ last_frame.cell.array,
        np.zeros((len(last_frame), 3)),
        rtol=0,
        atol=1e-6,
    ):
        return False
    if not check_time:
        return True
    left_time, right_time = _time_fs(last_frame), _time_fs(saved_state)
    return (
        left_time is None
        or right_time is None
        or bool(np.isclose(left_time, right_time, rtol=0, atol=1e-6))
    )


def restartable_endpoint(
    evidence: Evidence,
    last_frame,
    *,
    names: Sequence[str] = ("restartable_final_state", "final_state"),
    boundary_state=None,
) -> bool:
    """Check the submitted end state, accepting a complete last frame as one.

    A separately linked state takes precedence: if it is stale or malformed it
    cannot be silently replaced by the trajectory's last frame. A named final
    state must contain exactly one frame.
    """
    supplied = [name for name in names if name in evidence.manifest["artifacts"]]
    if supplied:
        saved = evidence.trajectory(supplied[0])
        if len(saved) != 1:
            raise EvidenceError("Save exactly one restartable final state")
        endpoint = saved[0]
    else:
        endpoint = last_frame if boundary_state is None else boundary_state
    return same_restartable_state(last_frame, endpoint)


def initial_com_removed(initial_state, *, atol: float = 1e-7) -> bool:
    """Inspect an explicitly saved initialized state, when available."""
    if not _restartable(initial_state):
        return False
    return close(
        initial_state.arrays["momenta"].sum(axis=0),
        np.zeros(3),
        rtol=0,
        atol=atol,
    )


def matches_pinned_cu32(atoms, *, lattice_parameter_A: float = 3.615) -> bool:
    """Recognize the bundled cubic 2x2x2 FCC Cu cell up to order/translation.

    Fractional coordinates allow a rigid rotation of the cell. This check uses
    the pinned lattice constant and full site pattern, rather than only Cu32.
    """
    if (
        len(atoms) != 32
        or not np.all(atoms.numbers == 29)
        or not bool(np.all(atoms.pbc))
    ):
        return False
    reference = bulk("Cu", "fcc", a=lattice_parameter_A, cubic=True).repeat((2, 2, 2))
    cell = finite_array(atoms.cell.array, shape=(3, 3))
    length = 2 * lattice_parameter_A
    if not (
        abs(np.linalg.det(cell)) > 1e-8
        and close(cell @ cell.T, np.eye(3) * length**2, rtol=0, atol=1e-5)
    ):
        return False
    candidate = finite_array(atoms.get_scaled_positions(wrap=True), shape=(32, 3))
    expected = reference.get_scaled_positions(wrap=True)
    candidate = np.mod(candidate - candidate[0] + expected[0], 1)
    delta = candidate[:, None, :] - expected[None, :, :]
    delta -= np.rint(delta)
    distances = np.linalg.norm(delta @ reference.cell.array, axis=2)
    nearest = np.argmin(distances, axis=1)
    return bool(
        len(np.unique(nearest)) == 32
        and np.all(distances[np.arange(32), nearest] <= 1e-5)
    )


def sampled_motion(frames: Sequence, times_fs: Sequence[float]) -> bool:
    """Reject a claimed multi-step trajectory made of one copied state.

    This only detects a clear contradiction, not whether the chosen thermostat
    and forces actually produced every saved step.
    """
    times = finite_array(times_fs, ndim=1)
    if len(frames) != len(times) or len(times) < 2 or np.any(np.diff(times) <= 0):
        return False
    if not all(_restartable(atoms) for atoms in frames):
        return False
    first = frames[0]
    return any(
        not same_restartable_state(first, atoms, check_time=False)
        for atoms in frames[1:]
    )


def recorded_velocity_protocol(evidence: Evidence) -> bool:
    """Reject recorded contradictions, without claiming execution attestation.

    Missing optional counters are not new deliverables. The initial seed,
    temperature and COM removal are checked separately against the saved state.
    Thermostat updates are not manual velocity resets.
    """
    expected = {
        "velocity_initializations": 1,
        "velocity_reinitializations_during_run": 0,
        "velocity_rescalings_during_run": 0,
        "manual_velocity_resets": 0,
        "manual_velocity_rescalings": 0,
        "reinitialize_velocities_during_run": False,
        "rescale_velocities_during_run": False,
    }
    for record in (evidence.settings, evidence.settings.get("md", {})):
        for name, target in expected.items():
            if name not in record:
                continue
            value = record[name]
            if type(value) is not type(target) or value != target:
                raise EvidenceError(
                    f"Recorded {name} contradicts the no-reset velocity protocol"
                )
    return True


def fixed_cell_log_fields(table, frames, *, timestep_fs: float) -> bool:
    """Check the generated prompt's step/T/P/density logging requirement.

    Density is recomputed from the fixed cell. Pressure need not be near zero
    in NVT, and saving stress tensors is not an additional requirement.
    """
    required = {"step", "time_fs", "temperature_K", "density_g_cm3"}
    missing = required - set(table.columns)
    if missing:
        raise EvidenceError(
            f"Thermal log is missing required fields: {', '.join(sorted(missing))}"
        )
    pressure_columns = set(table.columns) & {
        "pressure_GPa",
        "pressure_bar",
        "pressure_atm",
    }
    if not pressure_columns:
        raise EvidenceError("Thermal log must record measured pressure with its units")
    columns = sorted(required | pressure_columns)
    finite_array(table[columns].to_numpy(dtype=float), ndim=2)
    steps = table.step.to_numpy(dtype=float)
    if (
        len(table) < 2
        or np.any(steps < 0)
        or np.any(steps != np.floor(steps))
        or np.any(np.diff(steps) < 0)
        or not close(
            table.time_fs.to_numpy() - table.time_fs.iloc[0],
            (steps - steps[0]) * timestep_fs,
            rtol=0,
            atol=1e-6,
        )
        or np.any(table.temperature_K <= 0)
    ):
        raise EvidenceError(
            "Thermal log steps, elapsed times or temperatures are inconsistent"
        )
    for _, repeated in table.groupby("step"):
        if not np.allclose(
            repeated[columns], repeated[columns].iloc[0], rtol=1e-6, atol=1e-7
        ):
            raise EvidenceError("Repeated log samples disagree")
    density = frames[0].get_masses().sum() * 1.66053906660 / frames[0].get_volume()
    if not close(
        table.density_g_cm3,
        np.full(len(table), density),
        rtol=1e-4,
        atol=1e-6,
    ):
        raise EvidenceError(
            "Logged density disagrees with the saved fixed cell and masses"
        )
    return True


def equilibration_readiness(
    evidence: Evidence, frames, times_fs, trace, *, dof: int
) -> bool:
    """Require an assessment and screen the final state and terminal samples.

    The agent can select a terminal window. Otherwise the latter half is a
    screening default, not a prescribed run length or reporting recipe.
    Heuristic limits request review, never silently award credit or a zero.
    """
    assessments = [
        source[key]
        for source in (evidence.settings, evidence.results)
        for key in ("readiness", "readiness_assessment")
        if key in source
    ]
    for key in ("readiness", "readiness_assessment"):
        if key in evidence._artifacts:
            path = evidence.artifact(key)
            assessments.append(
                evidence.json(key) if path.suffix == ".json" else path.read_text()
            )
    if not assessments and evidence.manifest.get("report"):
        assessments.append(evidence._path(evidence.manifest["report"]).read_text())
    if not assessments or not all(
        (isinstance(value, str) and bool(value.strip()))
        or (
            isinstance(value, dict)
            and any(
                isinstance(value.get(key), str) and bool(value[key].strip())
                for key in ("note", "assessment", "rationale", "conclusion")
            )
        )
        for value in assessments
    ):
        raise EvidenceError(
            "Save a readiness assessment supported by the thermal trace and final state"
        )
    times = finite_array(times_fs, ndim=1)
    temperature = np.asarray(
        [
            np.sum(atoms.get_momenta() ** 2 / atoms.get_masses()[:, None])
            / (dof * units.kB)
            for atoms in frames
        ]
    )
    window = [float((times[0] + times[-1]) / 2), float(times[-1])]
    windows = [
        value["window_fs"]
        for value in assessments
        if isinstance(value, dict) and "window_fs" in value
    ]
    if windows:
        window = finite_array(windows[0], shape=(2,))
        if not all(close(value, window, rtol=0, atol=1e-6) for value in windows):
            raise EvidenceError("Readiness assessments select conflicting windows")
        if not times[0] <= window[0] < window[1] or not close(
            window[1], times[-1], rtol=0, atol=1e-6
        ):
            raise EvidenceError(
                "Readiness must assess a terminal window ending at the final state"
            )
    tail = temperature[times >= window[0]]
    allowance = 6 * 300 * np.sqrt(2 / dof)
    scientific_screen(
        abs(temperature[-1] - 300) <= allowance,
        "Final state temperature does not establish readiness near 300 K",
    )
    scientific_screen(
        len(tail) >= 2, "Too few terminal samples to assess equilibration"
    )
    cut = len(tail) // 2
    scientific_screen(
        abs(tail.mean() - 300) <= allowance
        and abs(tail[:cut].mean() - tail[cut:].mean()) <= allowance,
        "Terminal temperature samples do not establish equilibration near 300 K",
    )
    logged = trace.drop_duplicates(subset="time_fs")
    for value in assessments:
        if not isinstance(value, dict):
            continue
        if value.get("ready_for_nve") is False:
            raise EvidenceError(
                "The assessment reports that the state is not ready for NVE"
            )
        # Check existing common summaries when supplied, without requiring them.
        expected = {
            "final_temperature_K": float(logged.temperature_K.iloc[-1]),
            "last_10_log_mean_temperature_K": float(
                logged.temperature_K.iloc[-10:].mean()
            ),
        }
        for key, actual in expected.items():
            if key in value and not result_close(value[key], actual, atol=1e-6):
                raise EvidenceError(
                    f"Readiness {key} disagrees with the saved thermal trace"
                )
    return True


def _raw_log(evidence: Evidence, names: Sequence[str]) -> pd.DataFrame | None:
    name = next(
        (name for name in names if name in evidence.manifest["artifacts"]), None
    )
    if name is None:
        return None
    path = evidence.artifact(name)
    if path.suffix.lower() in (".csv", ".json"):
        return evidence.table(name)
    if path.suffix.lower() in (".log", ".txt", ".dat"):
        try:
            return pd.read_csv(path, sep=r"\s+", comment="#")
        except (OSError, ValueError) as exc:
            raise EvidenceError("Cannot read the saved raw thermal log") from exc
    raise UnsupportedEvidence("Raw thermal log format needs an independent adapter")


def _time_column(table: pd.DataFrame, timestep_fs: float) -> np.ndarray:
    if "time_fs" in table:
        return finite_array(table["time_fs"], ndim=1)
    if "step" in table:
        return finite_array(table["step"], ndim=1) * timestep_fs
    raise EvidenceError("Thermal log must record time_fs or step")


def pressure_evidence_matches(
    evidence: Evidence,
    *,
    frames: Sequence,
    frame_times_fs: Sequence[float],
    trace: pd.DataFrame,
    stage_id: str,
    timestep_fs: float,
    log_names: Sequence[str] = ("md_log", "log", "raw_log"),
) -> bool:
    """Verify task 7 pressure from saved stress and any distinct raw log.

    Stress is a numerical cross-check; a separate raw log only checks reporting
    consistency. If neither is available, pressure needs independent review.
    """
    rows = trace.loc[trace["stage"] == stage_id]
    if len(rows) < 2:
        return False
    trace_times = _time_column(rows, timestep_fs)
    if np.any(np.diff(trace_times) <= 0):
        return False
    pressure = finite_array(rows["pressure_GPa"], ndim=1)
    raw = _raw_log(evidence, log_names)
    has_stress = [
        "stress" in getattr(getattr(atoms, "calc", None), "results", {})
        for atoms in frames
    ]
    if not any(has_stress) and raw is None:
        raise UnsupportedEvidence(
            "Pressure has neither saved frame stress nor a distinct raw log"
        )
    if len(frames) != len(frame_times_fs):
        raise EvidenceError("Frame times do not match trajectory length")
    checked_stress = 0
    for atoms, frame_time, stored in zip(
        frames, frame_times_fs, has_stress, strict=True
    ):
        if not stored:
            continue
        hits = np.flatnonzero(np.isclose(trace_times, frame_time, rtol=0, atol=1e-6))
        if len(hits) != 1:
            continue
        measured = -np.mean(atoms.get_stress(include_ideal_gas=True)[:3]) / units.GPa
        if not np.isclose(pressure[hits[0]], measured, rtol=1e-4, atol=1e-6):
            return False
        checked_stress += 1
    if any(has_stress) and checked_stress < 2 and raw is None:
        raise UnsupportedEvidence(
            "Too few saved stress values align with the thermal trace"
        )
    if raw is None:
        return True

    raw_path = next(
        evidence.artifact(name)
        for name in log_names
        if name in evidence.manifest["artifacts"]
    )
    if raw_path == evidence.artifact("thermal_trace", "raw_trace") and not any(
        has_stress
    ):
        raise UnsupportedEvidence("Thermal trace and raw log are the same artifact")
    raw_times = _time_column(raw, timestep_fs)
    if np.any(np.diff(raw_times) <= 0):
        return False
    matches = [
        (index, int(hits[0]))
        for index, time in enumerate(raw_times)
        if len(hits := np.flatnonzero(np.isclose(trace_times, time, rtol=0, atol=1e-6)))
        == 1
    ]
    if len(matches) < 2:
        return False
    log_indices, trace_indices = np.asarray(matches).T
    for column in ("temperature_K", "pressure_GPa", "volume_A3"):
        if column not in raw or column not in rows:
            return False
        if not close(
            finite_array(raw[column], ndim=1)[log_indices],
            finite_array(rows[column], ndim=1)[trace_indices],
            rtol=1e-4,
            atol=1e-6,
        ):
            return False
    return True
