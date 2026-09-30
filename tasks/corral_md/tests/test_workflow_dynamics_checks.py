"""Adversarial checks for Level 1 dynamics evidence shared by several tasks."""

import json

import numpy as np
import pandas as pd
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from corral_md.workflow_scoring.common import Evidence, UnsupportedEvidence
from corral_md.workflow_scoring.dynamics import (
    initial_com_removed,
    matches_pinned_cu32,
    pressure_evidence_matches,
    restartable_endpoint,
    same_restartable_state,
    sampled_motion,
)


def _moving_al():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    momenta = np.arange(1, 13, dtype=float).reshape(4, 3) * 0.01
    atoms.set_momenta(momenta)
    atoms.info["time_fs"] = 100.0
    return atoms


def test_restartable_state_accepts_wrapping_but_rejects_stale_or_missing_momenta():
    last = _moving_al()
    saved = last.copy()
    saved.positions[0] += saved.cell.array[0]
    saved.positions += 4e-9
    saved.arrays["momenta"] += 4e-9
    assert same_restartable_state(last, saved)

    saved.arrays["momenta"][0, 0] += 0.001
    assert not same_restartable_state(last, saved)
    saved = last.copy()
    del saved.arrays["momenta"]
    assert not same_restartable_state(last, saved)
    assert not initial_com_removed(last)
    centered = last.copy()
    centered.set_momenta(last.get_momenta() - last.get_momenta().mean(axis=0))
    assert initial_com_removed(centered)


def test_linked_final_state_takes_precedence_over_complete_trajectory(tmp_path):
    last = _moving_al()
    trajectory = tmp_path / "trajectory.traj"
    write(trajectory, [last])
    final = tmp_path / "final.traj"
    stale = last.copy()
    stale.positions[0, 0] += 0.01
    write(final, [stale])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": {
                    "trajectory": "trajectory.traj",
                    "final_state": "final.traj",
                }
            }
        )
    )
    evidence = Evidence(manifest)
    assert not restartable_endpoint(evidence, evidence.trajectory("trajectory")[-1])

    manifest.write_text(json.dumps({"artifacts": {"trajectory": "trajectory.traj"}}))
    evidence = Evidence(manifest)
    assert restartable_endpoint(evidence, evidence.trajectory("trajectory")[-1])


def test_copied_frames_with_only_changed_times_are_not_dynamics():
    first = _moving_al()
    frames = [first.copy() for _ in range(3)]
    for atoms, time in zip(frames, (0.0, 100.0, 200.0), strict=True):
        atoms.info["time_fs"] = time
    assert not sampled_motion(frames, [0.0, 100.0, 200.0])
    frames[2].positions[0, 0] += 0.01
    assert sampled_motion(frames, [0.0, 100.0, 200.0])


@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".log", " ")])
def test_raw_pressure_log_matches_trace_and_rejects_changed_pressure(
    tmp_path, suffix, separator
):
    frames = [_moving_al() for _ in range(3)]
    times = np.array([100.0, 200.0, 300.0])
    trace = pd.DataFrame(
        {
            "stage": ["300K"] * 3,
            "time_fs": times,
            "temperature_K": [298.0, 301.0, 300.0],
            "pressure_GPa": [0.01, -0.02, 0.03],
            "volume_A3": [100.0, 101.0, 102.0],
        }
    )
    raw = trace.drop(columns="stage").copy()
    if suffix == ".log":
        raw = raw.drop(columns="time_fs")
        raw.insert(0, "step", [100, 200, 300])
    log_path = tmp_path / f"md{suffix}"
    raw.to_csv(log_path, sep=separator, index=False)
    trace_path = tmp_path / "trace.csv"
    trace.to_csv(trace_path, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"artifacts": {"thermal_trace": "trace.csv", "log": log_path.name}})
    )
    evidence = Evidence(manifest)
    kwargs = {
        "frames": frames,
        "frame_times_fs": times,
        "trace": trace,
        "stage_id": "300K",
        "timestep_fs": 1.0,
    }
    assert pressure_evidence_matches(evidence, **kwargs)
    bad = trace.copy()
    bad.loc[1, "pressure_GPa"] = 1.0
    assert not pressure_evidence_matches(evidence, **{**kwargs, "trace": bad})


def test_pressure_from_stress_and_missing_evidence_review(tmp_path):
    times = np.array([100.0, 200.0, 300.0])
    frames = [_moving_al() for _ in times]
    values = np.array([0.01, -0.02, 0.03])
    for atoms, pressure in zip(frames, values, strict=True):
        atoms.set_momenta(np.zeros((len(atoms), 3)))
        atoms.calc = SinglePointCalculator(
            atoms, stress=np.array([-pressure * units.GPa] * 3 + [0.0] * 3)
        )
    trace = pd.DataFrame(
        {
            "stage": ["300K"] * 3,
            "time_fs": times,
            "temperature_K": [300.0] * 3,
            "pressure_GPa": values,
            "volume_A3": [frames[0].get_volume()] * 3,
        }
    )
    trace_path = tmp_path / "trace.csv"
    trace.to_csv(trace_path, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {"thermal_trace": "trace.csv"}}))
    evidence = Evidence(manifest)
    kwargs = {
        "frames": frames,
        "frame_times_fs": times,
        "trace": trace,
        "stage_id": "300K",
        "timestep_fs": 1.0,
    }
    assert pressure_evidence_matches(evidence, **kwargs)
    trace.loc[2, "pressure_GPa"] += 0.1
    assert not pressure_evidence_matches(evidence, **kwargs)
    for atoms in frames:
        atoms.calc = None
    with pytest.raises(UnsupportedEvidence):
        pressure_evidence_matches(evidence, **kwargs)


def test_cu32_pinned_fcc_geometry_accepts_order_and_translation():
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2))
    assert matches_pinned_cu32(atoms)
    shuffled = atoms[np.random.default_rng(7).permutation(len(atoms))]
    shuffled.positions += [0.31, -0.27, 0.18]
    assert matches_pinned_cu32(shuffled)
    shuffled.positions[0, 0] += 0.02
    assert not matches_pinned_cu32(shuffled)
    assert not matches_pinned_cu32(
        bulk("Cu", "fcc", a=3.62, cubic=True).repeat((2, 2, 2))
    )
