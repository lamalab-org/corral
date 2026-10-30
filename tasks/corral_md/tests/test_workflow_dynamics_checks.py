"""Adversarial checks for Level 1 dynamics evidence shared by several tasks."""

import numpy as np
from ase.build import bulk
from corral_md.scoring.dynamics import (
    matches_pinned_cu32,
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


def test_copied_frames_with_only_changed_times_are_not_dynamics():
    first = _moving_al()
    frames = [first.copy() for _ in range(3)]
    for atoms, time in zip(frames, (0.0, 100.0, 200.0), strict=True):
        atoms.info["time_fs"] = time
    assert not sampled_motion(frames, [0.0, 100.0, 200.0])
    frames[2].positions[0, 0] += 0.01
    assert sampled_motion(frames, [0.0, 100.0, 200.0])


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
