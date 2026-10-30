"""Read-only checks for saved Level 1 molecular-dynamics states and traces.

These checks establish agreement among retained artifacts. They do not establish
that an unrecorded dynamics step or velocity reset never occurred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.build import bulk

from .common import close, finite_array

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
