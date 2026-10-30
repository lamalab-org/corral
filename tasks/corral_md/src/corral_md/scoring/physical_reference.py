"""Physical reference bounds, independent of an agent's submitted observations.

Historical constants are taken from the matching physical quantities in commit
75d7b3560b7ab922a63fe4fbff9c01d73f83dff3. Task numbers alone are not a mapping:
that commit's Level 1 tasks differ from the current Level 1 tasks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .common import EvidenceError, finite_array

REFERENCE_COMMIT = "75d7b3560b7ab922a63fe4fbff9c01d73f83dff3"
REFERENCE_RTOL = 0.10
PALLADIUM_ZPE_EV_PER_ATOM = 0.025437
ALUMINUM_EXPANSION_PER_K = 9.56133318e-5


def require_physical_reference(
    actual, target, quantity: str, *, source_commit: str | None = REFERENCE_COMMIT
) -> dict:
    """Require every value to lie within inclusive +/-10% of its reference.

    The reference, never the submission, sets the tolerance. Negative targets
    retain their sign; a zero reference permits only zero. The tiny relative
    roundoff allowance preserves decimal boundaries without a unit-dependent floor.
    """
    observed, expected = finite_array(actual), finite_array(target)
    if observed.shape != expected.shape:
        raise EvidenceError(f"{quantity}: physical reference shape mismatch")
    allowance = REFERENCE_RTOL * np.abs(expected)
    roundoff = 8 * np.finfo(float).eps * np.abs(expected)
    lower, upper = expected - allowance, expected + allowance
    if not np.all(np.abs(observed - expected) <= allowance + roundoff):
        raise EvidenceError(
            f"{quantity} is outside the physical reference range (+/-10%): "
            f"observed {observed.tolist()}, reference {expected.tolist()}, "
            f"allowed [{lower.tolist()}, {upper.tolist()}]"
        )
    return {
        "reference": expected.tolist(),
        "relative_tolerance": REFERENCE_RTOL,
        "allowed_lower": lower.tolist(),
        "allowed_upper": upper.tolist(),
        **({"source_commit": source_commit} if source_commit is not None else {}),
    }


@lru_cache(maxsize=1)
def reference_spectra() -> tuple:
    """Read the five original, unmodified Al spectra bundled with the scorer."""
    root = Path(__file__).parent / "references" / "task_6"
    spectra = []
    for seed in (42, 123, 456, 789, 1024):
        name = f"vdos_seed_{seed}.csv"
        data = np.loadtxt(root / name, delimiter=",", skiprows=1)
        frequency, power = data[:, 0].copy(), data[:, 1].copy()
        frequency.setflags(write=False)
        power.setflags(write=False)
        spectra.append((name, frequency, power))
    return tuple(spectra)


def require_reference_spectrum(frequency, power) -> dict:
    """Require the historical 0.9 cosine similarity on aligned physical axes.

    Integrate on the union of the frequency grids so that changing the sampling
    cadence does not compare different frequencies or hide out-of-range power.
    The original task used a spectrum threshold, not scalar moment targets.
    """
    frequency = finite_array(frequency, ndim=1)
    power = finite_array(power, shape=frequency.shape)
    if (
        len(frequency) < 3
        or frequency[0] < 0
        or not np.all(np.diff(frequency) > 0)
        or np.any(power < 0)
        or not np.any(power > 0)
    ):
        raise EvidenceError("Invalid spectrum for physical reference comparison")
    best, matched = -np.inf, None
    for name, reference_frequency, reference_power in reference_spectra():
        grid = np.union1d(frequency, reference_frequency)
        observed = np.interp(grid, frequency, power, left=0, right=0)
        expected = np.interp(
            grid, reference_frequency, reference_power, left=0, right=0
        )
        # Rescaling avoids overflow without changing cosine similarity.
        observed /= observed.max()
        expected /= expected.max()
        norm = np.sqrt(
            np.trapezoid(observed**2, grid) * np.trapezoid(expected**2, grid)
        )
        similarity = float(np.trapezoid(observed * expected, grid) / norm)
        if similarity > best:
            best, matched = similarity, name
    if not np.isfinite(best) or best < 0.9 - 8 * np.finfo(float).eps:
        raise EvidenceError(
            "VDOS disagrees with the historical physical reference spectra: "
            f"best cosine similarity {best:.8g}; required >= 0.9"
        )
    return {
        "maximum_cosine_similarity": min(best, 1.0),
        "minimum_cosine_similarity": 0.9,
        "matched_reference": matched,
        "source_commit": REFERENCE_COMMIT,
    }
