"""Physical bounds reject self-consistent answers with the wrong magnitude."""

import numpy as np
import pytest
from corral_md.scoring.common import EvidenceError
from corral_md.scoring.physical_reference import (
    reference_spectra,
    require_physical_reference,
    require_reference_spectrum,
)


@pytest.mark.parametrize("target", [1576.88, 1.74e-8, -0.11081421171444082])
@pytest.mark.parametrize("fraction", [-0.1, 0, 0.1])
def test_reference_bounds_are_inclusive_and_keep_negative_sign(target, fraction):
    require_physical_reference(target + fraction * abs(target), target, "quantity")


@pytest.mark.parametrize("target", [1576.88, 1.74e-8, -0.11081421171444082])
@pytest.mark.parametrize("fraction", [-0.100001, 0.100001, 2])
def test_just_outside_reference_range_fails(target, fraction):
    with pytest.raises(EvidenceError, match="physical reference range"):
        require_physical_reference(target + fraction * abs(target), target, "quantity")


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), [], [1, 2]])
def test_invalid_reference_quantities_fail(value):
    with pytest.raises(EvidenceError):
        require_physical_reference(value, 1, "quantity")


def test_every_vector_component_must_qualify_and_zero_has_no_absolute_floor():
    require_physical_reference([0, 270, -11], [0, 300, -10], "quantities")
    with pytest.raises(EvidenceError):
        require_physical_reference([0, 270, -11.01], [0, 300, -10], "quantities")
    with pytest.raises(EvidenceError):
        require_physical_reference(1e-100, 0, "zero")


def test_dynamic_reference_is_not_misattributed_to_historical_commit():
    result = require_physical_reference(1, 1, "model", source_commit=None)
    assert "source_commit" not in result


@pytest.mark.parametrize("index", range(5))
def test_every_original_spectrum_matches_itself_at_any_amplitude(index):
    name, frequency, power = reference_spectra()[index]
    result = require_reference_spectrum(frequency, power * 17)
    assert result["maximum_cosine_similarity"] == pytest.approx(1)
    assert result["matched_reference"] == name


def test_spectrum_grid_refinement_preserves_physical_comparison():
    _, frequency, power = reference_spectra()[0]
    grid = np.union1d(frequency, (frequency[:-1] + frequency[1:]) / 2)
    result = require_reference_spectrum(grid, np.interp(grid, frequency, power))
    assert result["maximum_cosine_similarity"] > 0.999


def test_same_power_at_wrong_frequencies_is_rejected():
    _, frequency, power = reference_spectra()[0]
    with pytest.raises(EvidenceError, match="physical reference spectra"):
        require_reference_spectrum(frequency * 20, power)


def test_power_outside_reference_frequency_domain_is_not_ignored():
    _, frequency, power = reference_spectra()[0]
    frequency = np.r_[frequency, 1000, 1500]
    power = np.r_[power, power.max() * 1000, 0]
    with pytest.raises(EvidenceError, match="physical reference spectra"):
        require_reference_spectrum(frequency, power)
