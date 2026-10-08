"""Calculator metadata must not change declared scientific parameters."""

import copy
from types import SimpleNamespace

import pytest
from corral_md.calculator_settings import (
    CalculatorSettingsError,
    UnsupportedCalculatorSettings,
    parse_calculator_settings,
)
from corral_md.workflow_scoring.level1_trusted import _model_parameters
from corral_md.workflow_scoring.verification import (
    Plan,
    UnsupportedEvidence,
    calculator_settings,
)


@pytest.mark.parametrize(
    "container", [None, "model_settings", "calculator", "calculator_args"]
)
@pytest.mark.parametrize(
    "alias", ["default_dtype", "dtype", "precision", "model_dtype"]
)
def test_supported_layouts_preserve_parameters_and_pin_the_trusted_model(
    container, alias
):
    options = {alias: "float32", "device": "cuda", "dispersion": False}
    settings = {container: options} if container else options
    settings = {**settings, "model": "/untrusted/other.model"}
    if container != "calculator":
        settings["calculator"] = "mace.calculators.MACECalculator"
    original = copy.deepcopy(settings)
    assert parse_calculator_settings(settings) == {
        "default_dtype": "float32",
        "device": "cuda",
        "dispersion": False,
    }
    expected = {
        "model": "teacher.model",
        "default_dtype": "float32",
        "dispersion": False,
    }
    evidence = SimpleNamespace(settings=settings)
    assert calculator_settings(evidence) == expected
    assert _model_parameters(evidence, 5) == expected
    assert settings == original


def test_matching_duplicates_are_accepted_and_roles_stay_independent():
    settings = {
        "dtype": "float64",
        "model_settings": {"precision": "float64"},
        "calculator_args": {"default_dtype": "float64"},
        "teacher_model_settings": {"dtype": "float32", "dispersion": True},
        "student_model_settings": {"dtype": "float64", "dispersion": False},
    }
    assert parse_calculator_settings(settings) == {"default_dtype": "float64"}
    assert parse_calculator_settings(settings, role="teacher") == {
        "default_dtype": "float32",
        "dispersion": True,
    }
    assert parse_calculator_settings(settings, role="student") == {
        "default_dtype": "float64",
        "dispersion": False,
    }
    assert parse_calculator_settings(settings, role="md") == {}


@pytest.mark.parametrize(
    ("settings", "location"),
    [
        ({"model_settings": []}, "settings.model_settings"),
        ({"calculator_args": None}, "settings.calculator_args"),
        ({"calculator": " "}, "settings.calculator"),
        ({"dtype": "half"}, "settings.dtype"),
        ({"dispersion": "false"}, "settings.dispersion"),
        ({"device": False}, "settings.device"),
        (
            {"dtype": "float32", "model_settings": {"dtype": "float64"}},
            "settings.model_settings.dtype",
        ),
        (
            {"dispersion": True, "calculator": {"dispersion": False}},
            "settings.calculator.dispersion",
        ),
        (
            {"device": "cpu", "calculator_args": {"device": "cuda"}},
            "settings.calculator_args.device",
        ),
    ],
)
def test_malformed_or_conflicting_options_fail_instead_of_becoming_pending(
    settings, location
):
    with pytest.raises(CalculatorSettingsError) as error:
        parse_calculator_settings(settings)
    assert error.value.location == location
    evidence = SimpleNamespace(settings=settings)
    plan = Plan(evidence, 5, "nonce")
    plan.attempt(
        "calculator", lambda: calculator_settings(evidence), targets=["forces"]
    )
    assert plan.notes[0]["status"] == "failed"
    assert plan.notes[0]["targets"] == ["forces"]


def test_unknown_calculator_options_still_require_independent_verification_support():
    settings = {"model_settings": {"custom_potential": True}}
    with pytest.raises(UnsupportedCalculatorSettings):
        parse_calculator_settings(settings)
    with pytest.raises(UnsupportedEvidence):
        calculator_settings(SimpleNamespace(settings=settings))
