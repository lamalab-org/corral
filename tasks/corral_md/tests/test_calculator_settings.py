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


def test_recorded_default_d3_cutoffs_are_supported_without_changing_the_calculator():
    settings = {
        "dispersion": True,
        "model_settings": {
            "dispersion": True,
            "dispersion_xc": "pbe",
            "damping": "bj",
            "dispersion_cutoff_angstrom": 21.167088422553647,
            "dispersion_cutoff_bohr": 40.0,
            "dispersion_cn_cutoff": "torch-dftd default",
        },
    }
    assert parse_calculator_settings(settings) == {"dispersion": True}
    assert calculator_settings(SimpleNamespace(settings=settings))["dispersion"] is True

    settings["model_settings"]["dispersion_cutoff_bohr"] = 35.0
    with pytest.raises(UnsupportedCalculatorSettings):
        parse_calculator_settings(settings)


@pytest.mark.parametrize("cutoff", [40.0 * 0.52917721, 40.0 * 0.52917726])
def test_recorded_d3_damping_alias_and_bohr_conversion(cutoff):
    settings = {"model_settings": {
        "dispersion": True,
        "dispersion_damping": "bj",
        "dispersion_xc": "pbe",
        "dispersion_cutoff_A": cutoff,
    }}
    assert _model_parameters(SimpleNamespace(settings=settings), 3) == {
        "model": "teacher.model", "default_dtype": "float64", "dispersion": True,
    }
    for changes in (
        {"damping": "zero"},
        {"dispersion_damping": "zero"},
        {"dispersion_cutoff_A": cutoff + 1e-5},
        {"dispersion": False},
    ):
        altered = copy.deepcopy(settings)
        altered["model_settings"].update(changes)
        with pytest.raises(UnsupportedCalculatorSettings):
            parse_calculator_settings(altered)


def test_identity_calculator_options_are_accepted_but_nondefault_values_stay_pending():
    options = {
        "default_dtype": "float64",
        "enable_cueq": False,
        "energy_units_to_eV": 1.0,
        "length_units_to_A": 1.0,
        "model_type": "MACE",
        "compile_mode": None,
    }
    settings = {"model_settings": options, "calculator_args": {"model_paths": "input/teacher.model"}}
    assert parse_calculator_settings(settings) == {"default_dtype": "float64"}
    for key, value in (
        ("enable_cueq", True),
        ("energy_units_to_eV", 2.0),
        ("length_units_to_A", 2.0),
        ("model_type", "Other"),
        ("compile_mode", "max-autotune"),
    ):
        changed = copy.deepcopy(settings)
        changed["model_settings"][key] = value
        with pytest.raises(UnsupportedCalculatorSettings):
            parse_calculator_settings(changed)


@pytest.mark.parametrize("role", [None, "teacher"])
@pytest.mark.parametrize("uppercase", [False, True])
def test_explicit_default_dispersion_parameters_use_the_pinned_calculator(
    role, uppercase
):
    key = "model_settings" if role is None else "teacher_model_settings"
    settings = {
        key: {
            "dispersion": True,
            "damping": "BJ" if uppercase else "bj",
            "dispersion_xc": "PBE" if uppercase else "pbe",
            "dispersion_cutoff_A": 21.1670884268,
            "default_dtype": "float64",
        }
    }
    assert calculator_settings(SimpleNamespace(settings=settings), role=role) == {
        "model": "teacher.model",
        "default_dtype": "float64",
        "dispersion": True,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("damping", "zero"),
        ("dispersion_xc", "b3lyp"),
        ("dispersion_cutoff_A", 40.0),
        ("dispersion_cutoff", True),
    ],
)
@pytest.mark.parametrize("container", [None, "model_settings"])
def test_nondefault_dispersion_is_not_silently_replaced(key, value, container):
    options = {"dispersion": True, key: value}
    settings = {container: options} if container else options
    with pytest.raises(UnsupportedCalculatorSettings):
        parse_calculator_settings(settings)


@pytest.mark.parametrize("dispersion", [None, False, True])
def test_top_level_dispersion_defaults_preserve_the_declared_option(dispersion):
    settings = {"damping": "bj", "dispersion_cutoff_A": 21.167088422553647}
    expected = {}
    if dispersion is not None:
        settings["dispersion"] = dispersion
        expected["dispersion"] = dispersion
    assert parse_calculator_settings(settings) == expected


def test_role_calculator_and_note_are_metadata_with_explicit_options():
    settings = {
        "md_model_settings": {
            "calculator": "mace.calculators.MACECalculator",
            "note": "Student learned the total energy; no extra dispersion",
            "dispersion": False,
            "default_dtype": "float64",
        }
    }
    assert parse_calculator_settings(settings, role="md") == {
        "dispersion": False,
        "default_dtype": "float64",
    }


@pytest.mark.parametrize("key", ["calculator", "note"])
def test_role_metadata_cannot_hide_a_parameter_object(key):
    with pytest.raises(CalculatorSettingsError):
        parse_calculator_settings(
            {"teacher_model_settings": {key: {"custom": 1}}}, role="teacher"
        )


def test_dispersion_damping_alias_and_rounded_bohr_conversion():
    assert parse_calculator_settings(
        {
            "model_settings": {
                "dispersion_damping": "bj",
                "dispersion_cutoff_A": 40 * 0.52917726,
                "dispersion": True,
            }
        }
    ) == {"dispersion": True}
