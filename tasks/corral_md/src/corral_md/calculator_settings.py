"""Parse public calculator metadata without selecting an execution backend."""

from collections.abc import Mapping
from math import isclose


class CalculatorSettingsError(ValueError):
    """Malformed or contradictory calculator parameters in saved settings."""

    def __init__(self, location: str, message: str):
        self.location = location
        super().__init__(f"{location}: {message}")


class UnsupportedCalculatorSettings(CalculatorSettingsError):
    """Additional calculator options need an independent verification adapter."""


_D3_KEYS = {
    "dispersion_xc",
    "damping",
    "dispersion_damping",
    "dispersion_cutoff_A",
    "dispersion_cutoff_Angstrom",
    "dispersion_cutoff",
    "dispersion_cutoff_angstrom",
    "dispersion_cutoff_bohr",
    "dispersion_cn_cutoff",
    "d3_calculator",
    "d3_attributes",
}
_D3_CUTOFF_A = 21.167088422553647  # MACE-MP-0's 40 Bohr default.
# Retained submissions also express 40 Bohr using this rounded conversion.
# Recognize that exact representation without broadening the cutoff tolerance.
_D3_CUTOFF_A_ROUNDED_CONVERSION = 40.0 * 0.52917726
_D3_CUTOFF_BOHR = 40.0


def _default_d3_options(config: Mapping) -> bool:
    """Recognize only the pinned mace_mp D3(BJ) defaults, not arbitrary D3 settings."""
    if not any(key in config for key in _D3_KEYS):
        return False
    for key, expected in (
        ("dispersion_xc", "pbe"),
        ("damping", "bj"),
        ("dispersion_damping", "bj"),
        (
            "d3_calculator",
            "torch_dftd.torch_dftd3_calculator.TorchDFTD3Calculator",
        ),
    ):
        if key in config:
            value = config[key]
            if key != "d3_calculator" and isinstance(value, str):
                value = value.lower()
            if value != expected:
                return False

    def cutoff_matches(value) -> bool:
        return (
            type(value) in (int, float)
            and any(
                isclose(value, cutoff, rel_tol=0, abs_tol=1e-6)
                for cutoff in (_D3_CUTOFF_A, _D3_CUTOFF_A_ROUNDED_CONVERSION)
            )
        )

    for key in ("dispersion_cutoff_A", "dispersion_cutoff_Angstrom", "dispersion_cutoff_angstrom", "dispersion_cutoff"):
        if key in config and not cutoff_matches(config[key]):
            return False
    if "dispersion_cutoff_bohr" in config and not (
        type(config["dispersion_cutoff_bohr"]) in (int, float)
        and isclose(
            config["dispersion_cutoff_bohr"],
            _D3_CUTOFF_BOHR,
            rel_tol=0,
            abs_tol=1e-6,
        )
    ):
        return False
    if (
        "dispersion_cn_cutoff" in config
        and config["dispersion_cn_cutoff"] != "torch-dftd default"
    ):
        return False
    attributes = config.get("d3_attributes")
    if attributes is not None:
        if not isinstance(attributes, Mapping):
            return False
        defaults = {
            "damping": "bj",
            "abc": False,
            "old": False,
            "bidirectional": True,
            "use_cache": False,
        }
        if any(key not in defaults and key != "cutoff" for key in attributes):
            return False
        if any(
            attributes[key] != expected
            for key, expected in defaults.items()
            if key in attributes
        ):
            return False
        if "cutoff" in attributes and not cutoff_matches(attributes["cutoff"]):
            return False
    return True


def parse_calculator_settings(settings: Mapping, *, role: str | None = None) -> dict:
    """Read declared options, leaving absent options for the caller to default.

    A calculator class name is metadata. Parameter dictionaries may use the
    documented model_settings field or the calculator/calculator_args aliases.
    Role-specific dictionaries describe separate calculators and do not inherit
    generic parameters. Submitted model paths never select a trusted backend.
    """
    sources = []
    if role:
        keys = (f"{role}_model_settings",)
    else:
        sources.append(("settings", settings))
        keys = ("model_settings", "calculator", "calculator_args")
    for key in keys:
        if key not in settings:
            continue
        value = settings[key]
        location = f"settings.{key}"
        if key == "calculator" and isinstance(value, str) and value.strip():
            continue
        if not isinstance(value, Mapping):
            expected = (
                "a JSON object or a nonempty class-name string"
                if key == "calculator"
                else "a JSON object"
            )
            raise CalculatorSettingsError(location, f"Expected {expected}")
        sources.append((location, value))

    dtype_aliases = ("default_dtype", "dtype", "precision", "model_dtype")
    allowed = {
        *dtype_aliases,
        "device",
        "dispersion",
        "model",
        "model_path",
        "checkpoint",
        "model_paths",
        "calculator",
        "note",
    }
    result, locations, unknown = {}, {}, []
    for location, config in sources:
        if location != "settings":
            recognized_d3 = (
                _D3_KEYS
                if config.get("dispersion", settings.get("dispersion")) is True
                and _default_d3_options(config)
                else set()
            )
            identity_options = {
                "enable_cueq": False,
                "energy_units_to_eV": 1.0,
                "length_units_to_A": 1.0,
                "model_type": "MACE",
                "compile_mode": None,
                "foundation_loader": "mace.calculators.mace_mp",
                "mace_version": "0.3.13",
            }
            unknown.extend(
                f"{location}.{key}"
                for key in config
                if key not in allowed
                and key not in recognized_d3
                and not (key in identity_options and type(config[key]) is type(identity_options[key]) and config[key] == identity_options[key])
            )
            for key in ("calculator", "note"):
                if key in config and (
                    not isinstance(config[key], str) or not config[key].strip()
                ):
                    raise CalculatorSettingsError(
                        f"{location}.{key}", "Expected a nonempty metadata string"
                    )
        elif any(key in config for key in _D3_KEYS) and not _default_d3_options(config):
            raise UnsupportedCalculatorSettings(
                location,
                "Nondefault dispersion parameters need a verification adapter",
            )
        for key in (*dtype_aliases, "device", "dispersion"):
            if key not in config:
                continue
            value = config[key]
            field = f"{location}.{key}"
            canonical = "default_dtype" if key in dtype_aliases else key
            if canonical == "default_dtype":
                if not isinstance(value, str) or value not in {"float32", "float64"}:
                    raise CalculatorSettingsError(field, "Expected float32 or float64")
            elif key == "dispersion":
                if type(value) is not bool:
                    raise CalculatorSettingsError(field, "Expected a boolean")
            elif not isinstance(value, str) or not value.strip():
                raise CalculatorSettingsError(
                    field, "Expected a nonempty device string"
                )
            if canonical in result and result[canonical] != value:
                raise CalculatorSettingsError(
                    field, f"Conflicts with {locations[canonical]}"
                )
            result[canonical], locations[canonical] = value, field
    if unknown:
        raise UnsupportedCalculatorSettings(
            unknown[0],
            "Additional calculator settings need a verification adapter: "
            + ", ".join(unknown),
        )
    return result
