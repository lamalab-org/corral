"""Parse public calculator metadata without selecting an execution backend."""

from collections.abc import Mapping


class CalculatorSettingsError(ValueError):
    """Malformed or contradictory calculator parameters in saved settings."""

    def __init__(self, location: str, message: str):
        self.location = location
        super().__init__(f"{location}: {message}")


class UnsupportedCalculatorSettings(CalculatorSettingsError):
    """Additional calculator options need an independent verification adapter."""


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
    }
    result, locations, unknown = {}, {}, []
    for location, config in sources:
        if location != "settings":
            unknown.extend(f"{location}.{key}" for key in config if key not in allowed)
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
