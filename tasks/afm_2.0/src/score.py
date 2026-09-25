"""Score AFM submissions from raw NID artifacts, without instrument access.

Task lengths and reported roughness are in nm, line times in seconds, and
friction in V. Task tolerance applies to both settings and measurements as a
fraction of the expected value (0.01 = 1%), with no absolute allowance. Zero
targets, counts, modes and setpoint units must match exactly.
"""

import json
import math
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
from loguru import logger
from NSFopen.read import read

LENGTH_UNITS = {"m": 1.0, "mm": 1e-3, "µm": 1e-6, "um": 1e-6, "nm": 1e-9, "pm": 1e-12}
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "µs": 1e-6, "us": 1e-6, "ns": 1e-9}
VOLT_UNITS = {"V": 1.0, "mV": 1e-3, "µV": 1e-6, "uV": 1e-6}
ROUGHNESS = {"rms_roughness", "mean_roughness"}
FRICTION = {"average_friction", "rms_friction"}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _finite_number(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Booleans are not measurements")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Measurements must be finite")
    return number


def clean_unit_string(value):
    return str(value).replace("Â", "").replace("μ", "µ").strip()


def _quantity(value):
    match = re.fullmatch(rf"\s*({NUMBER})\s*([^\d\s]*)\s*", clean_unit_string(value))
    if not match:
        raise ValueError(f"Invalid numeric quantity: {value}")
    return _finite_number(match[1]), match[2]


def _convert(value, units):
    number, unit = _quantity(value)
    if unit not in units:
        raise ValueError(f"Unsupported unit: {unit}")
    return number * units[unit]


def to_meters(value):
    return _convert(value, LENGTH_UNITS)


def to_seconds(value):
    return _convert(value, TIME_UNITS)


def parse_setpoint(value):
    number, unit = _quantity(value)
    if unit == "%":
        return {"value": number, "unit": "%"}
    if unit in VOLT_UNITS:
        return {"value": number * VOLT_UNITS[unit], "unit": "V"}
    raise ValueError(f"Unsupported setpoint unit: {unit}")


def _mode(value):
    text = clean_unit_string(value).lower().replace("-", " ")
    aliases = {
        "contact": 2,
        "contact mode": 2,
        "static force": 2,
        "lateral force": 2,
        "dynamic": 3,
        "dynamic force": 3,
        "phase contrast": 4,
        "tapping": 4,
        "tapping mode": 4,
    }
    return aliases[text] if text in aliases else _finite_number(text)


def extract_params(afm):
    """Extract available DataSet-Info fields in the task's nm/s/degree units.

    Missing fields stay missing and fail comparison if required by the task.
    No value is supplied by the submission or by the live instrument.
    """
    info = afm.param[("HeaderDump", "DataSet-Info")]
    fields = {
        "pgain": ("P-Gain", _finite_number),
        "igain": ("I-Gain", _finite_number),
        "dgain": ("D-Gain", _finite_number),
        "times_per_line": ("Time/Line", to_seconds),
        "points_per_line": ("Points", _finite_number),
        "lines_per_frame": ("Lines", _finite_number),
        "centre_x": ("X-Pos", lambda v: to_meters(v) * 1e9),
        "centre_y": ("Y-Pos", lambda v: to_meters(v) * 1e9),
        "rotation": ("Rotation", lambda v: _convert(v, {"°": 1, "deg": 1, "": 1})),
        "image_height": ("Image size", lambda v: to_meters(v) * 1e9),
        "image_width": ("Image size", lambda v: to_meters(v) * 1e9),
        "mode": ("Op. mode", _mode),
        "tip": ("Cantilever type", clean_unit_string),
        "setpoint": ("Setpoint", parse_setpoint),
    }
    params = {
        key: convert(info[field])
        for key, (field, convert) in fields.items()
        if field in info
    }
    # Channel extents describe rectangular scans more precisely than Image size.
    channel = _channel_header(afm, "Forward", "Z-Axis")
    for key, dim in (("image_width", "Dim0"), ("image_height", "Dim1")):
        if f"{dim}Range" in channel:
            params[key] = (
                _convert(
                    f"{channel[f'{dim}Range']} {channel[f'{dim}Unit']}", LENGTH_UNITS
                )
                * 1e9
            )
    return params


def _channel_header(afm, direction, channel):
    header = afm.param["HeaderDump"]
    matches = [
        entry
        for _, entry in header.items()
        if isinstance(entry, dict)
        and entry.get("Frame") == f"Scan {direction.lower()}"
        and entry.get("Dim2Name") == channel
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one header for {direction}/{channel}")
    return matches[0]


def check_params(expected, actual, tolerance=0.01):
    """Compare NID settings within tolerance * abs(task target)."""
    tolerance = _finite_number(tolerance)
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    for key, expected_value in expected.items():
        target = expected_value
        if key not in actual:
            return False
        value = actual[key]
        if key == "setpoint":
            if value["unit"] != target["unit"]:
                return False
            value, target = value["value"], target["value"]
        elif key == "mode":
            if _mode(value) != _mode(target):
                return False
            continue
        elif key == "tip":
            # NID stores the cantilever name, not its SDK GUID.
            if str(value).strip() != str(target).strip():
                return False
            continue
        value, target = _finite_number(value), _finite_number(target)
        if key in {"points_per_line", "lines_per_frame"}:
            if value != target or value != int(value) or value <= 0:
                return False
        elif not _close(value, target, tolerance):
            return False
    return True


def _image_array(afm, direction, channel, shape, units):
    array = np.asarray(afm.data["Image"][direction][channel], dtype=float)
    if (
        array.ndim != 2
        or array.size == 0
        or array.shape != shape
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"Invalid {direction}/{channel} image; expected {shape}")
    header = _channel_header(afm, direction, channel)
    if (int(header["Lines"]), int(header["Points"])) != shape:
        raise ValueError("Channel dimensions do not match acquisition settings")
    unit = clean_unit_string(header["Dim2Unit"])
    if unit not in units:
        raise ValueError(f"Unsupported {channel} unit: {unit}")
    return array * units[unit]


def measure_image(
    afm, metrics, *, shape=None, require_lateral=False, friction_absolute=False
):
    """Compute raw-image roughness in nm and friction in V using channel units.

    NSFopen scales samples by Dim2Range/Dim2Min; it does not convert Dim2Unit.
    Convert the declared unit once, then compute the requested measurements.
    """
    if shape is None:
        header = _channel_header(afm, "Forward", "Z-Axis")
        shape = (int(header["Lines"]), int(header["Points"]))
    height = _image_array(afm, "Forward", "Z-Axis", shape, LENGTH_UNITS) * 1e9
    values = {}
    if ROUGHNESS.intersection(metrics):
        centered = height - height.mean()
        values["rms_roughness"] = float(np.sqrt(np.mean(centered**2)))
        values["mean_roughness"] = float(np.mean(np.abs(centered)))
    if require_lateral or FRICTION.intersection(metrics):
        forward = _image_array(afm, "Forward", "Friction force", shape, VOLT_UNITS)
        backward = _image_array(afm, "Backward", "Friction force", shape, VOLT_UNITS)
        # Keep NSFopen's array orientation, as in Image_Analyzer.
        friction = (forward - backward) / 2
        values["average_friction"] = float(
            np.mean(np.abs(friction) if friction_absolute else friction)
        )
        values["rms_friction"] = float(np.sqrt(np.mean(friction**2)))
    return {metric: values[metric] for metric in metrics}


def score_topography(tolerance, final_params, **options):
    """Check one raw path or the configured sequence of raw image paths."""
    return _scorer(tolerance, final_params, (), **options)


def score_roughness(
    tolerance, final_params, metrics=("rms_roughness", "mean_roughness"), **options
):
    """Check configured height roughness measurements in nm and NID settings."""
    if not metrics or not set(metrics) <= ROUGHNESS:
        raise ValueError("Select RMS and/or mean roughness")
    return _scorer(tolerance, final_params, metrics, **options)


def score_friction(tolerance, final_params, metrics=("average_friction",), **options):
    """Check configured friction measurements in V and NID settings."""
    if not metrics or not set(metrics) <= FRICTION:
        raise ValueError("Select average and/or RMS friction")
    return _scorer(tolerance, final_params, metrics, **options)


def score_roughness_and_friction(
    tolerance,
    final_params,
    metrics=("rms_roughness", "mean_roughness", "average_friction"),
    **options,
):
    """Require both roughness and friction measurements, plus NID settings."""
    if (
        not (set(metrics) & ROUGHNESS and set(metrics) & FRICTION)
        or not set(metrics) <= ROUGHNESS | FRICTION
    ):
        raise ValueError("Select both roughness and friction metrics")
    return _scorer(tolerance, final_params, metrics, **options)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate submission key: {key}")
        result[key] = value
    return result


def _scorer(
    tolerance,
    final_params,
    metrics,
    percent_change_reference=None,
    require_lateral=False,
    friction_absolute=False,
):
    tolerance = _finite_number(tolerance)
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    # A mapping describes the single level-1 artifact; an ordered list describes
    # each level-2 artifact. There is no separate duplicated final-image target.
    sequence = deepcopy(
        final_params if isinstance(final_params, list) else [final_params]
    )
    metrics = tuple(metrics)
    if (
        not sequence
        or not final_params
        or not all(isinstance(p, dict) and p for p in sequence)
    ):
        raise ValueError("Expected nonempty acquisition settings")
    count = len(sequence)
    if percent_change_reference is not None and (
        isinstance(percent_change_reference, bool)
        or not isinstance(percent_change_reference, int)
        or not 1 <= percent_change_reference <= count
        or not metrics
    ):
        raise ValueError("Invalid percentage-change reference acquisition")
    fields = {f"path_{i}" for i in range(1, count + 1)}
    fields.update(f"{m}_{i}" for m in metrics for i in range(1, count + 1))
    if percent_change_reference is not None:
        fields.update(
            f"{m}_percent_change_{i}" for m in metrics for i in range(1, count + 1)
        )

    def score_fn(result):
        try:
            if count == 1 and not metrics:
                report = {"path_1": result}
            else:
                report = (
                    json.loads(result, object_pairs_hook=_unique_object)
                    if isinstance(result, str)
                    else result
                )
            if not isinstance(report, dict) or set(report) != fields:
                raise ValueError(f"Expected submission fields: {sorted(fields)}")
            seen = set()
            measurements = []
            for i, expected in enumerate(sequence, 1):
                path = report[f"path_{i}"]
                if not isinstance(path, str):
                    raise ValueError("Expected an absolute NID path")
                path = Path(path.strip())
                if (
                    not path.is_absolute()
                    or path.suffix.lower() != ".nid"
                    or not path.is_file()
                ):
                    raise ValueError("Expected an existing absolute .nid file")
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    raise ValueError("Each acquisition requires a separate file")
                seen.add(identity)
                afm = read(str(path))
                actual = extract_params(afm)
                if not check_params(expected, actual, tolerance):
                    raise ValueError(
                        f"Acquisition {i} settings differ from task settings"
                    )
                measured = measure_image(
                    afm,
                    metrics,
                    shape=(
                        int(actual["lines_per_frame"]),
                        int(actual["points_per_line"]),
                    ),
                    require_lateral=require_lateral,
                    friction_absolute=friction_absolute,
                )
                for metric in metrics:
                    if not _close(report[f"{metric}_{i}"], measured[metric], tolerance):
                        raise ValueError(f"Acquisition {i}: incorrect {metric}")
                measurements.append(measured)
            if percent_change_reference is not None:
                reference = measurements[percent_change_reference - 1]
                for i, measured in enumerate(measurements, 1):
                    for metric in metrics:
                        base = reference[metric]
                        expected = (
                            None
                            if base == 0
                            else 100 * (measured[metric] - base) / base
                        )
                        answer = report[f"{metric}_percent_change_{i}"]
                        if (expected is None and answer is not None) or (
                            expected is not None
                            and not _close(answer, expected, tolerance)
                        ):
                            raise ValueError(
                                f"Acquisition {i}: incorrect percentage change"
                            )
            return 1.0
        except Exception as exc:
            logger.warning(f"AFM scoring failed: {exc}")
            return 0.0

    return score_fn


def _close(answer, measured, tolerance):
    measured = _finite_number(measured)
    margin = tolerance * abs(measured)
    return measured - margin <= _finite_number(answer) <= measured + margin
