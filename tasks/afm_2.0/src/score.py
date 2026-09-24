"""AFM submission scoring and measurement helpers.

Scorer factories return a callable that evaluates a submitted result.
File helpers inspect NID data; instrument helpers read live Nanosurf settings.
RMS roughness is returned in the original data units without conversion.
"""

import gc
import json
import math
import platform
import re
from pathlib import Path

import nanosurf
import numpy as np
from loguru import logger
from NSFopen.read import read
from skimage.metrics import structural_similarity as ssim

# ----------------------------------------------------------
# Safe pythoncom import (Windows only)
# ----------------------------------------------------------
if platform.system() == "Windows":
    import pythoncom
else:
    pythoncom = None


# ----------------------------------------------------------------------------
# Submission scorers: numeric and text answers
# ----------------------------------------------------------------------------


def check_numerical(target: float, tolerance: float, final_params):
    """
    Returns a scoring function that checks if a result is within a
    percentage-based tolerance of the target.

    Args:
        target (float): The target value.
        tolerance (float): Fractional tolerance (e.g., 0.1 means ±10% of target).

    Returns:
        score_fn (function): A function that accepts a result and returns 1.0
            if it's within tolerance, else 0.0.
    """

    def score_fn(result: str) -> float:
        try:
            if isinstance(result, str):
                result = result.strip()
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict) and "answer" in parsed_result:
                        answer = float(parsed_result["answer"])
                    else:
                        answer = float(result)
                except json.JSONDecodeError:
                    answer = float(result)
            else:
                if isinstance(result, dict) and "answer" in result:
                    answer = float(result["answer"])
                else:
                    answer = float(result)
        except (ValueError, TypeError):
            return 0.0  # Not a valid number

        abs_tol = tolerance * target
        if target - abs_tol <= answer <= target + abs_tol:
            score = check_params(final_params)
            logger.info(f"score for params {score}")
            return 1.0 * score
        return 0.0

    return score_fn


def check_indentation(target: str):
    """Build a scorer that compares stripped text to the target, ignoring case."""

    def score_fn(result: str) -> float:
        try:
            isinstance(result, str)
            result = result.strip()
            if result.lower() == target.lower():
                return 1.0
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0

    return score_fn


# ----------------------------------------------------------------------------
# Submission scorers: instrument settings and files
# ----------------------------------------------------------------------------


def check_params_function(final_params):
    """Build a scorer that checks live instrument settings, ignoring the answer."""

    def score_fn(_result: str) -> float:
        try:
            logger.info(f"Checking params with final_params: {final_params}")
            # Validate against the live settings via check_params below
            score = check_params(final_params)
            logger.info(f"check params score: {score}")
            return float(score)  # Ensure the returned score is a float
        except Exception as e:
            logger.error(f"Error during scoring: {e}")
            return 0.0  # Return 0.0 in case of an error

    return score_fn  # Return the scoring function itself


def check_file_exists(final_params):
    """Build a scorer requiring both matching instrument settings and a NID path."""

    def score_fn(result: str) -> float:
        try:
            logger.info(f"Checking params with final_params: {final_params}")
            # Validate against the live settings via check_params below
            score = check_params(final_params)
            score2 = check_nid_file_exists(result)
            logger.info(f"check params score: {score}")
            return float(score * score2)  # Ensure the returned score is a float
        except Exception as e:
            logger.error(f"Error during scoring: {e}")
            return 0.0  # Return 0.0 in case of an error

    return score_fn  # Return the scoring function itself


# ----------------------------------------------------------------------------
# Submission scorers: image measurements
# ----------------------------------------------------------------------------


def check_image_quality(tolerance, final_params):
    """Build a scorer requiring trace/retrace similarity and matching settings."""

    def score_fn(result):
        try:
            if isinstance(result, str):
                result = result.strip()
                afm = read(result)
                data = afm.data
                im_file_fw = data["Image"]["Forward"]["Z-Axis"]
                im_file_bw = data["Image"]["Backward"]["Z-Axis"]
                similarity_index, _diff = ssim(
                    im_file_bw,
                    im_file_fw,
                    full=True,
                    data_range=im_file_bw.max() - im_file_bw.min(),
                )
                if similarity_index >= tolerance:
                    score = check_params(final_params)
                    logger.info(f"check params score : {score}")
                    return 1.0 * score
                return 0.0
        except (ValueError, TypeError):
            return 0.0

    return score_fn


def check_roughness_function(tolerance: float, final_params):
    """
    Parses a submission_format string from the LLM, handling minor formatting issues.
    Runs check_roughness(path) for each valid pair of (rms_roughness_n, path_n).
    Submitted RMS values must use the original data units; no scaling is applied.
    Returns 1 if all RMS values pass the tolerance check AND
    check_params(final_params)==1, else 0.
    """

    def score_fn(result: str) -> float:
        try:
            cleaned = result.replace(
                ";", ","
            ).strip()  # Fix common LLM formatting mistakes

            # Ensure JSON braces are balanced
            if not cleaned.startswith("{"):
                cleaned = "{" + cleaned
            if not cleaned.endswith("}"):
                cleaned = cleaned + "}"

            data = json.loads(cleaned)  # Try parsing JSON
            # Detect all indices dynamically (e.g., 1, 2, 3, ...)
            indices = sorted(
                {
                    int(re.findall(r"\d+", key)[0])
                    for key in data
                    if key.startswith("rms_roughness_")
                }
            )

            if not indices:
                logger.warning("No RMS entries found")
                return 0

            all_passed = True  # Track if all checks pass
            for i in indices:
                rough_key = f"rms_roughness_{i}"
                path_key = f"path_{i}"

                if rough_key in data and path_key in data:
                    try:
                        rms = float(data[rough_key])
                    except ValueError:
                        logger.warning(f"Invalid RMS value for entry {i}")
                        all_passed = False
                        continue

                    path = data[path_key]
                    logger.info(f"Entry {i}: RMS={rms}, Path={path}")

                    check_output = check_roughness(
                        path
                    )  # Run user-defined check function
                    abs_tol = tolerance * rms
                    logger.info(
                        f"Tolerance: ±{tolerance}, Check Output: {check_output}"
                    )

                    # Check if the measured value is within tolerance
                    if not (rms - abs_tol <= check_output <= rms + abs_tol):
                        logger.warning(f"Entry {i} failed tolerance check.")
                        all_passed = False

            # Additional condition: check_params(final_params) must be 1
            if check_params(final_params) != 1:
                logger.warning("check_params(final_params) != 1")
                logger.info(check_params(final_params))
                all_passed = False

            return 1 if all_passed else 0

        except json.JSONDecodeError:
            logger.warning("Could not parse JSON — please check the submission format.")
            return 0
        except Exception as e:
            logger.warning(f"Error: {e}")
            return 0

    return score_fn


# ----------------------------------------------------------------------------
# Level 1 scorers: one acquisition and its reported measurements
# ----------------------------------------------------------------------------


def score_single_topography(tolerance: float, final_params):
    """Score an absolute NID path and the requested topography settings."""
    return _single_image_scorer(tolerance, final_params)


def score_single_rms_roughness(tolerance: float, final_params):
    """Score path_1 and rms_roughness_1 in the original image units."""
    return _single_image_scorer(tolerance, final_params, ("rms_roughness",))


def score_single_mean_roughness(tolerance: float, final_params):
    """Score path_1 and mean_roughness_1 (mean absolute height deviation)."""
    return _single_image_scorer(tolerance, final_params, ("mean_roughness",))


def score_single_topography_roughness(tolerance: float, final_params):
    """Score both roughness metrics; check the derived line time via final_params."""
    return _single_image_scorer(
        tolerance, final_params, ("rms_roughness", "mean_roughness")
    )


def score_single_average_friction(tolerance: float, final_params):
    """Score average_friction_1 from half the trace/retrace difference."""
    return _single_image_scorer(
        tolerance, final_params, ("average_friction",), lateral=True
    )


def score_single_rms_friction(tolerance: float, final_params):
    """Score rms_friction_1, without subtracting the friction signal's mean."""
    return _single_image_scorer(
        tolerance, final_params, ("rms_friction",), lateral=True
    )


def score_single_lateral_roughness(tolerance: float, final_params):
    """Require lateral channels and score both roughness metrics from height."""
    return _single_image_scorer(
        tolerance, final_params, ("rms_roughness", "mean_roughness"), lateral=True
    )


def score_single_roughness_and_friction(tolerance: float, final_params):
    """Score both height roughness metrics and signed average friction."""
    return _single_image_scorer(
        tolerance,
        final_params,
        ("rms_roughness", "mean_roughness", "average_friction"),
        lateral=True,
    )


def _single_image_scorer(tolerance, final_params, metrics=(), *, lateral=False):
    """Build a fail-closed scorer for the Level 1 path or numbered JSON format.

    tolerance is the fractional error allowed in reported metrics. Instrument
    settings use check_params' own tolerances. Settings are checked live, not
    reconstructed from acquisition history. No metric unit conversion is applied.
    """
    tolerance = _finite_number(tolerance)
    if tolerance < 0:
        raise ValueError("Scoring tolerance must be nonnegative")
    final_params = dict(final_params)

    def score_fn(result) -> float:
        try:
            if metrics:
                report = json.loads(result) if isinstance(result, str) else result
                expected_keys = {"path_1", *(f"{metric}_1" for metric in metrics)}
                if not isinstance(report, dict) or set(report) != expected_keys:
                    raise ValueError(
                        f"Expected submission fields: {sorted(expected_keys)}"
                    )
                submitted = {
                    metric: _finite_number(report[f"{metric}_1"]) for metric in metrics
                }
                path = report["path_1"]
            else:
                submitted = {}
                path = result

            if not isinstance(path, str) or not path.strip():
                raise ValueError("An absolute NID file path is required")
            path = Path(path.strip())
            if (
                not path.is_absolute()
                or path.suffix.lower() != ".nid"
                or not path.is_file()
            ):
                raise ValueError("Expected an existing absolute .nid file path")

            shape = (final_params["lines_per_frame"], final_params["points_per_line"])
            image = read(str(path)).data["Image"]
            height = _image_array(image["Forward"]["Z-Axis"], shape)
            measured = {}
            if "rms_roughness" in metrics or "mean_roughness" in metrics:
                centered = height - np.mean(height)
                rows, columns = height.shape
                measured["rms_roughness"] = float(
                    np.sqrt(np.sum(centered**2) / (rows * columns))
                )
                measured["mean_roughness"] = float(
                    np.sum(np.abs(centered)) / (rows * columns)
                )
            if lateral:
                friction = _friction_signal(image, shape)
                measured["average_friction"] = float(np.mean(friction))
                measured["rms_friction"] = float(np.sqrt(np.mean(friction**2)))

            for metric, answer in submitted.items():
                expected = _finite_number(measured[metric])
                # A relative comparison keeps small, unconverted signals meaningful.
                if not math.isclose(answer, expected, rel_tol=tolerance, abs_tol=0.0):
                    logger.warning(f"{metric}: submitted {answer}, measured {expected}")
                    return 0.0
            return float(check_params(final_params))
        except Exception as exc:
            logger.warning(f"Level 1 scoring failed: {exc}")
            return 0.0

    return score_fn


def _finite_number(value):
    """Accept numeric values/strings, rejecting booleans, NaN, and infinity."""
    if isinstance(value, bool | np.bool_):
        raise ValueError("Boolean values are not measurements")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Measurements must be finite")
    return number


def _image_array(values, shape):
    """Require a finite, complete image matching the configured resolution."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.size == 0 or array.shape != shape:
        raise ValueError(f"Expected image shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Image contains non-finite values")
    return array


def _friction_signal(image, shape):
    """Match Image_Analyzer: half the forward/backward Friction force difference.

    Preserve the reader's array orientation and units; do not take absolute values.
    """
    trace = _image_array(image["Forward"]["Friction force"], shape)
    retrace = _image_array(image["Backward"]["Friction force"], shape)
    return 0.5 * (trace - retrace)


# ----------------------------------------------------------------------------
# NID file and roughness helpers
# ----------------------------------------------------------------------------


def check_nid_file_exists(path):
    """Return 1 for a .nid file or a directory containing a .nid entry; else 0."""
    p = Path(path)
    if p.is_dir():
        # Path is a directory → check for any .nid files inside it
        for filepath in p.iterdir():
            if filepath.suffix == ".nid":
                logger.info(f"NID file found in directory: {filepath.name}")
                return 1
        logger.info(f"No .nid files found in the specified directory: {path}")
        return 0
    elif p.is_file():
        # Path is a file → check if it ends with .nid
        if p.suffix == ".nid":
            logger.info(f"NID file found: {p.name}")
            return 1
        else:
            logger.info(f"File exists but is not a .nid file: {p.name}")
            return 0
    else:
        logger.info(f"Path does not exist: {path}")
        return 0


def check_roughness(path):
    """
    Calculate RMS roughness from the latest .nid file in a directory
    or from a specified .nid file.

    Parameters:
        path (str): Path to a .nid file or a directory containing .nid files.

    Returns:
        float: RMS roughness value in the original data units (no conversion).
    """

    p = Path(path).resolve()  # Normalize path
    # Case 1: If path is a directory
    if p.is_dir():
        nid_files = list(p.glob("*.nid"))
        if not nid_files:
            raise FileNotFoundError(f"No .nid files found in directory: {p}")
        p = max(nid_files, key=lambda x: x.stat().st_mtime)

    # Case 2: If path is a file
    elif p.is_file():
        if p.suffix.lower() != ".nid":
            raise ValueError(f"The specified file is not a .nid file: {p}")
    else:
        # Path exists neither as a file nor a directory
        raise FileNotFoundError(f"The specified path is not valid: {p}")

    afm = read(str(p))  # At this point, `p` is a valid .nid file
    # Extract data and parameters
    data = afm.data
    _param = afm.param

    try:
        z = data["Image"]["Forward"]["Z-Axis"]
    except KeyError as e:
        raise KeyError(f"Missing key in AFM data: {e}") from e

    # Calculate RMS roughness
    z_mean = np.mean(z)
    rms_m = np.sqrt(np.mean((z - z_mean) ** 2))
    return float(rms_m)


# ----------------------------------------------------------------------------
# Live instrument settings
# ----------------------------------------------------------------------------


def get_params():
    """Read live settings; image dimensions and center coordinates are in nanometers."""
    if pythoncom:
        pythoncom.CoInitialize()
    _tip_guid_map = {
        "AN2_200": "{BD61D124-8350-4464-BFE4-1D8A156E4913}",
        "GLA_1": "{9E2BA28D-D843-41bf-8F62-05502B3EDB18}",
        "ACL_A": "{ABB75273-9543-431a-B681-C79B533DD9E6}",
        "ANSCM": "{40AEA787-942C-4d48-A389-DA81571F009C}",
        "SICON_A": "{F7A339A7-E29F-42a9-B7AA-D69C54363B76}",
        "XYNCHR": "{DD3DFE39-455E-40a1-801E-5D5B14CE4080}",
        "XYCONTR": "{12ADC816-C7B1-48f8-8B9E-5E579151CF50}",
        "ContAl_G": "{ED5A15E6-D3B0-4e64-8C50-809335D3E143}",
        "Multi75E_G": "{9593403B-A476-49a9-AA1F-9C3AEDAC0178}",
        "Multi75M_G": "{03D0715C-A520-4976-A5E2-4FC3078E3821}",
        "Multi75Al_G": "{443A2EDC-5C9C-4d60-843F-C6688BEA1DEA}",
        "Tap190Al_G": "{041FB80E-A179-4170-B5A4-A4EA1CC0A965}",
        "Tap150Al_G": "{E0F31C86-6BB8-496b-AC7E-F55C62EAB635}",
        "USC_F1_2_k7_3": "{19AEEE43-478F-4D16-BDB7-2EE256EAF4A4}",
        "USC_F0_3_k0_3": "{16FAEEB6-A887-46F6-A418-81A9EBBCB6C3}",
        "Dyn190Al": "{E9CE0D2D-F59E-4B44-A74F-B78C11575E9F}",
        "Stat0_2LAuD": "{A4A16538-CCD1-4BB1-B048-7B4F0F1B31BD}",
        "CONTR": "{89E92173-96FB-4ff9-94D8-42296D00D980}",
        "CONTSCR": "{5A687B3E-A75A-4b22-BD70-40ABB931F00E}",
        "CONTSCPt": "{1E95D12B-1DDB-4ace-B3AF-BE9C0D52D4FC}",
        "EFMR": "{986305AC-64B5-462e-B37E-6BD5AE447BE3}",
        "LFMR": "{C61FCA2C-6D5D-4105-9FDE-640D263E229F}",
        "MFMR": "{9499F49F-920F-47ec-80B6-883F683FF056}",
        "NCLR": "{62633FD4-0555-4cee-A8B4-B82F4CEFBB48}",
        "PPP_FMR": "{EBA2B75C-AA94-4451-AD36-1388CDABF5E8}",
        "pq_SCONT": "{8D28AE10-E1DD-49E0-8CC6-ABD7CEDF57B0}",
        "qp_CONT": "{0996E3AC-ABF6-4A22-B320-4BF749288156}",
        "qp_fast_CB1": "{3F3DD96B-F838-45B6-AA8C-B54F66ED9571}",
        "qp_fast_CB2": "{964280C3-70F7-4E22-AA60-734E672D7A02}",
        "qp_fast_CB3": "{CCF4B65D-F3D8-4A40-9108-53468ECBA1B4}",
    }
    spm = nanosurf.SPM()
    application = spm.application
    scan = application.Scan
    opmode = application.OperatingMode
    zcontrol = application.ZController
    head = application.ScanHead
    tip = head.CantileverByGUID
    # tip = None
    # # Reverse lookup: find key (tip name) for current GUID
    # for tip_name, guid in tip_guid_map.items():
    #     if guid.lower() == current_guid.lower():  # Case-insensitive match
    #         tip = tip_name

    params = {
        "pgain": zcontrol.PGain,
        "igain": zcontrol.IGain,
        "dgain": zcontrol.DGain,
        "image_height": scan.ImageHeight * 1e9,
        "image_width": scan.ImageWidth * 1e9,
        "times_per_line": scan.Scantime,
        "points_per_line": scan.Points,
        "lines_per_frame": scan.Lines,
        "rotation": scan.rotation,
        "centre_x": scan.CenterPosX * 1e9,
        "centre_y": scan.CenterPosY * 1e9,
        "setpoint": zcontrol.SetPoint,
        "tip": tip,
        "mode": opmode.OperatingMode,
    }
    # Nanosurf: dynamic/phase-contrast setpoints are percentages; contact
    # setpoints are volts only when SetPointForceUnitMode is DefUnitMode_V (0).
    if params["mode"] in (3, 4):
        params["setpoint_p"] = params["setpoint"]
    elif params["mode"] == 2 and zcontrol.SetPointForceUnitMode == 0:
        params["setpoint_v"] = params["setpoint"]
    del zcontrol
    del scan
    del application
    del spm
    gc.collect()
    if pythoncom:
        pythoncom.CoUninitialize()
    return params


def check_params(gt_params, rel_tol=1e-2, abs_tol=1e-3):
    """Compare expected settings with live values, allowing numeric tolerances."""
    current_params = get_params()

    for key in gt_params:
        logger.info(f"param {key}")
        if key not in current_params:
            logger.warning(f"Missing instrument parameter: {key}")
            return 0.0
        current_val = current_params[key]
        logger.info(f"current {current_val}")
        gt_val = gt_params[key]
        logger.info(f"gt val {gt_val}")

        # Counts and operating modes must match exactly, even if the SDK uses floats.
        if key in {"points_per_line", "lines_per_frame", "mode"}:
            if isinstance(current_val, bool) or current_val != gt_val:
                return 0.0
            continue

        # Use math.isclose for floats
        if isinstance(gt_val, float) or isinstance(current_val, float):
            try:
                current_val = _finite_number(current_val)
                gt_val = _finite_number(gt_val)
            except (ValueError, TypeError):
                return 0.0
            if not math.isclose(current_val, gt_val, rel_tol=rel_tol, abs_tol=abs_tol):
                logger.warning(f"Mismatch in {key}: {current_val} != {gt_val}")
                return 0.0
        else:
            if current_val != gt_val:
                logger.warning(f"Mismatch in {key}: {current_val} != {gt_val}")
                return 0.0

    return 1.0


def check_gain():
    """Return the live proportional, integral, and derivative gains."""
    spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreAFM()
    application = spm.application
    _scan = application.Scan
    _opmode = application.OperatingMode
    zcontrol = application.ZController
    _head = application.ScanHead
    return [zcontrol.PGain, zcontrol.IGain, zcontrol.DGain]


def check_image_size():
    """Return the live image height and width in nanometers."""
    # load application
    spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreAFM()
    application = spm.application
    # all variables
    scan = application.Scan
    _opmode = application.OperatingMode
    _zcontrol = application.ZController
    _head = application.ScanHead
    return [scan.ImageHeight * 1e9, scan.ImageWidth * 1e9]


def check_scan_mode():
    """Return whether the instrument is currently scanning."""
    spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreA FM()
    application = spm.application
    scan = application.Scan
    return scan.IsScanning


def check_tip():
    """Return the selected cantilever GUID."""
    spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreAFM()
    application = spm.application
    # all variables
    _scan = application.Scan
    _opmode = application.OperatingMode
    _zcontrol = application.ZController
    head = application.ScanHead
    return head.CantileverByGUID


# ----------------------------------------------------------------------------
# Scalar comparison helper
# ----------------------------------------------------------------------------


def check_scalar(gt, ag):
    """
    Check if 'ag' is within ±20% of 'gt'.

    Parameters:
        gt (float): Ground truth value
        ag (float): Agent-predicted or measured value

    Returns:
        bool: True if ag is within 20% of gt, False otherwise
    """
    tolerance = 0.20 * abs(gt)
    return abs(ag - gt) <= tolerance
