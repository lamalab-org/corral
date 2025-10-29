#!/usr/bin/env python
import gc
import json
import math
import os
import re
from pathlib import Path

import nanosurf
import numpy as np
import pythoncom
from loguru import logger
from NSFopen.read import read
from scipy.optimize import curve_fit
from skimage.metrics import structural_similarity as ssim


def check_numerical(target: float, tolerance: float, final_params):
    """
    Returns a scoring function that checks if a result is within a percentage-based tolerance of the target.

    Args:
        target (float): The target value.
        tolerance (float): Fractional tolerance (e.g., 0.1 means ±10% of target).

    Returns:
        score_fn (function): A function that accepts a result and returns 1.0 if it's within tolerance, else 0.0.
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


def check_roughness_function(tolerance: float, final_params):
    """
    Parses a submission_format string from the LLM, handling minor formatting issues.
    Runs check_roughness(path) for each valid pair of (rms_roughness_n, path_n).
    Returns 1 if all RMS values pass the tolerance check AND check_params(final_params)==1, else 0.
    """

    def score_fn(result: str) -> float:
        try:
            # Fix common LLM formatting mistakes
            cleaned = result.replace(";", ",").strip()

            # Ensure JSON braces are balanced
            if not cleaned.startswith("{"):
                cleaned = "{" + cleaned
            if not cleaned.endswith("}"):
                cleaned = cleaned + "}"

            # Try parsing JSON
            data = json.loads(cleaned)

            # Detect all indices dynamically (e.g., 1, 2, 3, ...)
            indices = sorted(
                {
                    int(re.findall(r"\d+", key)[0])
                    for key in data
                    if key.startswith("rms_roughness_")
                }
            )

            all_passed = True  # Track if all checks pass

            for i in indices:
                rough_key = f"rms_roughness_{i}"
                path_key = f"path_{i}"

                if rough_key in data and path_key in data:
                    try:
                        rms = float(data[rough_key])
                    except ValueError:
                        logger.warning(f"⚠️ Invalid RMS value for entry {i}")
                        all_passed = False
                        continue

                    path = data[path_key]
                    logger.info(f"🔹 Entry {i}: RMS={rms}, Path={path}")

                    # Run user-defined check function
                    check_output = check_roughness(path)

                    abs_tol = tolerance * rms
                    logger.info(
                        f"Tolerance: ±{tolerance}, Check Output: {check_output}"
                    )

                    # Check if the measured value is within tolerance
                    if not (rms - abs_tol <= check_output <= rms + abs_tol):
                        logger.error(f"❌ Entry {i} failed tolerance check.")
                        all_passed = False

            # ✅ Additional condition: check_params(final_params) must be 1
            if check_params(final_params) != 1:
                logger.error("❌ check_params(final_params) != 1")
                logger.error(check_params(final_params))
                all_passed = False

            return 1 if all_passed else 0

        except json.JSONDecodeError:
            logger.error(
                "❌ Could not parse JSON — please check the submission format."
            )
            return 0
        except Exception as e:
            logger.warning(f"⚠️ Error: {e}")
            return 0

    return score_fn


def check_params_function(final_params):
    def score_fn(result: str) -> float:  # noqa: ARG001
        try:
            logger.info(f"Checking params with final_params: {final_params}")
            score = check_params(
                final_params
            )  # Assuming check_params is defined elsewhere
            logger.info(f"check params score: {score}")
            return float(score)  # Ensure the returned score is a float
        except Exception as e:
            logger.error(f"Error during scoring: {e}")
            return 0.0  # Return 0.0 in case of an error

    return score_fn  # Return the scoring function itself


def check_file_exists(final_params):
    def score_fn(result: str) -> float:
        try:
            logger.info(f"Checking params with final_params: {final_params}")
            score = check_params(
                final_params
            )  # Assuming check_params is defined elsewhere
            score2 = check_nid_file_exists(result)
            logger.info(f"check params score: {score}")
            return float(score * score2)  # Ensure the returned score is a float
        except Exception as e:
            logger.error(f"Error during scoring: {e}")
            return 0.0  # Return 0.0 in case of an error

    return score_fn  # Return the scoring function itself


def check_image_quality(tolerance, final_params):
    def score_fn(result):
        try:
            if isinstance(result, str):
                result = result.strip()
                afm = read(result)
                data = afm.data
                im_file_fw = data["Image"]["Forward"]["Z-Axis"]
                im_file_bw = data["Image"]["Backward"]["Z-Axis"]
                similarity_index, diff = ssim(
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


def check_indentation(target: str):
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


def check_nid_file_exists(path):
    if Path(path).is_dir():
        # path is a directory → check for any .nid files inside it
        for filename in Path(path).iterdir():
            if filename.suffix == ".nid":  # Use .suffix for Path objects
                logger.info(f"NID file found in directory: {filename}")
                return 1
        logger.info(f"No .nid files found in the specified directory: {path}")
        return 0
    elif Path(path).is_file():
        # Path is a file → check if it ends with .nid
        if path.endswith(".nid"):
            logger.info(f"NID file found: {Path(path).name}")
            return 1
        else:
            logger.info(f"File exists but is not a .nid file: {Path(path).name}")
            return 0
    else:
        logger.info(f"Path does not exist: {path}")
        return 0


def get_params():
    pythoncom.CoInitialize()
    spm = nanosurf.SPM()
    application = spm.application
    scan = application.Scan
    opmode = application.OperatingMode
    zcontrol = application.ZController
    head = application.ScanHead
    tip = head.CantileverByGUID
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
        "centre_x": scan.CenterPosX,
        "centre_y": scan.CenterPosY,
        "setpoint": zcontrol.SetPoint,
        "tip": tip,
        "mode": opmode.OperatingMode,
    }
    del zcontrol
    del scan
    del application
    del spm
    gc.collect()
    pythoncom.CoUninitialize()
    return params


def check_params(gt_params, rel_tol=1e-4, abs_tol=1e-9):
    current_params = get_params()

    for key in gt_params:
        logger.info(f"param {key}")
        current_val = current_params.get(key)
        logger.info(f"current {current_val}")
        gt_val = gt_params[key]
        logger.info(f"gt val {gt_val}")

        # Use math.isclose for floats
        if isinstance(gt_val, float) or isinstance(current_val, float):
            if not math.isclose(
                float(current_val), float(gt_val), rel_tol=rel_tol, abs_tol=abs_tol
            ):
                logger.warning(f"Mismatch in {key}: {current_val} != {gt_val}")
                return 0.0
        else:
            if current_val != gt_val:
                logger.warning(f"Mismatch in {key}: {current_val} != {gt_val}")
                return 0.0

    return 1.0


def check_gain():
    spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreAFM()
    application = spm.application
    zcontrol = application.ZController
    return [zcontrol.PGain, zcontrol.IGain, zcontrol.DGain]


def check_image_size():
    # load application
    spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreAFM()
    application = spm.application

    # all variables
    scan = application.Scan
    return [scan.ImageHeight * 1e9, scan.ImageWidth * 1e9]


def check_scan_mode():
    spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreA FM()
    application = spm.application
    scan = application.Scan
    return scan.IsScanning


def check_tip():
    spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreAFM()
    application = spm.application

    # all variables
    head = application.ScanHead
    return head.CantileverByGUID


def check_scalar(gt, ag):
    """
    Check if 'ag' is within ±10% of 'gt'.

    Parameters:
        gt (float): Ground truth value
        ag (float): Agent-predicted or measured value

    Returns:
        bool: True if ag is within 10% of gt, False otherwise
    """
    tolerance = 0.20 * abs(gt)
    return abs(ag - gt) <= tolerance


def check_roughness(path):
    """
    Calculate RMS roughness from the latest .nid file in a directory
    or from a specified .nid file.

    Parameters:
        path (str): Path to a .nid file or a directory containing .nid files.

    Returns:
        float: RMS roughness value.
    """

    # Normalize path
    path = Path(path).resolve()

    # Case 1: If path is a directory
    if path.is_dir():
        nid_files = list(path.glob("*.nid"))
        if not nid_files:
            raise FileNotFoundError(f"No .nid files found in directory: {path}")
        path = max(nid_files, key=os.path.getmtime)

    # Case 2: If path is a file
    elif path.is_file():
        if not path.lower().endswith(".nid"):
            raise ValueError(f"The specified file is not a .nid file: {path}")
    else:
        # Path exists neither as a file nor a directory
        raise FileNotFoundError(f"The specified path is not valid: {path}")

    # At this point, `path` is a valid .nid file
    afm = read(path)

    # Extract data and parameters
    data = afm.data

    try:
        z = data["Image"]["Forward"]["Z-Axis"]
    except KeyError as e:
        raise KeyError(f"Missing key in AFM data: {e}") from e

    # Calculate RMS roughness
    z_mean = np.mean(z)
    return np.sqrt(np.mean((z - z_mean) ** 2))


def fit_power_law(area, roughness):
    area = np.array(area, dtype=float)
    roughness = np.array(roughness, dtype=float)

    def power_func(A, C, k):
        return C * A**k

    popt, _ = curve_fit(power_func, area, roughness, maxfev=10000)
    C, k = popt
    return C, k


def check_equation(final_params, tolerance):
    def score_fn(result: str) -> float:
        try:
            data = json.loads(result)
            logger.info(f"Checking params with final_params: {final_params}")
            logger.info(f"Raw submission {result}")
            logger.info(f"Parsed data: {data['equation']}, {data['Rb']}, {data['A']}")
            # Fit power law
            C, k = fit_power_law(data["A"], data["Rb"])
            fitted_eq = f"Rb = {C:.4f} * A**{k:.4f}"
            logger.info(f"Fitted power law: {fitted_eq}")
            match = re.search(
                r"Rb\s*=\s*(?:([0-9.]+)\s*\*\s*)?A\s*\*\*\s*([0-9.]+)", data["equation"]
            )
            if match:
                C_orig = float(match.group(1) if match.group(1) else 1)
                k_orig = float(match.group(2))
                return (
                    1
                    if (abs(C - C_orig) <= tolerance and abs(k - k_orig) <= tolerance)
                    and (check_params(final_params) == 1)
                    else 0
                )
            else:
                raise ValueError("Equation format not recognized")
        except Exception as e:
            logger.error(f"Error during scoring: {e}")
            return 0.0  # Return 0.0 in case of an error

    return score_fn  # Return the scoring function itself
