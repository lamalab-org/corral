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
from scipy.optimize import curve_fit
from skimage.metrics import structural_similarity as ssim

# ----------------------------------------------------------
# Safe pythoncom import (Windows only)
# ----------------------------------------------------------
if platform.system() == "Windows":
    import pythoncom
else:
    pythoncom = None


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

                    # Run user-defined check function
                    check_output = check_roughness(path, rms)

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


def auto_match_unit(rms_meters: float, llm_value: float) -> float:
    """
    Scale rms_meters to match the order-of-magnitude of llm_value.

    """
    if llm_value == 0 or rms_meters == 0:
        return rms_meters

    scale = llm_value / rms_meters
    exponent = int(np.round(np.log10(abs(scale))))

    # safety (prevents absurd scaling)
    exponent = max(-15, min(15, exponent))

    return rms_meters * (10**exponent)


def check_params_function(final_params):
    def score_fn(_result: str) -> float:
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


def get_params():
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
    if pythoncom:
        pythoncom.CoUninitialize()
    return params


def check_params(gt_params, rel_tol=1e-2, abs_tol=1e-3):
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
    _scan = application.Scan
    _opmode = application.OperatingMode
    zcontrol = application.ZController
    _head = application.ScanHead
    return [zcontrol.PGain, zcontrol.IGain, zcontrol.DGain]


def check_image_size():
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
    spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreA FM()
    application = spm.application
    scan = application.Scan
    return scan.IsScanning


def check_tip():
    spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreAFM()
    application = spm.application

    # all variables
    _scan = application.Scan
    _opmode = application.OperatingMode
    _zcontrol = application.ZController
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


def check_roughness(path, llm_rms):
    """
    Calculate RMS roughness from the latest .nid file in a directory
    or from a specified .nid file.

    Parameters:
        path (str): Path to a .nid file or a directory containing .nid files.

    Returns:
        float: RMS roughness value.
    """

    # Normalize path
    p = Path(path).resolve()

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

    # At this point, `p` is a valid .nid file
    afm = read(str(p))

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
    return float(auto_match_unit(rms_m, llm_rms))


def fit_power_law(area, roughness):
    area = np.array(area, dtype=float)
    roughness = np.array(roughness, dtype=float)

    def power_func(A, C, k):
        return C * A**k

    popt, _ = curve_fit(power_func, area, roughness, maxfev=10000)
    C, k = popt
    return C, k


def check_mathematical_eq(final_params, tolerance):
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
