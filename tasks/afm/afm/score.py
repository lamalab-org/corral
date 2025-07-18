import gc
import math

import nanosurf
from loguru import logger
from NSFopen.read import read
from skimage.metrics import structural_similarity as ssim


def get_params():
    import pythoncom

    pythoncom.CoInitialize()
    tip_guid_map = {
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
    zcontrol = application.ZController
    head = application.ScanHead
    current_guid = head.CantileverByGUID
    tip = None
    # Reverse lookup: find key (tip name) for current GUID
    for tip_name, guid in tip_guid_map.items():
        if guid.lower() == current_guid.lower():  # Case-insensitive match
            tip = tip_name

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
        "setpoint": zcontrol.SetPoint,
        "tip": tip,
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
        current_val = current_params.get(key)
        gt_val = gt_params[key]

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


def check_image_quality(path):
    afm = read(path)
    data = afm.data
    im_file_fw = data["Image"]["Forward"]["Z-Axis"]
    im_file_bw = data["Image"]["Backward"]["Z-Axis"]
    similarity_index, diff = ssim(
        im_file_bw,
        im_file_fw,
        full=True,
        data_range=im_file_bw.max() - im_file_bw.min(),
    )
    if similarity_index >= 0.80:
        return 1.0
    return 0.0


# def binary_image_score(img1_path, img2_path, metric='ssim', threshold=0.95, resize_to=None):
#     """
#     Return 1 if images are similar enough, else 0.

#     Parameters:
#         img1_path (str): Path to first image.
#         img2_path (str): Path to second image.
#         metric (str): 'ssim' or 'mse'.
#         threshold (float): Threshold for similarity.
#         resize_to (tuple): Resize dimensions (height, width), if needed.

#     Returns:
#         int: 1 if similar, 0 if not.
#     """
#     def normalize_image(img):
#         img = img.astype(np.float32)
#         return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

#     img1 = imread(img1_path, as_gray=True)
#     img2 = imread(img2_path, as_gray=True)

#     if resize_to:
#         img1 = resize(img1, resize_to, anti_aliasing=True)
#         img2 = resize(img2, resize_to, anti_aliasing=True)

#     img1 = normalize_image(img1)
#     img2 = normalize_image(img2)

#     if metric == 'ssim':
#         score, _ = ssim(img1, img2, data_range=1.0, full=True)  # FIX: Added data_range=1.0
#         return int(score >= threshold)
#     elif metric == 'mse':
#         score = np.mean((img1 - img2) ** 2)
#         return int(score <= threshold)
#     else:
#         raise ValueError("Unsupported metric. Use 'ssim' or 'mse'.")

# # score = binary_image_score("Question_23_b_pure.png", "Question_23_b_pure.png", metric='ssim', threshold=0.95)
# # print(score)
