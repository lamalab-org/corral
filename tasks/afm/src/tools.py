"""
AFM Tools Module
This module provides tools for operating an Atomic Force Microscope (AFM) using the Nanosurf API.
It includes functionalities for grain detection, scanning, document retrieval, image optimization,
and code execution for AFM operations.
"""

from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from loguru import logger
from NSFopen.read import read
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from afm_contract import reference
from afm_runtime import runtime
from aila_image_process import *  # noqa: F403
from corral.core.tool import tool
from score import measure_image
from tool_utils import MyProblem


@tool
def visualize_grain_boxes(image_path: str) -> list:
    """
    [BRIEF] Detects and visualizes grains in a microscopy image by drawing bounding boxes and indexing each grain. [/BRIEF]

    [DETAILED] This tool analyzes an input microscopy image to detect individual grains (microstructural features), assigns each grain a unique index, and draws bounding boxes around them. It returns a list of grain bounding boxes and saves an annotated image ("annotated.png") in the current working directory for visual verification. The coordinates are based on micrometer-scale measurements, making this tool ideal for downstream scanning, segmentation, or grain-wise material analysis. [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - When identifying and isolating grains within a microscopy image for further processing.
        - When preparing for grain-wise scanning or measurement tasks.
        - When you need visual confirmation of grain detection and layout.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Provide a valid image file path that captures a topographical or microstructural view of a sample. [/PREREQUISITE]
        2. [CURRENT] Use this tool to detect and index each grain, draw bounding boxes, and extract coordinate data. [/CURRENT]
        3. [FOLLOW_UP] Use the returned grain coordinates (and `image_path`) with the `scan_grain_area` tool to zoom in and scan a specific grain. This enables precise, localized AFM scanning on individual grain features. In case if no bounding boxes are generated, image quality may be low and needs to be improved with optimal P,I,D gains using `Image_optimizer` tool[/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Uses `image_process(image_path)` to extract bounding box information from the image.
        - Each grain is given a unique index and surrounded by a rectangle (bounding box).
        - A matplotlib figure is generated where each grain is visualized in cyan with its index label.
        - The figure is saved as `"annotated.png"` in the current Nanosurf working directory, though this file is not used programmatically in later steps.
        - The final output is a list of bounding boxes with format: `(index, x1, y1, x2, y2)` where x1/y1 is bottom-left, x2/y2 is top-right (in microns).
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `visualize_grain_boxes(image_path="afm_scan.nid")`,
        `visualize_grain_boxes(image_path="/data/images/grain_map_12.nid")`
    ]
    [/SYNTACTICAL]

    Args:
        image_path (str):
            [ARGS_BRIEF] Path to the microscopy image file. [/ARGS_BRIEF]
            [ARGS_DETAILED] File path pointing to the image to be processed. The image should represent a microstructural scan with visible grains to enable detection. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: A string ending in `.nid`. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] `"sample_grains.nid"`, `"/scans/run5_scan_topo.nid"` [/ARGS_EXAMPLES]

    Returns:
        list:
            [RETURNS_BRIEF] A list of indexed grain bounding boxes. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] Each entry in the list is a tuple of the form `(index, x1, y1, x2, y2)`, where `index` is the grain number (starting from 1), and `(x1, y1)` and `(x2, y2)` represent the bottom-left and top-right coordinates of the bounding box in microns. This data can be used for grain-by-grain analysis or scan targeting. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] `[(1, 2.1, 3.4, 4.2, 5.5), (2, 6.0, 7.2, 7.8, 8.9)]` [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If the image file is missing, corrupted, or if grain detection fails. [/ERROR_WHEN]
            [ERROR_DETAILS] Could be caused by invalid file path, unsupported image format. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the image exists at the given path. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - Grain detection relies on the quality of the image; poor contrast or noise can reduce accuracy.
    [/LIMITATIONS]
    """

    plt.switch_backend("Agg")  # Use non-GUI backend for saving

    indexed_boxes, extents, Z_flat2, _labeled = image_process(image_path)  # noqa: F405

    fig, ax = plt.subplots()
    ax.imshow(Z_flat2, cmap="afmhot", origin="lower", extent=extents)

    # List to store (index, x1, y1, x2, y2)
    box_coords = []

    for index, x, y, w, h in indexed_boxes:
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor="cyan", facecolor="none"
        )
        ax.add_patch(rect)

        # Label box center
        center_x = x + w / 2
        center_y = y + h / 2

        base_fontsize = 5
        scale_factor = 0.3
        font_size = base_fontsize + scale_factor * min(w, h)

        ax.text(
            center_x,
            center_y,
            str(index),
            color="cyan",
            fontsize=font_size,
            ha="center",
            va="center",
        )

        # Store (index, bottom-left, top-right)
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        box_coords.append((index, x1, y1, x2, y2))

    ax.set_xlabel(r"X [$\mu$m]")
    ax.set_ylabel(r"Y [$\mu$m]")
    ax.set_title("Grains with Bounding Boxes")

    current_path = runtime().call("snapshot")["Current Working directory"]
    output_path = Path(current_path) / "annotated.png"
    try:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    logger.info(f"Image annotated with bounding boxes saved at {current_path}.")

    return box_coords


@tool
def scan_grain_area(grain_id: int, image_path: str) -> None:
    """
    [BRIEF] Scans the AFM image region corresponding to a specific grain, identified by its grain id. [/BRIEF]

    [DETAILED] This tool performs a targeted AFM scan on a selected grain from an image. The grain must be previously indexed using the `visualize_grain_boxes` tool. It adjusts the scan parameters to zoom into the area around the grain and initiates a new AFM scan. Upon completion, the resulting scan is saved, and the path to the latest `.nid` file is returned. [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - After detecting grains using `visualize_grain_boxes`.
        - When you want to perform a high-resolution scan of a specific grain region.
        - When preparing to analyze or manipulate a particular grain feature in isolation.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Run `visualize_grain_boxes` with a `.nid` image to obtain the indexed grain list. Choose a specific `grain_id` from the output. [/PREREQUISITE]
        2. [CURRENT] Use this tool to zoom into the bounding box of the selected grain and start an AFM scan on that localized area. [/CURRENT]
        3. [FOLLOW_UP] The resulting `.nid` file (path returned by this tool) can be used for further measurement, or grain-level material analysis. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Loads current scan configuration from the AFM.
        - Identifies the bounding box of the specified `grain_id`.
        - Adjusts the scan window (`width`, `height`, `center_x`, `center_y`) of the AFM to zoom into the selected grain area.
        - Starts a new AFM frame scan using these updated parameters.
        - Waits for the scan to finish and returns the most recently generated `.nid` file path. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `scan_grain_area(grain_id=2, image_path="annotated.nid")`,
        `scan_grain_area(grain_id=1, image_path="/data/images/topo_01.nid")`
    ]
    [/SYNTACTICAL]

    Args:
        grain_id (int):
            [ARGS_BRIEF] Index of the grain to be scanned. [/ARGS_BRIEF]
            [ARGS_DETAILED] This should correspond to the `index` returned by `visualize_grain_boxes`. Indexing starts from 1 and follows the order of detection. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: Positive integer. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "1", "5", "12" [/ARGS_EXAMPLES]

        image_path (str):
            [ARGS_BRIEF] Path to the `.nid` image that contains the grain. [/ARGS_BRIEF]
            [ARGS_DETAILED] This should be the same image file used in `visualize_grain_boxes`, from which the grain indices were derived. It must be in `.nid` format. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: A string ending in `.nid`. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "grains_overview.nid", "/data/surface1.nid" [/ARGS_EXAMPLES]

    Returns:
        str:
            [RETURNS_BRIEF] Path to the latest `.nid` file produced from the scan. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] After the scan completes, this tool returns the full path to the `.nid` file corresponding to the localized grain scan. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] `"/data/scans/local_grain_12.nid"` [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        FileNotFoundError:
            [ERROR_WHEN] If no `.nid` file is found after the scan completes. [/ERROR_WHEN]
            [ERROR_DETAILS] Could occur if the scan did not finish successfully or saving failed. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the AFM is connected and functioning correctly, and the image path is correct. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - Assumes the grain index exists in the image and that the image has not been altered since bounding box extraction.
        - Only one grain can be scanned per call.
        - Depends on consistent file naming/timestamps to determine the "latest" `.nid` file.
    [/LIMITATIONS]
    """

    snapshot = runtime().call("snapshot")
    # Runtime/task geometry is nm; the image helper consumes and returns µm.
    current_size = (snapshot["image_width"] / 1000, snapshot["image_height"] / 1000)
    current_center = (snapshot["centre_x"] / 1000, snapshot["centre_y"] / 1000)
    _boxes, extents, _Z_flat2, labeled = image_process(image_path)  # noqa: F405
    params = get_subscan_parameters(  # noqa: F405
        grain_id=grain_id,
        labeled_mask=labeled,
        extents=extents,
        current_center=current_center,
        current_size=current_size,
    )
    return runtime().call(
        "update_and_acquire",
        {
            "image_width": params["width"] * 1000,
            "image_height": params["height"] * 1000,
            "centre_x": params["center_x"] * 1000,
            "centre_y": params["center_y"] * 1000,
        },
    )


@tool
def Document_Retrieval(query: str) -> str:
    """Return the complete local managed-afm-v1 API reference.

    Use for connection, settings, single-frame acquisition, metrics and submission
    conventions. The same small versioned reference is returned for every question;
    this is not a search of the vendor manual. No network or API key is needed.

    Args:
        query: A nonempty AFM API question.

    Returns:
        The bundled reference for this managed instrument adapter.
    """
    return reference(query)


@tool
def Image_optimizer(baseline: bool = False) -> str:
    """
    [BRIEF] Optimizes AFM image sharpness using a genetic algorithm by tuning PID control parameters. The image file corresponding to the optimal P,I,D parameters is the latest .nid file in the current working directory.[/BRIEF]

    [DETAILED] This tool automatically tunes the Proportional, Integral, and Derivative (PID) gains of the Z-controller to enhance Atomic Force Microscopy (AFM) image clarity. It uses a genetic algorithm to evaluate image sharpness and iteratively sample PID combinations. During each iteration (generation x population), a new image is acquired and saved. The image corresponding to the optimal PID parameters is the latest .nid file in the current working directory at the end of the run. The optional `baseline` argument enables baseline correction, which can improve results on samples with tilt or drift. [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - When AFM images appear blurred, distorted, or exhibit tracking instability.
        - When PID control tuning is needed for optimal imaging performance.
        - When baseline artifacts may be affecting image quality.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Image has visual quality issues. [/PREREQUISITE]
        2. [CURRENT] Run this tool to start P,I,D tuning via genetic optimization. Optionally enable baseline correction. [/CURRENT]
        3. [FOLLOW_UP] Inspect the latest `.nid` file in the working directory — this file corresponds to the best image produced during optimization, and can be used for downstream analysis. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Defines a custom optimization problem which evaluates image quality based on PID settings.
        - For each (P, I, D) sample, captures and stores a new `.nid` image file in the working directory.
        - Uses the `pymoo` genetic algorithm library to explore combinations of gains and minimize error.
        - The latest saved `.nid` file corresponds to the best-performing PID values and is the one to be used for downstream analysis.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `Image_optimizer()`,
        `Image_optimizer(baseline=True)`
    ]
    [/SYNTACTICAL]

    Args:
        baseline (bool):
            [ARGS_BRIEF] Whether to enable baseline correction during optimization. [/ARGS_BRIEF]
            [ARGS_DETAILED] When set to `True`, baseline leveling is applied to each image before it is evaluated for sharpness. This improves results in cases where scanner drift or sample tilt introduces bias. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: Boolean. Default is `False`. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] "True" [/ARGS_EXAMPLES]

    Returns:
        str:
            [RETURNS_BRIEF] Best PID settings and corresponding image error value. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] Returns a summary of the optimal Proportional, Integral, and Derivative gains found via the genetic algorithm, along with the error metric. The image associated with these gains is saved as the most recent `.nid` file in the current working directory. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] "Best solution found: [Pgain Igain Dgain] = [120 8000 12], [Error] = 0.014" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the `baseline` argument is not a boolean. [/ERROR_WHEN]
            [ERROR_DETAILS] Prevents misconfiguration by enforcing a valid argument type. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Use `baseline=True` or `baseline=False` only. Avoid using strings or numbers. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - Uses a small population and limited generations to keep optimization fast, which may reduce solution quality.
        - The `.nid` file associated with the best PID settings is not explicitly labeled — must check the most recently saved file in the working directory.
    [/LIMITATIONS]
    """

    if not isinstance(baseline, bool):
        raise ValueError(
            f"Invalid type for 'baseline': {type(baseline).__name__}. Expected a boolean."
        )

    problem = MyProblem(baseline=baseline)
    termination = get_termination("n_gen", 1)
    algorithm = GA(pop_size=1, eliminate_duplicates=True)
    res = minimize(problem, algorithm, termination, seed=1, verbose=True)

    return f"Best solution found: \n[Pgain Igain Dgain] = {res.X}\n[Error] = {res.F}"


@tool
def Code_Executor(code: str) -> dict:
    """Execute Python on the configured real AFM through its managed connection.

    Import nanosurf and use nanosurf.SPM(). StartFrameUp/Down and acquire_frame()
    block until a new NID is saved and verified, then return its absolute path.
    Calls are serialized on one COM owner thread. Each call has a fresh Python
    namespace; define variables again. Do not probe application types, import raw
    COM, start continuous scans or use external instrument-control processes.
    Stop hardware commands on InstrumentUnavailable; operator recovery is required.
    ScriptError means cleanup/health succeeded and the code may be corrected.

    Args:
        code: Python source applying the task's settings or acquiring a frame.

    Returns:
        stdout, stderr, and observed parameters (geometry in nm, time in seconds).
        Preserve absolute image paths printed by your script for submission.
    """
    return runtime().call("execute", code)


@tool
def Image_Analyzer(
    path: str,
    dynamic_code: str = "",
    calculate_friction: bool = False,
    calculate_mean_roughness: bool = False,
    calculate_rms_roughness: bool = False,
    friction_absolute: bool = False,
    calculate_rms_friction: bool = False,
    direction: str = "Forward",
    channel: str = "Z-Axis",
) -> dict[str, Any]:
    """Read a Nanosurf NID and calculate the benchmark's raw-image metrics.

    Roughness uses unlevelled Forward/Z-Axis height in nm: Ra=mean(abs(h-mean(h))),
    Rq=sqrt(mean((h-mean(h))**2)). Friction uses f=(Forward-Backward)/2 in V from
    Friction force channels, without flipping the stored arrays. Average friction
    is mean(f), or mean(abs(f)) with friction_absolute. RMS friction is
    sqrt(mean(f**2)), without subtracting its mean.

    Args:
        path: Absolute raw NID path returned by the acquisition tool.
        dynamic_code: Legacy Python assigning image_data from data or afm. Changes
            only the returned raw array, never metric channels or definitions.
            Prefer direction/channel for selecting a raw array.
        calculate_friction: Return average_friction in V.
        calculate_mean_roughness: Return mean_roughness (Ra) in nm.
        calculate_rms_roughness: Return rms_roughness (Rq) in nm.
        friction_absolute: Return mean friction magnitude instead of signed mean.
        calculate_rms_friction: Return rms_friction in V.
        direction: Forward or Backward, for returned image_data only.
        channel: NID channel name for image_data only; default Z-Axis.

    Returns:
        Requested metrics and metric_units, plus image_data in its original file
        unit. Raw channel selection does not change metric calculations.
    """
    if direction not in ("Forward", "Backward"):
        raise ValueError("direction must be Forward or Backward")
    afm = read(path)
    data = afm.data
    image_data = data["Image"][direction][channel]
    if dynamic_code:
        namespace = {"afm": afm, "data": data, "image_data": image_data}
        exec(dynamic_code, globals(), namespace)
        image_data = namespace["image_data"]
    metrics = [
        metric
        for metric, requested in (
            ("average_friction", calculate_friction),
            ("mean_roughness", calculate_mean_roughness),
            ("rms_roughness", calculate_rms_roughness),
            ("rms_friction", calculate_rms_friction),
        )
        if requested
    ]
    measured = (
        measure_image(afm, metrics, friction_absolute=friction_absolute)
        if metrics
        else {}
    )
    return {
        "status": "Success",
        "message": f"Raw image {path} processed successfully.",
        "image_data": image_data,
        **measured,
        "metric_units": {m: "V" if "friction" in m else "nm" for m in measured},
    }


def get_params():
    """Read parameters on the same COM owner used for setup and acquisition."""
    return runtime().call("snapshot")
