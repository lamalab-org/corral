'''
AFM Tools Module
This module provides tools for operating an Atomic Force Microscope (AFM) using the Nanosurf API.
It includes functionalities for grain detection, scanning, document retrieval, image optimization, 
and code execution for AFM operations.   
'''

# import modal
# from modal import Image
from typing import Dict, Any
from corral.base import Tool, ToolArgument
# from corral.utils import MODAL_TOOL_REGISTRY, modal_tool, tool
from corral.utils import tool
from tool_utils import Document_Retriever
from aila_image_process import *
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import gc


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
        3. [FOLLOW_UP] Use the returned grain coordinates (and `image_path`) with the `scan_grain_area` tool to zoom in and scan a specific grain. This enables precise, localized AFM scanning on individual grain features. [/FOLLOW_UP]
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
            [BRIEF] Path to the microscopy image file. [/BRIEF]
            [DETAILED] File path pointing to the image to be processed. The image should represent a microstructural scan with visible grains to enable detection. [/DETAILED]
            [SYNTACTICAL] Format: A string ending in `.nid`. [/SYNTACTICAL]
            [EXAMPLES] `"sample_grains.nid"`, `"/scans/run5_scan_topo.nid"` [/EXAMPLES]

    Returns:
        list:
            [BRIEF] A list of indexed grain bounding boxes. [/BRIEF]
            [DETAILED] Each entry in the list is a tuple of the form `(index, x1, y1, x2, y2)`, where `index` is the grain number (starting from 1), and `(x1, y1)` and `(x2, y2)` represent the bottom-left and top-right coordinates of the bounding box in microns. This data can be used for grain-by-grain analysis or scan targeting. [/DETAILED]
            [EXAMPLES] `[(1, 2.1, 3.4, 4.2, 5.5), (2, 6.0, 7.2, 7.8, 8.9)]` [/EXAMPLES]

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

    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend for saving
    
    indexed_boxes, extents, Z_flat2, labeled = image_process(image_path)
    
    fig, ax = plt.subplots()
    clip = [np.percentile(Z_flat2, percent) for percent in [3, 91]]
    ax.imshow(Z_flat2, cmap='afmhot', origin='lower', extent=extents)

    # List to store (index, x1, y1, x2, y2)
    box_coords = []

    for index, x, y, w, h in indexed_boxes:
        
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)

        # Label box center
        center_x = x + w / 2
        center_y = y + h / 2

        base_fontsize = 5
        scale_factor = 0.3
        font_size = base_fontsize + scale_factor * min(w, h)

        ax.text(center_x, center_y, str(index),
                color='cyan', fontsize=font_size, ha='center', va='center')

        # Store (index, bottom-left, top-right)
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        box_coords.append((index, x1, y1, x2, y2))

    ax.set_xlabel(r'X [$\mu$m]')
    ax.set_ylabel(r'Y [$\mu$m]')
    ax.set_title("Grains with Bounding Boxes")

    # Save to file
    import pythoncom
    pythoncom.CoInitialize()
    import nanosurf
    spm = nanosurf.SPM()
    application = spm.application
    current_path = application.GetGalleryHistoryDirectoryPath
    output_path = os.path.join(current_path, "annotated.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"image annotated with bounding boxes saved at {current_path}.")
    plt.close(fig)
    del application
    del spm
    gc.collect()
    pythoncom.CoUninitialize()
    
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
        3. [FOLLOW_UP] The resulting `.nid` file (path returned by this tool) can be used for further image optimization, measurement, or grain-level material analysis. [/FOLLOW_UP]
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
            [BRIEF] Index of the grain to be scanned. [/BRIEF]
            [DETAILED] This should correspond to the `index` returned by `visualize_grain_boxes`. Indexing starts from 1 and follows the order of detection. [/DETAILED]
            [EXAMPLES] "1", "5", "12" [/EXAMPLES]

        image_path (str):
            [BRIEF] Path to the `.nid` image that contains the grain. [/BRIEF]
            [DETAILED] This should be the same image file used in `visualize_grain_boxes`, from which the grain indices were derived. It must be in `.nid` format. [/DETAILED]
            [EXAMPLES] "grains_overview.nid", "/data/surface1.nid" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] Path to the latest `.nid` file produced from the scan. [/BRIEF]
            [DETAILED] After the scan completes, this tool returns the full path to the `.nid` file corresponding to the localized grain scan. [/DETAILED]
            [EXAMPLES] `"/data/scans/local_grain_12.nid"` [/EXAMPLES]

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

    import pythoncom
    pythoncom.CoInitialize()
    import nanosurf
    import time
    import os

    # Use absolute path as key for consistency
    abs_image_path = os.path.abspath(image_path)
        
    # Query current scan parameters from the SPM
    spm = nanosurf.SPM()
    application = spm.application
    current_working_directory = application.GetGalleryHistoryDirectoryPath
    scan = application.Scan
    current_size = (scan.ImageWidth * 1e9, scan.ImageHeight * 1e9)
    current_center = (scan.CenterPosX * 1e9, scan.CenterPosY * 1e9)

    # Process the image and compute subscan parameters
    boxes, extents, Z_flat2, labeled = image_process(image_path)

    params = get_subscan_parameters(
        grain_id=grain_id,
        labeled_mask=labeled,
        extents=extents,
        current_center=current_center,
        current_size=current_size
    )

    # Update scan settings
    scan.ImageWidth = params["width"] * 1e-9
    scan.ImageHeight = params["height"] * 1e-9
    scan.CenterPosX = params["center_x"] * 1e-9
    scan.CenterPosY = params["center_y"] * 1e-9
    scan.StartFrameUp()


    # Wait while scanning is in progress
    while scan.IsScanning:
        print("Scanning in progress...")
        time.sleep(5)
        
    nid_files = glob.glob(os.path.join(current_working_directory, "*.nid"))
    if not nid_files:
        raise FileNotFoundError("No .nid files found after scan.")
    latest_nid = max(nid_files, key=os.path.getmtime)
    
    print(f"Scan complete. Latest saved .nid file: {latest_nid}")
    return latest_nid


@tool
def Document_Retrieval(query: str) -> str:
    """
    [BRIEF] This tool retrieves code snippets from a database that are specifically designed for operating an Atomic Force Microscope (AFM) machine. The retrieved code is intended to be used as a reference for controlling the AFM. [/BRIEF]

    [DETAILED] This tool searches a structured knowledge base containing Python code examples and routines used to operate Atomic Force Microscopes (AFMs). These include initialization commands, scanning procedures, configuration settings, and parameter adjustments. This helps users automate or modify AFM procedures quickly and reliably. [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - When looking for specific Python code to control an AFM (e.g., start a scan, set PID gains, configure scan modes).
        - When building a larger AFM automation script and need reference routines.
        - When troubleshooting or experimenting with control procedures and need examples. 
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Identify a control action you need code for, such as "set P gain" or "start scan." [/PREREQUISITE]
        2. [CURRENT] Use this tool to retrieve a relevant code snippet. Provide a clear, descriptive query. [/CURRENT]
        3. [FOLLOW_UP] Use the retrieved code directly or modify it. Then pass it to `Code_Executor(code=...)` to execute the AFM routine. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Receives a query string and performs keyword or semantic search on an indexed database of trusted AFM Python scripts.
        - Returns one or more relevant code snippets, typically containing object-level interactions with nanosurf Python APIs.
        - The database includes examples related to scan configuration, imaging, calibration, feedback loops, and motion control.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `Document_Retrieval(query="initialize AFM and load application")`,
        `Document_Retrieval(query="set PID gains")`,
        `Document_Retrieval(query="start scan in contact mode")`,
        `Document_Retrieval(query="capture an image")`
    ]
    [/SYNTACTICAL]

    Args:
        query (str):
            [BRIEF] Search string describing the desired AFM control routine. [/BRIEF]
            [DETAILED] This is a plain-language or command-style string that specifies what kind of control code is needed. The system matches it against documented and tested control snippets used in prior AFM workflows. [/DETAILED]
            [SYNTACTICAL] Format: String containing keywords or phrases (e.g., "initialize AFM", "set Z controller gains"). [/SYNTACTICAL]
            [EXAMPLES] "load application", "configure scan parameters", "set contact mode", "move head to (x=5, y=5)" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] The AFM control code snippet matching the query. [/BRIEF]
            [DETAILED] Returns a formatted string of Python code that matches the request. This code is typically suitable for direct use with the `Code_Executor` tool and interacts with the AFM system using nanosurf’s Python API. [/DETAILED]
            [EXAMPLES] 
                `"spm = nanosurf.SPM()\nscan = spm.application.Scan\nscan.StartFrameUp()"`,
                `"zcontrol.PGain = 120\nzcontrol.IGain = 7000\nzcontrol.DGain = 8"` 
            [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If no relevant code snippet can be found or if an internal error occurs during lookup. [/ERROR_WHEN]
            [ERROR_DETAILS] Might occur if the query is too vague, malformed, or if the database is temporarily unavailable. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try rephrasing the query with more precise terms or retrying after some time. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The database contains only AFM-related routines; generic Python queries are not supported.
    [/LIMITATIONS]
    """

    result = Document_Retriever.invoke(query)
    return result

def Image_optimizer(baseline: bool = False) -> str:
    """
    [BRIEF] Optimizes AFM image sharpness using a genetic algorithm by tuning PID control parameters. The image file corresponding to the optimal P,I,D parameters is the latest .nid file in the current working directory.[/BRIEF]

    [DETAILED] This tool automatically tunes the Proportional, Integral, and Derivative (PID) gains of the Z-controller to enhance Atomic Force Microscopy (AFM) image clarity. It uses a genetic algorithm to evaluate image sharpness and iteratively sample PID combinations. During each iteration (generation × population), a new image is acquired and saved. The image corresponding to the optimal PID parameters is the latest .nid file in the current working directory at the end of the run. The optional `baseline` argument enables baseline correction, which can improve results on samples with tilt or drift. [/DETAILED]

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
        - The last saved `.nid` file corresponds to the best-performing PID values and is the one to be used for downstream analysis.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `Image_optimizer()`,
        `Image_optimizer(baseline=True)`
    ]
    [/SYNTACTICAL]

    Args:
        baseline (bool):
            [BRIEF] Whether to enable baseline correction during optimization. [/BRIEF]
            [DETAILED] When set to `True`, baseline leveling is applied to each image before it is evaluated for sharpness. This improves results in cases where scanner drift or sample tilt introduces bias. [/DETAILED]
            [SYNTACTICAL] Format: Boolean. Default is `False`. [/SYNTACTICAL]
            [EXAMPLES] "True" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] Best PID settings and corresponding image error value. [/BRIEF]
            [DETAILED] Returns a summary of the optimal Proportional, Integral, and Derivative gains found via the genetic algorithm, along with the error metric. The image associated with these gains is saved as the most recent `.nid` file in the current working directory. [/DETAILED]
            [EXAMPLES] "Best solution found: [Pgain Igain Dgain] = [120 8000 12], [Error] = 0.014" [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the `baseline` argument is not a boolean. [/ERROR_WHEN]
            [ERROR_DETAILS] Prevents misconfiguration by enforcing a valid argument type. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Use `baseline=True` or `baseline=False` only. Avoid using strings or numbers. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - Uses a small population and limited generations to keep optimization fast, which may reduce solution quality.
        - The `.nid` file associated with the best PID settings is not explicitly labeled — users must check the most recently saved file in the working directory.
    [/LIMITATIONS]
    """

    import os
    import glob
    from pymoo.termination import get_termination
    from tool_utils import MyProblem
    from pymoo.optimize import minimize
    from pymoo.algorithms.soo.nonconvex.ga import GA

    if not isinstance(baseline, bool):
        raise ValueError(f"Invalid type for 'baseline': {type(baseline).__name__}. Expected a boolean.")

    try:

        import pythoncom
        pythoncom.CoInitialize()

        problem = MyProblem(baseline=baseline)
    
        termination = get_termination("n_gen", 3)
        algorithm = GA(pop_size=5, eliminate_duplicates=True)
    
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=True)

    finally:
        pass

    return "Best solution found: \n[Pgain Igain Dgain] = %s\n[Error] = %s" % (res.X, res.F)

@tool
def Code_Executor(code: str) -> int:
    """
    [BRIEF] Executes Python code for controlling Atomic Force Microscopes (AFM). [/BRIEF]
    
    [DETAILED] This tool executes raw Python code intended for operating an Atomic Force Microscope (AFM). It is primarily used to run or test code snippets retrieved using the `Document_Retriever` tool, typically for automation, configuration, or interaction with AFM hardware through an API. Because the execution directly controls AFM operations, the input code must be validated and handled cautiously. This tool ensures COM initialization for compatibility with AFM control libraries. [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - When you need to execute control commands or configuration routines on an AFM.
        - When you retrieve a code snippet from a lab protocol or document and need to test or apply it.
        - When automating AFM workflows through scripting. 
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Retrieve a control script using the `Document_Retriever`. [/PREREQUISITE]
        2. [CURRENT] Use `Code_Executor` to run the code directly on the AFM system. [/CURRENT]
        3. [FOLLOW_UP] Optionally analyse the resulting images or data generated after the code execution for downstream analysis. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Imports and initializes the COM interface via `pythoncom.CoInitialize()` to ensure compatibility with AFM hardware APIs.
        - Executes the given Python code string via `exec()`.
        - Returns success or error information based on execution outcome.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `Code_Executor(code=
                import nanosurf
                import time
                import os

                # Load application
                spm = nanosurf.SPM()
                application = spm.application

                # Get access to different components
                scan = application.Scan
                opmode = application.OperatingMode
                zcontrol = application.ZController
                head = application.ScanHead

                # Set Z controller parameters (PID gains)
                print("Setting PID gains...")
                zcontrol.PGain = 100
                zcontrol.IGain = 6000
                zcontrol.DGain = 10

                scan.StartFrameUp()

                del spm)`
    ]
    [/SYNTACTICAL]

    Args:
        code (str):
            [BRIEF] Python code string to be executed. [/BRIEF]
            [DETAILED] Raw Python instructions that will be executed in the current runtime context. This code typically contains AFM operation logic and must be syntactically correct and safe to run. [/DETAILED]
            [SYNTACTICAL] Format: Valid Python code as a string. [/SYNTACTICAL]
            [EXAMPLES] "import nanosurf\nspm=nanosurf.SPM()\napplication = spm.application\nscan=application.scan\nscan.StartFrameUp()\ndel spm\n" [/EXAMPLES]

    Returns:
        int:
            [BRIEF] Status or error information from code execution. [/BRIEF]
            [DETAILED] Returns a success message or the captured exception details if execution fails. The result may be logged or used to troubleshoot control scripts. [/DETAILED]
            [EXAMPLES] 
                `"code executed successfully"`  
                `"Error: NameError: name 'afm' is not defined"` [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If the provided Python code raises any runtime or syntax error. [/ERROR_WHEN]
            [ERROR_DETAILS] All exceptions are caught and returned as output. Common issues include undefined variables, invalid syntax, or hardware communication errors. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Review the error message, validate code syntax, confirm all libraries are imported, and ensure the hardware interface is properly initialized. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - Executes arbitrary Python code; misuse can cause hardware damage or safety risks.
        - Does not sandbox or secure code — all commands are executed with full runtime permissions.
        - Designed only for AFM-related Python control code; general-purpose code execution is not allowed.
    [/LIMITATIONS]
    """
    try:
        # Execute the code
        import pythoncom
        pythoncom.CoInitialize()
        exec(code)
        output='code executed successfully'
    except Exception as e:
        print("Error:", e) 
        output=e
    return output



@tool
def Image_Analyzer(path: str = None, dynamic_code: str = None, calculate_friction: bool = False, calculate_mean_roughness: bool = False, calculate_rms_roughness: bool = False) -> Dict[str, Any]:
    """
    [BRIEF] Analyzes AFM `.nid` image files from Nanosurf instruments and optionally computes surface metrics such as average friction, mean roughness, and RMS roughness. [/BRIEF]
    
    [DETAILED] This tool processes Atomic Force Microscopy (AFM) image files in `.nid` format captured using Nanosurf devices. It leverages the Nanosurf API to read high-resolution topographical and force-channel data. In addition to extracting image data, the tool can optionally compute key surface metrics:
    - Average friction (via forward/backward scan differences),
    - Mean roughness (Ra),
    - Root-mean-square roughness (Rq).
    It supports custom logic through the `dynamic_code` parameter, allowing flexible access to alternate scan channels or directions such as 'Deflection', 'Friction Force', or 'Backward' images. The tool is well-suited for automated AFM workflows in surface characterization and materials research. 
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - When analyzing `.nid` AFM image files from Nanosurf instruments.
        - When surface roughness or friction needs to be quantified from image data.
        - When specific imaging channels or scan directions must be dynamically selected.
        - When building workflows for materials analysis that require AFM image interpretation.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Pass the path of the .nid file of interest. [/PREREQUISITE]
        2. [CURRENT] Use this tool to load the file and analyze image data; optionally provide custom code to select imaging channels or enable metric calculations. [/CURRENT]
        3. [FOLLOW_UP] Use the returned roughness/friction values for materials characterization, or visualize/compare the extracted image data across experiments. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Reads `.nid` file using Nanosurf’s `NSFopen.read`.
        - Extracts `Z-Axis` image data by default from the `Forward` scan.
        - Optionally executes user-defined code via the `dynamic_code` argument to switch channels or apply processing.
        - Computes requested surface metrics (friction, Ra, Rq) using standard definitions.
        - Returns a structured result including raw image data, processing status, and computed values.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `Image_Analyzer(path="sample1.nid")`,
        `Image_Analyzer(path="scan_02.nid", calculate_friction=True)`,
        `Image_Analyzer(path="scan_03.nid", dynamic_code='image_data = data["Image"]["Backward"]["Deflection"]', calculate_mean_roughness=True)`,
        `Image_Analyzer(path="friction_test.nid", calculate_friction=True, calculate_rms_roughness=True)`,
        `Image_Analyzer(path="surface_scan.nid", dynamic_code='image_data = data["Image"]["Forward"]["Friction force"]')`
    ]
    [/SYNTACTICAL]

    Args:
        path (str):
            [BRIEF] Path to the `.nid` AFM image file. [/BRIEF]
            [DETAILED] Full file path to a Nanosurf `.nid` AFM image file. This file is read and processed to extract imaging data. [/DETAILED]
            [SYNTACTICAL] Format: "string ending in .nid" [/SYNTACTICAL]
            [EXAMPLES] Examples: "scan.nid", "/data/images/sample3.nid" [/EXAMPLES]

        dynamic_code (str):
            [BRIEF] Custom Python code to modify image data access logic. [/BRIEF]
            [DETAILED] Executed at runtime to override the default image channel. Useful for switching to different imaging modes such as Deflection or Friction Force, or for accessing Backward scan data. [/DETAILED]
            [SYNTACTICAL] Format: A string of valid Python code that assigns a value to `image_data`, based on the internal data dictionary structure. Must be compatible with the Nanosurf AFM `.nid` data schema. [/SYNTACTICAL]
            [EXAMPLES] 'image_data = data["Image"]["Backward"]["Deflection"]' [/EXAMPLES]

        calculate_friction (bool):
            [BRIEF] If True, computes the average friction force. [/BRIEF]
            [DETAILED] Computes the average of the difference between Forward and Backward friction force images. [/DETAILED]
            [SYNTACTICAL] Format: Boolean flag. Default is `False`. Set to `True` to trigger friction force computation. [/SYNTACTICAL]
            [EXAMPLES] "True" [/EXAMPLES]

        calculate_mean_roughness (bool):
            [BRIEF] If True, computes mean surface roughness (Ra). [/BRIEF]
            [DETAILED] Calculates the arithmetic average of absolute deviations from the mean surface height. [/DETAILED]
            [SYNTACTICAL] Format: Boolean flag. Default is `False`. Set to `True` to enable mean roughness computation. [/SYNTACTICAL]
            [EXAMPLES] "True" [/EXAMPLES]

        calculate_rms_roughness (bool):
            [BRIEF] If True, computes root-mean-square surface roughness (Rq). [/BRIEF]
            [DETAILED] Measures the standard deviation of the surface height distribution. Useful for quantifying surface texture. [/DETAILED]
            [SYNTACTICAL] Format: Boolean flag. Default is `False`. Set to `True` to enable RMS roughness computation. [/SYNTACTICAL]
            [EXAMPLES] "False" [/EXAMPLES]

    Returns:
        Dict[str, Any]:
            [BRIEF] Dictionary with image data, computation results, and status messages. [/BRIEF]
            [DETAILED] Contains raw image data extracted from the file, and optionally, values for average friction, mean roughness, and RMS roughness if requested. In case of error, includes a detailed message. [/DETAILED]
            [EXAMPLES] Example outputs: 
                - "{"status": "Success", "image_data": [...], "mean_roughness": 2.4e-9}"
                - "{"status": "Error", "message": "An error occurred: File not found"}" [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If there is any failure in reading or processing the `.nid` file or executing user code. [/ERROR_WHEN]
            [ERROR_DETAILS] General exception handler catches file errors, API issues, or invalid dynamic code. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Check that the file path is correct, the file is a valid `.nid` format, and the dynamic code is syntactically correct and contextually relevant. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - Only works with `.nid` files produced by Nanosurf instruments and supported by the `NSFopen` package.
        - Relies on user-provided `dynamic_code` being safe and correctly scoped.
    [/LIMITATIONS]
    """
    import os
    import glob
    from NSFopen.read import read
    import numpy as np

    try:
        # Read the file
        afm = read(path)
        
        # Extract data and parameters
        data = afm.data  # Raw data
        param = afm.param  # Parameters
        
        # Assuming 'Image', 'Forward', and 'Z-Axis' are keys in the data structure
        image_data = data['Image']['Forward']['Z-Axis']
        
        # If dynamic code is provided, execute it. image_data = data['Image']['Forward']['Z-Axis'] cange Forward to Backward if asked. Z-Axis to Deflection or Friction force if asked. 
        if dynamic_code:
            # Safely execute the dynamic code
            try:
                exec(dynamic_code)
                # After executing the dynamic code, `image_data` should be processed accordingly
                print("Dynamic code executed successfully.")
            except Exception as e:
                print(f"Error executing dynamic code: {e}")
                return {"status": "Error", "message": f"Error executing dynamic code: {str(e)}"}
        
        # Calculate Average Friction if requested
        if calculate_friction:
            friction = 0.5 * (data['Image']['Forward']['Friction force'] - data['Image']['Backward']['Friction force'])
            average_friction = np.mean(friction)
            print(f"Average Friction: {average_friction}")
        
        # Calculate Mean Roughness if requested
        if calculate_mean_roughness:
            z = data['Image']['Forward']['Z-Axis']
            z_mean = np.mean(z)
            absolute_differences = np.abs(z - z_mean)
            total_sum = np.sum(absolute_differences)
            M, N = z.shape
            mean_roughness = total_sum / (M * N)
            print(f"Mean Roughness: {mean_roughness}")
        
        # Calculate RMS Roughness if requested
        if calculate_rms_roughness:
            z = data['Image']['Forward']['Z-Axis']
            z_mean = np.mean(z)
            squared_differences = (z - z_mean) ** 2
            total_sum = np.sum(squared_differences)
            M, N = z.shape
            rms_roughness = np.sqrt(total_sum / (M * N))
            print(f"RMS Roughness: {rms_roughness}")
        
        # Return the image data along with status
        result = {"status": "Success", "message": f"Raw Image {path} processed successfully.", "image_data": image_data}
        
        # Include calculated metrics in the result if they were calculated
        if calculate_friction:
            result["average_friction"] = average_friction
        if calculate_mean_roughness:
            result["mean_roughness"] = mean_roughness
        if calculate_rms_roughness:
            result["rms_roughness"] = rms_roughness
        
        return result
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"status": "Error", "message": f"An error occurred: {str(e)}"}
