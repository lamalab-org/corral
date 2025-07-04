import modal
from modal import Image
from typing import Dict, Any
from corral.base import Tool, ToolArgument
from corral.utils import MODAL_TOOL_REGISTRY, modal_tool, tool
from tool_utils import Document_Retriever
from aila_image_process import *
import matplotlib.pyplot as plt
import numpy as np

# app = modal.App("corral-test")

@tool
def visualize_grain_boxes(image_path: str=None) -> list:
    """
    [BRIEF]  
    Detects grains in an AFM image and overlays bounding boxes with indexed labels for each grain. Returns the box coordinates for later use.  
    [/BRIEF]

    [DETAILED]  
    This tool performs image analysis on an AFM `.nid` image file to detect grain-like regions. Each detected region is enclosed in a bounding box, and a unique index (starting from 1) is assigned to each. The tool renders these labeled boxes on top of the topographic image and saves the visualization as a `.png` file.  
    This helps in visually identifying grain locations and mapping them to their respective indices, which are required by other tools like `scan_grain_area`.  
    Returns a list of box coordinates for programmatic access to the regions.  
    If no path is given, the latest `.nid` file in the current directory is used. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use after scanning a surface to detect and visualize grains.
    - Use to prepare for grain-specific scanning by indexing each grain.
    - Use if you want a saved image with grain boundaries annotated.
    - Do not use if no valid `.nid` image is available in the working directory.  
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Accepts a path to a `.nid` image or uses the most recent `.nid` file in the directory.
    - Processes the image to detect grains using an internal image analysis function.
    - Creates bounding boxes around detected grains and labels each with an index.
    - Saves the visualized result as a `.png` image for reference.
    - Returns a list of box coordinates for use in downstream tools.  
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    - visualize_grain_boxes()  
    - visualize_grain_boxes(image_path="sample_surface.nid")  
    [/SYNTACTICAL]
    
    Args:
        image_path (str, optional):  
            [BRIEF] Path to the input AFM image file (.nid format). Defaults to latest file. [/BRIEF]  
            [DETAILED]  
            The path to the `.nid` file to be processed. If not provided, the tool automatically selects the most recently modified `.nid` file in the current working directory.  
            This file should represent a previously scanned AFM image.  
            [/DETAILED]  
            [SYNTACTIC] Format: String path ending in `.nid` or None [/SYNTACTIC]  
            [EXAMPLES]  
            - `"test_scan_04.nid"`  
            - `"scans/grain_surface.nid"`  
            - `None` (auto-selects latest file)  
            [/EXAMPLES]

    Returns:
        list:  
            [BRIEF] A list of tuples representing bounding boxes and their associated grain indices. [/BRIEF]  
            [DETAILED]  
            Each element in the list is a tuple of the form `(index, x1, y1, x2, y2)`, where:  
            - `index`: Unique grain ID (starts from 1).  
            - `(x1, y1)`: Bottom-left corner of the bounding box.  
            - `(x2, y2)`: Top-right corner of the bounding box.  
            This output can be passed into tools like `scan_grain_area` to enable targeted scanning of grains.  
            [/DETAILED]  
            [EXAMPLES]  
            - `[(1, 4.2, 3.1, 8.5, 6.4), (2, 9.0, 2.0, 12.0, 5.5)]`  
            - `[(index, x_min, y_min, x_max, y_max), ...]`  
            [/EXAMPLES]

    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend for saving
    
    if image_path is None:
        # Search for the latest .nid file
        nid_files = glob.glob("*.nid")
        if not nid_files:
            raise FileNotFoundError("No .nid image files found in the current directory.")
        image_path = max(nid_files, key=os.path.getmtime)

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
    output_path = os.path.splitext(image_path)[0] + "_annotated.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return box_coords

_image_scan_cache = {}


@tool
def scan_grain_area(grain_id: int, image_path: str = None) -> None:
    """
    [BRIEF]  
    Scans a specified grain region from an AFM image by adjusting scan parameters based on grain position.  
    Reuses cached scan parameters when available and dynamically processes image data to define sub-scan boundaries.  
    [/BRIEF]

    [DETAILED]  
    This tool automates localized AFM scanning for a specific grain area identified in a .nid image.  
    It first checks for existing cached scan parameters (scan size and center position) based on the image path to avoid redundant queries to the SPM.  
    If the grain is being scanned for the first time, it reads current AFM scan parameters and uses them to calculate a sub-region scan centered around the selected grain.  
    The tool modifies the scan width, height, and center position accordingly and triggers a new scan. Once scanning is completed, it returns the filename of the newly saved .nid file.
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use to trigger a high-resolution sub-scan focused on a specific grain/region in a previously scanned AFM image.
    - Use if image processing has labeled grains and you want to scan grain #i.
    - Use in automated grain-wise scanning workflows.
    - Do not use if no image with grain data is available or image_path is incorrect.  
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Accepts a grain ID and optional image path.
    - If no image path is given, selects the latest .nid file in the working directory.
    - If scan parameters for the image are cached, reuses them. Otherwise, queries current scan size and center from the AFM and caches them.
    - Uses image processing to identify the labeled grain region and compute sub-scan parameters.
    - Updates the AFM's scan dimensions and position, then initiates the scan.
    - Waits until scanning completes, then returns the name of the saved .nid file.  
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    - scan_grain_area(grain_id=3)
    - scan_grain_area(grain_id=1, image_path="grains.nid")
    [/SYNTACTICAL]

    Args:
        grain_id (int):  
            [BRIEF] ID of the grain to scan (index of the labeled bounding box). [/BRIEF]  
            [DETAILED]  
            This is the index of the target grain as computed by the image processing function.  
            Indexing starts from 1 (not 0). The index maps to labeled grain regions identified by connected components or bounding boxes.  
            [/DETAILED]  
            [SYNTACTIC] Format: Integer ≥ 1 [/SYNTACTIC]  
            [EXAMPLES] 1 (first grain), 2 (second grain), 10 (tenth grain) [/EXAMPLES]

        image_path (str, optional):  
            [BRIEF] Path to the image (.nid) containing labeled grains. Defaults to latest file. [/BRIEF]  
            [DETAILED]  
            If not provided, the tool will automatically detect and use the most recently modified `.nid` file in the current directory.  
            This file must contain grain label data compatible with the image processing pipeline.  
            [/DETAILED]  
            [SYNTACTIC] Format: String path ending in `.nid` or None [/SYNTACTIC]  
            [EXAMPLES]  
            - `"sample_scan_01.nid"`  
            - `"images/test_image.nid"`  
            - `None` (uses most recent file)  
            [/EXAMPLES]

    Returns:
        str:  
            [BRIEF] Path to the newly saved .nid file after the sub-scan completes. [/BRIEF]  
            [DETAILED]  
            The function returns the full path to the latest .nid file created as a result of scanning the selected grain area.  
            This file contains the new AFM scan centered around the target grain with updated size and position parameters.  
            This is useful for further post-processing or visualization.  
            [/DETAILED]  
            [EXAMPLES]  
            - `"grain_subscan_03_20250704.nid"`  
            - `"C:/scans/afm/latest_grain_7.nid"`  
            [/EXAMPLES]

    """

    import pythoncom
    pythoncom.CoInitialize()
    import nanosurf
    import time
    import os
    if image_path is None:
        nid_files = glob.glob("*.nid")
        if not nid_files:
            raise FileNotFoundError("No .nid files found in the current directory.")
        image_path = max(nid_files, key=os.path.getmtime)
        print(f"No image_path provided. Using latest .nid file: {image_path}")

    # Use absolute path as key for consistency
    abs_image_path = os.path.abspath(image_path)

    # Check if parameters are already cached for this image
    if abs_image_path in _image_scan_cache:
        current_center, current_size = _image_scan_cache[abs_image_path]
        print(f"Using cached scan parameters for {abs_image_path}")
    else:
        
        # Query current scan parameters from the SPM
        spm = nanosurf.SPM()
        application = spm.application
        scan = application.Scan

        current_size = (scan.ImageWidth * 1e6, scan.ImageHeight * 1e6)
        current_center = (scan.CenterPosX * 1e6, scan.CenterPosY * 1e6)

        # Cache the values for future use
        _image_scan_cache[abs_image_path] = (current_center, current_size)
        print(f"Caching scan parameters for {abs_image_path}")

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
    scan.ImageWidth = params["width"] * 1e-6
    scan.ImageHeight = params["height"] * 1e-6
    scan.CenterPosX = params["center_x"] * 1e-6
    scan.CenterPosY = params["center_y"] * 1e-6
    scan.StartFrameUp()

    # Wait while scanning is in progress
    while scan.IsScanning:
        print("Scanning in progress...")
        time.sleep(5)
        
    nid_files = glob.glob("*.nid")
    if not nid_files:
        raise FileNotFoundError("No .nid files found after scan.")
    latest_nid = max(nid_files, key=os.path.getmtime)
    
    print(f"Scan complete. Latest saved .nid file: {latest_nid}")
    return latest_nid


@tool
def Document_Retrieval(query: str) -> str:
    """
    [BRIEF] Retrieves AFM-specific Python code snippets from a codebase or database based on user queries. [/BRIEF]

    [DETAILED]  
    This tool serves as a search interface for accessing a curated collection of Python code snippets used to operate an Atomic Force Microscope (AFM).  
    By inputting a natural language or keyword-based query, users can retrieve predefined and tested code segments that automate or control different AFM operations, such as PID tuning, image acquisition, friction analysis, scan setup, etc.  
    The output is intended for immediate inspection, editing, or execution (typically passed to the `Code_Executor` tool).  
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need example Python code for controlling or automating AFM tasks.
    - Best suited for retrieving scripts to use with the `Code_Executor` tool.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - 1: Accepts a user-defined query string describing the desired AFM operation.
    - 2: Passes the query to a backend document retrieval system (`Document_Retriever.invoke()`).
    - 3: Searches a predefined collection of AFM control code snippets using keyword and semantic matching.
    - 4: Returns the best-matching snippet as a string for further processing or execution.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    - Example 1: Document_Retrieval("code to capture image.")
    - Example 2: Document_Retrieval("code to start AFM scan")
    - Example 3: Document_Retrieval("set image size")
    [/SYNTACTICAL]

    Args:
    query (str):  
        [BRIEF] The keyword or natural language phrase describing the AFM code needed. [/BRIEF]  
        [DETAILED]  
        This is a string-based search query used to match and retrieve relevant code snippets for operating the AFM.  
        It may contain action words (e.g. "scan", "capture image") or parameter-based requests (e.g., "set PID to 0.1").  
        The quality of results depends on the clarity and specificity of the query.  
        [/DETAILED]  
        [SYNTACTIC] Format: String with natural language or keywords. [/SYNTACTIC]  
        [EXAMPLES]  
        - `"start imaging"`  
        - `"beging scan"`  
        [/EXAMPLES]

    Returns:
    str:  
        [BRIEF] A Python code snippet relevant to the query. [/BRIEF]  
        [DETAILED]  
        The return value is a string containing a valid and functional Python code snippet that matches the user's query.  
        This code can typically be passed directly to an execution tool (e.g., `Code_Executor`) or used as a reference for manual modification.  
        [/DETAILED]  
        [EXAMPLES]  
        - `"afm.set_pid_gains(p=0.2, i=0.05, d=0.01)"`  
        - `"afm.start_scan(mode='contact', resolution=512)"`  
        [/EXAMPLES]

    """
    result = Document_Retriever.invoke(query)
    return result


@tool
def Image_optimizer(baseline: bool) -> str:
    """
    [BRIEF] Optimizes P, I, D gains using a genetic algorithm to enhance AFM image clarity based on optional baseline correction. [/BRIEF]

    [DETAILED]  
    This tool performs parameter optimization of the Proportional (P), Integral (I), and Derivative (D) gains to improve Atomic Force Microscopy (AFM) image quality.  
    Using the `pymoo` genetic algorithm (`GA`), it searches for optimal control gains that minimize image blurring or noise artifacts. The user can toggle baseline correction behavior via the `baseline` flag.  
    It automatically selects the most recent AFM image from a specified directory and uses a custom problem definition (`MyProblem`) to guide the optimization process.  
    The result is a set of gain values that produce the clearest image possible under current conditions.  
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when AFM image output appears blurry or degraded in quality.
    - Best suited for experiments requiring real-time gain tuning for AFM stability or feedback response.
    - Avoid if PID tuning is already optimized or set or baseline correction is not applicable.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Changes working directory to a predefined image storage path.
    - Loads the latest AFM image based on creation time.
    - Initializes and runs a genetic algorithm (pymoo GA) with custom objective function defined in `MyProblem`.
    - Returns the optimal P/I/D gain values and associated error.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    - Example 1: Image_optimizer(baseline=True)
    - Example 2: Image_optimizer(baseline=False)
    [/SYNTACTICAL]

    Args:
    baseline (bool):  
        [BRIEF] Whether to apply baseline correction. [/BRIEF]  
        [DETAILED]  
        If `True`, includes baseline correction in the PID optimization process.  
        This can help correct drift or tilt in AFM surface data.  
        If `False`, the optimization is performed without correcting for baseline effects.  
        [/DETAILED]  
        [SYNTACTIC] Format: Boolean — either `True` or `False`. [/SYNTACTIC]  
        [EXAMPLES] Examples: `True` (enable correction), `False` (skip correction) [/EXAMPLES]

        
    Returns:
        str:  
            [BRIEF] A string describing the optimal PID gains and error metric. [/BRIEF]  
            [DETAILED]  
            After optimization, the function returns a formatted string containing the best-performing P, I, and D gains (`res.X`) and the associated error value (`res.F`).  
            This output can be used to update the AFM control system or log tuning outcomes.  
            [/DETAILED]  
            [EXAMPLES]  
            - `"Best solution found: \n[Pgain Igain Dgain] = [0.5 0.1 0.05]\n[Error] = 0.0032"`  
            - `"Best solution found: \n[Pgain Igain Dgain] = [0.3 0.2 0.1]\n[Error] = 0.0019"`  
            [/EXAMPLES]

    """
    import os
    import glob
    from pymoo.termination import get_termination
    from tools_utils import MyProblem
    from pymoo.optimize import minimize
    from pymoo.algorithms.soo.nonconvex.ga import GA

    original_path = os.getcwd()
    # new_path = "./test_directory"
    new_path = "./afm_images/tool_calling/claude_37"

    try:
        os.chdir(new_path)
        print(f"Current working directory: {os.getcwd()}")
        
        # Your code that needs to be executed in the new directory goes here
        import pythoncom
        pythoncom.CoInitialize()
    
        list_of_files = glob.glob(new_path+'/*') 
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
    
    
        problem = MyProblem(baseline=baseline)
    
        termination = get_termination("n_gen", 5)
        algorithm = GA(pop_size=3, eliminate_duplicates=True)
    
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=True)

    finally:
        # Restore the original working directory
        os.chdir(original_path)
        print(f"Returned to original working directory: {os.getcwd()}")

    return "Best solution found: \n[Pgain Igain Dgain] = %s\n[Error] = %s" % (res.X, res.F)

@tool
def Code_Executor(code: str) -> int:
    """

    [BRIEF]
    Executes Python code intended to operate an Atomic Force Microscope (AFM), ensuring compatibility via pythoncom initialization. AFM is sensitive, so handle with care.
    [/BRIEF]

    [DETAILED]  
    This tool securely executes Python scripts related to AFM operation. It is optimized to run hardware-interfacing scripts—especially those using COM-based APIs—by initializing the COM environment via `pythoncom.CoInitialize()`.  
    It is designed to work seamlessly with output from tools like `Document_Retriever`, which provide AFM operation code. Any runtime errors during execution are captured and returned to aid debugging.  
    This is a powerful tool and should only be used when the code is trusted and verified for safe interaction with AFM hardware.  
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when executing AFM operation scripts retrieved from tools such as `Document_Retriever`.
    - Best suited for scripts that control Nanosurf or COM-based AFM instruments.
    - Avoid when input code has not been validated or may have side effects unrelated to AFM hardware.
    - Recommended for safe, automated operation of AFM instruments in lab automation workflows.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Initializes the COM interface using `pythoncom.CoInitialize()` to prepare for hardware API interaction.
    - Executes the user-provided Python code using the built-in `exec()` function.
    - Any exceptions are caught, printed, and returned to the user for diagnostics.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    - Example 1:
    Code_Executor("import time\nprint('Moving probe...')\ntime.sleep(1)")
    - Example 2:
    Code_Executor('from nanosurf import afm\nafm.connect()')
    [/SYNTACTICAL]

    Args:
        code (str): 
            [BRIEF] Python script string to work with AFM. [/BRIEF]  
            [DETAILED] A string containing valid Python code, typically for AFM hardware control.  
            The code may use libraries such as `nanosurf`, and must assume COM support via `pythoncom`. 
            [/DETAILED]  
            [SYNTACTIC] Format: string. [/SYNTACTIC]  
            [EXAMPLES]  
            - `"from nanosurf import afm\nafmsystem = afm.connect()\nafmsystem.scan_start()"`  
            - `"print('AFM initialized')"`  
            [/EXAMPLES]

    
    Returns:
        str: 
            [BRIEF] Execution status or error message. [/BRIEF]  
            [DETAILED]  
            If the script executes without errors, a success message string is returned.  
            If an exception occurs during execution, the exception message is returned instead.  
            Useful for debugging scripts that interact with AFM hardware.  
            [/DETAILED]  
            [EXAMPLES]  
            - `"code executed successfully"`  
            - `"NameError: name 'afm' is not defined"`  
            [/EXAMPLES]

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
def Image_Analyzer(path: str = None, filename: str = None, dynamic_code: str = None, calculate_friction: bool = False, calculate_mean_roughness: bool = False, calculate_rms_roughness: bool = False) -> Dict[str, Any]:
    """
    [BRIEF]
    Extracts, processes, and analyzes surface topography data from AFM .nid image files captured using Nanosurf instruments, and optionally computes surface roughness metrics including average friction, mean roughness (Ra), and RMS roughness (Rq).
    [/BRIEF]

    [DETAILED]
    This tool is designed to extract and analyze surface topography data from Atomic Force Microscopy (AFM) .nid image files generated by Nanosurf instruments. It utilizes the Nanosurf API to read image data and optionally compute quantitative surface characteristics. Depending on the parameters specified, it can calculate:
    1) Average Friction from forward and backward scan signals
    2) Mean Roughness (Ra) as the average absolute deviation from the mean surface height
    3) RMS Roughness (Rq) as the root mean square of surface height deviations
    Additionally, the dynamic_code parameter allows advanced users to inject custom Python logic — such as selecting specific image channels (e.g., Z-Axis, Deflection, Friction Force), scan directions (e.g., Forward, Backward), or performing tasks like extracting, plotting, and saving image data using libraries like Matplotlib — making it suitable for a wide range of AFM imaging scenarios.
    [/DETAILED]
    
    [PROCEDURAL] 
    When to use this tool:
    -Use when you need to extract and analyze surface topography data from AFM .nid files generated by Nanosurf instruments.
    -Best suited for workflows that involve surface morphology characterization, roughness analysis, or quantitative friction studies.
    -Recommended for use in automated pipelines where image selection, processing, and custom analysis (e.g., dynamic plotting or filtering) are needed.    
    [/PROCEDURAL]

    [CONTEXTUAL]
    How this tool works:
    - Determines the file to load using the path and filename parameters. 
    - Uses the Nanosurf Python API (NSFopen.read) to parse the .nid file and extract structured imaging data.
    - Selects the default data channel (Z-Axis) from the Forward scan direction.
    - Executes optional computations like average friction, mean roughness (Ra), and RMS roughness (Rq) based on flags.
    - Optional Python code provided via dynamic_code. This makes it suitable for exploratory data analysis or research workflows, like plotting topological data .etc.
    [/CONTEXTUAL]


    [SYNTACTICAL]
    - Example 1: Image_Analyzer(path = "./nid_files/", filename="scan1.nid", calculate_mean_roughness=True)
    - Example 2: Image_Analyzer(path = "./nid_files/", calculate_friction=True, calculate_rms_roughness=True)
    - Example 3 : Image_Analyzer(
                    path="/data/afm/",
                    dynamic_code="image_data = data['Image']['Backward']['Deflection']",
                    calculate_friction=True,
                    calculate_rms_roughness=True
                )
    [/SYNTACTICAL]

    Args:
        path (str): 
            [BRIEF] Directory path which contains the nid file. Defaults to current working directory. [/BRIEF]  
            [DETAILED] If no `filename` is specified, the tool will use this path to locate and analyze the latest `.nid` file based on creation time. Supports automated workflows or batch processing where the newest scan needs to be processed. [/DETAILED]  
            [SYNTACTIC] Format: string representing a valid absolute or relative directory path. [/SYNTACTIC]  
            [EXAMPLES] Examples: "/data/afm_scans", ".", "../scans" [/EXAMPLES]  

        filename (str):  
            [BRIEF] Specific AFM `.nid` file to analyze. Overrides directory search. [/BRIEF]  
            [DETAILED] When provided, the tool directly loads this file from the specified `path` (or current directory if `path` is None). Useful for targeting specific files rather than using the latest in a folder. [/DETAILED]  
            [SYNTACTIC] Format: string with filename, must end in ".nid" [/SYNTACTIC]  
            [EXAMPLES] Examples: "test_sample.nid", "scan_001.nid" [/EXAMPLES]  

        dynamic_code (str):  
            [BRIEF] Inline Python code to dynamically process image data. Optional. [/BRIEF]  
            [DETAILED] Enables advanced users to inject custom logic for image transformation or selection, such as switching scan directions, using different channels, plotting, or saving results. Executed in the context where `data` and `image_data` are accessible. [/DETAILED]  
            [SYNTACTIC] Format: string containing valid Python code. Multiline supported via triple quotes. [/SYNTACTIC]  
            [EXAMPLES] Examples:  
                `"image_data = data['Image']['Backward']"`  
                `\"\"\"import matplotlib.pyplot as plt\nplt.imshow(image_data)\nplt.savefig('out.png')\"\"\"`  
            [/EXAMPLES]  

        calculate_friction (bool):  
            [BRIEF] Whether to compute average friction using forward and backward channels. [/BRIEF]  
            [DETAILED] Calculates 0.5 × (forward friction − backward friction) averaged across the full image. Useful for tribological analysis. [/DETAILED]  
            [SYNTACTIC] Format: Boolean flag (True/False) [/SYNTACTIC]  
            [EXAMPLES] Examples: True, False [/EXAMPLES]  

        calculate_mean_roughness (bool):  
            [BRIEF] Whether to compute mean roughness (Ra) from Z-Axis data. [/BRIEF]  
            [DETAILED] Computes the average absolute deviation from the mean surface height. A standard metric for surface quality. [/DETAILED]  
            [SYNTACTIC] Format: Boolean flag (True/False) [/SYNTACTIC]  
            [EXAMPLES] Examples: True (include Ra), False (omit) [/EXAMPLES]  

        calculate_rms_roughness (bool):  
            [BRIEF] Whether to compute RMS roughness (Rq) from Z-Axis data. [/BRIEF]  
            [DETAILED] Calculates the square root of the mean squared deviation from the mean height, providing a weighted measure of surface variation. [/DETAILED]  
            [SYNTACTIC] Format: Boolean flag (True/False) [/SYNTACTIC]  
            [EXAMPLES] Examples: True (include Rq), False (omit) [/EXAMPLES]

    Returns:
        dict: 
            [BRIEF] A dictionary containing processing status, image data, and optional surface metrics. [/BRIEF]  
            [DETAILED]  
            The returned dictionary always includes a `"status"` key (`"Success"` or `"Error"`) and a `"message"` summarizing the outcome.  
            On success, it includes the raw topography array in `"image_data"` (typically a 2D NumPy-like array from the Z-Axis channel).  
            If metric flags were enabled, the result also includes:  
            - `"average_friction"` (float): Mean value of friction force  
            - `"mean_roughness"` (float): Ra value in surface units  
            - `"rms_roughness"` (float): Rq value in surface units  
            In case of errors (e.g., file not found or bad dynamic code), a descriptive message is returned with status `"Error"`.  
            [/DETAILED]  
            [EXAMPLES]  
            Example success response:  
            ```python
            {
                "status": "Success",
                "message": "Raw Image /path/to/sample.nid processed successfully.",
                "image_data": [[0.012, 0.015], [0.017, 0.014]],
                "mean_roughness": 0.0035,
                "rms_roughness": 0.0042
            }
            ```  
            
            Example error response:  
            ```python
            {
                "status": "Error",
                "message": "The specified file does not exist."
            }
            ```  
            [/EXAMPLES]
            

    """
    import os
    import glob
    from NSFopen.read import read
    import numpy as np
    if path is None:
        path = os.getcwd()
    
    # Determine the file to display
    if filename:
        file_to_display = os.path.join(path, filename)
        if not os.path.isfile(file_to_display):
            print(f"File not found: {file_to_display}")
            return {"status": "Error", "message": "The specified file does not exist."}
    else:
        # Get the list of all files in the directory
        list_of_files = glob.glob(os.path.join(path, '*'))
        
        if not list_of_files:
            print("No files found in the specified directory.")
            return {"status": "Error", "message": "No files found in the directory."}
        
        # Find the latest file based on creation time
        file_to_display = max(list_of_files, key=os.path.getctime)
    
    print(f"File to display: {file_to_display}")

    try:
        # Read the file
        afm = read(file_to_display)
        
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
        result = {"status": "Success", "message": f"Raw Image {file_to_display} processed successfully.", "image_data": image_data}
        
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
