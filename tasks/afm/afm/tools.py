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
    Detects grains in the input image and generates bounding boxes around each one.

    Each detected grain is assigned a unique index (starting from 1), which can be
    used to reference and process specific grains in subsequent steps (e.g., scanning).

    Args:
         image_path (str, optional): Path to the input image file. If None, uses the latest saved .nid file.

    Returns:
        list: List of bounding boxes as (index, x1, y1, x2, y2), where
              (x1, y1) is the bottom-left and (x2, y2) is the top-right corner.
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
    Scans the area corresponding to the specified grain in the image.

    If the image was previously scanned, cached scan size and center values are reused.

    Args:
        grain_id (int): ID of the grain (bounding box index starting from 1).
        image_path (str, optional): Path to the image file containing grains. If None, uses the latest .nid file.

    Returns:
        None
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
    This tool retrieves code snippets from a database that are specifically designed for
    operating an Atomic Force Microscope (AFM) machine. The retrieved code is intended to be
    used as a reference for controlling the AFM.

    Args:
        query (str): The query string to search for relevant code snippets in the database.

    Returns:
        str: The retrieved code snippet.
    """
    result = Document_Retriever.invoke(query)
    return result


@tool
def Image_optimizer(baseline: bool) -> str:
    """
    This tool optimizes the parameters (P/I/D gains) based on baseline correction
    settings to provide the best solution for image clarity. Use this tool if the image 
    appears blurry or unclear and you want to enhance its sharpness.
    
    Args:
        baseline: If True, use baseline correction. If False, do not use baseline correction.
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
    Use this tool only to run Python code for operating an Atomic Force Microscope.
    Use code from 'Document_Retriever' and correct it as needed. This code controls the AFM, so handle it with care.
    
    Args:
        code (str): The Python code to be executed.
    
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
    This tool is specifically designed to extract and analyze image data from Atomic Force Microscopy (AFM) .nid image files captured using Nanosurf instruments. Leveraging the Nanosurf API, it processes high-resolution AFM images and optionally computes key surface properties such as:
    - Average Friction
    - Mean Roughness
    - RMS Roughness
    The tool supports both static file access (via a provided filename) and dynamic selection of the latest AFM image file in a specified directory. It also allows execution of custom image-processing logic through an optional dynamic_code parameter, enabling flexible access to different imaging channels (e.g., Z-Axis, Deflection, Friction Force) and scan directions (Forward/Backward).
    
    Args:
        path (str): The directory path to search for the latest file. Defaults to None.
        filename (str): The specific image file to display. Defaults to None.
        dynamic_code (str): A string containing Python code to process the image data. Defaults to None.
        calculate_friction (bool): Whether to calculate average friction. Defaults to False.
        calculate_mean_roughness (bool): Whether to calculate mean roughness. Defaults to False.
        calculate_rms_roughness (bool): Whether to calculate RMS roughness. Defaults to False.

    Returns:
    - dict: A dictionary containing the status, image data, or an error message.
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
