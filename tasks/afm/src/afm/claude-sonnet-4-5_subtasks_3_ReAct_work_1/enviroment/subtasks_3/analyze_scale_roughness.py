import numpy as np
import json
import sys
sys.path.append('C:/Program Files/Nanosurf/NanoSurf Report/NSFopen')
from NSFopen import read

# Load the three scan files
file_paths = [
    'C:/Users/Admin/Desktop/corral/corral/tasks/afm/src/afm/claude-sonnet-4-5_subtasks_3_ReAct_work_1/enviroment/subtasks_3/image_capture_1_subtask_level_3_trial_0/scan_10x10um_22713.nid',
    'C:/Users/Admin/Desktop/corral/corral/tasks/afm/src/afm/claude-sonnet-4-5_subtasks_3_ReAct_work_1/enviroment/subtasks_3/image_capture_2_subtask_level_3_trial_6/scan_10x10um_22726.nid',
    'C:/Users/Admin/Desktop/corral/corral/tasks/afm/src/afm/claude-sonnet-4-5_subtasks_3_ReAct_work_1/enviroment/subtasks_3/image_capture_3_subtask_level_3_trial_6/scan_10x10um_22727.nid'
]

# Function to calculate RMS roughness
def calculate_rms_roughness(image_data):
    mean_height = np.mean(image_data)
    rms = np.sqrt(np.mean((image_data - mean_height) ** 2))
    return rms

# Collect data for different scan areas
all_areas = []
all_roughness = []

for file_path in file_paths:
    try:
        # Read the file
        data = read(file_path)
        image_data = data['Image']['Forward']['Z-Axis']
        
        # Full image is 128x128 pixels, 10x10 µm
        full_size = image_data.shape[0]
        pixel_size = 10.0 / full_size  # µm per pixel
        
        # Analyze different sub-region sizes
        sizes = [128, 96, 64, 48, 32, 24, 16]
        
        for size in sizes:
            if size <= full_size:
                # Extract center sub-region
                start = (full_size - size) // 2
                end = start + size
                sub_region = image_data[start:end, start:end]
                
                # Calculate area in µm²
                area = (size * pixel_size) ** 2
                
                # Calculate RMS roughness
                rms = calculate_rms_roughness(sub_region)
                
                all_areas.append(area)
                all_roughness.append(rms)
                
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert to numpy arrays
all_areas = np.array(all_areas)
all_roughness = np.array(all_roughness)

# Remove any zero or negative roughness values for fitting
valid_mask = all_roughness > 0
valid_areas = all_areas[valid_mask]
valid_roughness = all_roughness[valid_mask]

print(f"Total data points: {len(all_areas)}")
print(f"Valid data points (Rq > 0): {len(valid_areas)}")
print(f"Areas: {all_areas}")
print(f"Roughness: {all_roughness}")

# Fit power law: Rq = C * A^H
if len(valid_areas) > 2:
    # Take logarithm: log(Rq) = log(C) + H * log(A)
    log_areas = np.log(valid_areas)
    log_roughness = np.log(valid_roughness)
    
    # Linear fit
    coeffs = np.polyfit(log_areas, log_roughness, 1)
    H = coeffs[0]  # Hurst exponent
    log_C = coeffs[1]
    C = np.exp(log_C)
    
    print(f"\nFitted power law: Rq = {C} * A^{H}")
    print(f"Hurst exponent H = {H}")
    print(f"Coefficient C = {C}")
    
    # Create result
    result = {
        "equation": f"Rb = {C} * A**{H}",
        "Rb": all_roughness.tolist(),
        "A": all_areas.tolist()
    }
else:
    # If all roughness values are zero or near-zero, use a simple constant model
    print("\nAll roughness values are near zero. Using constant model.")
    mean_roughness = np.mean(all_roughness)
    result = {
        "equation": f"Rb = {mean_roughness}",
        "Rb": all_roughness.tolist(),
        "A": all_areas.tolist()
    }

# Save result
with open('C:/Users/Admin/Desktop/corral/corral/tasks/afm/src/afm/claude-sonnet-4-5_subtasks_3_ReAct_work_1/enviroment/subtasks_3/roughness_relationship_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResult saved to roughness_relationship_result.json")
print(json.dumps(result, indent=2))