import sys
sys.path.append('C:/Users/Admin/Desktop/corral/mat-agent-bench/tasks/afm/src/afm/claude-sonnet-4-5_task_3_ReAct/enviroment')

from NSFopen import read
import numpy as np

# List of scan files from trial_2
scan_files = [
    'C:/Users/Admin/Desktop/corral/mat-agent-bench/tasks/afm/src/afm/claude-sonnet-4-5_task_3_ReAct/enviroment/tasks_3/afm_experiment_level_3_trial_2/scan_task3_73703.nid',
    'C:/Users/Admin/Desktop/corral/mat-agent-bench/tasks/afm/src/afm/claude-sonnet-4-5_task_3_ReAct/enviroment/tasks_3/afm_experiment_level_3_trial_2/scan_task3_73704.nid',
    'C:/Users/Admin/Desktop/corral/mat-agent-bench/tasks/afm/src/afm/claude-sonnet-4-5_task_3_ReAct/enviroment/tasks_3/afm_experiment_level_3_trial_2/scan_task3_73705.nid',
    'C:/Users/Admin/Desktop/corral/mat-agent-bench/tasks/afm/src/afm/claude-sonnet-4-5_task_3_ReAct/enviroment/tasks_3/afm_experiment_level_3_trial_2/scan_task3_73706.nid',
    'C:/Users/Admin/Desktop/corral/mat-agent-bench/tasks/afm/src/afm/claude-sonnet-4-5_task_3_ReAct/enviroment/tasks_3/afm_experiment_level_3_trial_2/scan_task3_73707.nid',
    'C:/Users/Admin/Desktop/corral/mat-agent-bench/tasks/afm/src/afm/claude-sonnet-4-5_task_3_ReAct/enviroment/tasks_3/afm_experiment_level_3_trial_2/scan_task3_73708.nid'
]

results = []

for scan_file in scan_files:
    try:
        data = read(scan_file)
        
        # Get scan parameters
        scan_width = data['Image']['Forward']['Width']
        scan_height = data['Image']['Forward']['Height']
        
        # Calculate scan area in µm²
        scan_area_m2 = scan_width * scan_height
        scan_area_um2 = scan_area_m2 * 1e12  # Convert m² to µm²
        
        # Get Z-Axis data
        image_data = data['Image']['Forward']['Z-Axis']
        
        # Calculate RMS roughness (Rq)
        mean_height = np.mean(image_data)
        rms_roughness = np.sqrt(np.mean((image_data - mean_height)**2))
        
        results.append({
            'file': scan_file.split('/')[-1],
            'width_m': scan_width,
            'height_m': scan_height,
            'area_um2': scan_area_um2,
            'rms_roughness_m': rms_roughness,
            'rms_roughness_nm': rms_roughness * 1e9
        })
        
        print(f"File: {scan_file.split('/')[-1]}")
        print(f"  Scan size: {scan_width*1e6:.1f} µm × {scan_height*1e6:.1f} µm")
        print(f"  Scan area: {scan_area_um2:.1f} µm²")
        print(f"  RMS roughness: {rms_roughness:.6e} m ({rms_roughness*1e9:.3f} nm)")
        print()
        
    except Exception as e:
        print(f"Error processing {scan_file}: {e}")
        print()

# Sort by area
results.sort(key=lambda x: x['area_um2'])

print("\n=== Summary (sorted by area) ===")
print("Area (µm²) | RMS Roughness (nm)")
for r in results:
    print(f"{r['area_um2']:10.1f} | {r['rms_roughness_nm']:10.3f}")

# Prepare data for fitting
A = [r['area_um2'] for r in results]
Rb = [r['rms_roughness_m'] for r in results]

print(f"\nA = {A}")
print(f"Rb = {Rb}")