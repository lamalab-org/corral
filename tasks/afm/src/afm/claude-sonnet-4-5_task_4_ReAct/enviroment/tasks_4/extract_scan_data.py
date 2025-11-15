import NSFopen
import numpy as np
import os
import json

# Directory containing the scan files
base_dir = r'C:\Users\Admin\Desktop\corral\mat-agent-bench\tasks\afm\src\afm\claude-sonnet-4-5_task_4_ReAct\enviroment\tasks_4\afm_experiment_level_4_trial_0'

# Get all .nid files
nid_files = [f for f in os.listdir(base_dir) if f.endswith('.nid')]
nid_files.sort()

print(f"Found {len(nid_files)} .nid files")
print("\nAnalyzing files...\n")

results = []

for nid_file in nid_files:
    file_path = os.path.join(base_dir, nid_file)
    
    try:
        # Read the .nid file
        data = NSFopen.read(file_path)
        
        # Extract image data (Z-Axis from Forward scan)
        image_data = data["Image"]["Forward"]["Z-Axis"]
        
        # Get scan parameters
        scan_params = data["Scan"]
        width = scan_params.get("Width", None)  # in meters
        height = scan_params.get("Height", None)  # in meters
        
        if width and height:
            # Calculate area in µm²
            area_m2 = width * height
            area_um2 = area_m2 * 1e12  # convert m² to µm²
            
            # Calculate RMS roughness (Rq)
            mean_height = np.mean(image_data)
            rq = np.sqrt(np.mean((image_data - mean_height) ** 2))
            
            results.append({
                'file': nid_file,
                'area_um2': area_um2,
                'area_m2': area_m2,
                'rq_m': rq,
                'rq_nm': rq * 1e9,
                'width_m': width,
                'height_m': height
            })
            
            print(f"{nid_file}:")
            print(f"  Width: {width*1e6:.2f} µm, Height: {height*1e6:.2f} µm")
            print(f"  Area: {area_um2:.2f} µm²")
            print(f"  Rq: {rq*1e9:.4f} nm ({rq:.4e} m)")
            print()
    
    except Exception as e:
        print(f"Error processing {nid_file}: {e}")
        print()

# Sort by area
results.sort(key=lambda x: x['area_um2'])

print("\n" + "="*60)
print("Summary (sorted by area):")
print("="*60)

for r in results:
    print(f"Area: {r['area_um2']:6.2f} µm² | Rq: {r['rq_nm']:8.4f} nm | File: {r['file']}")

# Save results
with open('scan_data_extracted.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to scan_data_extracted.json")