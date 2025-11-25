import NSFopen

# File paths
files = [
    r"C:\Users\Admin\Desktop\corral\corral\tasks\afm\src\afm\claude-sonnet-4-5_subtasks_3_tool_calling\enviroment\subtasks_3\image_capture_1_subtask_level_3_trial_0\20251123_Microforcep_HBSS_18uN_T1_zoomout_11395.nid",
    r"C:\Users\Admin\Desktop\corral\corral\tasks\afm\src\afm\claude-sonnet-4-5_subtasks_3_tool_calling\enviroment\subtasks_3\image_capture_2_subtask_level_3_trial_7\topographic_scan_20um_11416.nid",
    r"C:\Users\Admin\Desktop\corral\corral\tasks\afm\src\afm\claude-sonnet-4-5_subtasks_3_tool_calling\enviroment\subtasks_3\image_capture_3_subtask_level_3_trial_7\topographic_30um_11417.nid"
]

results = []
for file_path in files:
    try:
        data = NSFopen.read(file_path)
        # Get scan size
        scan_size_x = data['Scan']['Range']['X']
        scan_size_y = data['Scan']['Range']['Y']
        # Calculate area in µm²
        area = scan_size_x * scan_size_y * 1e12  # Convert from m² to µm²
        results.append({
            'file': file_path.split('\\')[-1],
            'scan_size_x': scan_size_x * 1e6,  # Convert to µm
            'scan_size_y': scan_size_y * 1e6,  # Convert to µm
            'area': area
        })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

for r in results:
    print(f"File: {r['file']}")
    print(f"  Scan size: {r['scan_size_x']:.2f} µm × {r['scan_size_y']:.2f} µm")
    print(f"  Area: {r['area']:.2f} µm²")
    print()
