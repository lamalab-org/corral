import nanosurf
import time
import os

# Load application
spm = nanosurf.SPM()  # or .C3000(), .CX(), or .CoreAFM()
application = spm.application

# Get access to different components
scan = application.Scan
opmode = application.OperatingMode
zcontrol = application.ZController
head = application.ScanHead

# Set Z controller parameters (PID gains)
print("Setting PID gains...")
zcontrol.PGain = 100    # P gain = 100
zcontrol.IGain = 6000   # I gain = 6000
zcontrol.DGain = 10     # D gain = 10

# Set the output file path and name
output_path = r"C:\Users\Admin\Desktop\corral\mat-agent-bench\tasks\afm\afm\afm_images"
file_name = "task_3_00303"
full_path = os.path.join(output_path, file_name)

# Set the filename for the AFM software
print(f"Setting filename to: {file_name}")
application.SetGalleryHistoryFilenameMask(file_name)

# Start the scan
print("Starting scan...")
scan.StartFrameUp()  # Scanning from bottom to top
# Wait for scan to complete
scanning = scan.IsScanning
while scanning:
    print(f"Scanning in progress... Remaining lines: {scan.Lines - scan.Currentline}")
    time.sleep(3)
    scanning = scan.IsScanning



print("Scanning finished")
print(f"Image saved as: {full_path}.nid")

# Clean up
del spm
