import nanosurf

# Load application
spm = nanosurf.SPM()
application = spm.application

# All variables
scan = application.Scan
zcontrol = application.ZController

# Set scan parameters
scan.ImageWidth = 10e-6  # [m]
scan.ImageHeight = 10e-6  # [m]
scan.Scantime = 0.4  # [s] time per line (51.2 s / 128 lines)
scan.Points = 128  # points per line
scan.Lines = 128  # lines per frame

# Set Z controller parameters to minimize tip damage
zcontrol.PGain = 3100
zcontrol.IGain = 3500

# Start scan
application.StartScan()

# Wait for scan to complete
application.WaitForScanComplete()

# Get the path to the saved image
image_path = application.GetLastImagePath()

# Clean up
spm = None

print(image_path)