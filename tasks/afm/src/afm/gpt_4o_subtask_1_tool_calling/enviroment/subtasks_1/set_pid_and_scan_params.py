#imports 
import nanosurf

#load application
spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreAFM()
application = spm.application

#all variables
scan = application.Scan
zcontrol = application.ZController

# Set scan parameters
scan.ImageWidth = 10e-6  # [m]
scan.ImageHeight = 10e-6  # [m]
scan.Scantime = 0.55  # [s] time per line
scan.Points = 256  # points per line
scan.Lines = 256  # lines per frame
scan.CenterPosX = 5e-6  # [m]
scan.CenterPosY = 5e-6  # [m]
scan.SlopeX = 0.0  # degree
scan.SlopeY = 0.0  # degree
scan.Overscan = 0  # [%]
scan.Rotation = 0  # degree

# Set Z controller parameters
zcontrol.SetPoint = 1  # [%/V]
zcontrol.PGain = 3100
zcontrol.IGain = 3500
zcontrol.DGain = 10

# No cleanup needed
