#imports 
import nanosurf

#load application
spm = nanosurf.SPM()  # or .C3000() or .CX(), or .CoreAFM()
application = spm.application

#all variables
scan = application.Scan
zcontrol = application.ZController

# Set scan parameters
scan.Scantime = 0.1 # [s] time per line 
scan.Lines = 128 # lines per frame

# Set Z controller parameters
zcontrol.PGain = 100
zcontrol.IGain = 6000
zcontrol.DGain = 10

del spm