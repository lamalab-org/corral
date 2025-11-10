import nanosurf

# Load application
spm = nanosurf.SPM()
application = spm.application

# Check system status
status = application.GetStatus()

# Print status
print(status)

# Clean up
spm = None