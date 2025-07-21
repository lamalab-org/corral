import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from NSFopen.read import nid_read

# Step 1: Load the NID file
afm = nid_read("sample_data/Sample.nid")

# Step 2: Print keys to check available data
logger.info("Available data keys: %s", afm.data.keys())

# Step 3: Extract Z-AxisSensor image from the Forward scan
z_forward_image = afm.data[("Image", "Backward", "Z-AxisSensor")]

# Step 4: Convert to NumPy array (if it's not already)
z_forward_image = np.array(z_forward_image)

# Step 5: Compute number of square grids (pixels)
num_squares = z_forward_image.shape[0] * z_forward_image.shape[1]
logger.info(f"Number of square grids: {num_squares}")

# # Step 6: Normalize the image for consistent color scaling (optional but useful)
# z_norm = (z_forward_image - np.min(z_forward_image)) / (np.max(z_forward_image) - np.min(z_forward_image))

# Step 7: Save as a pure image (no axis, no title, no colorbar)
plt.imsave("Question_23_b_pure.png", z_forward_image, cmap="afmhot")
logger.info("Pure image saved as 'Question_23_b_pure.png'")
