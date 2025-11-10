import math

# RMS roughness values
Rb_values = [8.80400268217022e-08, 2.612748494296943e-07, 3.873876761017551e-07]

# Calculate the average RMS roughness
k = sum(Rb_values) / len(Rb_values)

# Print the result
print(k)