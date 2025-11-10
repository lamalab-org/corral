import math

# Given data
Rq1 = 8.819189527413199e-08
Rq2 = 2.6127e-07
A1 = 100  # Assumed scan area in µm²
A2 = 100  # Assumed scan area in µm²

# Calculate k for both data points
k1 = Rq1 / A1
k2 = Rq2 / A2

# Print the results
print(f'k1: {k1}')
print(f'k2: {k2}')

# Assuming a linear relationship Rb = k * A
# Use the average k for the final equation
k_avg = (k1 + k2) / 2

# Final equation
final_equation = f'Rb = {k_avg} * A'

# Output the final equation
print(final_equation)