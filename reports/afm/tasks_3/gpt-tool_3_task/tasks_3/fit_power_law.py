import numpy as np
from scipy.optimize import curve_fit

# Hypothetical data
A = np.array([1, 2, 3, 4, 5])
Rq = np.array([3.3e-07, 3.0e-07, 2.8e-07, 2.6e-07, 2.5e-07])

# Power-law function
def power_law(x, k, b):
    return k * x ** b

# Fit the power-law model
params, _ = curve_fit(power_law, A, Rq)

# Extract parameters
k, b = params

# Print the equation
print(f"Rq = {k} * A**{b}")

# Save the results
with open('C:/Users/Admin/Desktop/corral/mat-agent-bench/tasks/afm/src/afm/enviroment/tasks_3/power_law_results.txt', 'w') as f:
    f.write(f"Rq = {k} * A**{b}\n")
    f.write(f"k: {k}, b: {b}\n")