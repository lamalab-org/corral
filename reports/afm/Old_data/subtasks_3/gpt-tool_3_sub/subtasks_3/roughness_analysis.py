import math

import numpy as np
from scipy.stats import linregress

# Given RMS roughness values
Rb_values = [3.1925371294018595e-07, 6.052114876060461e-07, 8.711400831928807e-07]

# Corresponding scan areas (assuming unit area per pixel)
A_values = [16384, 16384, 16384]

# Logarithmic transformation
log_Rb = np.log(Rb_values)
log_A = np.log(A_values)

# Perform linear regression on the log-transformed data
slope, intercept, r_value, p_value, std_err = linregress(log_A, log_Rb)

# Calculate k from the intercept
k = math.exp(intercept)

# Print the results
equation = f"Rb = {k} * A**{slope}"
print({"equation": equation, "Rb": Rb_values, "A": A_values})
