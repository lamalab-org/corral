import math
import numpy as np
from scipy.optimize import curve_fit

# Data points
A = np.array([100, 400, 900])  # µm²
Rb = np.array([0.0, 1.9155383705064574e-14, 2.205592860413776e-14])  # m

# Since first point is 0, let's fit using points 2 and 3
A_fit = A[1:]
Rb_fit = Rb[1:]

# Power law: Rb = c * A^alpha
def power_law(A, c, alpha):
    return c * A**alpha

# Fit the power law
params, _ = curve_fit(power_law, A_fit, Rb_fit)
c, alpha = params

print(f"Power law fit: Rb = {c} * A**{alpha}")
print(f"c = {c}")
print(f"alpha = {alpha}")

# Verify the fit
for i, a in enumerate(A):
    predicted = power_law(a, c, alpha)
    actual = Rb[i]
    print(f"A={a}: Predicted Rb={predicted:.6e}, Actual Rb={actual:.6e}")

# Try logarithmic fit
def log_fit(A, c, d):
    return c * np.log(A) + d

params_log, _ = curve_fit(log_fit, A_fit, Rb_fit)
c_log, d_log = params_log
print(f"\nLogarithmic fit: Rb = {c_log} * log(A) + {d_log}")

# Verify log fit
for i, a in enumerate(A):
    predicted = log_fit(a, c_log, d_log)
    actual = Rb[i]
    print(f"A={a}: Predicted Rb={predicted:.6e}, Actual Rb={actual:.6e}")

# Try linear fit
def linear_fit(A, c, d):
    return c * A + d

params_lin, _ = curve_fit(linear_fit, A_fit, Rb_fit)
c_lin, d_lin = params_lin
print(f"\nLinear fit: Rb = {c_lin} * A + {d_lin}")

# Verify linear fit
for i, a in enumerate(A):
    predicted = linear_fit(a, c_lin, d_lin)
    actual = Rb[i]
    print(f"A={a}: Predicted Rb={predicted:.6e}, Actual Rb={actual:.6e}")