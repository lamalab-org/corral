import numpy as np
import math
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# Data from AFM scans
A = np.array([1, 4, 9, 16, 25])  # Scan area in µm²
Rb = np.array([2.614e-08, 3.090e-08, 3.823e-08, 5.007e-08, 5.678e-08])  # RMS roughness in meters

print("Data:")
print(f"Area (µm²): {A}")
print(f"Roughness (m): {Rb}")
print(f"Roughness (nm): {Rb * 1e9}")
print()

# Test different models

# Model 1: Power law - Rb = a * A^b
def power_law(A, a, b):
    return a * A**b

try:
    params_power, _ = curve_fit(power_law, A, Rb)
    Rb_pred_power = power_law(A, *params_power)
    r_power, _ = pearsonr(Rb, Rb_pred_power)
    rmse_power = np.sqrt(np.mean((Rb - Rb_pred_power)**2))
    print(f"Power Law: Rb = {params_power[0]:.6e} * A**{params_power[1]:.6f}")
    print(f"  R² = {r_power**2:.6f}, RMSE = {rmse_power:.6e}")
    print()
except Exception as e:
    print(f"Power law fit failed: {e}")
    print()

# Model 2: Logarithmic - Rb = a + b * log(A)
def logarithmic(A, a, b):
    return a + b * np.log(A)

try:
    params_log, _ = curve_fit(logarithmic, A, Rb)
    Rb_pred_log = logarithmic(A, *params_log)
    r_log, _ = pearsonr(Rb, Rb_pred_log)
    rmse_log = np.sqrt(np.mean((Rb - Rb_pred_log)**2))
    print(f"Logarithmic: Rb = {params_log[0]:.6e} + {params_log[1]:.6e} * log(A)")
    print(f"  R² = {r_log**2:.6f}, RMSE = {rmse_log:.6e}")
    print()
except Exception as e:
    print(f"Logarithmic fit failed: {e}")
    print()

# Model 3: Linear - Rb = a + b * A
def linear(A, a, b):
    return a + b * A

try:
    params_linear, _ = curve_fit(linear, A, Rb)
    Rb_pred_linear = linear(A, *params_linear)
    r_linear, _ = pearsonr(Rb, Rb_pred_linear)
    rmse_linear = np.sqrt(np.mean((Rb - Rb_pred_linear)**2))
    print(f"Linear: Rb = {params_linear[0]:.6e} + {params_linear[1]:.6e} * A")
    print(f"  R² = {r_linear**2:.6f}, RMSE = {rmse_linear:.6e}")
    print()
except Exception as e:
    print(f"Linear fit failed: {e}")
    print()

# Model 4: Square root - Rb = a + b * sqrt(A)
def sqrt_model(A, a, b):
    return a + b * np.sqrt(A)

try:
    params_sqrt, _ = curve_fit(sqrt_model, A, Rb)
    Rb_pred_sqrt = sqrt_model(A, *params_sqrt)
    r_sqrt, _ = pearsonr(Rb, Rb_pred_sqrt)
    rmse_sqrt = np.sqrt(np.mean((Rb - Rb_pred_sqrt)**2))
    print(f"Square Root: Rb = {params_sqrt[0]:.6e} + {params_sqrt[1]:.6e} * sqrt(A)")
    print(f"  R² = {r_sqrt**2:.6f}, RMSE = {rmse_sqrt:.6e}")
    print()
except Exception as e:
    print(f"Square root fit failed: {e}")
    print()

print("\n=== Best Fit Selection ===")
print("Based on R² values, the best model will be selected.")