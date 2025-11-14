import numpy as np
import math
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Data from AFM scans
A = np.array([1, 4, 9, 16, 25])  # Scan areas in µm²
Rq = np.array([5.349e-08, 3.595e-08, 4.577e-08, 4.901e-08, 5.358e-08])  # RMS roughness in meters

# Convert Rq to nanometers for easier interpretation
Rq_nm = Rq * 1e9

print("Data Summary:")
print("="*60)
for i in range(len(A)):
    print(f"Area: {A[i]:5.1f} µm²  ->  Rq: {Rq_nm[i]:6.2f} nm")
print("="*60)

# Test different models
print("\nTesting different models:")
print("="*60)

# Model 1: Linear (Rq = a + b*A)
def linear(x, a, b):
    return a + b * x

try:
    popt_linear, _ = curve_fit(linear, A, Rq)
    residuals_linear = Rq - linear(A, *popt_linear)
    ss_res_linear = np.sum(residuals_linear**2)
    ss_tot = np.sum((Rq - np.mean(Rq))**2)
    r2_linear = 1 - (ss_res_linear / ss_tot)
    print(f"1. Linear: Rq = {popt_linear[0]:.6e} + {popt_linear[1]:.6e}*A")
    print(f"   R² = {r2_linear:.6f}")
except:
    print("1. Linear: Failed to fit")

# Model 2: Power law (Rq = a * A^b)
def power_law(x, a, b):
    return a * np.power(x, b)

try:
    popt_power, _ = curve_fit(power_law, A, Rq, p0=[1e-8, 0.1])
    residuals_power = Rq - power_law(A, *popt_power)
    ss_res_power = np.sum(residuals_power**2)
    r2_power = 1 - (ss_res_power / ss_tot)
    print(f"2. Power law: Rq = {popt_power[0]:.6e} * A**{popt_power[1]:.6f}")
    print(f"   R² = {r2_power:.6f}")
except:
    print("2. Power law: Failed to fit")

# Model 3: Logarithmic (Rq = a + b*log(A))
def logarithmic(x, a, b):
    return a + b * np.log(x)

try:
    popt_log, _ = curve_fit(logarithmic, A, Rq)
    residuals_log = Rq - logarithmic(A, *popt_log)
    ss_res_log = np.sum(residuals_log**2)
    r2_log = 1 - (ss_res_log / ss_tot)
    print(f"3. Logarithmic: Rq = {popt_log[0]:.6e} + {popt_log[1]:.6e}*math.log(A)")
    print(f"   R² = {r2_log:.6f}")
except:
    print("3. Logarithmic: Failed to fit")

# Model 4: Square root (Rq = a + b*sqrt(A))
def sqrt_model(x, a, b):
    return a + b * np.sqrt(x)

try:
    popt_sqrt, _ = curve_fit(sqrt_model, A, Rq)
    residuals_sqrt = Rq - sqrt_model(A, *popt_sqrt)
    ss_res_sqrt = np.sum(residuals_sqrt**2)
    r2_sqrt = 1 - (ss_res_sqrt / ss_tot)
    print(f"4. Square root: Rq = {popt_sqrt[0]:.6e} + {popt_sqrt[1]:.6e}*A**0.5")
    print(f"   R² = {r2_sqrt:.6f}")
except:
    print("4. Square root: Failed to fit")

# Model 5: Exponential (Rq = a * exp(b*A))
def exponential(x, a, b):
    return a * np.exp(b * x)

try:
    popt_exp, _ = curve_fit(exponential, A, Rq, p0=[1e-8, 0.01])
    residuals_exp = Rq - exponential(A, *popt_exp)
    ss_res_exp = np.sum(residuals_exp**2)
    r2_exp = 1 - (ss_res_exp / ss_tot)
    print(f"5. Exponential: Rq = {popt_exp[0]:.6e} * math.exp({popt_exp[1]:.6e}*A)")
    print(f"   R² = {r2_exp:.6f}")
except:
    print("5. Exponential: Failed to fit")

print("="*60)

# Determine best model
models = {
    'Linear': (r2_linear, f"{popt_linear[0]:.6e} + {popt_linear[1]:.6e} * A"),
    'Power': (r2_power, f"{popt_power[0]:.6e} * A**{popt_power[1]:.6f}"),
    'Logarithmic': (r2_log, f"{popt_log[0]:.6e} + {popt_log[1]:.6e} * math.log(A)"),
    'Square root': (r2_sqrt, f"{popt_sqrt[0]:.6e} + {popt_sqrt[1]:.6e} * A**0.5"),
    'Exponential': (r2_exp, f"{popt_exp[0]:.6e} * math.exp({popt_exp[1]:.6e} * A)")
}

best_model = max(models.items(), key=lambda x: x[1][0])
print(f"\nBest fit model: {best_model[0]}")
print(f"R² = {best_model[1][0]:.6f}")
print(f"Equation: Rb = {best_model[1][1]}")

print("\n" + "="*60)
print("Final Results:")
print("="*60)
print(f"Equation: Rb = {best_model[1][1]}")
print(f"\nData points:")
print(f"A (µm²): {list(A)}")
print(f"Rb (m): {[f'{x:.6e}' for x in Rq]}")
