import numpy as np
import math
from scipy.optimize import curve_fit
import json

# Data from AFM scans
# Area in µm²
A = np.array([1, 4, 9, 25, 49, 100])

# Rq (RMS roughness) in meters
Rq_meters = np.array([2.3805028730457278e-08, 
                       5.556974732467009e-08, 
                       9.374404204637746e-08, 
                       1.66524512307693e-07, 
                       2.3752860265138498e-07, 
                       3.3361321924539366e-07])

# Convert to nanometers for easier interpretation
Rq_nm = Rq_meters * 1e9

print("Data:")
for i in range(len(A)):
    print(f"A = {A[i]} µm², Rq = {Rq_nm[i]:.2f} nm")

# Try different models
print("\n" + "="*50)
print("Testing different models:")
print("="*50)

# Model 1: Power law Rq = k * A^n
def power_law(A, k, n):
    return k * A**n

popt_power, _ = curve_fit(power_law, A, Rq_nm)
k_power, n_power = popt_power
Rq_pred_power = power_law(A, k_power, n_power)
rmse_power = np.sqrt(np.mean((Rq_nm - Rq_pred_power)**2))
print(f"\n1. Power Law: Rq = {k_power:.4f} * A^{n_power:.4f}")
print(f"   RMSE: {rmse_power:.4f} nm")

# Model 2: Linear Rq = a * A + b
def linear(A, a, b):
    return a * A + b

popt_linear, _ = curve_fit(linear, A, Rq_nm)
a_linear, b_linear = popt_linear
Rq_pred_linear = linear(A, a_linear, b_linear)
rmse_linear = np.sqrt(np.mean((Rq_nm - Rq_pred_linear)**2))
print(f"\n2. Linear: Rq = {a_linear:.4f} * A + {b_linear:.4f}")
print(f"   RMSE: {rmse_linear:.4f} nm")

# Model 3: Logarithmic Rq = a * log(A) + b
def logarithmic(A, a, b):
    return a * np.log(A) + b

popt_log, _ = curve_fit(logarithmic, A, Rq_nm)
a_log, b_log = popt_log
Rq_pred_log = logarithmic(A, a_log, b_log)
rmse_log = np.sqrt(np.mean((Rq_nm - Rq_pred_log)**2))
print(f"\n3. Logarithmic: Rq = {a_log:.4f} * log(A) + {b_log:.4f}")
print(f"   RMSE: {rmse_log:.4f} nm")

# Model 4: Square root Rq = a * sqrt(A) + b
def sqrt_model(A, a, b):
    return a * np.sqrt(A) + b

popt_sqrt, _ = curve_fit(sqrt_model, A, Rq_nm)
a_sqrt, b_sqrt = popt_sqrt
Rq_pred_sqrt = sqrt_model(A, a_sqrt, b_sqrt)
rmse_sqrt = np.sqrt(np.mean((Rq_nm - Rq_pred_sqrt)**2))
print(f"\n4. Square Root: Rq = {a_sqrt:.4f} * sqrt(A) + {b_sqrt:.4f}")
print(f"   RMSE: {rmse_sqrt:.4f} nm")

# Find best model
models = [
    ('Power Law', rmse_power, f"{k_power:.6f} * A ** {n_power:.6f}"),
    ('Linear', rmse_linear, f"{a_linear:.6f} * A + {b_linear:.6f}"),
    ('Logarithmic', rmse_log, f"{a_log:.6f} * math.log(A) + {b_log:.6f}"),
    ('Square Root', rmse_sqrt, f"{a_sqrt:.6f} * A ** 0.5 + {b_sqrt:.6f}")
]

models.sort(key=lambda x: x[1])

print("\n" + "="*50)
print("Best Model (lowest RMSE):")
print("="*50)
print(f"Model: {models[0][0]}")
print(f"RMSE: {models[0][1]:.4f} nm")
print(f"Equation: Rb = {models[0][2]}")

# Prepare output in the required format
# Convert Rq from nm back to the same units for consistency
result = {
    "equation": f"Rb = {models[0][2]}",
    "Rb": Rq_nm.tolist(),
    "A": A.tolist()
}

print("\n" + "="*50)
print("Final Result:")
print("="*50)
print(json.dumps(result, indent=2))

# Save to file
with open('roughness_analysis_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResult saved to roughness_analysis_result.json")