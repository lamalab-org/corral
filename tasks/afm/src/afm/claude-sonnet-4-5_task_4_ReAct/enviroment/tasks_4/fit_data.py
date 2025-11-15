import numpy as np
from scipy.optimize import curve_fit
import math

# Data points (A in um^2, Rq in nm)
A_data = np.array([1, 4, 9, 16, 25, 36])
Rq_data = np.array([33.16, 49.53, 96.58, 117.1, 165.7, 190.3])

print("Data points:")
for i in range(len(A_data)):
    print(f"A = {A_data[i]} um^2, Rq = {Rq_data[i]} nm")

# Try different models

# Model 1: Linear Rq = a*A + b
def linear(A, a, b):
    return a * A + b

popt_linear, _ = curve_fit(linear, A_data, Rq_data)
Rq_pred_linear = linear(A_data, *popt_linear)
rmse_linear = np.sqrt(np.mean((Rq_data - Rq_pred_linear)**2))
print(f"\nLinear model: Rq = {popt_linear[0]:.4f}*A + {popt_linear[1]:.4f}")
print(f"RMSE: {rmse_linear:.4f}")

# Model 2: Power law Rq = a*A^b
def power_law(A, a, b):
    return a * np.power(A, b)

popt_power, _ = curve_fit(power_law, A_data, Rq_data, p0=[10, 0.5])
Rq_pred_power = power_law(A_data, *popt_power)
rmse_power = np.sqrt(np.mean((Rq_data - Rq_pred_power)**2))
print(f"\nPower law model: Rq = {popt_power[0]:.4f}*A**{popt_power[1]:.4f}")
print(f"RMSE: {rmse_power:.4f}")

# Model 3: Square root Rq = a*sqrt(A) + b
def sqrt_model(A, a, b):
    return a * np.sqrt(A) + b

popt_sqrt, _ = curve_fit(sqrt_model, A_data, Rq_data)
Rq_pred_sqrt = sqrt_model(A_data, *popt_sqrt)
rmse_sqrt = np.sqrt(np.mean((Rq_data - Rq_pred_sqrt)**2))
print(f"\nSquare root model: Rq = {popt_sqrt[0]:.4f}*A**0.5 + {popt_sqrt[1]:.4f}")
print(f"RMSE: {rmse_sqrt:.4f}")

# Model 4: Logarithmic Rq = a*log(A) + b
def log_model(A, a, b):
    return a * np.log(A) + b

popt_log, _ = curve_fit(log_model, A_data, Rq_data)
Rq_pred_log = log_model(A_data, *popt_log)
rmse_log = np.sqrt(np.mean((Rq_data - Rq_pred_log)**2))
print(f"\nLogarithmic model: Rq = {popt_log[0]:.4f}*log(A) + {popt_log[1]:.4f}")
print(f"RMSE: {rmse_log:.4f}")

# Find best model
models = {
    'Linear': rmse_linear,
    'Power law': rmse_power,
    'Square root': rmse_sqrt,
    'Logarithmic': rmse_log
}

best_model = min(models, key=models.get)
print(f"\n\nBest model: {best_model} with RMSE = {models[best_model]:.4f}")

if best_model == 'Power law':
    print(f"\nFinal equation: Rb = {popt_power[0]:.6f} * A**{popt_power[1]:.6f}")
    print(f"Or in Python format: Rb = {popt_power[0]:.6f} * A**{popt_power[1]:.6f}")
elif best_model == 'Square root':
    print(f"\nFinal equation: Rb = {popt_sqrt[0]:.6f} * A**0.5 + {popt_sqrt[1]:.6f}")
    print(f"Or in Python format: Rb = {popt_sqrt[0]:.6f} * A**0.5 + {popt_sqrt[1]:.6f}")
elif best_model == 'Linear':
    print(f"\nFinal equation: Rb = {popt_linear[0]:.6f} * A + {popt_linear[1]:.6f}")
elif best_model == 'Logarithmic':
    print(f"\nFinal equation: Rb = {popt_log[0]:.6f} * math.log(A) + {popt_log[1]:.6f}")
