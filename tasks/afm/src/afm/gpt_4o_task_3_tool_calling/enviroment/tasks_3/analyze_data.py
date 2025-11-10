import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

# Load data
A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Rq = np.array([1.5, 2.12, 2.6, 3.0, 3.35, 3.67, 3.96, 4.24, 4.5, 4.74])

# Define a function for the relationship
# Assuming a logarithmic relationship: Rq = a * log(A) + b
def model(A, a, b):
    return a * np.log(A) + b

# Fit the model to the data
params, covariance = curve_fit(model, A, Rq)

# Extract parameters
a, b = params

# Generate fitted Rq values
Rq_fitted = model(A, a, b)

# Plot the data and the fit
plt.scatter(A, Rq, label='Data')
plt.plot(A, Rq_fitted, label=f'Fit: Rq = {a:.2f} * log(A) + {b:.2f}', color='red')
plt.xlabel('A (Scan Area)')
plt.ylabel('Rq (RMS Roughness)')
plt.legend()
plt.title('Rq vs. A')
plt.savefig('fit_plot.png')

# Save the equation and data
result = {
    "equation": f"Rb = {a:.2f} * math.log(A) + {b:.2f}",
    "Rb": Rq.tolist(),
    "A": A.tolist()
}

with open('result.json', 'w') as f:
    json.dump(result, f)
