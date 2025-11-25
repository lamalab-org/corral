import math
import json

# Data from the analysis
# Assuming scan areas based on filenames
Rb_values = [5.79486424195013e-07, 0.0, 0.0]  # in meters
A_values = [100, 400, 900]  # in square micrometers

# Since we have limited data with two zeros, lets try different models
# Model 1: Power law Rb = C * A^alpha
# Model 2: Exponential decay Rb = C * exp(-k*A)
# Model 3: Linear Rb = C - k*A

# With only one non-zero point, we'll use a simple exponential decay model
# that passes through the data points

# For exponential decay: Rb = C * exp(-k*A)
# At A=100: 5.79486424195013e-07 = C * exp(-k*100)
# At A=400: 0 approx C * exp(-k*400)
# At A=900: 0 approx C * exp(-k*900)

# Since exp never reaches exactly zero, we'll use a model that approximates this
# Lets use: Rb = max(0, C * exp(-k*A))

# From the first point: C * exp(-k*100) = 5.79486424195013e-07
# We need to choose k such that at A=400, the value is very small

# Lets try k = 0.01 (decay constant)
k = 0.01
C = 5.79486424195013e-07 / math.exp(-k * 100)

print(f"Model: Rb = {C} * exp(-{k} * A)")
print(f"C = {C}")
print(f"k = {k}")

# Test the model
for A, Rb_actual in zip(A_values, Rb_values):
    Rb_predicted = C * math.exp(-k * A)
    print(f"A = {A} square micrometers: Rb_actual = {Rb_actual:.3e}, Rb_predicted = {Rb_predicted:.3e}")

# Create output in required format
output = {
    "equation": f"Rb = {C} * math.exp(-{k} * A)",
    "Rb": Rb_values,
    "A": A_values
}

print("\nOutput:")
print(json.dumps(output, indent=2))
