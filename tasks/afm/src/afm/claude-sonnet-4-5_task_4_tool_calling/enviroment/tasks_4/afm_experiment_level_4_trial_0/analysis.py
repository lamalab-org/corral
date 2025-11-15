import math

# Data from the scans
# Areas in µm²
A_data = [1, 4, 25, 100, 400]
# Rq values in meters
Rq_data = [2.890e-08, 7.309e-08, 1.897e-07, 3.751e-07, 7.128e-07]

# Try power law model: Rq = a * A^b
log_A = []
log_Rq = []

for a in A_data:
    log_A.append(math.log(a))

for rq in Rq_data:
    log_Rq.append(math.log(rq))

# Calculate coefficients using least squares
n = len(A_data)
sum_log_A = sum(log_A)
sum_log_Rq = sum(log_Rq)
sum_log_A_sq = 0
for x in log_A:
    sum_log_A_sq += x**2

sum_log_A_log_Rq = 0
for i in range(n):
    sum_log_A_log_Rq += log_A[i] * log_Rq[i]

b = (n * sum_log_A_log_Rq - sum_log_A * sum_log_Rq) / (n * sum_log_A_sq - sum_log_A**2)
log_a = (sum_log_Rq - b * sum_log_A) / n
a = math.exp(log_a)

# Write results
with open('results.txt', 'w') as f:
    f.write(f"Power law fit:\n")
    f.write(f"a = {a}\n")
    f.write(f"b = {b}\n")
    f.write(f"Equation: Rb = {a} * A**{b}\n")
    f.write(f"\nData:\n")
    f.write(f"A (µm²): {A_data}\n")
    f.write(f"Rq (m): {Rq_data}\n")

print("Analysis complete")
