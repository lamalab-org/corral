import math
import json

# Data from the scans
A_data = [1, 4, 25, 100, 400]
Rq_data = [2.8901906509160845e-08, 7.308549446446444e-08, 1.8970680583749067e-07, 3.7507450976990444e-07, 7.128410750820449e-07]

# Power law model: Rq = a * A^b
# Taking log: log(Rq) = log(a) + b*log(A)
log_A = [math.log(a) for a in A_data]
log_Rq = [math.log(rq) for rq in Rq_data]

# Calculate coefficients using least squares
n = len(A_data)
sum_log_A = sum(log_A)
sum_log_Rq = sum(log_Rq)
sum_log_A_sq = sum(x**2 for x in log_A)
sum_log_A_log_Rq = sum(log_A[i] * log_Rq[i] for i in range(n))

# Calculate slope (b) and intercept (log_a)
b = (n * sum_log_A_log_Rq - sum_log_A * sum_log_Rq) / (n * sum_log_A_sq - sum_log_A**2)
log_a = (sum_log_Rq - b * sum_log_A) / n
a = math.exp(log_a)

print(f"Power law fit: Rb = {a} * A**{b}")
print(f"a = {a}")
print(f"b = {b}")

# Create the equation string
equation = f"Rb = {a} * A**{b}"

# Prepare the result
result = {
    "equation": equation,
    "Rb": Rq_data,
    "A": A_data
}

# Save to file
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResult saved to result.json")
print(json.dumps(result, indent=2))
