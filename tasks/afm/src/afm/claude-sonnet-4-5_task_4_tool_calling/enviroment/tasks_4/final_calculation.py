import math

# Data
A = [1, 4, 25, 100, 400]
Rq = [2.8901906509160845e-08, 7.308549446446444e-08, 1.8970680583749067e-07, 3.7507450976990444e-07, 7.128410750820449e-07]

# Log transform
log_A = [math.log(a) for a in A]
log_Rq = [math.log(rq) for rq in Rq]

print("log_A:", log_A)
print("log_Rq:", log_Rq)

# Sums
n = 5
sum_log_A = sum(log_A)
sum_log_Rq = sum(log_Rq)
sum_log_A_sq = sum(x**2 for x in log_A)
sum_log_A_log_Rq = sum(log_A[i] * log_Rq[i] for i in range(n))

print(f"\nn = {n}")
print(f"sum_log_A = {sum_log_A}")
print(f"sum_log_Rq = {sum_log_Rq}")
print(f"sum_log_A_sq = {sum_log_A_sq}")
print(f"sum_log_A_log_Rq = {sum_log_A_log_Rq}")

# Calculate b
b_numerator = n * sum_log_A_log_Rq - sum_log_A * sum_log_Rq
b_denominator = n * sum_log_A_sq - sum_log_A**2
b = b_numerator / b_denominator

print(f"\nb_numerator = {b_numerator}")
print(f"b_denominator = {b_denominator}")
print(f"b = {b}")

# Calculate a
log_a = (sum_log_Rq - b * sum_log_A) / n
a = math.exp(log_a)

print(f"\nlog_a = {log_a}")
print(f"a = {a}")

print(f"\n\nFinal equation: Rb = {a} * A**{b}")

# Verification
print("\nVerification:")
for i in range(n):
    predicted = a * (A[i] ** b)
    error = abs(predicted - Rq[i]) / Rq[i] * 100
    print(f"A={A[i]:3d}: Measured={Rq[i]:.3e}, Predicted={predicted:.3e}, Error={error:.2f}%")
