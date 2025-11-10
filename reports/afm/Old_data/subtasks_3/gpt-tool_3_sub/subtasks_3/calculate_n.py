import math

# Given RMS roughness values
Rq1 = 3.186646007556558e-7
Rq2 = 6.038765919512273e-7
Rq3 = 8.695911883498758e-7

# Assumed scan areas
A1 = 1
A2 = 2
A3 = 3

# Calculate k from the first equation
k = Rq1 / (A1 ** 1)  # Since A1 = 1, k = Rq1

# Calculate n using the second equation
n2 = math.log(Rq2 / k) / math.log(A2)

# Calculate n using the third equation
n3 = math.log(Rq3 / k) / math.log(A3)

# Average n
n = (n2 + n3) / 2

print(n)
