# Calculate the average Rb
Rb_values = [3.1925371294018595e-07, 6.052114876060461e-07, 8.711400831928807e-07]
average_Rb = sum(Rb_values) / len(Rb_values)

# Print the average Rb
equation = f"Rb = {average_Rb}"
print({"equation": equation, "Rb": Rb_values, "A": [16384, 16384, 16384]})
