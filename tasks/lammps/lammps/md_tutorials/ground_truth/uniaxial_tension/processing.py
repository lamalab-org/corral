import matplotlib.pyplot as plt
import numpy as np
import csv

path = '/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_5/mat-agent-bench/tasks/lammps/lammps/ground_truth/grain_boundary_fracture/data.Fe_110_sig3_fracture.txt'

def extract_max_stress(file_path) -> float:
    """
    Extracts the maximum tensile stress (in GPa) along the x-direction
    from a space-delimited stress-strain data file with a header.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        float: Maximum stress_xx value in GPa.

    Raises:
        Exception: If any error occurs while reading or processing the file.
    """
    import csv

    try:
        max_stress = float('-inf')

        with open(file_path, newline='') as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                # Remove empty strings caused by multiple spaces
                row_clean = [val for val in row if val.strip()]
                if len(row_clean) < 2:
                    continue  # skip if row is too short
                try:
                    stress_xx = float(row_clean[1])  # second column = stress_xx
                    if stress_xx > max_stress:
                        max_stress = stress_xx
                except ValueError:
                    continue  # skip malformed data

        return max_stress

    except Exception as e:
        raise Exception(f"Error while processing file '{file_path}': {e}")
    
print("Maximum tensile stress in GPa:", extract_max_stress(path))

# results = []
# with open('Al_tens_100.def1.txt',newline='') as file:
#     reader = csv.reader(file, delimiter=' ')
#     next(reader)  # Skip header row.
#     for row in reader:
#         row2 = [float(i) for i in row]
#         results.append(row2)
#         print(row2)

# results2 = np.transpose(results)

# print("maximum tensile stress: ", max(results2[1]))

# plt.plot(results2[0],results2[1], '-or', label='Stress in X', lw=2, markersize = 5, mec = 'r', mfc = 'r')
# plt.plot(results2[0],results2[2], '-ob', label='Stress in Y', lw=2, markersize = 5, mec = 'b', mfc = 'b')
# plt.plot(results2[0],results2[3], '-og', label='Stress in Z', lw=2, markersize = 5, mec = 'g', mfc = 'g')
# plt.xlabel('Strain',fontsize=16)
# plt.ylabel('Stress (GPa)',fontsize=16)
# plt.title('Stress versus Strain',fontsize=16)
# plt.legend(fontsize=12)
# plt.gca().set_aspect('auto')
# plt.ylim(0,10)
# plt.show()