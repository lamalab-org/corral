import modal 

# input_file = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_2/ground_truth/task_3/task_3.in"

# # input_file = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_2/ground_truth/incorrect_file/input.in"

# # input_file = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/hybrid_relax.lmp"

# with open(input_file, "r") as file:
#     content = file.read()

# run_lammps = modal.Function.lookup("simagent", "run_lammps")

# # # print(run_lammps)

# output_files = ["dump.fe.xyz"]

# Call run_lammps with error handling
# try:
#     output = run_lammps.remote(content, output_files, directory_path = None)
#     print("Simulation successful!")
#     # print("Output:", output)
# except ValueError as ve:
#     print(ve)
# except Exception as e:
#     print("An unexpected error occurred:")
#     print(e)

run_bash_command = modal.Function.lookup("simagent", "run_bash_command")

# run_bash_command.remote("rm", ["-r", "/results/gpt_4o_mini"])

for dir in ["/results/gpt_4o_test_results"]:
    run_bash_command.remote("rm", ["-r", dir])
# run_bash_command.remote("rm", ["-r", "/results/gpt_4"])

# try:
#     # output = run_bash_command.remote("rm", ["-r", "/results/surface_energy"])
#     # print("Output:", output)
#     # output = run_bash_command.remote("rm", ["-r", "/results/energy_minimisation"])
#     # print("Output:", output)
#     # output = run_bash_command.remote("rm", ["-r", "/results/cohesive_energy"])
#     # print("Output:", output)
#     # output = run_bash_command.remote("rm", ["-r", "/results/kappa"])
#     # print("Output:", output)

#     # change_directory.remote("/results/")
#     output = run_bash_command.remote("ls", ["/potentials/EAM"], directory_path = None)
#     print(output)

# except ValueError as ve:
#     print(ve)
# except Exception as e:
#     print("An unexpected error occurred:")
#     print(e)

# check_directory = modal.Function.lookup("simagent", "check_directory")

# output = check_directory.remote("/results/")

# print(output)


# vol = modal.Volume.from_name("simulations")

# #data = vol.read_file("lattice_generation/task_2/input.in")

# data = b""
# for chunk in vol.read_file("lattice_generation/task_1/input.in"):
#     data += chunk

# print(data)






