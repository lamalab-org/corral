from jinja2 import Template
import json

# Define lattice structures and minimizers
lattices = [
    {"type": "face centered cubic", "element": "Aluminum", "lattice_constant": 4.05, "potential" : "Embedded Atom Method (EAM)"},
    {"type": "body centered cubic", "element": "Iron", "lattice_constant": 2.87, "potential" : "Embedded Atom Method (EAM)"},
    {"type": "hexagonal closed packed", "element": "Magnesium", "lattice_constant": 3.21, "potential" : "Embedded Atom Method (EAM)"},
    {"type": "diamond", "element": "carbon", "lattice_constant": 3.57, "potential" : "tersoff"},
]

minimisers = ["conjugate_gradient", "newton", "steepest descent", "lbfgs"]

# Main task template
task_template = Template("""
{
    "id": "task_{{ task_id }}",
    "name": "energy_minimisation",
    "keywords": [
        "molecular_dynamics_simulations",
        "lammps",
        "{{ lattice.type }}",
        "{{ lattice.element }}",
        "{{ minimiser }}"
    ],
    "metrics": ["binary"],
    "input": [
        {
            "prompt": "Simulate a {{ lattice.type }} lattice structure for {{ lattice.element }} using LAMMPS with a lattice constant of {{ lattice.lattice_constant }} angstrom. The simulation should use metal units, define a 5x5x5 simulation box, and set periodic boundary conditions to True. Perform energy minimisation using the {{ lattice.potential }} potential and use {{ minimiser }} as the minimiser, with force tolerance of 1e-10, energy tolerance of 1e-10, maximum iterations of 1e+5 and maximum evaluations of 1e+5. As the final output, give the total energy after minimisation in eV units."
        }
    ],
    "output": [
        {
            "type": "numerical",
            "target": "-1679.99",
            "threshold": "1e-1"
        }
    ],
    "tools": ["run_lammps", "run_bash_command"],
    "subtasks": [
        {
            "id": "subtask_{{ task_id }}_1",
            "name": "initialisation",
            "keywords": ["molecular_dynamics_simulations", "lammps"],
            "metrics": ["binary"],
            "input": [
                {
                    "prompt": "Simulate a {{ lattice.type }} lattice structure for {{ lattice.element }} using LAMMPS with a lattice constant of {{ lattice.lattice_constant }} angstrom and do not perform energy minimisation. The simulation should use metal units, define a 5x5x5 simulation box and set periodic boundary conditions to True. Save this structure using write_data command in a file in xyz format. As the final output, give the absolute path of the structure file."
                }
            ],
            "output": [
                {
                    "type": "structure",
                    "target": "/path/to/initial_structure_{{ task_id }}.xyz",
                    "threshold": "0"
                }
            ],
            "tools": ["run_lammps", "run_bash_command"]
        },
        {
            "id": "subtask_{{ task_id }}_2",
            "name": "simulation_settings",
            "keywords": ["molecular_dynamics_simulations", "lammps"],
            "metrics": ["binary"],
            "input": [
                {
                    "prompt": "You are given the absolute path of the structure file of {{ lattice.element }} {{ lattice.type }} lattice. Read that structure file and perform energy minimisation using (lattice.potential) potential with {{ minimiser }} method in LAMMPS, with force tolerance of 1e-10, energy tolerance of 1e-10, maximum iterations of 1e+5 and maximum evaluations of 1e+5. After energy minimisation, save the final energy after minimisation in a text file, and as the final output, give the absolute path of the text file.",
                    "input_from_task": ["subtask_{{ task_id }}_1"]
                }
            ],
            "output": [
                {
                    "type": "text_file",
                    "target": "/path/to/total_energy_{{ task_id }}.txt",
                    "threshold": "1e-1"
                }
            ],
            "tools": ["run_lammps", "run_bash_command"]
        },
        {
            "id": "subtask_{{ task_id }}_3",
            "name": "final_output",
            "keywords": ["molecular_dynamics_simulations", "lammps"],
            "metrics": ["binary"],
            "input": [
                {
                    "prompt": "You are given the absolute path of the text file which contains total energy after minimisation of the {{ lattice.element }} {{ lattice.type }} lattice. As the final output, give its content.",
                    "input_from_task": ["subtask_{{ task_id }}_2"]
                }
            ],
            "output": [
                {
                    "target": "-1679.99",
                    "threshold": "1e-1"
                }
            ],
            "tools": ["run_bash_command"]
        }
    ]
}
""")

# Generate and save task JSON files
task_id = 1
for lattice in lattices:
    for minimiser in minimisers:
        task_json = task_template.render(task_id=task_id, lattice=lattice, minimiser=minimiser)
        filename = f"task_{task_id}.json"
        with open(filename, "w") as f:
            f.write(task_json)
        print(f"Generated {filename}")
        task_id += 1
