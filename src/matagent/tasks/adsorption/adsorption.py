from __future__ import annotations

import json
import subprocess
from typing import Dict, List, TypedDict
from typing import get_origin, get_args

import numpy as np

from schema import (
    BaseTypedDict,
    create_typed_dict,
    matches_typed_dict_type,
    AdsorbateOutput,
    AtomicStructure,
    DFTSettings,
    EnergyOutput,
    RelaxationOutput,
    SurfaceOutput,
)

# Task 1: Surface Generation
class SurfaceGenerationTask(BaseTypedDict):
    material_id: str
    miller_indices: List[int]
    min_depth: float
    min_width: float
    vacuum: float
    dft_settings: DFTSettings


# Task 2: Adsorbate Placement
class AdsorbatePlacementTask(BaseTypedDict):
    surface: AtomicStructure
    surface_atoms: List[int]
    molecule: str
    binding_atoms: List[int]
    site_types: List[str]


# Task 3: Structure Relaxation
class RelaxationTask(BaseTypedDict):
    structure: AtomicStructure
    dft_settings: DFTSettings
    force_threshold: float
    energy_threshold: float
    max_steps: int


# Task 4: Energy Calculation
class EnergyCalculationTask(BaseTypedDict):
    combined_structure: AtomicStructure
    surface_structure: AtomicStructure
    adsorbate_reference: str
    reference_energies: Dict[str, float]


class TaskFamily:
    standard_version = "0.5.0"

    required_environment_variables = ["VASP_PSEUDO_DIR", "MATERIALS_API_KEY"]

    @staticmethod
    def get_permissions(t: Task) -> list[str]:
        return ["full_internet"]

    @staticmethod
    def install() -> None:
        """Install required packages"""
        import os
        os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"

        subprocess.check_call(
            [
                "pip",
                "install",
                "pymatgen==2024.2.23",
                "ase==3.22.1",
                "catkit==0.5.4",
                "mpcontribs-client",
            ]
        )

    @staticmethod
    def start(t: Task) -> None:
        print("Starting task", t)

    # `get_tasks` returns a mapping from task names to arbitrary data that is specific to each task.
    # Tasks are not exposed to the agent. Agents only see the results of `get_instructions` below.
    @staticmethod
    def get_tasks() -> dict[str, Dict]:
        """Return all tasks in the workflow"""
        print("Returning all defined tasks")
        return {
            "surface_generation": create_typed_dict(SurfaceGenerationTask,
                material_id="mp-30",
                miller_indices=[1, 1, 1],
                min_depth=7.0,
                min_width=8.0,
                vacuum=12.0,
                dft_settings={
                    "xc": "PBE",
                    "encut": 400,
                    "kpts": [4, 4, 1],
                },
            ),
            "adsorbate_placement": create_typed_dict(AdsorbatePlacementTask,
                surface={},  # To be filled from previous task "structure.json"
                surface_atoms=[],
                molecule="CO",
                binding_atoms=[0],
                site_types=["top", "bridge", "hollow"],
            ),
            "relaxation": create_typed_dict(RelaxationTask,
                structure={},  # To be filled from previous task "combined.json"
                dft_settings={
                    "xc": "PBE",
                    "encut": 400,
                    "kpts": [4, 4, 1],
                },
                force_threshold=0.03,
                energy_threshold=1e-4,
                max_steps=200,
            ),
            "energy_calculation": create_typed_dict(EnergyCalculationTask,
                combined_structure={},  # To be filled from previous task "combined.json"
                surface_structure={},  # To be filled from previous task "surface.json"
                adsorbate_reference="CO",
                reference_energies={"CO_gas": -14.802, "clean_surface": -156.234},
            ),
        }

    @staticmethod
    def score_surface_generation(task: SurfaceGenerationTask, submission: str) -> float:
        """Score surface generation task"""
        try:
            result = json.loads(submission)
            score = 0.0

            # Check structure format
            if not all(k in result for k in SurfaceOutput.__annotations__):
                return score

            # Validate dimensions
            structure = result["structure"]
            cell = np.array(structure["cell"])
            lengths = np.linalg.norm(cell, axis=1)

            # Score components
            dim_score = 0.0
            if (
                lengths[0] >= task["min_width"]
                and lengths[1] >= task["min_width"]
                and lengths[2] >= task["min_depth"] + task["vacuum"]
            ):
                dim_score = 1.0

            # Surface atom identification score
            surf_score = 0.0
            if len(result["surface_atoms"]) > 0:
                # Validate surface atoms are at top of slab
                positions = np.array(structure["positions"])
                surf_z = positions[result["surface_atoms"], 2]
                if np.all(surf_z > np.percentile(positions[:, 2], 80)):
                    surf_score = 1.0

            # Energy convergence score
            energy_score = 0.0
            if "bulk_energy" in result and isinstance(result["bulk_energy"], float):
                energy_score = 1.0

            score = dim_score + surf_score + energy_score
            return score / 3.0

        except Exception as e:
            print(f"Error scoring surface generation: {str(e)}")
            return 0.0

    @staticmethod
    def score_adsorbate_placement(
        task: AdsorbatePlacementTask, submission: str
    ) -> float:
        """Score adsorbate placement task"""
        try:
            result = json.loads(submission)
            score = 0.0

            if not all(k in result for k in AdsorbateOutput.__annotations__):
                return score

            # Check multiple sites were identified
            site_score = 0.0
            if len(result["structures"]) >= 1:  # At least 1 unique sites
                site_score = 1

            # Validate binding distances
            # TODO
            distance_score = 0.0
            for struct in result["structures"]:
                positions = np.array(struct["positions"])
                # Calculate distances between binding atoms and surface
                distance_score += 1
            distance_score = distance_score / len(result["structures"])

            score = site_score + distance_score
            return score / 2.0

        except Exception as e:
            print(f"Error scoring adsorbate placement: {str(e)}")
            return 0.0

    @staticmethod
    def score_relaxation(task: RelaxationTask, submission: str) -> float:
        """Score relaxation task"""
        try:
            result = json.loads(submission)
            score = 0.0

            if not all(k in result for k in RelaxationOutput.__annotations__):
                return score

            # Convergence check
            if not result["converged"]:
                return score

            # Force convergence score
            force_score = 0.0
            max_force = np.max(np.abs(result["forces"]))
            if max_force <= task["force_threshold"]:
                force_score = 0.4
            elif max_force <= task["force_threshold"] * 1.5:
                force_score = 0.2

            # Energy convergence score
            energy_score = 0.0
            if len(result["energies"]) >= 2:
                energy_diff = abs(result["energies"][-1] - result["energies"][-2])
                if energy_diff <= task["energy_threshold"]:
                    energy_score = 0.4
                elif energy_diff <= task["energy_threshold"] * 10:
                    energy_score = 0.2

            # Efficiency score
            efficiency_score = 0.0
            if result["n_steps"] <= task["max_steps"]:
                efficiency_score = 0.2

            score = force_score + energy_score + efficiency_score
            return score

        except Exception as e:
            print(f"Error scoring relaxation: {str(e)}")
            return 0.0

    @staticmethod
    def score_energy_calculation(task: EnergyCalculationTask, submission: str) -> float:
        """Score energy calculation task"""
        try:
            result = json.loads(submission)
            score = 0.0

            if not all(k in result for k in EnergyOutput.__annotations__):
                return score

            # Energy components check
            components_score = 0.0
            required_components = {"combined", "surface", "adsorbate"}
            if all(c in result["energy_components"] for c in required_components):
                components_score = 1

            # Reference energy usage
            reference_score = 0.0
            if (
                abs(
                    result["energy_components"].get("adsorbate", 0)
                    - task["reference_energies"].get(
                        f"{task['adsorbate_reference']}_gas", 0
                    )
                )
                < 0.1
            ):
                reference_score = 1

            score = components_score + reference_score
            return score / 2.0

        except Exception as e:
            print(f"Error scoring energy calculation: {str(e)}")
            return 0.0

    @staticmethod
    def score(t: Dict, submission: str) -> float | None:
        """Overall workflow scoring function"""
        try:
            results = json.loads(submission)

            # Weight for each task
            weights = {
                "surface_generation": 0.25,
                "adsorbate_placement": 0.25,
                "relaxation": 0.25,
                "energy_calculation": 0.25,
            }

            total_score = 0.0
            completed_tasks = 0

            # Score each task if present
            if "surface_generation" in results:
                score = TaskFamily.score_surface_generation(
                    t["surface_generation"], json.dumps(results["surface_generation"])
                )
                total_score += score * weights["surface_generation"]
                completed_tasks += 1

            if "adsorbate_placement" in results:
                score = TaskFamily.score_adsorbate_placement(
                    t["adsorbate_placement"], json.dumps(results["adsorbate_placement"])
                )
                total_score += score * weights["adsorbate_placement"]
                completed_tasks += 1

            if "relaxation" in results:
                score = TaskFamily.score_relaxation(
                    t["relaxation"], json.dumps(results["relaxation"])
                )
                total_score += score * weights["relaxation"]
                completed_tasks += 1

            if "energy_calculation" in results:
                score = TaskFamily.score_energy_calculation(
                    t["energy_calculation"], json.dumps(results["energy_calculation"])
                )
                total_score += score * weights["energy_calculation"]
                completed_tasks += 1

            # Return None if no tasks completed
            if completed_tasks == 0:
                return None

            # Scale score by completion percentage
            completion_factor = completed_tasks / len(weights)
            final_score = total_score * completion_factor

            return final_score

        except Exception as e:
            print(f"Error in overall scoring: {str(e)}")
            return None

    @staticmethod
    def get_task_specific_instructions(task_name: str, task: Dict) -> str:
        """Get instructions for a specific task"""
        if task_name == "SurfaceGenerationTask":
            return f"""
            Generate a surface structure from the bulk material following these steps:

            1. Access the bulk structure with Materials Project ID: {task['material_id']}
            
            2. Create a surface with Miller indices {task['miller_indices']} meeting these requirements:
               - Minimum depth: {task['min_depth']} Å
               - Minimum width: {task['min_width']} Å
               - Vacuum spacing: {task['vacuum']} Å
            
            3. Identify surface atoms using these criteria:
               - Z-position within top 2 A of highest atom
               - Under-coordination relative to bulk
            
            4. Perform initial DFT calculation with settings:
               {json.dumps(task['dft_settings'], indent=2)}

            Submit your results as a JSON string with the following structure: 
            {{
                "structure": {{
                    "atomic_numbers": [...],
                    "positions": [...],
                    "cell": [...],
                    "pbc": [...]
                }},
                "surface_atoms": [list of surface atom indices],
                "area": float,                # Surface area in Å²
                "thickness": float,           # Slab thickness in Å
                "bulk_energy": float          # DFT energy of bulk structure
            }}
            """

        elif task_name == "AdsorbatePlacementTask":
            return f"""
            Place the adsorbate molecule on the provided surface following these steps:

            1. Process the input {task['molecule']} molecule
               - Binding atoms: {task['binding_atoms']}
               - Allowed site types: {task['site_types']}

            2. Identify potential binding sites on the surface
               - Use provided surface atoms list: {len(task['surface_atoms'])} atoms
               - Consider symmetrically distinct sites only

            3. Generate initial structures for each site type:
               - Maintain reasonable binding distances
               - Orient molecule appropriately
               - Consider multiple orientations per site

            Submit your results as a JSON string with:
            {{
                "structures": [
                    {{
                        "atomic_numbers": [...],
                        "positions": [...],
                        "cell": [...],
                        "pbc": [...]
                    }},
                    ...  # One for each site
                ],
                "binding_sites": [
                    {{
                        "type": "site type",
                        "coordinates": [x, y, z],
                        "surface_atoms": [indices]
                    }},
                    ...
                ],
                "site_types": ["top", "bridge", etc.]
            }}
            """

        elif task_name == "RelaxationTask":
            return f"""
            Perform structure relaxation with these specifications:

            1. Use provided DFT settings:
               {json.dumps(task['dft_settings'], indent=2)}

            2. Meet convergence criteria:
               - Force threshold: {task['force_threshold']} eV/Å
               - Energy threshold: {task['energy_threshold']} eV
               - Maximum steps: {task['max_steps']}

            3. Constraints:
               - Allow surface atoms to relax
               - Fix subsurface atoms
               - Allow full adsorbate relaxation

            Submit results as:
            {{
                "final_structure": {{
                    "atomic_numbers": [...],
                    "positions": [...],
                    "cell": [...],
                    "pbc": [...]
                }},
                "converged": bool,
                "forces": [[fx, fy, fz], ...],  # Final forces
                "energies": [e1, e2, ...],      # Energy trajectory
                "n_steps": int                   # Number of steps taken
            }}
            """

        elif task_name == "EnergyCalculationTask":
            return f"""
            Calculate adsorption energy using:

            1. Energy Components:
               - Combined system (provided structure)
               - Clean surface (provided)
               - Reference state: {task['adsorbate_reference']}

            2. Use reference energies:
               {json.dumps(task['reference_energies'], indent=2)}

            3. Calculate:
               - Adsorption energy: E_ads = E_combined - E_surface - E_ref
               - Binding energy relative to gas phase
               - Check stability

            Submit results as:
            {{
                "adsorption_energy": float,
                "binding_energy": float,
                "energy_components": {{
                    "combined": float,
                    "surface": float,
                    "adsorbate": float
                }},
                "is_stable": bool
            }}
            """

        return "Unknown task"

    # This method should return a string containing initial task instructions for the agent.
    @staticmethod
    def get_instructions(t: TypedDict) -> str:
        """Main instruction method that handles both single and multiple tasks"""
        # Determine if this is a single task or multiple tasks
        if matches_typed_dict_type(
            t,
            (
                SurfaceGenerationTask,
                AdsorbatePlacementTask,
                RelaxationTask,
                EnergyCalculationTask,
            ),
        ):
            # Single task case
            task_name = t["_type_marker"]
            print('Single task', task_name)
            return TaskFamily.get_task_specific_instructions(
                task_name, t
            )
        print('All tasks!')
        # Multiple tasks case - create workflow instructions
        workflow_instructions = """
        Complete the following adsorption energy calculation workflow.
        Submit results for each completed task before proceeding to the next.
        
        Overall Workflow:
        ----------------
        """

        for task_name, task_data in t.items():
            workflow_instructions += f"\n{task_name.upper()}:\n"
            workflow_instructions += (
                TaskFamily.get_task_specific_instructions(
                    task_name, task_data
                )
            )
            workflow_instructions += "\n" + "-" * 80 + "\n"

        workflow_instructions += """
        Submit your final results as a JSON string containing results from all completed tasks:
        {
            "surface_generation": { ... },  # Results from surface generation
            "adsorbate_placement": { ... }, # Results from adsorbate placement
            "relaxation": { ... },          # Results from relaxation
            "energy_calculation": { ... }   # Results from energy calculation
        }
        
        Each task's results should follow the format specified in its individual instructions.
        You may submit partial results if not all tasks are completed.
        """

        return workflow_instructions

    @staticmethod
    def get_next_task(previous_results: Dict) -> str | None:
        """Helper method to determine the next task in the workflow"""
        task_order = [
            "surface_generation",
            "adsorbate_placement",
            "relaxation",
            "energy_calculation",
        ]

        # Find first incomplete task
        for task in task_order:
            if task not in previous_results:
                return task

        return None

    @staticmethod
    def validate_task_dependencies(task_name: str, previous_results: Dict) -> bool:
        """Validate that prerequisites are met for a given task"""
        dependencies = {
            "surface_generation": [],
            "adsorbate_placement": ["surface_generation"],
            "relaxation": ["surface_generation", "adsorbate_placement"],
            "energy_calculation": [
                "surface_generation",
                "adsorbate_placement",
                "relaxation",
            ],
        }

        required_tasks = dependencies.get(task_name, [])
        return all(task in previous_results for task in required_tasks)


# Example creation of task objects

# Example Cu(111) surface structure
# cu111_surface = AtomicStructure(
#     atomic_numbers=[29, 29, 29, 29, 29, 29, 29, 29, 29],  # Cu atoms
#     positions=[
#         [0.0, 0.0, 0.0],
#         [2.56, 0.0, 0.0],
#         [1.28, 2.22, 0.0],
#         [0.0, 0.0, 2.08],
#         [2.56, 0.0, 2.08],
#         [1.28, 2.22, 2.08],
#         [0.0, 0.0, 4.16],
#         [2.56, 0.0, 4.16],
#         [1.28, 2.22, 4.16],
#     ],
#     cell=[[5.12, 0.0, 0.0], [-2.56, 4.44, 0.0], [0.0, 0.0, 20.0]],
#     pbc=[True, True, True],
# )

# # Example CO+Cu(111) combined structure
# co_cu111_structure = AtomicStructure(
#     atomic_numbers=[29, 29, 29, 29, 29, 29, 29, 29, 29, 6, 8],  # Cu + CO
#     positions=[
#         [0.0, 0.0, 0.0],
#         [2.56, 0.0, 0.0],
#         [1.28, 2.22, 0.0],
#         [0.0, 0.0, 2.08],
#         [2.56, 0.0, 2.08],
#         [1.28, 2.22, 2.08],
#         [0.0, 0.0, 4.16],
#         [2.56, 0.0, 4.16],
#         [1.28, 2.22, 4.16],
#         [0.0, 0.0, 6.16],  # C atom
#         [0.0, 0.0, 7.31],  # O atom
#     ],
#     cell=[[5.12, 0.0, 0.0], [-2.56, 4.44, 0.0], [0.0, 0.0, 20.0]],
#     pbc=[True, True, True],
# )

# tasks_co_cu111 = {
#     "surface_generation": SurfaceGenerationTask(
#         material_id="mp-id of cu",
#         miller_indices=[1, 1, 1],
#         min_depth=7.0,
#         min_width=8.0,
#         vacuum=12.0,
#         dft_settings=DFTSettings(xc="PBE", encut=400, kpts=[4, 4, 1]),
#     ),
#     "adsorbate_placement": AdsorbatePlacementTask(
#         surface=cu111_surface,
#         surface_atoms=[6, 7, 8],  # Top layer atoms
#         molecule="CO",
#         binding_atoms=[0],  # C atom binds
#         site_types=["top", "bridge", "fcc", "hcp"],
#     ),
#     "relaxation": RelaxationTask(
#         structure=co_cu111_structure,
#         dft_settings=DFTSettings(xc="PBE", encut=400, kpts=[4, 4, 1]),
#         force_threshold=0.03,
#         energy_threshold=1e-4,
#         max_steps=200,
#     ),
#     "energy_calculation": EnergyCalculationTask(
#         combined_structure=co_cu111_structure,
#         surface_structure=cu111_surface,
#         adsorbate_reference="CO",
#         reference_energies={"CO_gas": -14.802, "clean_surface": -156.234},
#     ),
# }
