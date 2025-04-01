import os
from pathlib import Path

import uvicorn
from loguru import logger
from score import (
    check_elastic_tensor,
)
from tools import (
    create_tools,
)

from corral.base import Environment
from corral.base_task import TaskDefinition, TaskEnvironment, TaskGroup
from corral.server import create_benchmark_server

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/temp")


def create_mp_rag_environments(work_dir: str = BASE_WORK_DIR) -> dict[str, Environment]:
    """Create environments for the MP-RAG benchmark tasks."""
    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)

    # Create task group with improved dependency management
    task_group = TaskGroup(
        group_id="mp_rag",
        tasks={
            # Easy tasks
            "NaCl_elastic_tensor": TaskDefinition(
                name="NaCl Elastic Tensor",
                description="What is the full elastic tensor of NaCl?",
                tools=["MP_elasticity_expert"],
                scoring_fn=check_elastic_tensor,
                scoring_inputs={"material": "NaCl"},
                submission_format={"answer": "elastic tensor string"},
                initial_input={},
            ),
            # "save_elastic_tensor": TaskDefinition(
            #     name="Save Elastic Tensor",
            #     description="Save the elastic tensor of NaCl as a text or json file.",
            #     tools=["write_file"],
            #     scoring_fn=check_elastic_tensor_file,
            #     scoring_inputs={"material": "NaCl"},
            #     submission_format={"answer": "/path/to/elastic_tensor.txt"},
            #     input_from_tasks=["NaCl_elastic_tensor"],
            #     initial_input={
            #         "work_dir": str(work_dir_path),
            #     },
            # ),
            # "retrieve_simple_bandgap": TaskDefinition(
            #     name="Retrieve Simple Bandgap",
            #     description="Retrieve the simple bandgap of BaTiO3.",
            #     tools=["MP_electronic_expert"],
            #     scoring_fn=check_simple_bandgap,
            #     scoring_inputs={"material": "BaTiO3"},
            #     submission_format={"answer": "bandgap in eV"},
            #     initial_input={},
            # ),
            # "retrieve_simple_formation_energy": TaskDefinition(
            #     name="Retrieve Simple Formation Energy",
            #     description="Retrieve the simple formation energy of Ag2O3.",
            #     tools=["MP_thermo_expert"],
            #     scoring_fn=check_simple_formation_energy,
            #     scoring_inputs={"material": "Ag2O3"},
            #     submission_format={"answer": "formation energy in eV"},
            #     initial_input={},
            # ),
            # # Medium tasks
            # "retrieve_bandgap_formation_energy": TaskDefinition(
            #     name="Retrieve Bandgap and Formation Energy",
            #     description="Return a material with a bandgap between 1.8 and 2.0 eV and a formation energy between -1.0 and -0.5 eV with less than 4 or less atoms in its unit cell.",
            #     tools=["MP_electronic_expert", "MP_thermo_expert"],
            #     scoring_fn=check_bandgap_formation_energy,
            #     scoring_inputs={
            #         "bandgap_low": 1.8,
            #         "bandgap_high": 2.0,
            #         "formation_energy_low": -1.0,
            #         "formation_energy_high": -0.5,
            #         "max_atoms": 4,
            #     },
            #     submission_format={"answer": "material_formula"},
            #     initial_input={},
            # ),
            # "retrieve_bandgap_bulk_modulus": TaskDefinition(
            #     name="Retrieve Bandgap and Bulk Modulus",
            #     description="Return a material with a bandgap between 1.8 and 2.0 eV and a bulk modulus between 100 and 200 GPa.",
            #     tools=["MP_electronic_expert", "MP_mechanical_expert"],
            #     scoring_fn=check_bandgap_bulk_modulus,
            #     scoring_inputs={
            #         "bandgap_low": 1.8,
            #         "bandgap_high": 2.0,
            #         "bulk_modulus_low": 100,
            #         "bulk_modulus_high": 200,
            #     },
            #     submission_format={"answer": "material_formula"},
            #     initial_input={},
            # ),
            # "write_cif_material": TaskDefinition(
            #     name="Write CIF for Material",
            #     description="Write the CIF file for the material with a bandgap 1.8 and 2.0 eV and a formation energy between -1.0 and -0.5 eV with less than 4 or less atoms in its unit cell.",
            #     tools=["MP_structure_retriever", "write_file"],
            #     scoring_fn=check_cif_material,
            #     scoring_inputs={
            #         "bandgap_low": 1.8,
            #         "bandgap_high": 2.0,
            #         "formation_energy_low": -1.0,
            #         "formation_energy_high": -0.5,
            #         "max_atoms": 4,
            #     },
            #     submission_format={"answer": "/path/to/material.cif"},
            #     input_from_tasks=["retrieve_bandgap_formation_energy"],
            #     initial_input={
            #         "work_dir": str(work_dir_path),
            #     },
            # ),
            # # Hard tasks
            # "retrieve_bandgap_formation_bulk_modulus": TaskDefinition(
            #     name="Retrieve Bandgap, Formation Energy, and Bulk Modulus",
            #     description="Return a material with a bandgap between 1.8 and 2.0 eV, a formation energy between -1.0 and -0.5 eV, and a bulk modulus between 100 and 200 GPa.",
            #     tools=[
            #         "MP_electronic_expert",
            #         "MP_thermo_expert",
            #         "MP_mechanical_expert",
            #     ],
            #     scoring_fn=check_bandgap_formation_bulk_modulus,
            #     scoring_inputs={
            #         "bandgap_low": 1.8,
            #         "bandgap_high": 2.0,
            #         "formation_energy_low": -1.0,
            #         "formation_energy_high": -0.5,
            #         "bulk_modulus_low": 100,
            #         "bulk_modulus_high": 200,
            #     },
            #     submission_format={"answer": "material_formula"},
            #     initial_input={},
            # ),
            # "write_cif_material_bulk_modulus": TaskDefinition(
            #     name="Write CIF for Material with Bandgap, Formation Energy, and Bulk Modulus",
            #     description="Write the CIF file for the retrieved material.",
            #     tools=["MP_structure_retriever", "write_file"],
            #     scoring_fn=check_cif_material,
            #     submission_format={"answer": "/path/to/material.cif"},
            #     input_from_tasks=["retrieve_bandgap_formation_bulk_modulus"],
            #     initial_input={
            #         "work_dir": str(work_dir_path),
            #     },
            # ),
            # "create_vacancy_defect": TaskDefinition(
            #     name="Create Vacancy Defect",
            #     description="Generate a 2x2x2 supercell of the material and introduce a single vacancy defect (remove one atom). Save the resulting structure with the defect as a CIF file.",
            #     tools=["MP_structure_retriever", "write_file"],
            #     scoring_fn=check_cif_material,
            #     submission_format={"answer": "/path/to/defect_structure.cif"},
            #     input_from_tasks=[
            #         "retrieve_bandgap_formation_energy",
            #         "write_cif_material_bulk_modulus",
            #     ],
            #     initial_input={
            #         "work_dir": str(work_dir_path),
            #         "defect_concentration": 1,
            #     },
            # ),
        },
    )

    # Print task dependencies for reference
    logger.info("\nTask Dependencies:")
    for task_id, deps in task_group.get_task_dependencies().items():
        logger.info(f"- {task_id}: depends on {deps}")

    # Print ordering of tasks
    ordered_tasks = task_group.get_ordered_tasks()
    logger.info("\nTask Execution Order:")
    for i, task_id in enumerate(ordered_tasks):
        logger.info(f"{i+1}. {task_id}")

    # Create all available tools
    # This assumes you have a function to create the tools
    available_tools = create_tools()

    # Create environments for all tasks
    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = TaskEnvironment(
            task_id=task_id,
            task_group=task_group,
            available_tools=available_tools,
        )

    return environments


if __name__ == "__main__":
    # Create all environments with file system tools
    environments = create_mp_rag_environments()

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"  Depends on: {env.current_task.input_from_tasks}")

    # Create and run server
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
