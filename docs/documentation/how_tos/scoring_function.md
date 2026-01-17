
# How to prepare tasks and scoring function (Material Science example)

This section demonstrates how to define a task and implementing reward function, using a material science scenario.
Our example task will involve an agent retrieving a material structure from the `Materials Project` database and then attempting to create a specific crystal slab from it. The scoring function will use `pymatgen` to verify the generated slab.


## **`1. Prepare Task Json`**

Create a `material_tasks.json` file. This file will define the parameters for each material science task.

```json
{
    "task_1": {
        "input_params": {
            "mp_id": "mp-149",
            "miller_index": [
                1,
                0,
                0
            ],
            "min_slab_size": 10.0,
            "vacuum_size": 10.0,
            "num_layers": 3
        },
        "problem_description": "Retrieve the structure for Silicon (mp-149) from Materials Project. Create a (100) slab with at least 3 layers and a 10 Å vacuum layer. Submit the CIF string of the slab."
    },
    "task_2": {
        "input_params": {
            "mp_id": "mp-66",
            "miller_index": [
                1,
                1,
                1
            ],
            "min_slab_size": 8.0,
            "vacuum_size": 12.0,
            "num_layers": 4
        },
        "problem_description": "Retrieve the structure for Diamond (mp-66). Create a (111) slab with at least 4 layers and a 12 Å vacuum layer. Submit the CIF string of the slab."
    }
}

```

- Each key (e.g., "silicon_slab_task") is a unique ID for a task.
- The values are dictionaries containing task-specific parameters (`mp_id`, `miller_index`,`min_slab_size`, `vacuum_size`, `num_layers`,`problem_description`)

## **`2. Write scoring functions`**

```python
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.structure_matcher import StructureMatcher
from loguru import logger


def pymatgen_score(submitted_slab_cif: str, metadata: dict) -> float:
    """
    Scores the agent's submitted slab using pymatgen to verify its validity.
    """
    if submitted_slab_cif is None:
        return 0.0  # No slab submitted

    try:
        agent_slab = Structure.from_str(submitted_slab_cif, fmt="cif")
    except Exception as e:
        logger.info(f"Scoring error: Agent submitted invalid CIF string: {e}")
        return 0.0  # Invalid CIF submitted

    # --- Generate the target reference slab for comparison ---
    try:
        with MPRester() as mpr:
            bulk_structure = mpr.get_structure_by_material_id(metadata["mp_id"])

        slab_gen = SlabGenerator(
            initial_structure=bulk_structure,
            miller_index=metadata["miller_index"],
            min_slab_size=metadata["min_slab_size"],
            min_vacuum_size=metadata["vacuum_size"],
        )
        reference_slab = slab_gen.get_slab()
    except Exception as e:
        logger.info(f"Scoring error: Could not generate reference slab: {e}")
        return 0.0

    score_value = 0.0
    matcher = StructureMatcher(ltol=0.1, stol=0.1, angle_tol=5)  # Default tolerances

    # 1. Structural similarity (comparing primitive cell of slabs)
    if matcher.fit(
        agent_slab.get_primitive_structure(), reference_slab.get_primitive_structure()
    ):
        logger.info("Scoring: Primitive structures match. (+0.4)")
        score_value += 0.5

    return round(score_value, 2)


def pymatgen_simple_score(submitted_slab_cif: str, metadata: dict) -> float:
    """
    Scores the agent's submitted slab using pymatgen to verify its validity.
    """
    if submitted_slab_cif is None:
        return 0.0  # No slab submitted

    try:
        agent_slab = Structure.from_str(submitted_slab_cif, fmt="cif")
        logger.info(f"slab :{agent_slab}")
    except Exception as e:
        logger.info(f"Scoring error: Agent submitted invalid CIF string: {e}")
        return 0.0  # Invalid CIF submitted

    return 1
```
The scoring function here takes as input the answer submitted by the agent and a dict with metadata for creating ground truth, it then compares and returns a reward. User can write any such complex, binary or floating scoring functions.


## **`3. Task and Scoring in environment`**

Let us also see how an environment can be defined for such task and scoring use case.

```python
from corral.backend.env import Environment
from corral.backend.server import create_benchmark_server
import uvicorn

BASE_WORK_DIR = os.getenv("CORRAL_WORK_DIR")


class SimpleMaterialSlabEnvironment(Environment):
    def __init__(
        self,
        task_id: str,
        input_params: dict[str, Any],  # This now holds all the specific task parameters
        problem_description: str,
        base_work_dir=BASE_WORK_DIR,
    ):
        self.input_params = input_params
        self.problem_description = problem_description

        super().__init__(task_id, base_work_dir)

        # Add your custom material science tools
        self.add_tool()

    def get_task_prompt(self) -> str:
        """
        Returns the initial prompt for the agent, using the problem_description from JSON.
        """
        return (
            f"{self.problem_description}\n"
            "Use the tool to retrieve the bulk structure, "
            "then use a tool to create the slab with the specified parameters. "
            "Finally, submit final answer with the CIF string of the generated slab,"
            "in the format FINAL_ANSWER: cif_string"
        )

    def score(self) -> float:
        """
        Calls the standalone custom_slab_scoring_function with the agent's output
        and the task's metadata (input_params).
        """
        return pymatgen_score(
            submitted_slab_cif=self.state.submitted_answer,
            metadata=self.input_params,  # Pass all input_params as metadata for ground truth
        )
```


///info

Here is some custom tools for this task

```python
import json
import os

from corral.backend.tool import tool
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Structure

if "MP_API_KEY" not in os.environ:
    load_dotenv("../.env")


def load_structure(
    bulk_structure_path_or_string: str, from_path: bool = False
) -> Structure:
    """Loads pymatgen structure from CIF string."""
    from pymatgen.core import Structure

    try:
        if from_path:
            return Structure.from_file(bulk_structure_path_or_string)
        else:
            from pymatgen.core import Structure

            return Structure.from_str(bulk_structure_path_or_string, fmt="cif")
    except Exception as e:
        raise ValueError(f"Failed to parse structure CIF: {e}") from e


def get_bulk_polymorphs_data_func(composition: str) -> str:
    """
    Query the Materials Project database to find polymorphs for a given composition.
    This function returns a JSON string containing polymorph data including MP IDs, structures (CIF), energies above hull, formation_energy_per_atom, band gaps, densities, volumes, number of sites,symmetry, and stability. The results are sorted by energy above hull.

    Args:
        composition: Chemical composition (e.g., 'TiO2')
        api_key: Materials Project API key (optional if set in environment)

    Returns:
        JSON string containing polymorph data including MP IDs, structures (CIF), energies above hull, formation_energy_per_atom, band gaps, densities, volumes, number of sites, symmetry, and stability. (sorted by energy above hull)
    """

    # Use provided API key or get from environment
    mp_api_key = os.getenv("MP_API_KEY")
    if not mp_api_key:
        raise ValueError(
            "Materials Project API key not provided and not found in environment"
        )

    with MPRester(mp_api_key) as mpr:
        # Query for materials with the given composition
        docs = mpr.materials.summary.search(
            formula=composition,
            fields=[
                "material_id",
                "structure",
                "energy_above_hull",
                "formation_energy_per_atom",
                "band_gap",
                "density",
                "volume",
                "nsites",
                "symmetry",
                "is_stable",
            ],
        )

        # Convert structures to CIF for easy storage
        polymorph_data = []
        for doc in docs:
            structure_cif = doc.structure.to(fmt="cif")

            polymorph_data.append(
                {
                    "material_id": doc.material_id,
                    "cif": structure_cif,
                    "energy_above_hull": doc.energy_above_hull,
                    "formation_energy_per_atom": doc.formation_energy_per_atom,
                    "band_gap": doc.band_gap,
                    "density": doc.density,
                    "volume": doc.volume,
                    "nsites": doc.nsites,
                    "space_group": doc.symmetry.symbol,
                    "is_stable": doc.is_stable,
                }
            )

        # Sort by energy above hull (stability)
        polymorph_data = sorted(polymorph_data, key=lambda x: x["energy_above_hull"])

        return json.dumps(polymorph_data, indent=2)


@tool
def get_structure_from_mp_text(mp_id: str) -> str:
    """[BRIEF] Retrieve a pymatgen structure from Materials Project using its API and return CIF content as text. [/BRIEF]

    Args:
        mp_id: [ARGS_BRIEF] Materials Project identifier string. [/ARGS_BRIEF]
               [ARGS_DETAILED] The unique identifier used by Materials Project to catalog materials.
               Should be in the format "mp-XXXXX" where XXXXX is a numerical ID.
               This ID corresponds to a specific material entry in the Materials Project database. [/ARGS_DETAILED]
               [ARGS_SYNTACTIC] "mp-" followed by digits (e.g., "mp-149", "mp-20066") [/ARGS_SYNTACTIC]
               [ARGS_EXAMPLES] "mp-149" (Silicon), "mp-20066" (CO2), "mp-2" (Li) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] CIF content string containing the crystal structure data. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A properly formatted CIF (Crystallographic Information File) string containing all necessary information about the crystal structure including lattice parameters, atomic positions, space group, and symmetry operations.
             This format is widely compatible with crystallographic software and other structure analysis tools. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] "\n_chemical_formula_structural Si\n_cell_length_a 5.468..." [/RETURNS_EXAMPLES]


    """

    with MPRester(os.getenv("MP_API_KEY")) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=[str(mp_id)], fields=["structure"]
        )
        structure = docs[0].structure

    return structure.to(fmt="cif")


@tool
def create_slab_from_structure_text(
    structure_cif: str,
    miller_index: tuple = (1, 1, 1),
    min_slab_size: int = 12,
    min_vacuum_size: int = 5,
    primitive: bool = True,
) -> str:
    """[BRIEF] Create a surface slab from a bulk crystal structure with specified Miller indices and dimensions. [/BRIEF]

    [DETAILED] This tool generates a surface slab by cleaving a bulk crystal structure along a specified crystallographic plane.
    It creates a two-dimensional periodic surface model suitable for surface chemistry calculations, catalysis studies, and adsorption analysis.
    The tool automatically handles the creation of vacuum space above the surface and ensures proper termination of the crystal structure.
    This is essential for computational surface science studies. [/DETAILED]

    Args:
        structure_cif: [ARGS_BRIEF] CIF content string of the bulk crystal structure. [/ARGS_BRIEF]
                      [ARGS_DETAILED] A properly formatted CIF string containing the bulk crystal structure data including lattice parameters, atomic positions, and space group information.
                      This structure will be cleaved to create the surface. [/ARGS_DETAILED]
                      [ARGS_SYNTACTIC] string in valid CIF syntax [/ARGS_SYNTACTIC]
                      [ARGS_EXAMPLES] "# generated using pymatgen\ndata_Si\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.83996459\n_cell_length_b   3.83996459\n_cell_length_c   18.81190774\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   120.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   Si\n_chemical_formula_sum   Si8\n_cell_volume   240.22483885\n_cell_formula_units_Z   8\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Si  Si0  1  0.83333333  0.41666667  0.10416667  1.0\n  Si  Si1  1  0.50000000  0.75000000  0.06250000  1.0\n  Si  Si2  1  0.16666667  0.08333333  0.27083333  1.0\n  Si  Si3  1  0.83333333  0.41666667  0.22916667  1.0\n  Si  Si4  1  0.50000000  0.75000000  0.43750000  1.0\n  Si  Si5  1  0.16666667  0.08333333  0.39583333  1.0\n  Si  Si6  1  0.83333333  0.41666667  0.60416667  1.0\n  Si  Si7  1  0.50000000  0.75000000  0.56250000  1.0\n" [/ARGS_EXAMPLES]
        miller_index: [ARGS_BRIEF] Miller indices for the surface plane. Defaults to (1,1,1). [/ARGS_BRIEF]
                     [ARGS_DETAILED] A tuple of three integers specifying the crystallographic plane along which the structure will be cleaved.
                     These indices define the surface orientation and determine the atomic arrangement at the surface.
                     Common choices include (1,1,1), (1,0,0), and (1,1,0) for different surface orientations. [/ARGS_DETAILED]
                     [ARGS_SYNTACTIC] tuple of three integers (h, k, l) [/ARGS_SYNTACTIC]
                     [ARGS_EXAMPLES] (1,1,1), (1,0,0), (1,1,0) [/ARGS_EXAMPLES]
        min_slab_size: [ARGS_BRIEF] Minimum slab thickness in Angstroms. Defaults to 12. [/ARGS_BRIEF]
                      [ARGS_DETAILED] The minimum thickness of the slab in the direction perpendicular to the surface plane.
                      This parameter ensures that the slab has sufficient bulk-like character in the center while exposing the desired surface.
                      Larger values provide more accurate representation of bulk properties but increase computational cost. [/ARGS_DETAILED]
                      [ARGS_SYNTACTIC] positive integer representing thickness in Angstroms [/ARGS_SYNTACTIC]
                      [ARGS_EXAMPLES] 12, 15, 8[/ARGS_EXAMPLES]
        min_vacuum_size: [ARGS_BRIEF] Minimum vacuum spacing in Angstroms. Defaults to 5. [/ARGS_BRIEF]
                        [ARGS_DETAILED] The minimum vacuum space above the surface to prevent interactions between periodic images in surface calculations.
                        This parameter is crucial for accurate surface energy calculations and adsorption studies.
                        Larger values reduce spurious interactions but increase computational requirements. [/ARGS_DETAILED]
                        [ARGS_SYNTACTIC] positive integer representing vacuum thickness in Angstroms [/ARGS_SYNTACTIC]
                        [ARGS_EXAMPLES] 5 (minimal), 10 (standard), 15 (large) [/ARGS_EXAMPLES]
        primitive: [ARGS_BRIEF] Whether to create a primitive cell slab. Defaults to True. [/ARGS_BRIEF]
                  [ARGS_DETAILED] Controls whether to use the primitive cell or conventional cell for slab generation.
                  Primitive cells have the minimum number of atoms while maintaining the essential symmetry, leading to smaller, more efficient computational models.
                  Setting to False uses the conventional cell which may be larger but more intuitive. [/ARGS_DETAILED]
                  [ARGS_SYNTACTIC] boolean value (True/False) [/ARGS_SYNTACTIC]
                  [ARGS_EXAMPLES] True, False [/ARGS_EXAMPLES]
    """
    from pymatgen.core import Structure
    from pymatgen.core.surface import SlabGenerator

    # Load the structure from CIF string
    structure = Structure.from_str(structure_cif, fmt="cif")

    # Create the slab
    slab_gen = SlabGenerator(
        structure, miller_index, min_slab_size, min_vacuum_size, primitive=primitive
    )
    slab = slab_gen.get_slab()
    slab = slab.get_orthogonal_c_slab().get_sorted_structure()

    return slab.to(fmt="cif")
```

///

Now let's load our task into the `Corral` server.

```python
def load_environments_from_json(file_path: str) -> Dict[str, Environment]:
    """Loads environment instances from a JSON task definition file."""
    environments = {}
    with open(file_path, "r") as f:
        tasks_data = json.load(f)

    for task_id, params in tasks_data.items():
        environments[task_id] = SimpleMaterialSlabEnvironment(
            task_id=task_id,
            input_params=params["input_params"],  # Pass the entire 'input_params' dict
            problem_description=params["problem_description"],
        )
    return environments


# Load environments for all defined tasks
all_material_environments = load_environments_from_json("material_tasks.json")


# Create the Corral server
if __name__ == "__main__":
    app = create_benchmark_server(all_material_environments)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

This implementation provides a clear and robust way to manage tasks and integrate external scoring functions within a straightforward Environment class structure.
