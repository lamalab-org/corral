
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
