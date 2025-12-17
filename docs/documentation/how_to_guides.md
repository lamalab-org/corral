# 1. How to create a custom `Environment` in `Corral`

Every Corral environment is treated as a Python package. It would have standalone dependency and can be pip installed. Let us start by creating a python repository for an example environment.



`1. Create a python project directory`

All the files related to the `Environment`  would be inside this directory.

   ```bash
   mkdir my_new_env
   cd my_new_env
   ```


`2.  Initialize a python project`


You need a pyproject.toml file to define your environment's name, version, and its dependencies. Make sure to include corral as a dependency. We will use `uv` to manage our python dependency and project.


   ```bash
   uv init
   ```

This command would initialize a project by creating a `pyproject.toml` file

   ```toml
   [project]
    name = "new-env"
    version = "0.1.0"
    description = "Add your description here"
    readme = "README.md"
    requires-python = ">=3.12"
    dependencies = []
   ```

[project] section defines metadata for your Python package.

**name:** This should be a unique identifier for your environment.

**dependencies:** Lists the required Python packages. corral is essential, and you'll add any other libraries your tools or environment logic might use here.

Also remember to create a virtual environment for this project (corral environment)

   ```bash
   uv venv
   ```

Activate the virtual env with: `source .venv/bin/activate`


`3.  Adding project dependencies`

   ```bash
   uv add corral
   ```
This command would add corral to your dependency and install it in your virtual environment.


`4. Create tools for the Environment`

To help the agent solve the task in this `Environment` we can also add tools to the environment.

   ```python
   # my_new_env/tools.py
   from corral.backend.tool import tool


   @tool
   def my_custom_tool(input_param: str) -> str:
       """Description of what the tool does.

       Args:
           input_param: Description of the parameter

       Returns:
           Description of the return value
       """
       # Your tool implementation
       return f"Processed: {input_param}"
   ```


The `@tool` decorator registers the function as an `Corral-tool` for agents.


/// info
    open: True

  Docstring Format: The docstring is critical because it is used by Corral to automatically generate tool descriptions that agents (especially LLM-based agents) can understand and use. It must clearly describe what the tool does, its Args (parameters), and what it Returns.
///



`5. Implement environment class`


Corral has an `Environment` abstraction which takes care of all the internal logic related to execution of tools upon calling management of the state of environment etc. User can build their environment on top of this.

Now we'll define the core logic of your environment, including how tasks are set up. Corral allows you to define tasks directly in your env.py or load them dynamically, for example, from a JSON file. Loading from JSON is highly recommended for managing multiple tasks.


```python
# my_new_env/env.py
from corral.backend import Environment
from corral.backend.server import create_benchmark_server


class MyEnvironment(Environment):
    def __init__(self, task_id: str, problem: str, answer: str):
        self.problem = problem
        self.correct_answer = answer
        super().__init__(task_id)

        # Add your tools
        self.add_tool(my_custom_tool)

    def get_task_prompt(self) -> str:
        return f"Solve this problem: {self.problem}"

    def score(self) -> float:
        if self.state.submitted_answer is None:
            return 0.0
        return 1.0 if self.state.submitted_answer == self.correct_answer else 0.0
```

Now define a task for this environment,

```python
# Define your tasks
environments = {
    "task_1": MyEnvironment("task_1", "Problem 1", "Answer 1"),
    "task_2": MyEnvironment("task_2", "Problem 2", "Answer 2"),
}


# Create server
if __name__ == "__main__":
    app = create_benchmark_server(environments)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Here `task_1` is the task id which is unique to the task. `Problem 1`, `Answer 1` represent the problem statement and answer. Note here we can have complex strategies like a scoring function instead of string etc.

`create_benchmark_server` sets up and runs the `corral_server` with your newly defined environments, making them accessible via `http://localhost:8000.`

---

# 2. How to create a custom `Agent` in `Corral`


By extending the `BaseAgent` class in corral, you can implement unique reasoning processes, interaction patterns to create different agent scaffolds. This abstract class provides essential functionalities and a standardized interface, handling:
- Calls to Large Language Models (LLMs) via `LiteLLM`
- Standardizes the loading and filling of various prompt types using `PromptStore`
- Provides a mechanism to inject custom logic at various points in the agent's lifecycle using `Hooks`
- Result extraction.

Similar to environments, agents reside in their own Python package.

**`1. Initialize agent project`**

   ```bash
   mkdir my_custom_agent
   cd my_custom_agent
   uv init
   uv venv
   source .venv/bin/activate
   uv add corral
   ```


**`2. Agent Implementations`**

Now, create your agent's main Python file (e.g., agent.py) and start defining your class.

   ```python
   # src/corral/agents/my_agent.py
   from corral.agents import BaseAgent
   from corral.agents.prompt_utils import create_prompt
   from corral.agents.utils import LiteLLMMessage


   class MyAgent(BaseAgent):
       """
       A custom agent that implements a simplified "Think then Answer" approach.
       """

       def __init__(self, model: str, **kwargs):
           super().__init__(model, **kwargs)
           # Add your agent-specific initialization

       def run(self, interface: CorralRouter, task_id: str) -> str:
           # Get task information
           task_guide = interface.get_task_guide(task_id)

           # prepare the prompt for the agent now that you have task info

           #    self.messages = create_prompt(
           #        user_prompt=self.user_prompt,
           #        task_guide=task_guide,
           #    )

           # Your agent logic here
           # Use interface.execute_tool() to call tools

           # self.get_llm_response() Calls the LLM with the current self.messages

           # your parsing logic

           return "Your final answer"
   ```
The `__init()__` initialize the BaseAgent with common parameters.

The `.run()` is an abstract method in BaseAgent and must be implemented.

`CorralRouter` (is the interface agent has with the environment). Use it to,

  - `interface.get_task_guide(task_id)`: Get the initial problem description.

  - `interface.get_tool_definitions(task_id)`: Get the names and descriptions of tools the environment provides (formatted for LLMs).

  - `interface.execute_tool(task_id, tool_name, arguments)`: Call a tool in the environment and get its result.

`create_prompt()` helper function from `BaseAgent` assembles the initial prompt, including system prompt, user prompt, task guide, and examples. Users can provide their own implementation of this method if needed.


`self.get_llm_response()`, calls the LLM with the current self.messages and potentially available tools.

---


# 3. How to prepare tasks and scoring function (Material Science example)

This section demonstrates how to define a task and implementing reward function, using a material science scenario.
Our example task will involve an agent retrieving a material structure from the `Materials Project` database and then attempting to create a specific crystal slab from it. The scoring function will use `pymatgen` to verify the generated slab.


**`1. Prepare Task Json`**

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

**`2. Write scoring functions`**

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


**`3. Task and Scoring in environment`**

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

Now lets load our task into corral server.

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


# Create the corral server
if __name__ == "__main__":
    app = create_benchmark_server(all_material_environments)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

This implementation provides a clear and robust way to manage tasks and integrate external scoring functions within a straightforward Environment class structure.
