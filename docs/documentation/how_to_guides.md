*1. Write how to create an environment*
2. How to create tools in environment
3. How to create tasks (take input from json) in environment
4. How to create scoring functions for the task
5. How to benchmark existing agent in this environment


6. How to create a new agent with some scaffolds
7. How to create a multi agent scaffold
8. How to run this new agent in existing environment or tasks


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

  Docstring Format: The docstring is critical! It must clearly describe what the tool does, its Args (parameters), and what it Returns. This information is used by Corral to automatically generate tool descriptions that agents (especially LLM-based agents) can understand and use.
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

Now define a task to this environment,

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

# 2. How to create a custom `Agent` in `Corral`

1. **Create agent file**

   ```python
   # src/corral/agents/my_agent.py
   from corral.agents import BaseAgent
   from corral import CorralRunner


   class MyAgent(BaseAgent):
       def __init__(self, model: str, **kwargs):
           super().__init__(model, **kwargs)
           # Add your agent-specific initialization

       def run(self, interface: CorralRouter, task_id: str) -> str:
           # Get task information
           guide = interface.get_task_guide(task_id)

           # Your agent logic here
           # Use interface.execute_tool() to call tools

           return "Your final answer"
   ```

2. **Add to agent registry**

   ```python
   # src/corral/agents/__init__.py
   from .my_agent import MyAgent

   __all__ = ["MyAgent", ...]
   ```

3. **Test your agent**

   ```python
   from corral.agents.my_agent import MyAgent
   from corral import CorralRunner, CorralRouter

   agent = MyAgent(model="gpt-4o")
   interface = CorralRouter("http://localhost:8000")
   runner = CorralRunner(interface, agent)

   result = runner.bench()
   ```
