# Add a New Environment

1. **Create environment directory**

   ```bash
   mkdir -p tasks/my_new_env/my_new_env
   cd tasks/my_new_env
   ```

2. **Create pyproject.toml**

   ```toml
   [project]
   name = "my_new_env"
   version = "0.1.0"
   dependencies = [
       "corral",
       # Add your specific dependencies
   ]
   ```

3. **Create tools**

   ```python
   # tasks/my_new_env/my_new_env/tools.py
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

   Note that the docstring has to be formatted correctly for the tool to be registered properly. This means it has to include a description of the parameters and return values as in the example above.

4. **Implement environment class**

   ```python
   # tasks/my_new_env/my_new_env/env.py
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

# Adding a New Agent

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



## 📋 Advanced Usage

# Tool Creation

#### Standard Tools

```python
from corral.backend.tool import tool


@tool
def calculate_molecular_weight(formula: str) -> float:
    """Calculate molecular weight from chemical formula.

    Args:
        formula: Chemical formula (e.g., 'H2O', 'CH4')

    Returns:
        Molecular weight in g/mol
    """
    # Implementation here
    pass
```

#### [Modal](https://modal.com) Tools (Cloud Execution)

```python
from corral.utils.modal import modal_tool, MODAL_TOOL_REGISTRY
from modal import Image


@modal_tool(app=app, image=Image.debian_slim().pip_install("rdkit"), memory=1024)
def complex_calculation(data: str) -> str:
    """Run computationally intensive task in the cloud."""
    # This runs in Modal's cloud environment
    pass


# Access the tool
tool_instance = MODAL_TOOL_REGISTRY["complex_calculation"]
```
