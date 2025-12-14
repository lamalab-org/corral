1. Write how to create an environment
2. How to create tools in environment
3. How to create tasks (take input from json) in environment
4. How to create scoring functions for the task
5. How to benchmark existing agent in this environment


6. How to create a new agent with some scaffolds
7. How to create a multi agent scaffold
8. How to run this new agent in existing environment or tasks


# Add a new Environment

1. **Create environment directory**

   ```bash
   mkdir my_new_env
   cd my_new_env
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

   Note that the docstring has to be formatted correctly for the tool to be registered properly. This means it has to include a description of the parameters and return values as in the example above.

4. **Implement environment class**

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

# Add a new Agent

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
