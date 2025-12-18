# How to create a custom `Agent` in `Corral`


By extending the `BaseAgent` class in `Corral`, you can implement unique reasoning processes, interaction patterns to create different agent scaffolds. This abstract class provides essential functionalities and a standardized interface, handling:
- Calls to Large Language Models (LLMs) via `LiteLLM`
- Standardizes the loading and filling of various prompt types using `PromptStore`
- Provides a mechanism to inject custom logic at various points in the agent's lifecycle using `Hooks`
- Result extraction.

Similar to environments, agents reside in their own Python package.

## **`1. Initialize agent project`**

   ```bash
   mkdir my_custom_agent
   cd my_custom_agent
   uv init
   uv venv
   source .venv/bin/activate
   uv add corral
   ```


## **`2. Agent Implementations`**

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
The `__init__()` method initializes the BaseAgent with common parameters.

The `.run()` is an abstract method in BaseAgent and must be implemented.

`CorralRouter` (is the interface agent has with the environment). Use it to,

  - `interface.get_task_guide(task_id)`: Get the initial problem description.

  - `interface.get_tool_definitions(task_id)`: Get the names and descriptions of tools the environment provides (formatted for LLMs).

  - `interface.execute_tool(task_id, tool_name, arguments)`: Call a tool in the environment and get its result.

`create_prompt()` helper function from `BaseAgent` assembles the initial prompt, including system prompt, user prompt, task guide, and examples. Users can provide their own implementation of this method if needed.


`self.get_llm_response()`, calls the LLM with the current self.messages and potentially available tools.

---
