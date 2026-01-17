# Tutorial 3: Creating a Custom Agent

**What you'll learn**: Build a simple custom agent that works with `Corral` environments.

**Prerequisites**: Complete Tutorials 1 and 2

## Step 1: Understand the agent structure

Create `my_agent.py`:

```python
from corral.agents import BaseAgent
from corral.router import CorralRouter


class SimpleThinkAgent(BaseAgent):
    def __init__(self, model: str = "openai/gpt-4o", **kwargs):
        super().__init__(model=model, max_iterations=5, **kwargs)

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        history=None,
        task_prompt=None,
        examples=None,
        **kwargs
    ) -> str:
        # We'll implement this step by step
        pass
```

The `run` method is where your agent logic goes. It must return a string answer.

## Step 2: Get task information

Update the `run` method:

```python
def run(self, interface, task_id, **kwargs):
    # Get the task description and available tools
    task_guide = interface.get_task_guide(task_id)

    # Create initial message
    self.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": task_guide},
    ]

    print(f"Task guide received: {len(task_guide)} characters")

    # Continue in next step...
```

The `interface` gives you access to the environment. The `get_task_guide` method returns the task description with tool information.

## Step 3: Add the reasoning loop

Complete the `run` method:

```python
def run(self, interface, task_id, **kwargs):
    task_guide = interface.get_task_guide(task_id)

    self.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": task_guide},
    ]

    for iteration in range(self.max_iterations):
        print(f"\nIteration {iteration + 1}")

        # Get LLM response
        response = self.get_llm_response()
        content = response.content

        print(f"Agent thought: {content[:100]}...")

        # Check if agent provided an answer
        if "ANSWER:" in content:
            # Extract the answer
            answer = content.split("ANSWER:")[1].strip()
            print(f"Found answer: {answer}")
            return answer

        # Add response to history
        self.messages.append({"role": "assistant", "content": content})

    return "No answer found"
```

Notice we use `self.get_llm_response()` to call the LLM and check for "ANSWER:" to know when the agent is done.

## Step 4: Test your agent

Create `test_my_agent.py`:

```python
from corral.run import CorralRunner
from corral.router import CorralRouter
from my_agent import SimpleThinkAgent

# Make sure calculator_env.py is running!

interface = CorralRouter(base_url="http://localhost:8000")
agent = SimpleThinkAgent(model="openai/gpt-4o")
runner = CorralRunner(interface, agent)

result = runner.bench(task_ids=["simple_add"], trials_per_task=1)

print(f"\nYour agent scored: {result.average_score():.2f}")
```

Start the calculator environment in one terminal:

```bash
python calculator_env.py
```

Run your agent in another:

```bash
python test_my_agent.py
```

You'll see your agent's iteration-by-iteration thinking process.

## What you accomplished

- ✅ Created a custom agent from scratch
- ✅ Implemented the main reasoning loop
- ✅ Connected your agent to `Corral` environments
- ✅ Saw your agent solve tasks

### Experiment

Try modifying the agent to use tools. You'll need to:
1. Parse tool calls from the LLM response
2. Use `interface.execute_tool(task_id, tool_name, arguments)`
3. Add the tool result back to `self.messages`

---
