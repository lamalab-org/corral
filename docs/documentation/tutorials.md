## Tutorial 1: Your First Corral Benchmark

**What you'll learn**: By the end of this tutorial, you'll have created a simple environment, connected an agent, and run your first benchmark.

**Prerequisites**: Python 3.12+, basic Python knowledge

### Step 1: Set up your project

Create a new directory for this tutorial:

```bash
mkdir corral-tutorial
cd corral-tutorial
```

Create a virtual environment and install Corral:

```bash
uv init
uv venv
source .venv/bin/activate
uv add https://github.com/lamalab-org/corral
```

You should see installation messages. When complete, verify the installation:

```bash
python -c "import corral; print('Corral installed!')"
```

You should see: `Corral installed!`

### Step 2: Create your first environment

Create a file called `simple_env.py`:

```python
from corral.backend.env import Environment
from corral.backend.tool import tool
import os

BASE_WORK_DIR = os.getenv("CORRAL_WORK_DIR")


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


class SimpleEnvironment(Environment):
    def __init__(self, task_id: str, num1: float, num2: float, answer: float):
        self.num1 = num1
        self.num2 = num2
        self.answer = answer
        super().__init__(task_id, base_work_dir)
        self.add_tool(add_numbers)

    def get_task_prompt(self) -> str:
        return f"What is {self.num1} + {self.num2}? Use the add_numbers tool."

    def score(self) -> float:
        if self.state.submitted_answer is None:
            return 0.0
        try:
            result = float(self.state.submitted_answer)
            return 1.0 if abs(result - self.answer) < 0.001 else 0.0
        except:
            return 0.0
```

Notice that we created a tool using the `@tool` decorator and an environment that extends `Environment`.

### Step 3: Start the server

Add this code to the bottom of `simple_env.py`:

```python
from corral.backend.server import create_benchmark_server
import uvicorn

environments = {
    "task_1": SimpleEnvironment("task_1", 10, 15, 25),
    "task_2": SimpleEnvironment("task_2", 7, 8, 15),
}

if __name__ == "__main__":
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Now run the server:

```bash
python simple_env.py
```

You should see output like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Great! Your server is running. Keep this terminal open.

### Step 4: Create the benchmark runner

Open a new terminal, activate the same virtual environment, and create `run_benchmark.py`:

```python
from corral.run import CorralRunner
from corral.router import CorralRouter
from corral.agents import ReActAgent

# Connect to the server
interface = CorralRouter(base_url="http://localhost:8000")

# Create an agent
agent = ReActAgent(model="openai/gpt-4o")

# Create the runner
runner = CorralRunner(interface, agent)

# Run the benchmark
result = runner.bench(trials_per_task=1)

# Print results
print(f"\nResults:")
print(f"Average Score: {result.average_score():.2f}")
print(f"Pass@1: {result.pass_at_k(1):.2f}")
```

Before running this, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Now run the benchmark:

```bash
python run_benchmark.py
```

You should see the agent working through the tasks. The output will show LLM calls and tool executions. Finally, you'll see:

```
Results:
Average Score: 1.00
Pass@1: 1.00
```

Congratulations! You've run your first Corral benchmark.

### What you accomplished

- ✅ Created a simple environment with a tool
- ✅ Started a Corral server
- ✅ Connected an agent to the server
- ✅ Ran a benchmark and saw results

### Next steps

Try modifying the environment to use different numbers or add more tasks.

---

## Tutorial 2: Building a Multi-Tool Environment

**What you'll learn**: Create an environment with multiple tools and see how agents use them together.

**Prerequisites**: Complete Tutorial 1

### Step 1: Create tools for a calculator

Create `calculator_env.py`:

```python
from corral.backend.env import Environment
from corral.backend.tool import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b
```

Notice we created three separate tools. Each tool does one thing clearly.

### Step 2: Create the environment

Add to `calculator_env.py`:

```python
class CalculatorEnvironment(Environment):
    def __init__(self, task_id: str, problem: str, answer: float):
        super().__init__(task_id)
        self.problem = problem
        self.answer = answer

        # Add all three tools
        self.add_tool(add)
        self.add_tool(multiply)
        self.add_tool(subtract)

    def get_task_prompt(self) -> str:
        return f"Calculate: {self.problem}\nSubmit your answer as a number."

    def score(self) -> float:
        if self.state.submitted_answer is None:
            return 0.0
        try:
            result = float(self.state.submitted_answer)
            return 1.0 if abs(result - self.answer) < 0.001 else 0.0
        except:
            return 0.0
```

The environment now has three tools available. The agent will need to choose which ones to use.

### Step 3: Create complex tasks

Add to `calculator_env.py`:

```python
from corral.backend.server import create_benchmark_server
import uvicorn

environments = {
    "simple_add": CalculatorEnvironment("simple_add", "10 + 5", 15),
    "two_step": CalculatorEnvironment("two_step", "(10 + 5) * 2", 30),
    "three_step": CalculatorEnvironment("three_step", "10 * 3 - 5", 25),
}

if __name__ == "__main__":
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Notice the tasks increase in complexity. The three-step task requires the agent to use multiple tools.

### Step 4: Run with verbose output

Start the server:

```bash
python calculator_env.py
```

In another terminal, create `run_verbose.py`:

```python
from corral.run import CorralRunner
from corral.router import CorralRouter
from corral.agents import ReActAgent

interface = CorralRouter(base_url="http://localhost:8000")
agent = ReActAgent(model="openai/gpt-4o", max_iterations=10)
runner = CorralRunner(interface, agent)

result = runner.bench(
    task_ids=["three_step"], trials_per_task=1, verbose=True  # Enable verbose output
)

print(f"\nScore: {result.average_score():.2f}")

# Check tool usage
for trial in result.all_results:
    stats = trial.tool_statistics
    print(f"Tools used: {stats['tools_used']}")
    print(f"Total tool calls: {stats['total_calls']}")
```

Run it:

```bash
python run_verbose.py
```

Watch the agent's reasoning process. You'll see it call `multiply` first, then `subtract`. The agent conversation is saved in JSON files in your working directory.

### What you accomplished

- ✅ Created multiple related tools
- ✅ Built tasks requiring multi-step reasoning
- ✅ Enabled verbose output to see agent thinking
- ✅ Examined tool usage statistics

---

## Tutorial 3: Creating a Custom Agent

**What you'll learn**: Build a simple custom agent that works with Corral environments.

**Prerequisites**: Complete Tutorials 1 and 2

### Step 1: Understand the agent structure

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

### Step 2: Get task information

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

### Step 3: Add the reasoning loop

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

### Step 4: Test your agent

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

### What you accomplished

- ✅ Created a custom agent from scratch
- ✅ Implemented the main reasoning loop
- ✅ Connected your agent to Corral environments
- ✅ Saw your agent solve tasks

### Experiment

Try modifying the agent to use tools. You'll need to:
1. Parse tool calls from the LLM response
2. Use `interface.execute_tool(task_id, tool_name, arguments)`
3. Add the tool result back to `self.messages`

---
