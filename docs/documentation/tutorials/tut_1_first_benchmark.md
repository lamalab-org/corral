# Tutorial 1: Your First `Corral` Benchmark

**What you'll learn**: By the end of this tutorial, you'll have created a simple environment, connected an agent, and run your first benchmark.

**Prerequisites**: Python 3.12+, basic Python knowledge

## Step 1: Set up your project

Create a new directory for this tutorial:

```bash
mkdir corral-tutorial
cd corral-tutorial
```

Create a virtual environment and install `Corral`:

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

## Step 2: Create your first environment

Create a file called `simple_env.py`:

```python
from corral.backend.env import Environment
from corral.backend.tool import tool
import os

BASE_WORK_DIR = os.getenv("CORRAL_WORK_DIR")


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: first number
        b: second number
    """
    return a + b


class SimpleEnvironment(Environment):
    def __init__(
        self,
        task_id: str,
        num1: float,
        num2: float,
        answer: float,
        base_work_dir=BASE_WORK_DIR,
    ):
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
        except (TypeError, ValueError):
            return 0.0
```

Notice that we created a tool using the `@tool` decorator and an environment that extends `Environment`.

## Step 3: Start the server

`CORRAL_WORK_DIR` set this environment variable. This is where all the files will be written by the agents if there is any writing required for the task.

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

## Step 4: Create the benchmark runner

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
`generate_report` would save a comprehensive report of the run
```python
result.generate_report("results.json")
```

Congratulations! You've run your first `Corral` benchmark.

## What you accomplished

- ✅ Created a simple environment with a tool
- ✅ Started a `Corral` server
- ✅ Connected an agent to the server
- ✅ Ran a benchmark and saw results

## Next steps

Try modifying the environment to use different numbers or add more tasks.

Once you have several tasks (or run `trials_per_task > 1` for `pass@k`), running them one at a time gets slow. See [How to run benchmarks concurrently (`abench`)](../how_tos/concurrent_benchmarking.md) to overlap trials under limits you control.

---
