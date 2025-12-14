# Corral


A platform for development, deployment and evaluation of environments and agents.



Corral is built and maintained with the following foundational principles in mind:






- Reproducibility: Facilitating consistent and repeatable research outcomes.

- Control: Providing precise control over all environmental and agent variables.

- Observability: Ensuring consistent monitoring and recording of all system changes.

- Efficiency and Scalability: Designed to be lightweight while capable of scaling to complex demands.

- Simplicity and Usability: Engineered for ease of understanding and straightforward application.

- Flexibility: Avoiding implementation constraints to empower diverse approaches.



<!--
1. Framework of environment and agents should be compatible for active research, ie it should be reproducible, should allow control of anything that is a variables and should be able to monitor and record any change in the entire system with consistency.
2. It should be light-weight and scalable at the same time.
3. It should be easy to grasp and straightforward to apply.
4. It doesn’t impose implementation constraints.  -->




Corral has value for researchers, engineers, managers and teachers looking to use







# 🏗️ Available Environments

The framework includes several pre-built environments:

| Environment | Description |
|-------------|-------------|
| `samplemath` | Basic mathematical operations |
| `spectra_elu_easy` | Spectroscopy data analysis |
| `md_simulations` | Molecular dynamics setup |
| `catalyst` | Catalysis research tasks |
| `afm` | Atomic force microscopy |
| `md_tutorials` | MD tutorial completion |


# 🤖 Available Agents

The framework includes several built-in agent types:

### ReActAgent

Uses the ReAct (Reasoning and Acting) framework for step-by-step problem solving.

```python
from corral.agents import ReActAgent

agent = ReActAgent(
    model="gpt-4o",  # or "claude-3-5-sonnet-20241022" or any other model litellm supports
    temperature=0.1,
    max_iterations=10,
)
```

### ToolCallingAgent

Uses native function calling from LLM providers to solve tasks by leveraging built-in tool/function calling capabilities.

```python
from corral.agents import ToolCallingAgent

agent = ToolCallingAgent(
    model="gpt-4o",  # or "claude-3-5-sonnet-20241022" or any other model LiteLLM supports
    temperature=0.0,
    max_iterations=10,
)
```

### LLMPlanner

Uses hierarchical planning with high-level planning and low-level execution delegation to other agents.

```python
from corral.agents import LLMPlanner

agent = LLMPlanner(model="gpt-4o", temperature=0.1, max_iterations=5)
```


## 💾 Checkpoint System

The framework automatically saves checkpoints during benchmark runs.

Checkpoints are automatically searched and loaded when resuming interrupted runs.

## 🔧 Contributing



### Development Setup

1. **Install development dependencies**

   ```bash
   uv pip install -e .
   ```

2. **Install pre-commit hooks with commitizen commits**

   ```bash
   pre-commit install --hook-type commit-msg --hook-type pre-push
   ```


### Environment Configuration

For environments requiring file I/O:

```bash
export CORRAL_FS_PROTOCOL=local
export BASE_IO_PATH=/path/to/work/directory
```

### Evaluation Metrics

The framework provides comprehensive evaluation metrics:

```python
result = runner.bench(trials_per_task=10, k_values=[1, 3, 5])

# Access detailed results
print(f"Total score: {result.total_score}")
print(f"Pass@1: {result.pass_at_k[1]}")
print(f"Pass@3: {result.pass_at_k[3]}")
print(f"Average trials: {result.average_trials}")

# Per-task analysis
for task_id, task_result in result.task_results.items():
    print(f"Task {task_id}: {task_result.success_rate:.2f} success rate")
```

## 🤝 Community

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/lamalab-org/mat-agent-bench/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/lamalab-org/mat-agent-bench/discussions)
- **Contributing**: See our [Contributing Guide](CONTRIBUTING.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
