# MP RAG task

This task is a simple example of a task that uses the Material Project RAG Llamp tools.

## Example Environment with ChemBench scoring

Install environment

```bash
cd tasks/llamp_agent
uv venv --python 3.11.0
uv sync
```

Run the environment

```bash
cd tasks/llamp_agent/llamp_agent
python -m env
```

see the tasks

```bash
curl http://localhost:8000/tasks/
```
