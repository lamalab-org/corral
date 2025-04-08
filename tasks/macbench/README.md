# Example Environment with MaCBench scoring

Install environment

```bash
cd tasks/macbench
uv venv --python 3.11.0
uv sync
```

Run the environment

```bash
cd tasks/macbench/macbench
python -m env
```

see the tasks

```bash
curl http://localhost:8000/tasks/
```
