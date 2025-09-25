# Environment with Retrosynthesis Task

Install environment

```bash
cd tasks/retrosynthesis
uv venv --python 3.11.0
uv sync
```

Run the environment

```bash
cd tasks/retrosynthesis
python -m env
```

see the tasks

```bash
curl http://localhost:8000/tasks/
```
