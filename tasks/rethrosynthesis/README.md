# Environment with Rethrosynthesis Task

Install environment

```bash
cd tasks/rethrosynthesis
uv venv --python 3.11.0
uv sync
```

Run the environment

```bash
cd tasks/rethrosynthesis
python -m env
```

see the tasks

```bash
curl http://localhost:8000/tasks/
```
