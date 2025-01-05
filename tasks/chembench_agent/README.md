## Example Environment with ChemBench scoring

Install environment
```bash
cd tasks/chembench_agent
uv venv --python 3.11.0
uv sync
```


Run the environemnt
```bash
cd tasks/chembench_agent/chembench_agent
python -m env
```

see the tasks

```bash
curl http://localhost:8000/tasks/
```
