# AFM benchmarks with Corral

Run agents against a Nanosurf atomic force microscope and score their saved scans. The benchmark has 20 independent tasks: ten in [level 1](environments/level_1/tasks_json/) requiring one image each, and ten in [level 2](environments/level_2/tasks_json/) requiring three images each.

From the repository root, with the workstation's AFM Python environment activated and API credentials configured, run these commands sequentially in **Windows Command Prompt (`cmd.exe`)** to execute the entire benchmark:

```bat
corral bench --agent tool-calling --environment afm --model openai/gpt-4o --env-kwargs "{\"level\": 1}" --sandbox local --trials 1 --max-parallel 1 --max-attempts 1 --output-dir .corral\afm-runs
corral bench --agent tool-calling --environment afm --model openai/gpt-4o --env-kwargs "{\"level\": 2}" --sandbox local --trials 1 --max-parallel 1 --max-attempts 1 --output-dir .corral\afm-runs
```

Keep `--sandbox local` for access to Nanosurf/COM. AFM enforces one task at a time within a process; use only one Corral process per instrument. These commands run one trial per task without automatic retries.

Change `--agent`, `--model`, or `--trials` as needed. To run a specific task, add its full ID, such as `--task afm_experiment_level_1_task_1`, with the matching level.

## Setup

Use the Windows workstation's existing Python 3.11+ AFM environment and compatible Nanosurf/COM installation. AFM dependencies are declared in [pyproject.toml](pyproject.toml). Install the current Corral framework into that same environment and configure credentials and scan storage:

```bat
cd /d C:\path\to\corral
python -m pip install -e .
set "OPENAI_API_KEY=your-api-key"
set "CORRAL_WORK_DIR=C:\AFM\workspaces"
```

Replace the checkout path and API key. Document retrieval requires `OPENAI_API_KEY` even when the agent uses another provider; configure that provider's credentials too. If `corral` is unavailable on PATH, use `python -m corral.cli`.

## Results and scans

Each benchmark command prints its output directory under `.corral\afm-runs`. Open `report.json` for scores and errors. Keep the complete run directory to retain execution histories and saved scan snapshots.

Scans go into a separate task workspace under `CORRAL_WORK_DIR`, whose full path appears in the agent's prompt. Full Windows and UNC paths remain supported in submissions. Follow each task's submission format and save `.nid` files inside that workspace; evaluation reads the saved snapshots even if the original files are moved or removed.
