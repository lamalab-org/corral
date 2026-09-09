# Spectra Elucidation Task Environment

This directory contains the spectra elucidation task environment for Corral. It provides organic structure-elucidation benchmarks where agents use simulated spectroscopy tools, chemical reference helpers, and structure validators to infer a target molecule and submit a SMILES answer.

## Setup

Create and activate the virtual environment from this directory:

```bash
cd tasks/spectra_elucidation
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

If you prefer not to activate the environment, use `uv run` to prefix the commands below.

## Node.js Dependencies

The `mass_spectrometry_spectra` and `hsqc_nmr_spectra` tools call local Node.js predictors. Python dependencies are installed by `uv sync`, but their JavaScript dependencies must be installed separately.

> **Why aren't these in `pyproject.toml`?** `pyproject.toml`/`uv` only manage Python packages — they cannot install the Node.js runtime or npm packages. Node must therefore be installed through your OS package manager (e.g. `brew install node`, `apt-get install nodejs`, or [nodejs.org](https://nodejs.org)), and the JavaScript packages installed with `npm` as shown below. The predictors resolve the `node` executable from `PATH` (via `shutil.which("node")`), so any Node ≥ 20 on the `PATH` works.

Install Node.js 20 or newer, then create a small npm project for the predictor:

```bash
cd tasks/spectra_elucidation
mkdir -p CORRAL_WORK_DIR/js
cd CORRAL_WORK_DIR/js
npm init -y
npm pkg set type=module
npm i --omit=dev isotopic-distribution nmr-processing openchemlib
```

Point the spectra environment at that npm project before running tasks:

```bash
cd tasks/spectra_elucidation
export CORRAL_SPECTRA_JS_DIR="$PWD/CORRAL_WORK_DIR/js"
```

If `CORRAL_SPECTRA_JS_DIR` is not set, the code falls back to `/srv/js`. The unit tests mock these predictors, so passing tests do not prove the runtime Node.js setup is available.

## Inspect The Environment Definitions

Build and list the spectra-elucidation environment definitions from this directory:

```bash
cd tasks/spectra_elucidation
source .venv/bin/activate
export CORRAL_WORK_DIR="$PWD/CORRAL_WORK_DIR"
export CORRAL_SPECTRA_JS_DIR="$PWD/CORRAL_WORK_DIR/js"
python -m spectra_elucidation.env --level 1
```

To run level 2:

```bash
python -m spectra_elucidation.env --level 2
```

To run the chained subtask benchmark instead:

```bash
python -m spectra_elucidation.env --level 1 --subtask_level True
```

The inspection command accepts these options:

- `--level`: Benchmark level to load. Levels are stored under `environments/level_1` and `environments/level_2`.
- `--subtask_level`: Set to `True` to load `subtasks_json/` instead of `tasks_json/` for the selected level.

The environment also reads these variables:

- `CORRAL_WORK_DIR`: Base directory for task workspaces and generated files. Setting it explicitly to `tasks/spectra_elucidation/CORRAL_WORK_DIR` keeps task output inside this task directory.
- `CORRAL_SPECTRA_JS_DIR`: Directory containing the npm project with `isotopic-distribution`, `nmr-processing`, and `openchemlib`.

## Task Layout

Task definitions live under:

- `environments/level_1/tasks_json/`: complete molecule-identification tasks with fragment-support tooling.
- `environments/level_1/subtasks_json/`: chained subtasks for the same level 1 molecules.
- `environments/level_2/tasks_json/`: complete molecule-identification tasks without fragment hints.
- `environments/level_2/subtasks_json/`: chained subtasks for the same level 2 molecules.

Each complete task asks the agent to determine the target molecule as a SMILES string. Each subtask group decomposes the same problem into intermediate spectroscopy and chemistry questions, ending with a final molecule submission.

## Tools

The environment exposes tools for:

- Formula and SMILES validation: `get_formula_from_smiles`, `validate_smiles`.
- Reference chemistry: `retrieve_dbe_formula`, `retrieve_isotope_distribution`, `retrieve_protons_shifts`, `retrieve_aromatic_protons_shifts`, `retrieve_carbon_shifts`.
- Spectra simulation: `carbon_nmr_spectra`, `proton_nmr_spectra`, `ir_spectra`, `hsqc_nmr_spectra`, `mass_spectrometry_spectra`, `simulate_spectra`.
- Structure search and enumeration: `search_by_smiles`, `obtain_isomers_from_molecular_formula`, `return_possible_fragments`.

Some tools call external spectroscopy or chemistry services, so benchmark runs need network access.

## Subtask Flow

In subtask mode, each molecule is represented by 10 linked subtasks:

| Step | Goal | Main Scoring Function |
|------|------|-----------------------|
| 1 | Molecular formula | `score_formula_match` |
| 2 | Double bond equivalents | `validate_dbe_consistency` |
| 3 | Isotopic-distribution elements | `score_isotopic_distribution` |
| 4 | Carbon symmetry classes | `score_num_carbon_symmetry_classes` |
| 5 | Hydrogen symmetry classes | `score_num_hydrogen_symmetry_classes` |
| 6 | Aromatic carbon count | `score_num_aromatic_carbons` |
| 7 | Methyl group count | `score_num_ch3_groups` |
| 8 | Carbonyl group count | `score_num_carbonyl_groups` |
| 9 | Molecular fragments | `score_molecule_fragments` |
| 10 | Complete molecule SMILES | `score_molecule` |

Subtasks 1 through 8 can be solved independently. Subtask 9 consumes the earlier intermediate answers, and subtask 10 consumes the full chain.

## Notes

- IR simulation uses the code in [cheminfo-py/xtbservice](https://github.com/cheminfo-py/xtbservice); deploy that service yourself if the public API is unavailable.

## Testing

Run the task test suite from this directory:

```bash
cd tasks/spectra_elucidation
source .venv/bin/activate
pytest
```

or without activating the virtual environment:

```bash
cd tasks/spectra_elucidation
uv run pytest
```
