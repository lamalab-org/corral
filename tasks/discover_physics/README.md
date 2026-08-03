# DiscoverPhysics Task Environment

This directory contains the DiscoverPhysics environment for Corral. Agents run
experiments against a hidden physics simulator (particle trajectories under an
undisclosed force/field law) and submit Python code implementing the law they
inferred.

It wraps the upstream [DiscoverPhysics](https://github.com/SampsonML/DiscoverPhysics)
benchmark ([leaderboard](https://sampsonml.github.io/DiscoverPhysicsLeaderboard/)):
the `physchool` (simulator) and `scienceagent` (per-world executors + evaluators)
packages are reused as git dependencies rather than reimplemented — only the
Corral-side glue (`tools.py`, `env.py`, `score.py`) and the task JSONs are new.

## Scope and known differences from upstream

- **11 worlds, not 13.** `gravity, yukawa, fractional, coulomb_easy,
  extra_dimensions, circle, three_species, dark_matter, ether, hubble,
  oscillator`. Upstream's `configs/bench.yml` also lists `force_geography` and
  `running_coupling`, but neither has an entry in `scienceagent.worlds.WORLDS`
  or the `run_discovery.py` CLI as of this writing, so there is no executor or
  evaluator to wrap for them.
- **Trajectory-MSE scoring only.** Upstream also LLM-judges the agent's prose
  explanation against a per-world rubric. That needs a live LLM call made
  *during scoring* (not just during the agent's run), which no other Corral
  task does yet, so it's out of scope here — see `score.py` for detail.
- **No interactive `<run_mse_fit>` step.** Upstream lets the agent mid-round
  fit its candidate law's free parameters against its own collected data. This
  environment only exposes `run_experiment`; submissions are expected to be
  fully self-contained (constants hardcoded from what the agent observed), and
  no free-parameter fitting happens at scoring time either.
- **44 tasks**: 11 worlds x 2 seeds (`0, 1`) x 2 noise levels (`0.0, 0.05`),
  mirroring the seed/noise sweep shape of upstream's `configs/bench.yml`.
  Regenerate via `scripts/generate_tasks.py` after editing `content.py` /
  `worlds.py`.

## Setup

Create and activate the virtual environment from this directory:

```bash
cd tasks/discover_physics
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

`scienceagent`/`physchool` pull in `jax`; the first `uv sync` will take a
while.

## Run The Server

```bash
cd tasks/discover_physics
source .venv/bin/activate
python src/discover_physics/env.py
```

Options:

- `tasks_json_path`: optional path to a task JSON file or directory. Defaults
  to `environments/level_1/tasks_json` (all 44 tasks).
- `--host` / `--port`: default to `CORRAL_HOST`/`CORRAL_PORT` env vars, or
  `0.0.0.0:8000`.

## See The Tasks

```bash
curl http://localhost:8000/tasks/
```

## Regenerating task JSONs

```bash
cd tasks/discover_physics
PYTHONPATH=src python3 scripts/generate_tasks.py
```

This only needs the standard library plus this package's own modules —
`worlds.py` imports `scienceagent`/`physchool` lazily (inside
`build_world`/`build_evaluator`), not at module level, so generation doesn't
require `jax` to be installed.

## Notes

- `run_experiment` is the only tool. It reaches the hidden simulator via
  Corral's `hidden_args` mechanism (see `env.py`'s `_expose_world_config`,
  following the same pattern as `spectra_elucidation`/`wetlab`): each task
  JSON's `world_config` (world name, engine, noise_std, noise_seed) is bound
  to the trial via `setup_fn` and silently injected into `run_experiment`
  calls — the agent never sees it directly, only the tool's `experiments`
  argument.
- Submissions are a self-contained Python source string defining
  `discovered_law(...)` with the world-specific signature given in each
  task's description (`submission_format`). Scoring reconstructs the same
  hidden simulator (noise disabled, matching upstream `Evaluator.evaluate`),
  runs the world's `Evaluator` subclass against fixed held-out test cases, and
  maps `mean_pos_error` onto a 0-1 score.
- The `description` field in upstream's `WORLDS` registry (e.g. gravity's
  names the Laplacian operator outright) is intentionally never surfaced —
  only the generic `mission` text and the per-topology instructions
  (`prompts.py`, vendored verbatim from `PhysicsSchool/prompts/*.md` since
  those aren't installed by `pip install physchool`) reach the agent.
