# Retrosynthesis Task Environment

This directory contains the retrosynthesis task environment for Corral. It provides synthesis-planning benchmarks where agents query a reaction-template database, inspect candidate transformations, and submit precursor proposals for target molecules.

## Database Setup

The retrosynthesis environment uses two local, frozen databases:

- PostgreSQL with RDKit extensions for reaction templates.
- `retrosynthesis/data/buyables.sqlite` for commercial availability and prices.

Pricing lookups do not call supplier APIs or require API keys. The price of a
molecule is the frozen estimated cost in USD for 1 g, and a route's cost is the
sum of those prices for its leaf molecules.

### Quick Start

> **Note:** The Docker image is hosted on GHCR. If the package is private, authenticate first:
>
> ```bash
> echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
> ```
>
> A GitHub PAT with `read:packages` scope is sufficient. If the image is public, no token is required.

Run the setup script to pull and start the pre-seeded database:

```bash
cd tasks/retrosynthesis/docker
bash setup_retro_db.sh
```

On first startup, PostgreSQL initializes and restores both databases. Monitor progress with:

```bash
docker logs -f corral-retro-db
```

### Manual Pull

```bash
docker pull ghcr.io/lamalab-org/corral-retro-db:v1
docker run --name corral-retro-db \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -v corral-retro-db:/var/lib/postgresql/data \
  -d ghcr.io/lamalab-org/corral-retro-db:v1
```

Tasks 9 and 10 need the final three entries in the existing
`database_config/add_custom_reactions.py` list (Boc removal, quinazolinone
formation, and sulfone olefination). The published `v1` image predates these
entries. After restoration, with the Python environment installed and
`RETRO_DB_*` configured, import them from `tasks/retrosynthesis`:

```bash
PYTHONPATH=database_config python -c \
  'from add_custom_reactions import CUSTOM_REACTIONS, add_reactions_from_list; add_reactions_from_list(CUSTOM_REACTIONS[-3:])'
```

They have IDs 1914400–1914402 in the updated benchmark database. The importer
compares exact SMARTS when fingerprint hashes collide, preserving the older
Boc-removal template. Task data and level 2 hints are defined directly in the
existing level 2 and level 3 generators.

## Setup

Create and activate the virtual environment from this directory:

```bash
cd tasks/retrosynthesis
uv venv --python 3.11.0
source .venv/bin/activate
uv sync
```

The bundled buyables snapshot combines CoPriNet and ChemCost records with 20
repository-authored SMILES/price rows embedded directly by the database builder
for reference-route leaves those sources do not cover. The manual values are
frozen benchmark estimates, not live vendor quotes. SMILES are canonicalized
with RDKit while retaining stereochemistry and multicomponent structures.
Duplicate records are merged by taking the minimum USD/g price; the median,
observation count, and source names remain in the database for auditing.
Snapshot details and SHA256 hashes are recorded in
`retrosynthesis/data/buyables.metadata.json`.

To rebuild the snapshot from local source files:

```bash
uv run python scripts/build_buyables.py \
  --coprinet data/raw/test_set_PC.csv \
  --chemcost data/raw/chemcost.jsonl \
  --output retrosynthesis/data/buyables.sqlite
```

The 20 built-in manual rows are added automatically on every rebuild; no
separate task-price file or option is required.

An ASKCOS `buyables.json` or `buyables.json.gz` snapshot can optionally be
included with `--askcos`. To use a database outside the package, set
`RETRO_PRICE_DB_PATH=/absolute/path/to/buyables.sqlite`.

The level-3 price budgets for tasks 1–4, 9, and 10 are the reference-route total plus 10%,
rounded up to the nearest cent. Tests enforce both complete reference-leaf
coverage and this margin.

If you prefer not to activate the environment, use `uv run` to prefix the commands below.

## Inspect The Environment Definitions

Levels 2 and 3 each have ten full-route tasks. The two longest additions have
16 and 11 reaction nodes; level 2 includes descriptions for every reaction.
Task 9 uses [TREM2 Example 2](https://patents.google.com/patent/AU2023303060A1/en),
with the two N-debenzylations represented separately. Task 10 uses
[quinazoline Example 133](https://patents.google.com/patent/WO2026086892A1/en),
with aniline substitution and acetate hydrolysis represented separately.
These are reference-route counts, not proven minimum synthesis lengths.

Build and list the retrosynthesis environment definitions from this directory:

```bash
cd tasks/retrosynthesis
source .venv/bin/activate
python -m retrosynthesis.env --level 1
```

To run the subtask benchmark instead:

```bash
cd tasks/retrosynthesis
source .venv/bin/activate
python -m retrosynthesis.env --level 1 --subtask_level True
```

The inspection command accepts these options:

- `--level`: Benchmark level to load. Levels are stored under `environments/level_1`, `environments/level_2`, and `environments/level_3`.
- `--subtask_level`: Set to `True` to load `subtasks_json/` instead of `tasks_json/` for the selected level.

## Notes

- Before constructing definitions, the command validates database connectivity and schema. It fails early if the production database is unavailable or incomplete.
- Database credentials default to local PostgreSQL values, but can be overridden with `RETRO_DB_HOST`, `RETRO_DB_PORT`, `RETRO_DB_NAME`, `RETRO_DB_USER`, and `RETRO_DB_PASSWORD`.
- Pricing and availability use the bundled SQLite snapshot and never contact supplier services at runtime.
