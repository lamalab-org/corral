# Retrosynthesis Task Environment

This directory contains the retrosynthesis task environment for Corral. It provides synthesis-planning benchmarks where agents query a reaction-template database, inspect candidate transformations, and submit precursor proposals for target molecules.

## Database Setup

The retrosynthesis environment requires a PostgreSQL database with RDKit extensions containing two databases: `reactions_raw_db` and `reactions_production_db`.

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

## Setup

Create and activate the virtual environment from this directory:

```bash
cd tasks/retrosynthesis
uv venv --python 3.11.0
source .venv/bin/activate
uv sync
```

If you prefer not to activate the environment, use `uv run` to prefix the commands below.

## Run The Server

Start the retrosynthesis environment server from this directory:

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

The server also accepts these options:

- `--host`: Bind host. Defaults to `CORRAL_HOST` or `0.0.0.0`.
- `--port`: Bind port. Defaults to `CORRAL_PORT` or `8000`.
- `--level`: Benchmark level to load. Levels are stored under `environments/level_1`, `environments/level_2`, and `environments/level_3`.
- `--subtask_level`: Set to `True` to load `subtasks_json/` instead of `tasks_json/` for the selected level.

## See The Tasks

```bash
curl http://localhost:8000/tasks/
```

## Notes

- Before starting, the server validates database connectivity and schema. Startup fails early if the production database is unavailable or incomplete.
- Database credentials default to local PostgreSQL values, but can be overridden with `RETRO_DB_HOST`, `RETRO_DB_PORT`, `RETRO_DB_NAME`, `RETRO_DB_USER`, and `RETRO_DB_PASSWORD`.
- Optional pricing and availability utilities can also use `MOLPORT_API_KEY`, `CHEMSPACE_API_KEY`, and `MCULE_API_KEY`. See `.env.example` for the full configuration template.
