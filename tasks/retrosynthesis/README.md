# Environment with Retrosynthesis Task

## Database Setup

The retrosynthesis environment requires a PostgreSQL database with RDKit extensions containing two databases: `reactions_raw_db` and `reactions_production_db`.

### Quick Start (recommended)

> **Note:** The Docker image is hosted on GHCR. If the package is private, you must authenticate first:
> ```bash
> echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
> ```
> A [GitHub PAT](https://github.com/settings/tokens) with `read:packages` scope is sufficient.
> If the package has been made public, no token is needed.

Run the setup script to pull and start the pre-seeded database:

```bash
cd tasks/retrosynthesis/docker
bash setup_retro_db.sh
```

On first startup, PostgreSQL will initialize and restore both databases (this takes a few minutes). Monitor progress with:

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

## Install environment

```bash
cd tasks/retrosynthesis
uv venv --python 3.11.0
uv sync
```

## Run the environment

```bash
cd tasks/retrosynthesis
python -m env
```

## See the tasks

```bash
curl http://localhost:8000/tasks/
```
