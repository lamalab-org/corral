# Corral Docker

This folder contains Docker configurations for running Corral benchmarks.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TWO-CONTAINER SETUP                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────┐         ┌──────────────────────┐              │
│  │  Environment         │  HTTP   │  Agent Runner        │              │
│  │  (corral-envs)       │◄───────►│ (corral-agent-runner)│              │
│  │                      │         │                      │              │
│  │  - Serves tasks      │  :8000  │  - Runs benchmarks   │              │
│  │  - Provides tools    │         │  - Calls LLMs        │              │
│  │  - No LLM calls      │         │  - Logs to W&B       │              │
│  └──────────────────────┘         └──────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Docker Images

### 1. `corral-envs` - Environment Server

**What it does:** Runs a Corral environment that serves tasks and tools via HTTP API.

**Dockerfile:** `corral-envs/Dockerfile`

**Build arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `ENV_PACKAGE_URL` | Yes | Git URL or PyPI package (e.g., `git+https://github.com/org/my-env.git`) |
| `ENV_MODULE` | Yes | Python module path to run (e.g., `my_package.env`) |

**Build manually:**

```bash
docker build \
  -f .docker/corral-envs/Dockerfile \
  --build-arg ENV_PACKAGE_URL=git+https://github.com/lamalab-org/corral-materials.git \
  --build-arg ENV_MODULE=corral_materials.env \
  -t corral-materials:latest \
  .
```

**Run manually:**

```bash
docker run -d \
  --name corral-env \
  -p 8000:8000 \
  -e CORRAL_HOST=0.0.0.0 \
  -e CORRAL_PORT=8000 \
  corral-materials:latest
```

**Runtime environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CORRAL_HOST` | `0.0.0.0` | Server host |
| `CORRAL_PORT` | `8000` | Server port |
| `CORRAL_WORK_DIR` | `/workspace` | Working directory |

---

### 2. `corral-agent-runner` - Benchmark Runner

**What it does:** Runs AI agents against a Corral environment and benchmarks their performance.

**Dockerfile:** `corral-agent-runner/Dockerfile`

**Build manually:**

```bash
docker build \
  -f .docker/corral-agent-runner/Dockerfile \
  -t corral-agent-runner:latest \
  .
```

**Run manually:**

```bash
docker run -d \
  --name corral-agent \
  --network corral-network \
  -e BASE_URL=http://corral-env:8000 \
  -e MODEL=claude-3-5-sonnet-20241022 \
  -e AGENT_CLASS=ReActAgent \
  --env-file .env \
  corral-agent-runner:latest
```

**Runtime environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_URL` | `http://localhost:8000` | Environment server URL |
| `MODEL` | `claude-3-5-sonnet-20241022` | LLM model (any LiteLLM-supported model) |
| `AGENT_CLASS` | `ReActAgent` | Agent class name |
| `TEMPERATURE` | `0.0` | LLM temperature |
| `MAX_ITERATIONS` | `10` | Max agent iterations |
| `TRIALS_PER_TASK` | `5` | Trials per task |
| `TASK_IDS` | (all) | Comma-separated task IDs |
| `AGENT_PARAMS` | - | JSON string for extra agent kwargs |

**API Keys:** Pass any LiteLLM-supported API key via `--env-file` or `-e`:

```bash
# Using .env file (recommended)
docker run --env-file .env ...

# Or pass specific keys
docker run -e ANTHROPIC_API_KEY=... ...
docker run -e OPENAI_API_KEY=... ...
docker run -e GEMINI_API_KEY=... ...
docker run -e MISTRAL_API_KEY=... ...
```

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for all supported providers.

---

## How to Run

```bash
cd .docker

# Run both containers
ENV_IMAGE=ghcr.io/lamalab-org/corral-materials:latest \
MODEL=gpt-4 \
AGENT_CLASS=ReActAgent \
docker compose -f docker-compose.corral-two-container.yml up

# Run only environment
ENV_IMAGE=ghcr.io/lamalab-org/corral-materials:latest \
docker compose -f docker-compose.corral-two-container.yml up environment

# With extra agent params
AGENT_PARAMS='{"temperature": 0.7, "custom_param": "value"}' \
ENV_IMAGE=my-env:latest \
docker compose -f docker-compose.corral-two-container.yml up
```

## Building Images

### Build All Images

```bash
cd .docker
docker buildx bake -f docker-bake.hcl -f build.json --load
```

### Build Specific Image

```bash
# Build only agent-runner
docker buildx bake -f docker-bake.hcl -f build.json corral-agent-runner --load

# Build only envs base
docker buildx bake -f docker-bake.hcl -f build.json corral-envs --load
```

### Build for Multiple Platforms

```bash
# Build for both AMD64 and ARM64 (Intel + Apple Silicon)
PLATFORMS=linux/amd64,linux/arm64 docker buildx bake -f docker-bake.hcl -f build.json --push
```

---

## Development Workflow

**Fastest iteration:** Run environment in Docker, agent locally.

```bash
# 1. Start environment (once)
corral bench env --image my-env:latest --detach

# 2. Edit and run your agent code (repeat)
python my_agent_script.py
# ... make changes ...
python my_agent_script.py

# 3. Stop when done
corral bench stop
```
