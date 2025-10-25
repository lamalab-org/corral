# Docker Deployment Guide for Corral

This guide explains how to deploy Corral environments in sandboxed Docker containers for secure agent experimentation.

## Table of Contents

- [Docker Deployment Guide for Corral](#docker-deployment-guide-for-corral)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
  - [Building Docker Images](#building-docker-images)
    - [Generic Dockerfile Template](#generic-dockerfile-template)
    - [Run a Single Environment](#run-a-single-environment)
    - [Port Mapping](#port-mapping)

---

## Architecture

```markdown
┌─────────────────────────────────────────┐
│   Host Machine                          │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │  Agent Process                   │  │
│  │  (Python script on host)         │  │
│  │                                  │  │
│  │  CorralRouter(                   │  │
│  │    base_url="http://localhost:8001"│ │
│  │  )                               │  │
│  └──────────────┬───────────────────┘  │
│                 │ HTTP Requests         │
│                 │                       │
│  ┌──────────────▼───────────────────┐  │
│  │  Docker Container                │  │
│  │  Port 8000 → 8001 (host)         │  │
│  │                                  │  │
│  │  ┌────────────────────────────┐  │  │
│  │  │ FastAPI Server             │  │  │
│  │  │ (Environment)              │  │  │
│  │  └────────────────────────────┘  │  │
│  │                                  │  │
│  │  Volumes:                        │  │
│  │  /workspace (read-write)         │  │
│  │  /tmp (tmpfs)                    │  │
│  │                                  │  │
│  │  Root: Read-only                 │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (usually included with Docker Desktop)
- Corral repository cloned

## Building Docker Images

### Generic Dockerfile Template

Create a `Dockerfile` in each task directory (e.g., `tasks/catalyst/Dockerfile`):

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 corraluser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY tasks/ ./tasks/

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system -e .

# Install task-specific dependencies (if any)
RUN pip install --no-cache-dir -r tasks/catalyst/requirements.txt || true

# Create workspace directory
RUN mkdir -p /workspace && \
    chown -R corraluser:corraluser /workspace /app

# Switch to non-root user
USER corraluser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CORRAL_HOST=0.0.0.0
ENV CORRAL_PORT=8000
ENV CORRAL_WORK_DIR=/workspace

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/tasks || exit 1

# Run the environment server
CMD ["python", "-m", "tasks.corral_md.src.corral_md.env"]
```

### Run a Single Environment

```bash
# Build the image for a specific task
docker build -f tasks/corral_md/Dockerfile -t corral-md .

# Run the container
docker run -d \
  --name corral-md \
  -p 8001:8000 \
  -e CORRAL_WORK_DIR=/workspace \
  -e CORRAL_HOST=0.0.0.0 \
  -e CORRAL_PORT=8000 \
  -v md_workspace:/workspace:rw \
  --read-only \
  --tmpfs /tmp:size=1G,noexec \
  --security-opt=no-new-privileges:true \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --memory=2g \
  --memory-swap=2g \
  --cpus=2.0 \
  --pids-limit=100 \
  corral-md:latest

# Test the endpoint
curl http://localhost:8001/tasks

# View logs
docker logs corral-md

# Stop the container
docker stop corral-md
docker rm corral-md
```

---

### Port Mapping

Map different tasks to different host ports:

```bash
# Catalyst on port 8001
docker run -d -p 8001:8000 --name catalyst corral-catalyst:latest

# ML on port 8002
docker run -d -p 8002:8000 --name ml corral-ml:latest

# Corral MD on port 8003
docker run -d -p 8003:8000 --name md corral-md:latest
```

**Using the endpoints in your agent code:**

```python
from corral.router import CorralRouter

# Connect to specific environment by port
catalyst_interface = CorralRouter(base_url="http://localhost:8001")
ml_interface = CorralRouter(base_url="http://localhost:8002")
md_interface = CorralRouter(base_url="http://localhost:8003")
```
