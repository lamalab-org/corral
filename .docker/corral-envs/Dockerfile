# syntax=docker/dockerfile:1

# Corral Environment Server Base Docker Image
# Installs corral core and optionally an external environment package, then runs the environment server
#
# This image can be used in two ways:
#   1. As a BASE IMAGE: Build without ENV_PACKAGE_URL to get a base image with just corral installed.
#      Child images can then use FROM this image and install their own environment package.
#   2. As a STANDALONE IMAGE: Build with ENV_PACKAGE_URL to get a complete image with an environment.
#
# Build args:
#   ENV_PACKAGE_URL - (Optional) Git URL or PyPI package to install (e.g., git+https://github.com/org/my-env.git)
#   ENV_MODULE - (Optional at build time) Python module path to the env.py file (e.g., my_env.env)

FROM python:3.12-slim

LABEL maintainer="Corral Team"
LABEL description="Corral Environment Server Base Image"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG SYSTEM_USER="corraluser"
ARG SYSTEM_UID="1000"
ARG SYSTEM_GID="1000"

RUN groupadd -g "${SYSTEM_GID}" "${SYSTEM_USER}" && \
    useradd -m -u "${SYSTEM_UID}" -g "${SYSTEM_GID}" -s /bin/bash "${SYSTEM_USER}"

# Set working directory
WORKDIR /app

# Install uv (native installation - much faster than pip install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Build arguments for environment package
ARG ENV_PACKAGE_URL=""
ARG ENV_MODULE=""

# Install corral core from GitHub
RUN uv pip install --system git+https://github.com/lamalab-org/corral.git@main

# Install the external environment package (optional - skip if building base image)
RUN if [ -n "${ENV_PACKAGE_URL}" ]; then \
        echo "Installing environment package: ${ENV_PACKAGE_URL}" && \
        uv pip install --system "${ENV_PACKAGE_URL}"; \
    else \
        echo "No ENV_PACKAGE_URL provided - building base image without environment package"; \
    fi

# Store ENV_MODULE as environment variable
ENV ENV_MODULE=${ENV_MODULE}

# Create entrypoint script inline (self-contained, no external file needed)
RUN cat <<'EOF' > /usr/local/bin/entrypoint.sh
#!/bin/bash
set -e

if [ -z "${ENV_MODULE}" ]; then
    echo "ERROR: ENV_MODULE environment variable is required"
    exit 1
fi

export CORRAL_HOST="${CORRAL_HOST:-0.0.0.0}"
export CORRAL_PORT="${CORRAL_PORT:-8000}"
export CORRAL_WORK_DIR="${CORRAL_WORK_DIR:-/workspace}"

mkdir -p "${CORRAL_WORK_DIR}"

# Build command arguments starting with port
ARGS="--port ${CORRAL_PORT}"

# Parse CORRAL_ENV_ARGS JSON and convert to CLI arguments
if [ -n "$CORRAL_ENV_ARGS" ]; then
    EXTRA_ARGS=$(python3 -c "
import json
import os
import sys

try:
    args = json.loads(os.environ.get('CORRAL_ENV_ARGS', '{}'))
    parts = []
    for key, value in args.items():
        # Convert Python booleans to lowercase strings
        if isinstance(value, bool):
            value = str(value).lower()
        parts.append(f'--{key}')
        parts.append(str(value))
    print(' '.join(parts))
except json.JSONDecodeError as e:
    print(f'ERROR: Invalid JSON in CORRAL_ENV_ARGS: {e}', file=sys.stderr)
    sys.exit(1)
")
    if [ $? -ne 0 ]; then
        echo "Failed to parse CORRAL_ENV_ARGS"
        exit 1
    fi
    ARGS="$ARGS $EXTRA_ARGS"
fi

exec uv run python -m ${ENV_MODULE} $ARGS "$@"
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

# Create workspace directory with proper permissions
RUN mkdir -p /workspace && \
    chown -R "${SYSTEM_USER}:${SYSTEM_GID}" /workspace /app /usr/local/bin/entrypoint.sh

# Switch to non-root user
USER ${SYSTEM_USER}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CORRAL_HOST=0.0.0.0
ENV CORRAL_PORT=8000
ENV CORRAL_WORK_DIR=/workspace

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${CORRAL_PORT}/tasks || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
