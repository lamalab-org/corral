#!/usr/bin/env bash
# setup_retro_db.sh — Pull and run the corral retrosynthesis database
set -euo pipefail

IMAGE="ghcr.io/lamalab-org/corral-retro-db:v1"
CONTAINER_NAME="corral-retro-db"
VOLUME_NAME="corral-retro-db"
PORT="${RETRO_DB_PORT:-5432}"
PASSWORD="${POSTGRES_PASSWORD:-postgres}"

# If the package is private, authenticate to GHCR first:
#   echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
# Or set GITHUB_TOKEN env var and this script will do it automatically:
if [[ -n "${GITHUB_TOKEN:-}" && -n "${GITHUB_USER:-}" ]]; then
    echo "==> Logging in to ghcr.io as ${GITHUB_USER}..."
    echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GITHUB_USER}" --password-stdin
fi

echo "==> Pulling ${IMAGE}..."
docker pull "${IMAGE}"

# Stop and remove existing container if present
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "==> Removing existing container ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}"
fi

echo "==> Starting ${CONTAINER_NAME} on port ${PORT}..."
docker run \
    --name "${CONTAINER_NAME}" \
    -p "${PORT}:5432" \
    -e POSTGRES_PASSWORD="${PASSWORD}" \
    -v "${VOLUME_NAME}:/var/lib/postgresql/data" \
    -d "${IMAGE}"

echo ""
echo "Container started. On first run, database restoration takes a few minutes."
echo "Monitor progress with:  docker logs -f ${CONTAINER_NAME}"
echo ""
echo "Connection details:"
echo "  Host:     localhost"
echo "  Port:     ${PORT}"
echo "  User:     postgres"
echo "  Password: ${PASSWORD}"
echo "  Databases: reactions_raw_db, reactions_production_db"
