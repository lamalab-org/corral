#!/bin/bash
#
# Setup script for Phase A staging database
# Creates PostgreSQL database and installs RDKit extension (optional)
#

set -e

# Configuration
DB_NAME="reactions_raw_db"
DB_USER="${POSTGRES_USER:-postgres}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DOCKER_CONTAINER="${POSTGRES_CONTAINER:-pg-rdkit}"

echo "=========================================="
echo "Phase A Database Setup"
echo "=========================================="
echo ""
echo "Database: $DB_NAME"
echo "User:     $DB_USER"
echo "Host:     $DB_HOST"
echo "Port:     $DB_PORT"
echo "Container: $DOCKER_CONTAINER"
echo ""

# Check if Docker container is running
echo "Checking Docker container..."
if ! docker ps | grep -q "$DOCKER_CONTAINER"; then
    echo "ERROR: Docker container '$DOCKER_CONTAINER' is not running"
    echo ""
    echo "Please ensure the PostgreSQL container is running:"
    echo "  docker ps"
    echo ""
    echo "Or specify a different container:"
    echo "  export POSTGRES_CONTAINER=your_container_name"
    exit 1
fi

echo "✓ Docker container is running"
echo ""

# Check if PostgreSQL is accessible
echo "Checking PostgreSQL connection..."
if ! docker exec "$DOCKER_CONTAINER" psql -U "$DB_USER" -l > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to PostgreSQL in container"
    echo ""
    echo "Check your container settings:"
    echo "  docker exec $DOCKER_CONTAINER psql -U postgres -l"
    exit 1
fi

echo "✓ PostgreSQL is running"
echo ""

# Check if database exists
if docker exec "$DOCKER_CONTAINER" psql -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    echo "⚠️  Database '$DB_NAME' already exists"
    read -p "Drop and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Dropping database '$DB_NAME'..."
        docker exec "$DOCKER_CONTAINER" dropdb -U "$DB_USER" --if-exists "$DB_NAME"
    else
        echo "Using existing database"
    fi
fi

# Create database if it doesn't exist
if ! docker exec "$DOCKER_CONTAINER" psql -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    echo "Creating database '$DB_NAME'..."
    docker exec "$DOCKER_CONTAINER" createdb -U "$DB_USER" "$DB_NAME"
    echo "✓ Database created"
else
    echo "✓ Database exists"
fi

echo ""

# Try to install RDKit extension (optional)
echo "Attempting to install RDKit extension..."
if docker exec "$DOCKER_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS rdkit;" 2>/dev/null; then
    echo "✓ RDKit extension installed"
    echo ""
    echo "RDKit PostgreSQL cartridge is available!"
    echo "You can use molecule fingerprints and substructure searches in Phase B."
else
    echo "⚠️  RDKit extension not available"
    echo ""
    echo "This is OK for Phase A (staging only)."
    echo ""
    echo "To enable RDKit for Phase B:"
    echo "  1. Use a PostgreSQL container with RDKit support (e.g., mcs07/postgres-rdkit)"
    echo "  2. Re-run this script"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Update DB_CONFIG in phase_a_staging.py with your credentials"
echo "  2. Install Python dependencies:"
echo "     pip install -r requirements_phase_a.txt"
echo "  3. Run the staging script:"
echo "     python phase_a_staging.py"
echo ""
echo "Connection string for your script:"
echo "  host='$DB_HOST', port=$DB_PORT, database='$DB_NAME', user='$DB_USER'"
echo ""
echo "Note: PostgreSQL is running in Docker container '$DOCKER_CONTAINER'"
echo ""
