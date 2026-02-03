#!/bin/bash
# Setup script for Phase B production database

set -e  # Exit on any error

echo "============================================"
echo "Phase B - Production Database Setup"
echo "============================================"

# Database credentials
DB_HOST="localhost"
DB_PORT="5432"
DB_USER="postgres"
DB_NAME="reactions_production_db"
DOCKER_CONTAINER="${POSTGRES_CONTAINER:-pg-rdkit}"
export PGPASSWORD="postgres"

echo ""
echo "Prerequisites check..."

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

# Check if PostgreSQL is accessible
if ! docker exec "$DOCKER_CONTAINER" psql -U "$DB_USER" -l > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to PostgreSQL in container"
    exit 1
fi
echo "✓ PostgreSQL is running"

# Check if Phase A database exists
if ! docker exec "$DOCKER_CONTAINER" psql -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "reactions_raw_db"; then
    echo "ERROR: Phase A database (reactions_raw_db) not found"
    echo "Please run Phase A setup first: ./setup_phase_a_db.sh"
    exit 1
fi
echo "✓ Phase A database exists"

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."
python3 -c "import psycopg2; import rdkit; from rxnutils.chem.reaction import ChemicalReaction; from tqdm import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Missing Python dependencies"
    echo "Please install: pip install psycopg2-binary rdkit tqdm"
    echo "And ensure rxnutils is available"
    exit 1
fi
echo "✓ Python dependencies installed"

echo ""
echo "============================================"
echo "Ready to run Phase B pipeline"
echo "============================================"
echo ""
echo "Container: $DOCKER_CONTAINER"
echo ""
echo "To create and populate the production database, run:"
echo ""
echo "  python3 phase_b_production.py"
echo ""
echo "This will:"
echo "  1. Create reactions_production_db database"
echo "  2. Create production schema with RDKit extension"
echo "  3. Process all staging reactions"
echo "  4. Generate templates, fingerprints, bonds, and FGs"
echo "  5. Validate the setup"
echo ""
echo "Expected processing time: ~5-30 minutes depending on data size"
echo ""
echo "Note: PostgreSQL is running in Docker container '$DOCKER_CONTAINER'"
echo ""
