#!/bin/bash
set -e

echo "Creating databases..."
createdb -U "$POSTGRES_USER" reactions_raw_db
createdb -U "$POSTGRES_USER" reactions_production_db

echo "Restoring reactions_raw_db..."
gunzip -c /seed/reactions_raw_db.sql.gz | psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d reactions_raw_db
echo "reactions_raw_db restored."

echo "Restoring reactions_production_db..."
gunzip -c /seed/reactions_production_db.sql.gz | psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d reactions_production_db
echo "reactions_production_db restored."

echo "All databases restored successfully."
