# Reaction Database Setup Guide

Quick guide to set up and run both Phase A (staging) and Phase B (production) reaction databases.

---

## Prerequisites

### 1. Install Docker

**macOS:**

```bash
brew install --cask docker-desktop
```

**Linux (Ubuntu/Debian):**

```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (optional, to run without sudo)
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Install Python Dependencies

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install required packages
pip install psycopg2-binary rdkit tqdm pandas

# Ensure rxnutils is available
# (Already installed if working in this repository)
```

---

## Setup & Run

### Step 1: Start PostgreSQL with RDKit in Docker

Start a PostgreSQL container with RDKit extension installed:

```bash
# Pull the RDKit PostgreSQL cartridge image
docker pull informaticsmatters/rdkit-cartridge-debian:latest

# Start the Docker container
docker run --name pg-rdkit \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -d informaticsmatters/rdkit-cartridge-debian:latest

# Wait for PostgreSQL to initialize (8-10 seconds)
sleep 10

# Verify it's running
docker ps
```

**About this image:**

- Includes PostgreSQL with RDKit extension pre-installed
- Provides chemistry functions like `qmol_from_smarts()` and `rdkit_fp()`
- Required for Phase B chemistry operations (fingerprints, substructure search)

### Step 2: Validate RDKit Setup

After starting the container, verify that RDKit extension and required functions are available:

```bash
# Set password for psql commands
export PGPASSWORD=postgres

# Connect and check RDKit extension
psql -h localhost -U postgres -d postgres -c "CREATE EXTENSION IF NOT EXISTS rdkit;"

# Verify required functions exist
psql -h localhost -U postgres -d postgres -c "\df qmol_from_smarts"
psql -h localhost -U postgres -d postgres -c "\df rdkit_fp"
```

**Expected output:**

- `qmol_from_smarts` should be listed with signature: `qmol_from_smarts(cstring)`
- `rdkit_fp` should be listed with signature: `rdkit_fp(mol)`

If these functions are missing, ensure you're using the correct Docker image (see Step 1).

### Step 3: Validate Database Setup (Optional)

```bash
# Set environment variables for setup scripts
export POSTGRES_USER=postgres
export PGPASSWORD=postgres

# Run Phase A setup validation
./setup_phase_a_db.sh

# Run Phase B setup validation
./setup_phase_b_db.sh
```

### Step 4: Run Phase A - Staging Database

```bash
# Activate Python environment (if not already)
source .venv/bin/activate

# Load USPTO and ORD data into reactions_raw_db
python3 phase_a_staging.py
```

**What it does:**

- Creates `reactions_raw_db` database
- Loads and validates reaction data from CSV files
- Creates staging tables

### Step 5: Run Phase B - Production Database

```bash
# Generate chemistry artifacts and production schema
python3 phase_b_production.py
```

**What it does:**

- Creates `reactions_production_db` database
- Processes staging data from Phase A
- Generates reaction templates, fingerprints, bonds, and functional groups
- Uses RDKit extension for chemistry operations

---

## Verify Setup

```bash
# Connect to PostgreSQL
psql -h localhost -U postgres -d postgres

# List databases
\l

# Check Phase A tables
\c reactions_raw_db
\dt

# Check Phase B tables
\c reactions_production_db
\dt

# Exit
\q
```

---

## Managing Docker Container

```bash
# Stop the database
docker stop pg-rdkit

# Start it again (keeps all data)
docker start pg-rdkit

# View logs
docker logs pg-rdkit

# Remove container (WARNING: deletes all data!)
docker rm -f pg-rdkit
```

---

## Troubleshooting

### RDKit Function Not Found

**Problem:** Error message like `function qmol_from_smarts(unknown) does not exist`

**Solution:**

1. Ensure RDKit extension is enabled in your database:

   ```bash
   PGPASSWORD=postgres psql -h localhost -U postgres -d reactions_production_db \
     -c "CREATE EXTENSION IF NOT EXISTS rdkit;"
   ```

2. Verify the correct Docker image is running:

   ```bash
   docker ps
   # Should show: informaticsmatters/rdkit-cartridge-debian
   ```

3. Check that required functions are available:

   ```bash
   PGPASSWORD=postgres psql -h localhost -U postgres -d postgres \
     -c "\df qmol_from_smarts"
   PGPASSWORD=postgres psql -h localhost -U postgres -d postgres \
     -c "\df rdkit_fp"
   ```

4. If functions are missing, you may be using an incompatible image. Recreate with the correct image:

   ```bash
   docker rm -f pg-rdkit
   docker run --name pg-rdkit \
     -e POSTGRES_PASSWORD=postgres \
     -p 5432:5432 \
     -d informaticsmatters/rdkit-cartridge-debian:latest
   ```

### Port Already in Use

**Problem:** Error `port 5432 already in use`

**Solution:**

```bash
# Find what's using port 5432
lsof -i :5432  # macOS/Linux
netstat -ano | findstr :5432  # Windows

# Stop any existing PostgreSQL containers
docker stop pg-rdkit 2>/dev/null
docker rm pg-rdkit 2>/dev/null

# Start fresh container
docker run --name pg-rdkit \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -d informaticsmatters/rdkit-cartridge-debian:latest

# Or use a different port if needed
docker run --name pg-rdkit \
  -e POSTGRES_PASSWORD=postgres \
  -p 5433:5432 \
  -d informaticsmatters/rdkit-cartridge-debian:latest
```

### Connection Refused

**Problem:** `psql: could not connect to server: Connection refused`

**Solution:**

1. Wait 10-15 seconds after starting container
2. Check container is running: `docker ps`
3. Check container logs: `docker logs pg-rdkit`
4. Restart container if needed:

   ```bash
   docker restart pg-rdkit
   sleep 10
   ```

---

## Data Persistence

By default, data persists while the container exists (even when stopped). To persist data even after removing the container, use a volume:

```bash
docker run --name pg-rdkit \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -v pgdata:/var/lib/postgresql/data \
  -d informaticsmatters/rdkit-cartridge-debian:latest
```

---

## Quick Reference

| Component | Location |
|-----------|----------|
| Phase A Script | `phase_a_staging.py` |
| Phase B Script | `phase_b_production.py` |
| Configuration | `phase_a_config.ini` |
| Staging DB | `reactions_raw_db` |
| Production DB | `reactions_production_db` |
| Docker Image | `informaticsmatters/rdkit-cartridge-debian` |
| Default Port | `5432` |
| Default User | `postgres` |
| Default Password | `postgres` |

---

**Estimated Time:**

- Docker setup: 2-5 minutes
- Phase A: 2-10 minutes (depends on CSV size)
- Phase B: 5-30 minutes (depends on reaction count)
