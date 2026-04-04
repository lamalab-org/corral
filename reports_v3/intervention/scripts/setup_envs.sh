#!/usr/bin/env bash
#
# Set up virtual environments for all three benchmark environments.
#
# Spectra & Resistor: uv venv + uv sync (+ promptstore upgrade)
# Wetlab: micromamba (reaktoro is conda-only) + pip install corral & wetlab
#
# Also upgrades promptstore in the root corral venv to ensure prompts load.
#
# Usage:
#   ./scripts/setup_envs.sh           # Set up all environments
#   ./scripts/setup_envs.sh spectra   # Set up only spectra
#   ./scripts/setup_envs.sh wetlab    # Set up only wetlab
#   ./scripts/setup_envs.sh resistor  # Set up only resistor
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TASKS_DIR="$PROJECT_ROOT/tasks"

TARGET="${1:-all}"

setup_root() {
    echo "=== Setting up root corral venv ==="
    cd "$PROJECT_ROOT"
    uv lock --upgrade-package promptstore
    uv sync
    # boto3 needed for litellm Bedrock calls (not a declared dep)
    uv pip install boto3 2>&1 | tail -1
    echo "  Root venv ready: $PROJECT_ROOT/.venv"
}

setup_spectra() {
    echo "=== Setting up Spectra Elucidation ==="
    local env_dir="$TASKS_DIR/spectra_elucidation"
    cd "$env_dir"

    if [ ! -d ".venv" ]; then
        uv venv --python 3.11
    fi
    uv lock --upgrade-package promptstore
    uv sync
    echo "  Spectra ready: $env_dir/.venv"
}

setup_resistor() {
    echo "=== Setting up Resistor Network ==="
    local env_dir="$TASKS_DIR/resistor_network"
    cd "$env_dir"

    if [ ! -d ".venv" ]; then
        uv venv --python 3.12
    fi
    uv lock --upgrade-package promptstore
    uv sync
    echo "  Resistor ready: $env_dir/.venv"
}

setup_wetlab() {
    echo "=== Setting up WetLab ==="
    local env_dir="$TASKS_DIR/wetlab"
    cd "$env_dir"

    # reaktoro is conda-only — need micromamba or conda
    local MCMAMBA=""
    if command -v micromamba &>/dev/null; then
        MCMAMBA="micromamba"
    elif command -v mamba &>/dev/null; then
        MCMAMBA="mamba"
    elif command -v conda &>/dev/null; then
        MCMAMBA="conda"
    else
        echo "  No conda/mamba/micromamba found. Installing micromamba..."
        # Install micromamba to a local prefix
        local MAMBA_ROOT="$env_dir/.micromamba"
        mkdir -p "$MAMBA_ROOT/bin"
        # Detect platform
        local PLATFORM="osx-arm64"
        if [[ "$(uname -s)" == "Linux" ]]; then
            PLATFORM="linux-64"
        fi
        curl -Ls "https://micro.mamba.pm/api/micromamba/${PLATFORM}/latest" \
            | tar -xj -C "$MAMBA_ROOT" 2>/dev/null
        # The archive layout varies across versions; find the binary and
        # ensure it ends up at a predictable path.
        if [ ! -f "$MAMBA_ROOT/bin/micromamba" ]; then
            local found
            found=$(find "$MAMBA_ROOT" -name micromamba -type f | head -1)
            if [ -z "$found" ]; then
                echo "  ERROR: failed to download micromamba"
                return 1
            fi
            mv "$found" "$MAMBA_ROOT/bin/micromamba"
            chmod +x "$MAMBA_ROOT/bin/micromamba"
        fi
        MCMAMBA="$MAMBA_ROOT/bin/micromamba"
        echo "  Installed micromamba to $MCMAMBA"
    fi

    # Python 3.12 — required because corral uses StrEnum (Python >= 3.11).
    # reaktoro 2.13+ has py312 builds for both osx-arm64 and linux-64.
    local PYTHON_VER="3.12"

    if [ ! -d ".venv" ] || ! .venv/bin/python -c "import reaktoro" 2>/dev/null; then
        echo "  Creating wetlab venv with Python $PYTHON_VER + reaktoro..."
        rm -rf .venv
        $MCMAMBA create -y -p "$(pwd)/.venv" \
            -c conda-forge \
            "python=$PYTHON_VER" reaktoro
    fi

    # Install corral (with latest promptstore) + wetlab into the conda env via pip
    .venv/bin/pip install --upgrade promptstore 2>&1 | tail -1
    .venv/bin/pip install -e "$PROJECT_ROOT" -e . 2>&1 | tail -3

    echo "  WetLab ready: $env_dir/.venv"
}

setup_ml() {
    echo "=== Setting up ML ==="
    local env_dir="$TASKS_DIR/ml"
    cd "$env_dir"

    if [ ! -d ".venv" ]; then
        uv venv --python 3.12
    fi
    uv lock --upgrade-package promptstore
    uv sync
    echo "  ML ready: $env_dir/.venv"
}

setup_catalyst() {
    echo "=== Setting up Catalyst ==="
    local env_dir="$TASKS_DIR/catalyst"
    cd "$env_dir"

    if [ ! -d ".venv" ]; then
        uv venv --python 3.12
    fi
    uv lock --upgrade-package promptstore
    uv sync
    echo "  Catalyst ready: $env_dir/.venv"
}

setup_retrosynthesis() {
    echo "=== Setting up Retrosynthesis ==="
    local env_dir="$TASKS_DIR/retrosynthesis"
    cd "$env_dir"

    if [ ! -d ".venv" ]; then
        uv venv --python 3.11
    fi
    uv lock --upgrade-package promptstore
    uv sync
    echo "  Retrosynthesis ready: $env_dir/.venv"
}

setup_md() {
    echo "=== Setting up MD (Molecular Dynamics) ==="
    local env_dir="$TASKS_DIR/corral_md"
    cd "$env_dir"

    if [ ! -d ".venv" ]; then
        uv venv --python 3.11
    fi
    uv lock --upgrade-package promptstore
    uv sync
    echo "  MD ready: $env_dir/.venv"
}

case "$TARGET" in
    all)
        setup_root
        setup_spectra
        setup_resistor
        setup_wetlab
        setup_ml
        setup_catalyst
        setup_retrosynthesis
        setup_md
        ;;
    spectra)         setup_spectra ;;
    resistor)        setup_resistor ;;
    wetlab)          setup_wetlab ;;
    ml)              setup_ml ;;
    catalyst)        setup_catalyst ;;
    retrosynthesis)  setup_retrosynthesis ;;
    md)              setup_md ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [all|spectra|resistor|wetlab|ml|catalyst|retrosynthesis|md]"
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="
echo "Environment servers can now be started with:"
echo "  ./scripts/launch_sweep.sh --start-servers"
