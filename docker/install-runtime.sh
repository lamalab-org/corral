#!/bin/sh
# Shared Python/Conda image setup. SDK versions come from Corral's extras.
set -eu

CORRAL_EXTRAS=${CORRAL_EXTRAS:-}
python -m pip install --no-cache-dir --editable ".${CORRAL_EXTRAS:+[$CORRAL_EXTRAS]}" "$@"

case ",$CORRAL_EXTRAS," in
    *,claude,*)
        apt-get update
        apt-get install -y --no-install-recommends bubblewrap socat ripgrep
        rm -rf /var/lib/apt/lists/*
        ;;
esac
case ",$CORRAL_EXTRAS," in
    *,openhands,*)
        python -m playwright install chromium --with-deps
        rm -rf /var/lib/apt/lists/*
        ;;
esac

# Install everything before locking down the image's private directories.
python -m corral.runtime.permissions --prepare-image
