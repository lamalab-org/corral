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
        # OpenHands 1.35's browser discovery ignores PLAYWRIGHT_BROWSERS_PATH
        # and does not recognize Playwright's chrome-linux-arm64 directory.
        # Publish the installed binary on PATH without changing the SDK tools.
        python - <<'PY'
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

with sync_playwright() as playwright:
    chromium = Path(playwright.chromium.executable_path)
if not chromium.is_file() or not os.access(chromium, os.X_OK):
    raise RuntimeError(f"Playwright Chromium is not executable: {chromium}")
link = Path("/usr/local/bin/chromium")
if link.is_symlink():
    link.unlink()
link.symlink_to(chromium)
PY
        rm -rf /var/lib/apt/lists/*
        ;;
esac

# Install everything before locking down the image's private directories.
python -m corral.runtime.permissions --prepare-image
