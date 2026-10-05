"""Settings shared by every test."""

from __future__ import annotations

import os

# The tests build environments without the measured example baselines, which
# need live student servers.
os.environ.setdefault("CORRAL_INFERENCE_ALLOW_UNMEASURED", "1")
