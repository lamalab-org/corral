"""Settings shared by every test."""

from __future__ import annotations

import os

# Environments in tests may be built without measured example baselines, which
# take live students to measure.
os.environ.setdefault("CORRAL_INFERENCE_ALLOW_UNMEASURED", "1")
