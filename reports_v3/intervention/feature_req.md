# Feature Requests for Intervention Pipeline

## 1. Structured run logging

**Problem:** Current logs append to a single `run.log` via nohup stdout/stderr redirect. Multiple log sources (loguru from run_intervention.py, corral runner, wandb, litellm) interleave. If a run is accidentally restarted, logs from both runs mix in the same file. Out-of-order buffering makes timestamps unreliable.

**Proposed fix:**
- Each run should write to a timestamped log file (e.g. `run_20260402_004645.log`) instead of overwriting/appending to `run.log`
- The launch script should check if a run already completed (report JSON exists) before launching, to prevent accidental re-runs
- Add a structured JSON log alongside the human-readable log (one JSON object per line) for easier programmatic analysis
- Include run metadata (run_name, start_time, git_sha, config snapshot) in the first log line

**Priority:** Before next full experiment run
