# Resistor Network v4 native-agent Docker runs

This report covers Codex (GPT-5.6-Sol), Claude Code (Claude Opus 5.5), and OpenHands (Kimi K3) running in Corral Docker sandboxes. Each agent has 50 selected trials per level. `trials.csv` contains the final selected result for each of the 300 original trial slots. `attempts.csv` preserves all 380 Docker execution attempts, including infrastructure failures and retries. `runs.csv` summarizes selected scores and attempt counts.

The selected results incorporate the original 300-trial Docker batch and infrastructure recovery attempts. One Codex level-2 trial (task_7, trial index 4) uses its final disk-space recovery run, which scored 0. The earlier failed attempts remain in `attempts.csv`.

Scores are binary: 1 is a successful circuit submission, 0 is a miss. Iteration-limit and model errors remain failures. Each execution row includes its Langfuse session and deterministic Corral trace ID. Raw conversations and submitted circuit answers are intentionally omitted. In `attempts.csv`, `report_path` is relative to the external run archive: `primary/` means `resistor_network_v4_native_docker_ablation_20261021T151122Z/`, and `recovery/` means `resistor_network_v4_native_docker_recovery_20261022T083147Z/`.

The separate pre-Docker runs were executed outside the Docker sandbox and are not included here. Their workspaces had access different from the sandboxed agents, so they are not treated as a clean comparison.
