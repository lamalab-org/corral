# Inference optimization benchmark reports

Final benchmark: 14 runs × 50 trials = 700 trials.

- [summary.csv](summary.csv): run scores and report hashes.
- [trial_scores_langfuse.csv](trial_scores_langfuse.csv): scores, errors, session IDs, and trace IDs for all 700 final trials.
- [all_trial_scores_langfuse.csv](all_trial_scores_langfuse.csv): all 868 attempts, including smoke tests and discarded runs.
- [archive/](archive/README.md): smoke-test reports, discarded checkpoints, and launcher logs.

Trace IDs are derived from execution IDs and have not been verified against Langfuse. Archived retries may share IDs; use `source_path` to distinguish them.

Reports are complete JSON files compressed with gzip. Run `git lfs pull` to download large reports and archives, then read reports with `gzip -dc <file>.report.json.gz`.

| Run / Langfuse session ID | Average score | Report |
| --- | ---: | --- |
| `inference_opt_l1_react_claude55opus` | 72% | [JSON.gz](inference_opt_l1_react_claude55opus.report.json.gz) |
| `inference_opt_l1_react_gpt56sol` | 36% | [JSON.gz](inference_opt_l1_react_gpt56sol.report.json.gz) |
| `inference_opt_l2_react_claude55opus` | 84% | [JSON.gz](inference_opt_l2_react_claude55opus.report.json.gz) |
| `inference_opt_l2_react_gpt56sol` | 36% | [JSON.gz](inference_opt_l2_react_gpt56sol.report.json.gz) |
| `inference_opt_l1_codex_gpt56sol` | 54% | [JSON.gz](inference_opt_l1_codex_gpt56sol.report.json.gz) |
| `inference_opt_l2_codex_gpt56sol` | 52% | [JSON.gz](inference_opt_l2_codex_gpt56sol.report.json.gz) |
| `inference_opt_l1_claudecode_claude55opus` | 92% | [JSON.gz](inference_opt_l1_claudecode_claude55opus.report.json.gz) |
| `inference_opt_l2_claudecode_claude55opus` | 82% | [JSON.gz](inference_opt_l2_claudecode_claude55opus.report.json.gz) |
| `inference_opt_l1_openhands_kimik3` | 50% | [JSON.gz](inference_opt_l1_openhands_kimik3.report.json.gz) |
| `inference_opt_l2_openhands_kimik3` | 54% | [JSON.gz](inference_opt_l2_openhands_kimik3.report.json.gz) |
| `inference_opt_l1_aiscientist_claude55opus` | 70% | [JSON.gz](inference_opt_l1_aiscientist_claude55opus.report.json.gz) |
| `inference_opt_l2_aiscientist_claude55opus` | 68% | [JSON.gz](inference_opt_l2_aiscientist_claude55opus.report.json.gz) |
| `inference_opt_l1_react_kimik3` | 24% | [JSON.gz](inference_opt_l1_react_kimik3.report.json.gz) |
| `inference_opt_l2_react_kimik3` | 38% | [JSON.gz](inference_opt_l2_react_kimik3.report.json.gz) |
