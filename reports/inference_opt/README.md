# Inference optimization benchmark reports

Final reports for the October 7–10, 2026 batches: 14 runs, 50 trials each, 700 trials total. Failed trials are included in the average scores. Superseded and smoke-test runs are excluded.

The `.report.json.gz` files contain the complete original JSON reports, compressed with gzip. Read one with `gzip -dc <file>.report.json.gz` or Python `json.load(gzip.open(path, "rt"))`.

The two AI Scientist reports use Git LFS because of their size. Run `git lfs pull` after cloning to download them.

[summary.csv](summary.csv) records scores, completion times, and SHA-256 hashes of the uncompressed reports. [trial_scores_langfuse.csv](trial_scores_langfuse.csv) contains all 700 trial scores, errors, execution IDs, and Langfuse trace IDs. Session IDs are the run IDs below; trace IDs use Corral’s deterministic mapping and have not been checked against the remote Langfuse service.

| Run / Langfuse session ID | Average score | Failed / total | Report |
| --- | ---: | ---: | --- |
| `inference_opt_l1_react_claude55opus` | 72% | 0 / 50 | [JSON.gz](inference_opt_l1_react_claude55opus.report.json.gz) |
| `inference_opt_l1_react_gpt56sol` | 36% | 1 / 50 | [JSON.gz](inference_opt_l1_react_gpt56sol.report.json.gz) |
| `inference_opt_l2_react_claude55opus` | 84% | 1 / 50 | [JSON.gz](inference_opt_l2_react_claude55opus.report.json.gz) |
| `inference_opt_l2_react_gpt56sol` | 36% | 4 / 50 | [JSON.gz](inference_opt_l2_react_gpt56sol.report.json.gz) |
| `inference_opt_l1_codex_gpt56sol` | 54% | 0 / 50 | [JSON.gz](inference_opt_l1_codex_gpt56sol.report.json.gz) |
| `inference_opt_l2_codex_gpt56sol` | 52% | 0 / 50 | [JSON.gz](inference_opt_l2_codex_gpt56sol.report.json.gz) |
| `inference_opt_l1_claudecode_claude55opus` | 92% | 0 / 50 | [JSON.gz](inference_opt_l1_claudecode_claude55opus.report.json.gz) |
| `inference_opt_l2_claudecode_claude55opus` | 82% | 1 / 50 | [JSON.gz](inference_opt_l2_claudecode_claude55opus.report.json.gz) |
| `inference_opt_l1_openhands_kimik3` | 50% | 4 / 50 | [JSON.gz](inference_opt_l1_openhands_kimik3.report.json.gz) |
| `inference_opt_l2_openhands_kimik3` | 54% | 8 / 50 | [JSON.gz](inference_opt_l2_openhands_kimik3.report.json.gz) |
| `inference_opt_l1_aiscientist_claude55opus` | 70% | 0 / 50 | [JSON.gz](inference_opt_l1_aiscientist_claude55opus.report.json.gz) |
| `inference_opt_l2_aiscientist_claude55opus` | 68% | 0 / 50 | [JSON.gz](inference_opt_l2_aiscientist_claude55opus.report.json.gz) |
| `inference_opt_l1_react_kimik3` | 24% | 25 / 50 | [JSON.gz](inference_opt_l1_react_kimik3.report.json.gz) |
| `inference_opt_l2_react_kimik3` | 38% | 22 / 50 | [JSON.gz](inference_opt_l2_react_kimik3.report.json.gz) |
