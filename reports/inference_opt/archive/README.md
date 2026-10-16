# Smoke tests and discarded attempts

This archive preserves every file under the smoke/test and `_discarded` directories in `/data/alamparan/corral_ml_runs`, plus all top-level launcher scripts and logs, as inventoried on October 16, 2026. The original files remain in place. Final benchmark reports and their 700 trials are indexed in the [parent directory](../README.md).

There are 4 completed smoke-test trials and 164 discarded attempts. The discarded artifacts include empty Kimi replies, unsupported reasoning parameters, Claude Code MCP idle timeouts, lost test calls, OpenRouter payment errors, and superseded attempts. Directory names retain the original reason labels; unlabeled discarded directories have no inferred diagnosis.

| Archive | Contents |
| --- | --- |
| `20261008_2149_test_openhands.tar.gz` | OpenHands smoke test |
| `20261008_2157_test_aiscientist.tar.gz` | Initial AI Scientist smoke test |
| `20261009_0812_test_aiscientist.tar.gz` | AI Scientist L1 and L2 smoke tests |
| `20261007_2203_batch1_react_discarded.tar.gz` | Discarded React checkpoints and logs |
| `20261008_0748_batch2_codex_discarded.tar.gz` | Discarded Codex checkpoints and logs |
| `_discarded.tar.gz` | Discarded Claude Code and Kimi batches |
| `launchers_and_logs.tar.gz` | All batch launcher scripts and top-level logs |

Tarballs preserve original relative paths, including requests, results where available, metadata, commit databases, Langfuse delivery databases, and checkpoint objects. All 5,543 archived files were decompressed and checked against their original byte counts and SHA-256 hashes. [files.csv](files.csv) records each file, its archive, size, and hash. The four smoke-test reports are also available directly as `.report.json.gz` files. Discarded runs without reports retain their original checkpoints; no synthetic reports were created.

[all_trial_scores_langfuse.csv](../all_trial_scores_langfuse.csv) contains a row for every physical attempt, including attempts without a result. Trace IDs use `sha256("corral:" + execution_id)[:32]` and have not been verified against the remote Langfuse service. Retries can reuse IDs, so retain `source_path` when joining records.

Download tarballs with `git lfs pull`. Inspect one with `tar -tzf <archive>.tar.gz`, or extract into a separate directory with `tar -xzf <archive>.tar.gz -C <directory>`.
