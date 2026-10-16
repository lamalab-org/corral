# Smoke tests and discarded attempts

Preserved artifacts for 4 smoke-test trials and 164 discarded attempts. These are excluded from the [final benchmark results](../README.md).

| Archive | Contents |
| --- | --- |
| `20261008_2149_test_openhands.tar.gz` | OpenHands smoke test |
| `20261008_2157_test_aiscientist.tar.gz` | Initial AI Scientist smoke test |
| `20261009_0812_test_aiscientist.tar.gz` | AI Scientist L1 and L2 smoke tests |
| `20261007_2203_batch1_react_discarded.tar.gz` | Discarded React checkpoints and logs |
| `20261008_0748_batch2_codex_discarded.tar.gz` | Discarded Codex checkpoints and logs |
| `_discarded.tar.gz` | Discarded Claude Code and Kimi batches |
| `launchers_and_logs.tar.gz` | All batch launcher scripts and top-level logs |

[All-trial CSV](../all_trial_scores_langfuse.csv) provides session IDs, trace IDs, and checkpoint paths. [files.csv](files.csv) lists archived files and their SHA-256 hashes. Available smoke-test reports are also stored as `.report.json.gz` files.

Run `git lfs pull` to download the tarballs. Extract with `tar -xzf <archive>.tar.gz -C <directory>`.
