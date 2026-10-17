# Inference optimization reports

14 runs (7 agent/model pairs × 2 levels), 10 tasks × k = 5, 700 trials.
Layout, names and index columns follow [`../README.md`](../README.md).

| Agent | Model | L1 passes | L1 errors | L2 passes | L2 errors |
|---|---|---:|---:|---:|---:|
| ai-scientist | claude-opus-5-5 | 35/50 | 0 | 34/50 | 0 |
| claude-code | claude-opus-5-5 | 46/50 | 0 | 41/50 | 1 |
| codex | gpt-5.6-sol | 28/50 | 0 | 26/50 | 0 |
| openhands | kimi-k3 | 25/50 | 4 | 27/50 | 8 |
| react | claude-opus-5-5 | 36/50 | 0 | 42/50 | 1 |
| react | gpt-5.6-sol | 18/50 | 1 | 18/50 | 4 |
| react | kimi-k3 | 12/50 | 25 | 20/50 | 22 |

Includes 42 completed policy reevaluations after service errors during final
scoring; all 700 trials still count. [Rescore comparisons](provenance/rescores.csv)
preserve the original scores. Langfuse links refer to the original agent runs;
the policy reevaluations are recorded in the report evaluation metadata.

Errors are trials that ended with an agent or provider error; they count as
failures. Most are exhausted interaction budgets (react Kimi K3: 47 of 47).
react GPT-5.6 Sol has four prompts rejected by the provider's usage policy.

The AI Scientist reports (114 MB and 131 MB) are stored in Git LFS; run
`git lfs pull` before reading them.
