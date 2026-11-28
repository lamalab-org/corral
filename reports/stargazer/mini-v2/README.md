# Stargazer Mini v2 reports

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv) ·
[Selection protocol and audit](../../../corral_mini/README.md#stargazer-level-1-refresh-2026-11-28)

This snapshot contains 32 complete runs and 480 indexed trials. Claude and
OpenAI are complete. The four Kimi K2.5 reports are pending. At the last
server check, 187 of 220 new trials were usable, seven were running, and 26
needed a retry after OpenRouter credit errors.

| Level | Tasks |
|---|---|
| 1 | `seed300025_diff1`, `seed611761_diff5`, `seed630238_diff4` |
| 2 | `seed122_diff10`, `seed176_diff9`, `seed1_diff8` |

The campaign uses five trials per task and the same 18 agent/model pairs as
the other Mini v2 environments. Claude runs use Claude Code and ReAct, OpenAI
runs use Codex and ReAct, and Kimi runs use OpenHands and ReAct.

Claude and OpenAI need new trials only on `seed300025_diff1`. Kimi needs new
trials on all six tasks. The October and v4-additions Kimi runs used Kimi K3,
and the Mini v2 matrix uses K3.1 Pro, K2.5 and K2 0905.

| Family | Reusable Level 1 trials | Reusable Level 2 trials | New trials |
|---|---:|---:|---:|
| Claude | 70 | 90 | 20 |
| OpenAI | 70 | 90 | 20 |
| Kimi | 0 | 0 | 180 |
| **Total** | **140** | **180** | **220** |

The Claude and OpenAI Level 2 reports come from the original Stargazer
Mini v2 campaign (commit `a293ca086`), which used the same three Level 2
tasks. The Level 1 replacement tasks come from the
[v4 additions](../l1-v4-additions/). The October
[full reports](../full/) supply `seed300025_diff1` for Claude Code and ReAct
with Opus 5.5 and for Codex and ReAct with GPT-5.6 Sol. Each assembled Level 1
report preserves the source execution and Langfuse IDs and records its
per-task provenance. The October and v4 reports used different sandbox
memory limits for some agent/model pairs. See the source reports for settings.

## New trials

The top-up uses Stargazer at `9269652a4`, with 50 iterations,
no surrender, and the pinned task Docker images. Each trial uses 2 CPUs,
6 GB of memory and a 4096 PID limit. The three numerical-library thread
limits are set to 2.

Kimi uses OpenRouter with provider fallback disabled and a 32,768-token
output limit. Kimi K3.1 Pro uses Modal with medium reasoning. Kimi K2 0905
uses Novita with provider-default reasoning. For Kimi K2.5, the original
Novita route returned HTTP 404 on both harnesses because the endpoint was
unavailable. The top-up pins the same model to Amazon Bedrock in
`us-east-2` with medium reasoning. [OpenRouter endpoint listing](https://openrouter.ai/api/v1/models/moonshotai/kimi-k2.5/endpoints)
