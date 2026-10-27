# Cost estimate

Estimated API cost of the 116 runs in `reports/<environment>/runs.csv`, by environment and by agent and model. Token counts come from the `input_tokens` and `output_tokens` columns of `runs.csv` on `report_audit` at `d2882940c` (2026-10-27).

**Summary:** about **$24.8k** at list price with no cache discount, which is an upper bound. If about 90% of input tokens were served from the prompt cache, the cost is closer to **$6.4k**.

## Prices

USD per 1M tokens, looked up on 2026-10-27.

| Model | Input | Output | Cache read |
|---|---:|---:|---:|
| claude-opus-5-5 | $4.00 | $20.00 | $0.20 |
| gpt-5.6-sol | $4.00 | $20.00 | $0.40 (assumed) |
| kimi-k3 (OpenRouter) | $2.60 | $13.00 | $0.29 |

- GPT-5.6 Sol was cut from $5 / $30 in September 2026. The runs were made in October 2026, so this uses the lower price.
- Kimi K3 is listed at $3 / $15 on Moonshot's API and on some OpenRouter providers. Those rates would raise the Kimi rows by about 15%.
- I couldn't find a cache-read price for GPT-5.6 Sol. $0.40 assumes cached input costs 10% of the normal input price.

## Upper bound: list price, no cache discount

| Agent / model | afm | corral_md | inference_opt | psychometrics | resistor_network | retrosynthesis | spectra_elucidation | stargazer | wetlab | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ai-scientist / claude-opus-5-5 | — | $1,455 | $2,337 | $1,894 | $2,023 | $1,207 | $1,217 | $1,282 | $488 | **$11,903** |
| claude-code / claude-opus-5-5 | — | $290 | $233 | $59 | $262 | $772 | $25 | $43 | $76 | **$1,760** |
| codex / gpt-5.6-sol | $39 | $428 | $760 | $183 | $1,044 | $1,795 | $120 | $137 | $173 | **$4,678** |
| openhands / kimi-k3 | $42 | $259 | $157 | $85 | $517 | $477 | $105 | $125 | $147 | **$1,914** |
| react / claude-opus-5-5 | — | $181 | $142 | $48 | $229 | $471 | $22 | $75 | $66 | **$1,233** |
| react / gpt-5.6-sol | $33 | $182 | $308 | $54 | $563 | $828 | $22 | $34 | $46 | **$2,070** |
| react / kimi-k3 | — | $185 | $162 | $57 | $289 | $378 | $64 | $48 | $73 | **$1,255** |
| **Total** | **$114** | **$2,979** | **$4,099** | **$2,379** | **$4,927** | **$5,927** | **$1,576** | **$1,744** | **$1,068** | **$24,813** |


## Likely range: 90% of input tokens as cache reads

| Agent / model | afm | corral_md | inference_opt | psychometrics | resistor_network | retrosynthesis | spectra_elucidation | stargazer | wetlab | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ai-scientist / claude-opus-5-5 | — | $408 | $572 | $535 | $550 | $230 | $364 | $356 | $133 | **$3,147** |
| claude-code / claude-opus-5-5 | — | $68 | $52 | $17 | $64 | $143 | $11 | $13 | $21 | **$388** |
| codex / gpt-5.6-sol | $9 | $98 | $168 | $45 | $239 | $355 | $31 | $34 | $38 | **$1,015** |
| openhands / kimi-k3 | $10 | $63 | $38 | $23 | $156 | $104 | $50 | $32 | $36 | **$510** |
| react / claude-opus-5-5 | — | $63 | $45 | $27 | $68 | $81 | $12 | $40 | $24 | **$361** |
| react / gpt-5.6-sol | $15 | $74 | $80 | $23 | $194 | $170 | $13 | $14 | $16 | **$599** |
| react / kimi-k3 | — | $55 | $44 | $19 | $89 | $86 | $44 | $18 | $21 | **$376** |
| **Total** | **$34** | **$829** | **$999** | **$687** | **$1,359** | **$1,168** | **$525** | **$507** | **$289** | **$6,397** |


## Tokens

All runs together used 5.90B input tokens and 146M output tokens. Input tokens make up 83–97% of the list-price cost, so the cache hit rate matters most to the real cost.

| Agent / model | Input tokens | Output tokens | Input share of list cost |
|---|---:|---:|---:|
| ai-scientist / claude-opus-5-5 | 2,560M | 83.1M | 86% |
| claude-code / claude-opus-5-5 | 401M | 7.8M | 91% |
| codex / gpt-5.6-sol | 1,130M | 7.8M | 97% |
| openhands / kimi-k3 | 675M | 12.2M | 92% |
| react / claude-opus-5-5 | 255M | 10.7M | 83% |
| react / gpt-5.6-sol | 454M | 12.7M | 88% |
| react / kimi-k3 | 423M | 12.0M | 88% |

## Observations

- **ai-scientist is about half of the total.** It used 2.56B input tokens, around 200M per environment, and is not the top scorer in any environment.
- **codex is the second most expensive.** Its 1.13B input tokens went mostly to retrosynthesis and resistor_network.
- **retrosynthesis and resistor_network cost the most** of the environments. afm is cheap only because three of its agent and model pairs have no runs.

## Caveats

- The reports do not say how many input tokens were cache reads or cache writes, so the real cost falls between the two tables.
- Cache writes, which Anthropic charges at a premium, are not counted.
- Only the runs in the reports are counted. Superseded runs, failed runs and technical reruns are not, so the actual spend is higher.
- Most prices come from third-party aggregators rather than the providers' own pricing pages.
- Provider dashboards (Anthropic, OpenAI, OpenRouter) or Langfuse give the actual spend.

## Sources

- [Claude Opus 5.5 pricing (eesel)](https://www.eesel.ai/blog/claude-opus-5-5-pricing)
- [Anthropic Opus 5.5 price cut (techjacksolutions)](https://techjacksolutions.com/ai-brief/anthropic-cuts-claude-opus-5-5-api-costs-40-percent-launches/)
- [GPT-5.6 Sol (OpenAI docs)](https://developers.openai.com/api/docs/models/gpt-5.6-sol.md)
- [Amazon Bedrock GPT-5.6 Sol reduced pricing](https://aws.amazon.com/about-aws/whats-new/2026/08/bedrock-openai-gpt-56-sol-reduced-pricing/)
- [GPT-5.6 Sol pricing (eesel)](https://www.eesel.ai/blog/gpt-5-6-sol-pricing)
- [Kimi K3 on OpenRouter](https://openrouter.ai/moonshotai/kimi-k3/pricing)
- [Kimi K3 (pricepertoken)](https://pricepertoken.com/pricing-page/model/moonshotai-kimi-k3)
- [Simon Willison on Kimi](https://simonwillison.net/random/kimi/)
