"""
Compute the TODO statistics for the paper from analysis/results/data/reports.jsonl.

Token counting logic:
  For each assistant message in a trace, we count:
    - INPUT  = tiktoken count of ALL preceding messages (system + user + prior assistant)
    - OUTPUT = tiktoken count of that assistant message
  This mirrors the actual API usage: every generation call sends the full
  conversation prefix as input and produces one completion as output.

Outputs:
  1. Total tokens broken down by model and verbosity
  2. Number of configuration--environment pairs
  3. Estimated API cost for proprietary models (GPT-4o, Claude 4.5)
  4. Malformed-response / scaffold-error rates per model
"""

import json
from collections import defaultdict
from pathlib import Path

import tiktoken
from loguru import logger

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-4.5": {"input": 3.00, "output": 15.00},
    # gpt-oss-120b is self-hosted, no API cost
}

enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(enc.encode(str(text)))


DATA_PATH = Path(__file__).parent / "results" / "data" / "reports.jsonl"

config_env_pairs = set()

tokens = defaultdict(lambda: defaultdict(lambda: {"input": 0, "output": 0}))

scaffold_errors = defaultdict(lambda: {"errors": 0, "total_trials": 0})
trials_affected_count = defaultdict(lambda: {"affected": 0, "total": 0})

total_trials = 0

with DATA_PATH.open() as f:
    for line in f:
        rec = json.loads(line)
        model = rec["model"]
        agent_type = rec["agent_type"]
        env = rec["environment"]
        verbosity = rec.get("Tool Verbosity", "")

        config_env_pairs.add((model, agent_type, verbosity, env))

        for task_data in rec["Task Results"].values():
            for trial in task_data["trials"]:
                total_trials += 1
                msgs = trial.get("messages", [])

                # Tokenize every message once, reuse counts
                msg_tok = [count_tokens(m.get("content", "")) for m in msgs]

                # For each assistant turn: input = sum of all prior, output = this turn
                for i, m in enumerate(msgs):
                    if m.get("role") == "assistant":
                        tokens[model][verbosity]["input"] += sum(msg_tok[:i])
                        tokens[model][verbosity]["output"] += msg_tok[i]

                # Scaffold errors
                key = (model, agent_type)
                scaffold_errors[key]["total_trials"] += 1
                trial_has_error = False
                for m in msgs:
                    content = str(m.get("content", ""))
                    if m.get("role") == "user" and "No actions to execute" in content:
                        scaffold_errors[key]["errors"] += 1
                        trial_has_error = True
                if agent_type == "react":
                    trials_affected_count[model]["total"] += 1
                    if trial_has_error:
                        trials_affected_count[model]["affected"] += 1


def fmt_tokens(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def agg_model(model):
    inp = sum(v["input"] for v in tokens[model].values())
    out = sum(v["output"] for v in tokens[model].values())
    return inp, out


def agg_all():
    inp = sum(v["input"] for m in tokens for v in tokens[m].values())
    out = sum(v["output"] for m in tokens for v in tokens[m].values())
    return inp, out


def agg_verbosity(verb):
    inp = sum(tokens[m][verb]["input"] for m in tokens)
    out = sum(tokens[m][verb]["output"] for m in tokens)
    return inp, out


logger.info("=" * 80)
logger.info("PAPER STATISTICS  (from reports.jsonl)")
logger.info("=" * 80)

n_pairs = len(config_env_pairs)
logger.info(f"\nConfiguration--environment pairs (N): {n_pairs}")
logger.info(
    f"  (x 5 trials each = {n_pairs * 5} expected trials, actual = {total_trials})"
)

all_models = sorted(tokens.keys())
all_verbosities = sorted({v for m in tokens for v in tokens[m]})
grand_in, grand_out = agg_all()

logger.info(f"\n{'='*80}")
logger.info(
    "TOKEN COUNTS BY MODEL  (input = all prior msgs per call, output = assistant msg)"
)
logger.info(f"{'='*80}")
for m in all_models:
    inp, out = agg_model(m)
    logger.info(
        f"  {m:15s}:  input {inp:>14,}  output {out:>14,}  total {inp+out:>14,}"
    )
logger.info(
    f"  {'GRAND TOTAL':15s}:  input {grand_in:>14,}  output {grand_out:>14,}  total {grand_in+grand_out:>14,}"
)
logger.info(f"\n  => Total tokens (all models): ~{fmt_tokens(grand_in + grand_out)}")

logger.info(f"\n{'='*80}")
logger.info("TOKEN COUNTS BY VERBOSITY")
logger.info(f"{'='*80}")
for verb in all_verbosities:
    inp, out = agg_verbosity(verb)
    logger.info(f"\n  Verbosity: {verb}")
    for m in all_models:
        vi = tokens[m][verb]["input"]
        vo = tokens[m][verb]["output"]
        logger.info(
            f"    {m:15s}:  input {vi:>14,}  output {vo:>14,}  total {vi+vo:>14,}"
        )
    logger.info(
        f"    {'SUBTOTAL':15s}:  input {inp:>14,}  output {out:>14,}  total {inp+out:>14,}"
    )

logger.info(f"\n{'='*80}")
logger.info("FLAT TABLE: MODEL x VERBOSITY")
logger.info(f"{'='*80}")
header = (
    f"  {'Model':15s} {'Verbosity':15s} {'Input':>14s} {'Output':>14s} {'Total':>14s}"
)
logger.info(header)
logger.info(f"  {'-'*15} {'-'*15} {'-'*14} {'-'*14} {'-'*14}")
for m in all_models:
    for verb in all_verbosities:
        vi = tokens[m][verb]["input"]
        vo = tokens[m][verb]["output"]
        logger.info(f"  {m:15s} {verb:15s} {vi:>14,} {vo:>14,} {vi+vo:>14,}")

logger.info(f"\n{'='*80}")
logger.info("ESTIMATED API COST (proprietary models only)")
logger.info(f"{'='*80}")
total_cost = 0.0
for m, prices in PRICING.items():
    inp, out = agg_model(m)
    cost_in = inp / 1e6 * prices["input"]
    cost_out = out / 1e6 * prices["output"]
    cost = cost_in + cost_out
    total_cost += cost
    logger.info(
        f"  {m:15s}:  input ${cost_in:>10,.2f}  output ${cost_out:>10,.2f}  total ${cost:>10,.2f}"
    )
logger.info(f"  {'TOTAL':15s}:  ${total_cost:>10,.2f}")

logger.info("\n  Cost breakdown by verbosity:")
for verb in all_verbosities:
    verb_cost = 0.0
    for m, prices in PRICING.items():
        vi = tokens[m][verb]["input"]
        vo = tokens[m][verb]["output"]
        verb_cost += vi / 1e6 * prices["input"] + vo / 1e6 * prices["output"]
    logger.info(f"    {verb:15s}: ${verb_cost:>10,.2f}")

logger.info(f"\n{'='*80}")
logger.info("MALFORMED-RESPONSE RATES (ReAct scaffold errors)")
logger.info(f"{'='*80}")
react_totals = {}
for (m, at), val in scaffold_errors.items():
    if at == "react":
        react_totals[m] = val

logger.info(
    f"  {'Model':15s} {'Errors':>8s} {'Trials':>8s} {'Err/Trial':>10s} {'% Affected':>12s}"
)
for m in sorted(react_totals.keys()):
    v = react_totals[m]
    rate = v["errors"] / v["total_trials"] if v["total_trials"] else 0
    ta = trials_affected_count.get(m, {"affected": 0, "total": 0})
    pct = ta["affected"] / ta["total"] * 100 if ta["total"] else 0
    logger.info(
        f"  {m:15s} {v['errors']:>8d} {v['total_trials']:>8d} {rate:>10.3f} {pct:>11.1f}%"
    )

logger.info(f"\n{'='*80}")
logger.info("LATEX-READY VALUES")
logger.info(f"{'='*80}")
logger.info(f"  Tokens:  ~{fmt_tokens(grand_in + grand_out)}")
logger.info(f"  N (config--env pairs):  {n_pairs}")
logger.info(f"  API cost:  ${total_cost:,.0f}")
logger.info("")
logger.info("  Malformed-response rates (ReAct only):")
for m in ["gpt-oss-120b", "gpt-4o", "claude-4.5"]:
    if m in react_totals:
        v = react_totals[m]
        rate = v["errors"] / v["total_trials"] if v["total_trials"] else 0
        ta = trials_affected_count.get(m, {"affected": 0, "total": 0})
        pct = ta["affected"] / ta["total"] * 100 if ta["total"] else 0
        logger.info(
            f"    {m}: {rate:.2f} errors/trial, {pct:.1f}% of react trials affected"
        )
