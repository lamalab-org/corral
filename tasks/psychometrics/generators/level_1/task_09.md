# Task 09 — relationships between the two instruments

## Task

You are given responses from an online personality survey collected in several countries. Using US respondents, investigate the measurement-error-adjusted correlation between every dimension of one instrument and every dimension of the other.

Before estimating the correlations, assess response quality. In particular, consider respondents whose item answers show no variation. The self-rated `accuracy` field is available but should be evaluated as evidence rather than treated as a guaranteed indicator of careless responding. Responses are five-point ratings and `0` denotes a missing response.

## What to report

Submit one complete lavaan/semopy model covering both instruments and one latent
correlation for every cross-instrument pair. Estimate the correlations after
applying the response-quality rule you judge appropriate.

Name factors as you wish; dimensions are matched by the items they cover.

## Why this is non-trivial

Careless response patterns can leave the factor structure and global fit looking acceptable while distorting correlations, especially weak ones. Self-reported data quality need not identify the affected respondents, so response patterns themselves may be more informative than an auxiliary field.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t09_careless_responding.py [--verify|--naive]
```
