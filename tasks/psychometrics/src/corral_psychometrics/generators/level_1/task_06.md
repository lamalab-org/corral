# Task 06 — relationships between latent dimensions

## Task

You are given responses from an online personality survey containing two instruments. Using respondents from the United States, investigate how the dimensions of one instrument relate to the dimensions of the other.

Estimate the correlation between every cross-instrument dimension pair while accounting for measurement error. Also decide which pairs should not be treated as distinct constructs: use a correlation of `0.80` or above as the threshold for indistinguishability. Responses are five-point ratings and `0` denotes a missing response.

`codebook.md` gives the response coding and the text of every item.

## What to report

Submit one complete lavaan/semopy model covering both instruments, one latent correlation for every cross-instrument pair, and the pair or pairs that meet the indistinguishability threshold.

Name factors as you wish; dimensions are matched by the items they cover.

## Why this is non-trivial

Correlations between raw totals are attenuated by measurement error, and the attenuation is uneven when dimensions have different loading quality. A weak observed relationship may therefore need more careful interpretation, while a very high latent relationship raises a separate discriminant-validity issue.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t06_latent_relationships.py [--verify|--naive]
```
