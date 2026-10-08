# Task 10 — item integrity

## Task

You are given responses from an online personality survey collected in several countries. Using US respondents, investigate the 10 HSNS items and classify each item as exactly one of:

- `sound`: the item and its recorded data are usable;
- `mis_keyed`: the response scale was stored in reverse;
- `missing_as_neutral`: non-responses were recorded as the midpoint;
- `truncated_scale`: the upper part of the response scale was not recorded;
- `weak_item`: the data are intact but the item measures the trait poorly.

The Dirty Dozen items are present but are not part of this task. Responses are five-point ratings and `0` denotes a missing response.

`codebook.md` gives the response coding and the text of every item.

## What to report

Submit a complete lavaan/semopy model for the HSNS and a classification for every HSNS item.

## Why this is non-trivial

A low loading does not identify the cause of an item's poor performance. Data corruption can resemble a weak item, while some corruption is visible mainly in the response range or category frequencies. The population used for analysis also matters when deciding which item is unusual.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t10_item_integrity.py [--verify|--naive]
```
