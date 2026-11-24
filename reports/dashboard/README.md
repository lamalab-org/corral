# Results dashboard

A standalone HTML page that compares every agent and model pair across the
benchmark environments, including the full AFM reports:

- a leaderboard with a level toggle
- pass@k and pass^k curves
- how many of 5 tries solved each task
- cost per trial and per solved trial
- a per-level heatmap
- native agent vs ReAct on the same model
- passing vs failing trials

It is published at <https://lamalab-org.github.io/corral/artifact/> by the
`Deploy Site & Docs` workflow (`.github/workflows/docs.yaml`), which reads the
current `trials.csv` files from the checked-out repository and rebuilds the page
on every deploy. The default build includes all nine environments and 7,000 trials.

## Build it locally

```bash
python reports/dashboard/fetch_trials.py --dest _data/reports    # download every trials.csv
python reports/dashboard/build_dashboard.py --reports _data/reports
```

This writes `reports/dashboard/corral_dashboard.html`. On a branch that has the
reports, such as `main`, skip the fetch and run
`python reports/dashboard/build_dashboard.py`. Both scripts use only the
standard library. The page loads d3 from a CDN. To leave out specific environments,
pass `--exclude <environment> ...`; all environments are included by default.

| File | Purpose |
|---|---|
| `fetch_trials.py` | Downloads `reports/<environment>/trials.csv` from GitHub (`--ref` picks the branch). |
| `build_dashboard.py` | Embeds the trials in one HTML page. Model prices are at the top and match `reports/cost_estimate.md`. |
| `page.html`, `calcs.js`, `charts.js` | Layout, calculations and charts. |
| `dash_shim.js`, `theme.css` | Runtime and theme for the standalone page. |
| `corral_dashboard.html` | The last built page. |
