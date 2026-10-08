# Stargazer task bank

All tasks come from the Stargazer paper's benchmark, licensed CC BY 4.0.
`synthetic/` contains 28 tasks: the 20 tasks selected for benchmark Levels 1–2
and eight retained only for replaying the original Level 1 selection.

Records retain their Stargazer IDs and use the native JSON schema.
`meta.rv_semantics` identifies records that already use RV-only coordinates;
the loader converts REBOUND records to those coordinates while preserving
their noise realization.

`selection_manifest.json` records the paper attribution, task counts, selection
criteria, version, and membership for the two official levels. Level 1 selects
two single-planet tasks at each difficulty from 1 through 4, and two retained
two-planet tasks at difficulty 5. New tasks are copied from the pinned upstream
revision, choosing the first two reference-valid single-planet records by numeric
seed at each difficulty. Level 2 retains its ten tasks at difficulties 8–10.

`selectors/level_1_original.json` preserves the original ten-task selection at
difficulties 5–7. Pass it as `selector_path` to replay that selection. Custom
selectors can select from all 28 bundled tasks. Results on the revised default
Level 1 need a fresh model run; historical scores describe the old selection.

`reference_audit.json` is a deterministic Corral-generated report of every
bundled reference submission under the unchanged final scoring contract.
Regenerate or verify it with `python -m stargazer.audit` or
`python -m stargazer.audit --check`, respectively.

See `../THIRD_PARTY_NOTICES.md` for full attribution.
