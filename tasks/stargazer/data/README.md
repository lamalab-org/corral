# Stargazer task bank

All tasks come from the Stargazer paper's benchmark, licensed CC BY 4.0.
`synthetic/` contains exactly the 20 tasks selected for benchmark Levels 1–2.

Records retain their Stargazer IDs and use the native JSON schema.
`meta.rv_semantics` identifies records that already use RV-only coordinates;
the loader converts REBOUND records to those coordinates while preserving
their noise realization.

`selection_manifest.json` records the paper attribution, task counts, selection
criteria, and membership for the two official levels. Level 1 selects 10
synthetic tasks at difficulties 5–7; Level 2 selects 10 at difficulties 8–10.
Custom selectors can select from these 20 bundled tasks.

`reference_audit.json` is a deterministic Corral-generated report of every
bundled reference submission under the unchanged final scoring contract.
Regenerate or verify it with `python -m stargazer.audit` or
`python -m stargazer.audit --check`, respectively.

See `../THIRD_PARTY_NOTICES.md` for full attribution.
