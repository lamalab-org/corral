# Stargazer task bank

`synthetic/` contains the 100 released synthetic tasks (ten at each upstream
difficulty from 1 through 10). `real/` contains the 20 released archival RV
tasks. These JSON records were copied without modification from Stargazer
revision `3f617667472061e253288c7b26f0e70f186f2dff` and are licensed CC BY 4.0.

`reference_audit.json` is a deterministic Corral-generated report of every
published reference submission under the unchanged final scoring contract.
Regenerate or verify it with `python -m stargazer.audit` or
`python -m stargazer.audit --check`, respectively.

See `../THIRD_PARTY_NOTICES.md` for full attribution.
