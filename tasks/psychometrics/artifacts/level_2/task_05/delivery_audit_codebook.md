# Codebook - delivery audit

One row per delivered session. Join to `data.csv` on `session_id`. The fields are delivery characteristics, not respondent identifiers; different people can share a device family or a quality band.

| variable | description |
|---|---|
| `session_id` | identifier assigned to this delivery session |
| `device_family` | pseudonymous device-family grouping observed by the panel |
| `completion_band` | coarse completion-time grouping |
| `response_quality_band` | coarse panel quality grouping |
