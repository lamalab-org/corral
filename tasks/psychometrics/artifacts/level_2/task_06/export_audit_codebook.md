# Codebook - export audit

One row per item, from the export system's own logs. These are transmission records: they describe how each column was written out, not what the responses mean.

| variable | description |
|---|---|
| `item` | HSNS item |
| `export_batch` | batch the column was written in |
| `transmission_window` | when that batch was sent |
| `field_width` | characters allocated to the column |
| `rows_written` | rows the export reported writing |

Batch `B2` was re-sent the same evening after the first attempt was interrupted. The operator recorded no detail about what, if anything, differed in the second attempt.
