# Codebook - adaptive item bank

`data.csv` and `holdout.csv` are tab-separated and hold binary responses to the same twenty items, one row per respondent. The holdout sample was collected separately from the calibration sample.

| variable | description |
|---|---|
| `participant_id` | respondent identifier, unique within each file |
| `CAT01`-`CAT20` | item response, 0 = incorrect, 1 = correct |

`vendor_review.csv` records the vendor's review of its own item bank.

| variable | description |
|---|---|
| `item` | item identifier |
| `review_batch` | batch the item was reviewed in |
| `reviewed_on` | date of that review |
| `vendor_flag` | whether the vendor marked the item for attention |
| `bank_version` | version of the bank the review applies to |
