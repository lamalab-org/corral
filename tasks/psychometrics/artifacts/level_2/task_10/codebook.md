# Codebook - model identification

Each `dataset_*.csv` is tab-separated and holds an independent anonymised sample answering the same twelve items. Responses are continuous and unbounded, centered around zero; empty fields denote missing responses.

| variable | meaning |
|---|---|
| `respondent_id` | row identifier |
| response format | Q03, Q07 and Q10 used a shared reverse-worded format; their values have already been keyed in the same direction as the other items |
| `Q01` | continuous response; zero is valid |
| `Q02` | continuous response; zero is valid |
| `Q03` | continuous response; zero is valid |
| `Q04` | continuous response; zero is valid |
| `Q05` | continuous response; zero is valid |
| `Q06` | continuous response; zero is valid |
| `Q07` | continuous response; zero is valid |
| `Q08` | continuous response; zero is valid |
| `Q09` | continuous response; zero is valid |
| `Q10` | continuous response; zero is valid |
| `Q11` | continuous response; zero is valid |
| `Q12` | continuous response; zero is valid |

`candidate_models.csv` gives each model's description and lavaan-style syntax. The order of the catalogue carries no information.
