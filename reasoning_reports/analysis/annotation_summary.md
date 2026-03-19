# Reasoning annotation analysis

- Node probability definition: node_count / number_of_messages_in_original_trace

## By model + env + level

### claude_sonnet_45/catalyst/level_1

- Traces: 15 | Total messages: 321 | Mean messages/trace: 21.40

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1169 | 1.1153 |
| n_H | 0.0095 | 0.0093 |
| n_T | 0.4341 | 0.4330 |
| n_E | 0.5110 | 0.5109 |
| n_J | 0.0842 | 0.0841 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0781 | 0.0779 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.6267 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.3132 |
| refute_neglect_rate | N/A |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | N/A |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 1 | 0.0667 | 1 | 0.0667 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.2000 | 3 | 0.2000 |
| evidence_ignored | 15 | 1.0000 | 121 | 8.0667 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 1 | 0.0667 | 1 | 0.0667 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 15 | 1.0000 | 15 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.2000 | 3 | 0.2000 |
| evidence_ignored | 15 | 1.0000 | 121 | 8.0667 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 1 | 0.0667 | 1 | 0.0667 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 15 | 1.0000 | 15 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.2000 | 4 | 0.2667 |
| evidence_handling | 15 | 1.0000 | 121 | 8.0667 |
| experimental_strategy | 15 | 1.0000 | 16 | 1.0667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.2000 | 4 | 0.2667 |
| evidence_handling | 15 | 1.0000 | 121 | 8.0667 |
| experimental_strategy | 15 | 1.0000 | 16 | 1.0667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 0.0667 | 1 | 0.0667 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/md/level_1

- Traces: 10 | Total messages: 255 | Mean messages/trace: 25.50

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2450 | 1.0157 |
| n_H | 0.1263 | 0.0902 |
| n_T | 0.4984 | 0.3059 |
| n_E | 0.4882 | 0.4863 |
| n_J | 0.1059 | 0.0980 |
| n_U | 0.0029 | 0.0039 |
| n_C | 0.0234 | 0.0314 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8200 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.0000 |
| orphan_evidence_rate | 0.4125 |
| refute_neglect_rate | 0.3929 |
| hypothesis_switch_without_eval_rate | 1.0000 |
| scientificness_score | 0.2111 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 1 | 0.1000 | 1 | 0.1000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 1 | 0.1000 | 1 | 0.1000 |
| abductive | 1 | 0.1000 | 1 | 0.1000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 9 | 0.9000 | 15 | 1.5000 |
| evidence_ignored | 9 | 0.9000 | 89 | 8.9000 |
| judgment_without_evidence | 1 | 0.1000 | 1 | 0.1000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 3 | 0.3000 | 9 | 0.9000 |
| hypothesis_to_commitment_shortcut | 1 | 0.1000 | 1 | 0.1000 |
| test_without_evidence | 3 | 0.3000 | 27 | 2.7000 |
| no_belief_revision | 9 | 0.9000 | 9 | 0.9000 |
| orphan_evidence | 3 | 0.3000 | 5 | 0.5000 |
| confirmation_only | 1 | 0.1000 | 1 | 0.1000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 9 | 0.9000 | 15 | 1.5000 |
| evidence_ignored | 10 | 1.0000 | 89 | 8.9000 |
| judgment_without_evidence | 1 | 0.1000 | 1 | 0.1000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 4 | 0.4000 | 13 | 1.3000 |
| hypothesis_to_commitment_shortcut | 1 | 0.1000 | 1 | 0.1000 |
| test_without_evidence | 5 | 0.5000 | 27 | 2.7000 |
| no_belief_revision | 9 | 0.9000 | 9 | 0.9000 |
| orphan_evidence | 3 | 0.3000 | 5 | 0.5000 |
| confirmation_only | 1 | 0.1000 | 1 | 0.1000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.9000 | 25 | 2.5000 |
| evidence_handling | 10 | 1.0000 | 122 | 12.2000 |
| experimental_strategy | 9 | 0.9000 | 10 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.9000 | 29 | 2.9000 |
| evidence_handling | 10 | 1.0000 | 122 | 12.2000 |
| experimental_strategy | 9 | 0.9000 | 10 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.2000 | 2 | 0.2000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 1 | 0.1000 | 1 | 0.1000 |

### claude_sonnet_45/md/level_2

- Traces: 10 | Total messages: 307 | Mean messages/trace: 30.70

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1205 | 1.0358 |
| n_H | 0.0954 | 0.0814 |
| n_T | 0.2482 | 0.2476 |
| n_E | 0.5676 | 0.5375 |
| n_J | 0.1430 | 0.1205 |
| n_U | 0.0101 | 0.0065 |
| n_C | 0.0562 | 0.0423 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8400 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.0000 |
| orphan_evidence_rate | 0.3914 |
| refute_neglect_rate | 0.4167 |
| hypothesis_switch_without_eval_rate | 0.6667 |
| scientificness_score | 0.3290 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 1 | 0.1000 | 1 | 0.1000 |
| abductive | 4 | 0.4000 | 4 | 0.4000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 9 | 0.9000 | 20 | 2.0000 |
| evidence_ignored | 10 | 1.0000 | 120 | 12.0000 |
| judgment_without_evidence | 2 | 0.2000 | 2 | 0.2000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 1 | 0.1000 | 4 | 0.4000 |
| hypothesis_to_commitment_shortcut | 3 | 0.3000 | 3 | 0.3000 |
| test_without_evidence | 0 | 0.0000 | 5 | 0.5000 |
| no_belief_revision | 8 | 0.8000 | 8 | 0.8000 |
| orphan_evidence | 2 | 0.2000 | 2 | 0.2000 |
| confirmation_only | 3 | 0.3000 | 3 | 0.3000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 9 | 0.9000 | 20 | 2.0000 |
| evidence_ignored | 10 | 1.0000 | 120 | 12.0000 |
| judgment_without_evidence | 2 | 0.2000 | 2 | 0.2000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 2 | 0.2000 | 4 | 0.4000 |
| hypothesis_to_commitment_shortcut | 3 | 0.3000 | 3 | 0.3000 |
| test_without_evidence | 4 | 0.4000 | 5 | 0.5000 |
| no_belief_revision | 8 | 0.8000 | 8 | 0.8000 |
| orphan_evidence | 2 | 0.2000 | 2 | 0.2000 |
| confirmation_only | 3 | 0.3000 | 3 | 0.3000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.9000 | 27 | 2.7000 |
| evidence_handling | 10 | 1.0000 | 129 | 12.9000 |
| experimental_strategy | 9 | 0.9000 | 11 | 1.1000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.9000 | 27 | 2.7000 |
| evidence_handling | 10 | 1.0000 | 129 | 12.9000 |
| experimental_strategy | 9 | 0.9000 | 11 | 1.1000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.5000 | 5 | 0.5000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/retrosynthesis/level_1

- Traces: 15 | Total messages: 245 | Mean messages/trace: 16.33

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 0.9041 | 0.9429 |
| n_H | 0.0580 | 0.0653 |
| n_T | 0.2169 | 0.2163 |
| n_E | 0.3740 | 0.3959 |
| n_J | 0.1618 | 0.1878 |
| n_U | 0.0171 | 0.0204 |
| n_C | 0.0764 | 0.0571 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7333 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.6250 |
| orphan_evidence_rate | 0.0293 |
| refute_neglect_rate | 0.2778 |
| hypothesis_switch_without_eval_rate | 1.0000 |
| scientificness_score | 0.4530 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 1 | 0.0667 | 1 | 0.0667 |
| exploratory_to_confirmatory | 1 | 0.0667 | 1 | 0.0667 |
| bayesian_belief_updating | 2 | 0.1333 | 2 | 0.1333 |
| abductive | 5 | 0.3333 | 5 | 0.3333 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 1 | 0.0667 | 3 | 0.2000 |
| evidence_ignored | 8 | 0.5333 | 28 | 1.8667 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 1 | 0.0667 | 1 | 0.0667 |
| unresolved_contradiction | 2 | 0.1333 | 7 | 0.4667 |
| hypothesis_to_commitment_shortcut | 1 | 0.0667 | 1 | 0.0667 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 11 | 0.7333 | 11 | 0.7333 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 2 | 0.1333 | 3 | 0.2000 |
| evidence_ignored | 9 | 0.6000 | 28 | 1.8667 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 1 | 0.0667 | 1 | 0.0667 |
| unresolved_contradiction | 2 | 0.1333 | 7 | 0.4667 |
| hypothesis_to_commitment_shortcut | 1 | 0.0667 | 1 | 0.0667 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 11 | 0.7333 | 11 | 0.7333 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.1333 | 10 | 0.6667 |
| evidence_handling | 8 | 0.5333 | 28 | 1.8667 |
| experimental_strategy | 12 | 0.8000 | 13 | 0.8667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.2000 | 10 | 0.6667 |
| evidence_handling | 9 | 0.6000 | 28 | 1.8667 |
| experimental_strategy | 12 | 0.8000 | 13 | 0.8667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 6 | 0.4000 | 7 | 0.4667 |
| evidence_handling | 1 | 0.0667 | 1 | 0.0667 |
| experimental_strategy | 1 | 0.0667 | 1 | 0.0667 |

### claude_sonnet_45/retrosynthesis/level_2

- Traces: 15 | Total messages: 467 | Mean messages/trace: 31.13

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2421 | 1.3405 |
| n_H | 0.1073 | 0.1156 |
| n_T | 0.4225 | 0.4604 |
| n_E | 0.4907 | 0.5246 |
| n_J | 0.1468 | 0.1670 |
| n_U | 0.0289 | 0.0385 |
| n_C | 0.0458 | 0.0343 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9200 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.8852 |
| orphan_evidence_rate | 0.1401 |
| refute_neglect_rate | 0.5341 |
| hypothesis_switch_without_eval_rate | 1.0000 |
| scientificness_score | 0.5431 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 1 | 0.0667 | 1 | 0.0667 |
| ml_make_it_work | 3 | 0.2000 | 3 | 0.2000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 4 | 0.2667 | 4 | 0.2667 |
| abductive | 1 | 0.0667 | 1 | 0.0667 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 9 | 0.6000 | 33 | 2.2000 |
| evidence_ignored | 15 | 1.0000 | 121 | 8.0667 |
| judgment_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| dead_end_update | 4 | 0.2667 | 4 | 0.2667 |
| unresolved_contradiction | 3 | 0.2000 | 10 | 0.6667 |
| hypothesis_to_commitment_shortcut | 3 | 0.2000 | 3 | 0.2000 |
| test_without_evidence | 0 | 0.0000 | 5 | 0.3333 |
| no_belief_revision | 6 | 0.4000 | 6 | 0.4000 |
| orphan_evidence | 2 | 0.1333 | 2 | 0.1333 |
| confirmation_only | 2 | 0.1333 | 2 | 0.1333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 0.6667 | 33 | 2.2000 |
| evidence_ignored | 15 | 1.0000 | 121 | 8.0667 |
| judgment_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| dead_end_update | 4 | 0.2667 | 4 | 0.2667 |
| unresolved_contradiction | 7 | 0.4667 | 18 | 1.2000 |
| hypothesis_to_commitment_shortcut | 3 | 0.2000 | 3 | 0.2000 |
| test_without_evidence | 3 | 0.2000 | 5 | 0.3333 |
| no_belief_revision | 6 | 0.4000 | 6 | 0.4000 |
| orphan_evidence | 2 | 0.1333 | 2 | 0.1333 |
| confirmation_only | 2 | 0.1333 | 2 | 0.1333 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 0.6667 | 45 | 3.0000 |
| evidence_handling | 15 | 1.0000 | 129 | 8.6000 |
| experimental_strategy | 10 | 0.6667 | 13 | 0.8667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 11 | 0.7333 | 53 | 3.5333 |
| evidence_handling | 15 | 1.0000 | 129 | 8.6000 |
| experimental_strategy | 10 | 0.6667 | 13 | 0.8667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 6 | 0.4000 | 6 | 0.4000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 4 | 0.2667 | 4 | 0.2667 |

### claude_sonnet_45/retrosynthesis/level_3

- Traces: 15 | Total messages: 543 | Mean messages/trace: 36.20

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2764 | 1.2928 |
| n_H | 0.1066 | 0.1031 |
| n_T | 0.4017 | 0.4088 |
| n_E | 0.5121 | 0.5175 |
| n_J | 0.1875 | 0.1971 |
| n_U | 0.0335 | 0.0350 |
| n_C | 0.0350 | 0.0313 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9600 |
| loop_density | 0.0008 |
| update_grounding_rate | 0.6806 |
| orphan_evidence_rate | 0.0826 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.5667 |
| scientificness_score | 0.5773 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 1 | 0.0667 | 1 | 0.0667 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.0667 | 1 | 0.0667 |
| bayesian_belief_updating | 5 | 0.3333 | 5 | 0.3333 |
| abductive | 1 | 0.0667 | 1 | 0.0667 |
| triangulation | 1 | 0.0667 | 1 | 0.0667 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 12 | 0.8000 | 33 | 2.2000 |
| evidence_ignored | 15 | 1.0000 | 129 | 8.6000 |
| judgment_without_evidence | 2 | 0.1333 | 2 | 0.1333 |
| dead_end_update | 6 | 0.4000 | 6 | 0.4000 |
| unresolved_contradiction | 6 | 0.4000 | 14 | 0.9333 |
| hypothesis_to_commitment_shortcut | 3 | 0.2000 | 3 | 0.2000 |
| test_without_evidence | 0 | 0.0000 | 5 | 0.3333 |
| no_belief_revision | 3 | 0.2000 | 3 | 0.2000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 12 | 0.8000 | 33 | 2.2000 |
| evidence_ignored | 15 | 1.0000 | 129 | 8.6000 |
| judgment_without_evidence | 2 | 0.1333 | 2 | 0.1333 |
| dead_end_update | 6 | 0.4000 | 6 | 0.4000 |
| unresolved_contradiction | 7 | 0.4667 | 15 | 1.0000 |
| hypothesis_to_commitment_shortcut | 3 | 0.2000 | 3 | 0.2000 |
| test_without_evidence | 5 | 0.3333 | 5 | 0.3333 |
| no_belief_revision | 3 | 0.2000 | 3 | 0.2000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.8667 | 50 | 3.3333 |
| evidence_handling | 15 | 1.0000 | 136 | 9.0667 |
| experimental_strategy | 10 | 0.6667 | 12 | 0.8000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.8667 | 51 | 3.4000 |
| evidence_handling | 15 | 1.0000 | 136 | 9.0667 |
| experimental_strategy | 10 | 0.6667 | 12 | 0.8000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 6 | 0.4000 | 7 | 0.4667 |
| evidence_handling | 2 | 0.1333 | 2 | 0.1333 |
| experimental_strategy | 1 | 0.0667 | 1 | 0.0667 |

### claude_sonnet_45/spectra/level_1

- Traces: 22 | Total messages: 472 | Mean messages/trace: 21.45

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1720 | 1.2394 |
| n_H | 0.1500 | 0.1695 |
| n_T | 0.0755 | 0.0911 |
| n_E | 0.4501 | 0.4725 |
| n_J | 0.3808 | 0.4004 |
| n_U | 0.0486 | 0.0487 |
| n_C | 0.0670 | 0.0572 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8636 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.3556 |
| orphan_evidence_rate | 0.0805 |
| refute_neglect_rate | 0.1119 |
| hypothesis_switch_without_eval_rate | 0.3250 |
| scientificness_score | 0.4563 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 11 | 0.5000 | 11 | 0.5000 |
| abductive | 21 | 0.9545 | 21 | 0.9545 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 19 | 0.8636 | 54 | 2.4545 |
| evidence_ignored | 3 | 0.1364 | 44 | 2.0000 |
| judgment_without_evidence | 7 | 0.3182 | 19 | 0.8636 |
| dead_end_update | 2 | 0.0909 | 2 | 0.0909 |
| unresolved_contradiction | 3 | 0.1364 | 11 | 0.5000 |
| hypothesis_to_commitment_shortcut | 4 | 0.1818 | 5 | 0.2273 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 7 | 0.3182 | 7 | 0.3182 |
| orphan_evidence | 2 | 0.0909 | 4 | 0.1818 |
| confirmation_only | 1 | 0.0455 | 2 | 0.0909 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 19 | 0.8636 | 54 | 2.4545 |
| evidence_ignored | 17 | 0.7727 | 44 | 2.0000 |
| judgment_without_evidence | 7 | 0.3182 | 19 | 0.8636 |
| dead_end_update | 2 | 0.0909 | 2 | 0.0909 |
| unresolved_contradiction | 7 | 0.3182 | 12 | 0.5455 |
| hypothesis_to_commitment_shortcut | 4 | 0.1818 | 5 | 0.2273 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 7 | 0.3182 | 7 | 0.3182 |
| orphan_evidence | 2 | 0.0909 | 4 | 0.1818 |
| confirmation_only | 1 | 0.0455 | 2 | 0.0909 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 0.9091 | 67 | 3.0455 |
| evidence_handling | 8 | 0.3636 | 67 | 3.0455 |
| experimental_strategy | 11 | 0.5000 | 14 | 0.6364 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 0.9091 | 68 | 3.0909 |
| evidence_handling | 17 | 0.7727 | 67 | 3.0455 |
| experimental_strategy | 11 | 0.5000 | 14 | 0.6364 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 21 | 0.9545 | 32 | 1.4545 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/spectra/level_2

- Traces: 22 | Total messages: 508 | Mean messages/trace: 23.09

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2323 | 1.3268 |
| n_H | 0.1815 | 0.2146 |
| n_T | 0.1123 | 0.1358 |
| n_E | 0.4613 | 0.4843 |
| n_J | 0.3774 | 0.4016 |
| n_U | 0.0455 | 0.0453 |
| n_C | 0.0543 | 0.0453 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8818 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.4020 |
| orphan_evidence_rate | 0.1049 |
| refute_neglect_rate | 0.0738 |
| hypothesis_switch_without_eval_rate | 0.2667 |
| scientificness_score | 0.4694 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 12 | 0.5455 | 12 | 0.5455 |
| abductive | 20 | 0.9091 | 20 | 0.9091 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 17 | 0.7727 | 64 | 2.9091 |
| evidence_ignored | 6 | 0.2727 | 54 | 2.4545 |
| judgment_without_evidence | 10 | 0.4545 | 28 | 1.2727 |
| dead_end_update | 3 | 0.1364 | 3 | 0.1364 |
| unresolved_contradiction | 1 | 0.0455 | 12 | 0.5455 |
| hypothesis_to_commitment_shortcut | 3 | 0.1364 | 3 | 0.1364 |
| test_without_evidence | 0 | 0.0000 | 1 | 0.0455 |
| no_belief_revision | 5 | 0.2273 | 5 | 0.2273 |
| orphan_evidence | 2 | 0.0909 | 2 | 0.0909 |
| confirmation_only | 3 | 0.1364 | 3 | 0.1364 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 18 | 0.8182 | 64 | 2.9091 |
| evidence_ignored | 20 | 0.9091 | 54 | 2.4545 |
| judgment_without_evidence | 10 | 0.4545 | 28 | 1.2727 |
| dead_end_update | 3 | 0.1364 | 3 | 0.1364 |
| unresolved_contradiction | 9 | 0.4091 | 24 | 1.0909 |
| hypothesis_to_commitment_shortcut | 3 | 0.1364 | 3 | 0.1364 |
| test_without_evidence | 1 | 0.0455 | 1 | 0.0455 |
| no_belief_revision | 5 | 0.2273 | 5 | 0.2273 |
| orphan_evidence | 2 | 0.0909 | 2 | 0.0909 |
| confirmation_only | 3 | 0.1364 | 3 | 0.1364 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 19 | 0.8636 | 79 | 3.5909 |
| evidence_handling | 12 | 0.5455 | 85 | 3.8636 |
| experimental_strategy | 9 | 0.4091 | 11 | 0.5000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 19 | 0.8636 | 91 | 4.1364 |
| evidence_handling | 21 | 0.9545 | 85 | 3.8636 |
| experimental_strategy | 9 | 0.4091 | 11 | 0.5000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 22 | 1.0000 | 32 | 1.4545 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/wetlab/level1

- Traces: 10 | Total messages: 330 | Mean messages/trace: 33.00

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.3778 | 1.3818 |
| n_H | 0.1772 | 0.1788 |
| n_T | 0.2625 | 0.2667 |
| n_E | 0.4808 | 0.4788 |
| n_J | 0.3692 | 0.3697 |
| n_U | 0.0569 | 0.0576 |
| n_C | 0.0312 | 0.0303 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9800 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.2037 |
| orphan_evidence_rate | 0.1885 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.1875 |
| scientificness_score | 0.3955 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 4 | 0.4000 | 4 | 0.4000 |
| abductive | 10 | 1.0000 | 10 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 1.0000 | 56 | 5.6000 |
| evidence_ignored | 3 | 0.3000 | 38 | 3.8000 |
| judgment_without_evidence | 8 | 0.8000 | 14 | 1.4000 |
| dead_end_update | 2 | 0.2000 | 2 | 0.2000 |
| unresolved_contradiction | 2 | 0.2000 | 4 | 0.4000 |
| hypothesis_to_commitment_shortcut | 1 | 0.1000 | 1 | 0.1000 |
| test_without_evidence | 0 | 0.0000 | 1 | 0.1000 |
| no_belief_revision | 1 | 0.1000 | 1 | 0.1000 |
| orphan_evidence | 9 | 0.9000 | 17 | 1.7000 |
| confirmation_only | 1 | 0.1000 | 1 | 0.1000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 1.0000 | 56 | 5.6000 |
| evidence_ignored | 10 | 1.0000 | 38 | 3.8000 |
| judgment_without_evidence | 8 | 0.8000 | 14 | 1.4000 |
| dead_end_update | 2 | 0.2000 | 2 | 0.2000 |
| unresolved_contradiction | 3 | 0.3000 | 5 | 0.5000 |
| hypothesis_to_commitment_shortcut | 1 | 0.1000 | 1 | 0.1000 |
| test_without_evidence | 1 | 0.1000 | 1 | 0.1000 |
| no_belief_revision | 1 | 0.1000 | 1 | 0.1000 |
| orphan_evidence | 9 | 0.9000 | 17 | 1.7000 |
| confirmation_only | 1 | 0.1000 | 1 | 0.1000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 61 | 6.1000 |
| evidence_handling | 10 | 1.0000 | 70 | 7.0000 |
| experimental_strategy | 4 | 0.4000 | 4 | 0.4000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 62 | 6.2000 |
| evidence_handling | 10 | 1.0000 | 70 | 7.0000 |
| experimental_strategy | 4 | 0.4000 | 4 | 0.4000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 14 | 1.4000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/wetlab/level2

- Traces: 10 | Total messages: 562 | Mean messages/trace: 56.20

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.3594 | 1.3754 |
| n_H | 0.1736 | 0.1726 |
| n_T | 0.3294 | 0.3363 |
| n_E | 0.4738 | 0.4786 |
| n_J | 0.3187 | 0.3274 |
| n_U | 0.0428 | 0.0409 |
| n_C | 0.0211 | 0.0196 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 1.0000 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.0250 |
| orphan_evidence_rate | 0.2286 |
| refute_neglect_rate | 0.0333 |
| hypothesis_switch_without_eval_rate | 0.1648 |
| scientificness_score | 0.3475 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 10 | 1.0000 | 10 | 1.0000 |
| abductive | 10 | 1.0000 | 10 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 1.0000 | 92 | 9.2000 |
| evidence_ignored | 6 | 0.6000 | 78 | 7.8000 |
| judgment_without_evidence | 3 | 0.3000 | 7 | 0.7000 |
| dead_end_update | 4 | 0.4000 | 5 | 0.5000 |
| unresolved_contradiction | 0 | 0.0000 | 8 | 0.8000 |
| hypothesis_to_commitment_shortcut | 1 | 0.1000 | 3 | 0.3000 |
| test_without_evidence | 0 | 0.0000 | 3 | 0.3000 |
| no_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| orphan_evidence | 10 | 1.0000 | 24 | 2.4000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 1.0000 | 92 | 9.2000 |
| evidence_ignored | 10 | 1.0000 | 78 | 7.8000 |
| judgment_without_evidence | 3 | 0.3000 | 7 | 0.7000 |
| dead_end_update | 4 | 0.4000 | 5 | 0.5000 |
| unresolved_contradiction | 6 | 0.6000 | 13 | 1.3000 |
| hypothesis_to_commitment_shortcut | 1 | 0.1000 | 3 | 0.3000 |
| test_without_evidence | 3 | 0.3000 | 3 | 0.3000 |
| no_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| orphan_evidence | 10 | 1.0000 | 24 | 2.4000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 100 | 10.0000 |
| evidence_handling | 10 | 1.0000 | 112 | 11.2000 |
| experimental_strategy | 4 | 0.4000 | 8 | 0.8000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 105 | 10.5000 |
| evidence_handling | 10 | 1.0000 | 112 | 11.2000 |
| experimental_strategy | 4 | 0.4000 | 8 | 0.8000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 20 | 2.0000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/wetlab/level3

- Traces: 10 | Total messages: 702 | Mean messages/trace: 70.20

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.3853 | 1.3860 |
| n_H | 0.1855 | 0.1852 |
| n_T | 0.3286 | 0.3291 |
| n_E | 0.4917 | 0.4943 |
| n_J | 0.3292 | 0.3276 |
| n_U | 0.0355 | 0.0356 |
| n_C | 0.0148 | 0.0142 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 1.0000 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.1500 |
| orphan_evidence_rate | 0.1793 |
| refute_neglect_rate | 0.1389 |
| hypothesis_switch_without_eval_rate | 0.1667 |
| scientificness_score | 0.3791 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.1000 | 1 | 0.1000 |
| bayesian_belief_updating | 9 | 0.9000 | 9 | 0.9000 |
| abductive | 10 | 1.0000 | 10 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 1.0000 | 125 | 12.5000 |
| evidence_ignored | 6 | 0.6000 | 107 | 10.7000 |
| judgment_without_evidence | 8 | 0.8000 | 14 | 1.4000 |
| dead_end_update | 4 | 0.4000 | 8 | 0.8000 |
| unresolved_contradiction | 4 | 0.4000 | 17 | 1.7000 |
| hypothesis_to_commitment_shortcut | 5 | 0.5000 | 13 | 1.3000 |
| test_without_evidence | 0 | 0.0000 | 1 | 0.1000 |
| no_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| orphan_evidence | 9 | 0.9000 | 24 | 2.4000 |
| confirmation_only | 2 | 0.2000 | 3 | 0.3000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 1.0000 | 125 | 12.5000 |
| evidence_ignored | 10 | 1.0000 | 107 | 10.7000 |
| judgment_without_evidence | 8 | 0.8000 | 14 | 1.4000 |
| dead_end_update | 4 | 0.4000 | 8 | 0.8000 |
| unresolved_contradiction | 7 | 0.7000 | 24 | 2.4000 |
| hypothesis_to_commitment_shortcut | 5 | 0.5000 | 13 | 1.3000 |
| test_without_evidence | 1 | 0.1000 | 1 | 0.1000 |
| no_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| orphan_evidence | 9 | 0.9000 | 24 | 2.4000 |
| confirmation_only | 2 | 0.2000 | 3 | 0.3000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 145 | 14.5000 |
| evidence_handling | 10 | 1.0000 | 146 | 14.6000 |
| experimental_strategy | 6 | 0.6000 | 21 | 2.1000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 152 | 15.2000 |
| evidence_handling | 10 | 1.0000 | 146 | 14.6000 |
| experimental_strategy | 6 | 0.6000 | 21 | 2.1000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 19 | 1.9000 |
| evidence_handling | 1 | 0.1000 | 1 | 0.1000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/catalyst/level_1

- Traces: 15 | Total messages: 335 | Mean messages/trace: 22.33

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1345 | 1.0836 |
| n_H | 0.0105 | 0.0090 |
| n_T | 0.4632 | 0.4239 |
| n_E | 0.5207 | 0.5194 |
| n_J | 0.0702 | 0.0716 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0698 | 0.0597 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.6400 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.0237 |
| refute_neglect_rate | N/A |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | N/A |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 2 | 0.1333 | 2 | 0.1333 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 2 | 0.1333 | 2 | 0.1333 |
| evidence_ignored | 14 | 0.9333 | 113 | 7.5333 |
| judgment_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 1 | 0.0667 | 1 | 0.0667 |
| test_without_evidence | 0 | 0.0000 | 1 | 0.0667 |
| no_belief_revision | 15 | 1.0000 | 15 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 2 | 0.1333 | 2 | 0.1333 |
| evidence_ignored | 15 | 1.0000 | 113 | 7.5333 |
| judgment_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 1 | 0.0667 | 1 | 0.0667 |
| test_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| no_belief_revision | 15 | 1.0000 | 15 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.1333 | 3 | 0.2000 |
| evidence_handling | 15 | 1.0000 | 115 | 7.6667 |
| experimental_strategy | 15 | 1.0000 | 16 | 1.0667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.1333 | 3 | 0.2000 |
| evidence_handling | 15 | 1.0000 | 115 | 7.6667 |
| experimental_strategy | 15 | 1.0000 | 16 | 1.0667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.1333 | 2 | 0.1333 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/md/level_1

- Traces: 10 | Total messages: 390 | Mean messages/trace: 39.00

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1739 | 1.1795 |
| n_H | 0.0836 | 0.0872 |
| n_T | 0.3793 | 0.3821 |
| n_E | 0.5003 | 0.5000 |
| n_J | 0.1861 | 0.1872 |
| n_U | 0.0122 | 0.0128 |
| n_C | 0.0123 | 0.0103 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8800 |
| loop_density | 0.0000 |
| update_grounding_rate | 1.0000 |
| orphan_evidence_rate | 0.2306 |
| refute_neglect_rate | 0.0643 |
| hypothesis_switch_without_eval_rate | 1.0000 |
| scientificness_score | 0.5705 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 5 | 0.5000 | 5 | 0.5000 |
| bayesian_belief_updating | 4 | 0.4000 | 4 | 0.4000 |
| abductive | 9 | 0.9000 | 9 | 0.9000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 7 | 0.7000 | 19 | 1.9000 |
| evidence_ignored | 10 | 1.0000 | 95 | 9.5000 |
| judgment_without_evidence | 1 | 0.1000 | 2 | 0.2000 |
| dead_end_update | 1 | 0.1000 | 1 | 0.1000 |
| unresolved_contradiction | 4 | 0.4000 | 11 | 1.1000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 2 | 0.2000 |
| no_belief_revision | 6 | 0.6000 | 6 | 0.6000 |
| orphan_evidence | 1 | 0.1000 | 1 | 0.1000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 8 | 0.8000 | 19 | 1.9000 |
| evidence_ignored | 10 | 1.0000 | 95 | 9.5000 |
| judgment_without_evidence | 1 | 0.1000 | 2 | 0.2000 |
| dead_end_update | 1 | 0.1000 | 1 | 0.1000 |
| unresolved_contradiction | 6 | 0.6000 | 11 | 1.1000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 2 | 0.2000 | 2 | 0.2000 |
| no_belief_revision | 6 | 0.6000 | 6 | 0.6000 |
| orphan_evidence | 1 | 0.1000 | 1 | 0.1000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 30 | 3.0000 |
| evidence_handling | 10 | 1.0000 | 100 | 10.0000 |
| experimental_strategy | 7 | 0.7000 | 7 | 0.7000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 30 | 3.0000 |
| evidence_handling | 10 | 1.0000 | 100 | 10.0000 |
| experimental_strategy | 7 | 0.7000 | 7 | 0.7000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 13 | 1.3000 |
| evidence_handling | 5 | 0.5000 | 5 | 0.5000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/md/level_2

- Traces: 5 | Total messages: 209 | Mean messages/trace: 41.80

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2805 | 1.2823 |
| n_H | 0.1107 | 0.1100 |
| n_T | 0.4052 | 0.4067 |
| n_E | 0.5283 | 0.5311 |
| n_J | 0.2014 | 0.2010 |
| n_U | 0.0294 | 0.0287 |
| n_C | 0.0054 | 0.0048 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9600 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.5000 |
| orphan_evidence_rate | 0.2392 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.6667 |
| scientificness_score | 0.4534 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 2 | 0.4000 | 2 | 0.4000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 4 | 0.8000 | 4 | 0.8000 |
| bayesian_belief_updating | 3 | 0.6000 | 3 | 0.6000 |
| abductive | 5 | 1.0000 | 5 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 4 | 0.8000 | 12 | 2.4000 |
| evidence_ignored | 5 | 1.0000 | 53 | 10.6000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 2 | 0.4000 | 2 | 0.4000 |
| unresolved_contradiction | 2 | 0.4000 | 3 | 0.6000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 2 | 0.4000 |
| no_belief_revision | 1 | 0.2000 | 1 | 0.2000 |
| orphan_evidence | 1 | 0.2000 | 2 | 0.4000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 5 | 1.0000 | 12 | 2.4000 |
| evidence_ignored | 5 | 1.0000 | 53 | 10.6000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 2 | 0.4000 | 2 | 0.4000 |
| unresolved_contradiction | 3 | 0.6000 | 4 | 0.8000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 2 | 0.4000 | 2 | 0.4000 |
| no_belief_revision | 1 | 0.2000 | 1 | 0.2000 |
| orphan_evidence | 1 | 0.2000 | 2 | 0.4000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 15 | 3.0000 |
| evidence_handling | 5 | 1.0000 | 57 | 11.4000 |
| experimental_strategy | 3 | 0.6000 | 3 | 0.6000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 16 | 3.2000 |
| evidence_handling | 5 | 1.0000 | 57 | 11.4000 |
| experimental_strategy | 3 | 0.6000 | 3 | 0.6000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 10 | 2.0000 |
| evidence_handling | 4 | 0.8000 | 4 | 0.8000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/ml/level_1

- Traces: 15 | Total messages: 233 | Mean messages/trace: 15.53

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.0862 | 1.1030 |
| n_H | 0.0000 | 0.0000 |
| n_T | 0.4433 | 0.4506 |
| n_E | 0.4493 | 0.4549 |
| n_J | 0.1049 | 0.1073 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0887 | 0.0901 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.6000 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.0778 |
| refute_neglect_rate | N/A |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | N/A |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 0 | 0.0000 | 0 | 0.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 0 | 0.0000 | 0 | 0.0000 |
| evidence_ignored | 14 | 0.9333 | 67 | 4.4667 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 15 | 1.0000 | 15 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 0 | 0.0000 | 0 | 0.0000 |
| evidence_ignored | 14 | 0.9333 | 67 | 4.4667 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 15 | 1.0000 | 15 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 14 | 0.9333 | 67 | 4.4667 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 14 | 0.9333 | 67 | 4.4667 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/retrosynthesis/level_1

- Traces: 15 | Total messages: 239 | Mean messages/trace: 15.93

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.0190 | 1.0460 |
| n_H | 0.0401 | 0.0377 |
| n_T | 0.3512 | 0.3640 |
| n_E | 0.3822 | 0.4226 |
| n_J | 0.1531 | 0.1590 |
| n_U | 0.0019 | 0.0042 |
| n_C | 0.0906 | 0.0586 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7067 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.0000 |
| orphan_evidence_rate | 0.0300 |
| refute_neglect_rate | N/A |
| hypothesis_switch_without_eval_rate | 1.0000 |
| scientificness_score | 0.2895 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.0667 | 1 | 0.0667 |
| bayesian_belief_updating | 1 | 0.0667 | 1 | 0.0667 |
| abductive | 5 | 0.3333 | 5 | 0.3333 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 6 | 0.4000 | 6 | 0.4000 |
| evidence_ignored | 10 | 0.6667 | 39 | 2.6000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 3 | 0.2000 | 3 | 0.2000 |
| test_without_evidence | 0 | 0.0000 | 1 | 0.0667 |
| no_belief_revision | 14 | 0.9333 | 14 | 0.9333 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 6 | 0.4000 | 6 | 0.4000 |
| evidence_ignored | 10 | 0.6667 | 39 | 2.6000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 3 | 0.2000 | 3 | 0.2000 |
| test_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| no_belief_revision | 14 | 0.9333 | 14 | 0.9333 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 6 | 0.4000 | 9 | 0.6000 |
| evidence_handling | 10 | 0.6667 | 40 | 2.6667 |
| experimental_strategy | 14 | 0.9333 | 17 | 1.1333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 6 | 0.4000 | 9 | 0.6000 |
| evidence_handling | 11 | 0.7333 | 40 | 2.6667 |
| experimental_strategy | 14 | 0.9333 | 17 | 1.1333 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.3333 | 6 | 0.4000 |
| evidence_handling | 1 | 0.0667 | 1 | 0.0667 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/retrosynthesis/level_2

- Traces: 15 | Total messages: 295 | Mean messages/trace: 19.67

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.0418 | 1.1593 |
| n_H | 0.0809 | 0.0780 |
| n_T | 0.3172 | 0.3763 |
| n_E | 0.4012 | 0.4576 |
| n_J | 0.1630 | 0.1763 |
| n_U | 0.0132 | 0.0203 |
| n_C | 0.0662 | 0.0508 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8533 |
| loop_density | 0.0000 |
| update_grounding_rate | 1.0000 |
| orphan_evidence_rate | 0.0667 |
| refute_neglect_rate | 0.3333 |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | 0.5495 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 2 | 0.1333 | 2 | 0.1333 |
| exploratory_to_confirmatory | 3 | 0.2000 | 3 | 0.2000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 8 | 0.5333 | 8 | 0.5333 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 2 | 0.1333 | 2 | 0.1333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 7 | 0.4667 | 12 | 0.8000 |
| evidence_ignored | 12 | 0.8000 | 57 | 3.8000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 2 | 0.1333 | 6 | 0.4000 |
| hypothesis_to_commitment_shortcut | 2 | 0.1333 | 2 | 0.1333 |
| test_without_evidence | 0 | 0.0000 | 3 | 0.2000 |
| no_belief_revision | 10 | 0.6667 | 10 | 0.6667 |
| orphan_evidence | 1 | 0.0667 | 1 | 0.0667 |
| confirmation_only | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 7 | 0.4667 | 12 | 0.8000 |
| evidence_ignored | 14 | 0.9333 | 57 | 3.8000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 2 | 0.1333 | 6 | 0.4000 |
| hypothesis_to_commitment_shortcut | 2 | 0.1333 | 2 | 0.1333 |
| test_without_evidence | 2 | 0.1333 | 3 | 0.2000 |
| no_belief_revision | 10 | 0.6667 | 10 | 0.6667 |
| orphan_evidence | 1 | 0.0667 | 1 | 0.0667 |
| confirmation_only | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.5333 | 21 | 1.4000 |
| evidence_handling | 12 | 0.8000 | 61 | 4.0667 |
| experimental_strategy | 11 | 0.7333 | 12 | 0.8000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.5333 | 21 | 1.4000 |
| evidence_handling | 14 | 0.9333 | 61 | 4.0667 |
| experimental_strategy | 11 | 0.7333 | 12 | 0.8000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.5333 | 8 | 0.5333 |
| evidence_handling | 3 | 0.2000 | 3 | 0.2000 |
| experimental_strategy | 3 | 0.2000 | 4 | 0.2667 |

### gpt_4o/retrosynthesis/level_3

- Traces: 14 | Total messages: 732 | Mean messages/trace: 52.29

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1972 | 1.1926 |
| n_H | 0.0518 | 0.0492 |
| n_T | 0.4161 | 0.4167 |
| n_E | 0.5147 | 0.5260 |
| n_J | 0.1775 | 0.1803 |
| n_U | 0.0225 | 0.0150 |
| n_C | 0.0146 | 0.0055 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9286 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.5000 |
| orphan_evidence_rate | 0.1617 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 1.0000 |
| scientificness_score | 0.4667 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 2 | 0.1429 | 2 | 0.1429 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 4 | 0.2857 | 4 | 0.2857 |
| abductive | 1 | 0.0714 | 1 | 0.0714 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 7 | 0.5000 | 16 | 1.1429 |
| evidence_ignored | 14 | 1.0000 | 231 | 16.5000 |
| judgment_without_evidence | 4 | 0.2857 | 4 | 0.2857 |
| dead_end_update | 3 | 0.2143 | 4 | 0.2857 |
| unresolved_contradiction | 7 | 0.5000 | 18 | 1.2857 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 10 | 0.7143 |
| no_belief_revision | 5 | 0.3571 | 5 | 0.3571 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 9 | 0.6429 | 16 | 1.1429 |
| evidence_ignored | 14 | 1.0000 | 231 | 16.5000 |
| judgment_without_evidence | 4 | 0.2857 | 4 | 0.2857 |
| dead_end_update | 3 | 0.2143 | 4 | 0.2857 |
| unresolved_contradiction | 9 | 0.6429 | 18 | 1.2857 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 4 | 0.2857 | 10 | 0.7143 |
| no_belief_revision | 5 | 0.3571 | 5 | 0.3571 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 11 | 0.7857 | 34 | 2.4286 |
| evidence_handling | 14 | 1.0000 | 245 | 17.5000 |
| experimental_strategy | 8 | 0.5714 | 9 | 0.6429 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 12 | 0.8571 | 34 | 2.4286 |
| evidence_handling | 14 | 1.0000 | 245 | 17.5000 |
| experimental_strategy | 8 | 0.5714 | 9 | 0.6429 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.2857 | 5 | 0.3571 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 2 | 0.1429 | 2 | 0.1429 |

### gpt_4o/spectra/level_1

- Traces: 22 | Total messages: 296 | Mean messages/trace: 13.45

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 0.9779 | 0.9966 |
| n_H | 0.1241 | 0.1284 |
| n_T | 0.0997 | 0.1115 |
| n_E | 0.3719 | 0.3818 |
| n_J | 0.2936 | 0.2872 |
| n_U | 0.0083 | 0.0135 |
| n_C | 0.0802 | 0.0743 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7455 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.0000 |
| orphan_evidence_rate | 0.0347 |
| refute_neglect_rate | 0.2000 |
| hypothesis_switch_without_eval_rate | 0.4286 |
| scientificness_score | 0.3758 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.0455 | 1 | 0.0455 |
| bayesian_belief_updating | 3 | 0.1364 | 3 | 0.1364 |
| abductive | 21 | 0.9545 | 21 | 0.9545 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 11 | 0.5000 | 15 | 0.6818 |
| evidence_ignored | 8 | 0.3636 | 26 | 1.1818 |
| judgment_without_evidence | 1 | 0.0455 | 1 | 0.0455 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 3 | 0.1364 | 4 | 0.1818 |
| hypothesis_to_commitment_shortcut | 5 | 0.2273 | 5 | 0.2273 |
| test_without_evidence | 0 | 0.0000 | 1 | 0.0455 |
| no_belief_revision | 19 | 0.8636 | 19 | 0.8636 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 6 | 0.2727 | 6 | 0.2727 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 12 | 0.5455 | 15 | 0.6818 |
| evidence_ignored | 12 | 0.5455 | 26 | 1.1818 |
| judgment_without_evidence | 1 | 0.0455 | 1 | 0.0455 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 4 | 0.1818 | 4 | 0.1818 |
| hypothesis_to_commitment_shortcut | 5 | 0.2273 | 5 | 0.2273 |
| test_without_evidence | 1 | 0.0455 | 1 | 0.0455 |
| no_belief_revision | 19 | 0.8636 | 19 | 0.8636 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 6 | 0.2727 | 6 | 0.2727 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.5909 | 25 | 1.1364 |
| evidence_handling | 9 | 0.4091 | 28 | 1.2727 |
| experimental_strategy | 19 | 0.8636 | 24 | 1.0909 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 14 | 0.6364 | 25 | 1.1364 |
| evidence_handling | 12 | 0.5455 | 28 | 1.2727 |
| experimental_strategy | 19 | 0.8636 | 24 | 1.0909 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 21 | 0.9545 | 24 | 1.0909 |
| evidence_handling | 1 | 0.0455 | 1 | 0.0455 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/spectra/level_2

- Traces: 22 | Total messages: 302 | Mean messages/trace: 13.73

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 0.9610 | 0.9570 |
| n_H | 0.1107 | 0.1093 |
| n_T | 0.0940 | 0.0927 |
| n_E | 0.3700 | 0.3775 |
| n_J | 0.3038 | 0.2980 |
| n_U | 0.0055 | 0.0066 |
| n_C | 0.0771 | 0.0728 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7000 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.5000 |
| orphan_evidence_rate | 0.0275 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.5000 |
| scientificness_score | 0.4727 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 2 | 0.0909 | 2 | 0.0909 |
| abductive | 22 | 1.0000 | 22 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 0.6818 | 18 | 0.8182 |
| evidence_ignored | 5 | 0.2273 | 23 | 1.0455 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 2 | 0.0909 | 4 | 0.1818 |
| hypothesis_to_commitment_shortcut | 10 | 0.4545 | 10 | 0.4545 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 20 | 0.9091 | 20 | 0.9091 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 9 | 0.4091 | 9 | 0.4091 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 0.6818 | 18 | 0.8182 |
| evidence_ignored | 12 | 0.5455 | 23 | 1.0455 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 2 | 0.0909 | 4 | 0.1818 |
| hypothesis_to_commitment_shortcut | 10 | 0.4545 | 10 | 0.4545 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 20 | 0.9091 | 20 | 0.9091 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 9 | 0.4091 | 9 | 0.4091 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 16 | 0.7273 | 31 | 1.4091 |
| evidence_handling | 5 | 0.2273 | 23 | 1.0455 |
| experimental_strategy | 20 | 0.9091 | 30 | 1.3636 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 16 | 0.7273 | 31 | 1.4091 |
| evidence_handling | 12 | 0.5455 | 23 | 1.0455 |
| experimental_strategy | 20 | 0.9091 | 30 | 1.3636 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 22 | 1.0000 | 24 | 1.0909 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/wetlab/level1

- Traces: 10 | Total messages: 206 | Mean messages/trace: 20.60

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1467 | 1.1602 |
| n_H | 0.1599 | 0.1602 |
| n_T | 0.3007 | 0.3058 |
| n_E | 0.3927 | 0.3981 |
| n_J | 0.2157 | 0.2184 |
| n_U | 0.0279 | 0.0291 |
| n_C | 0.0497 | 0.0485 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9200 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.0000 |
| orphan_evidence_rate | 0.2922 |
| refute_neglect_rate | 0.1000 |
| hypothesis_switch_without_eval_rate | 0.0833 |
| scientificness_score | 0.3521 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.1000 | 1 | 0.1000 |
| bayesian_belief_updating | 6 | 0.6000 | 6 | 0.6000 |
| abductive | 10 | 1.0000 | 10 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 1.0000 | 29 | 2.9000 |
| evidence_ignored | 9 | 0.9000 | 39 | 3.9000 |
| judgment_without_evidence | 2 | 0.2000 | 3 | 0.3000 |
| dead_end_update | 1 | 0.1000 | 1 | 0.1000 |
| unresolved_contradiction | 1 | 0.1000 | 2 | 0.2000 |
| hypothesis_to_commitment_shortcut | 4 | 0.4000 | 5 | 0.5000 |
| test_without_evidence | 0 | 0.0000 | 2 | 0.2000 |
| no_belief_revision | 4 | 0.4000 | 4 | 0.4000 |
| orphan_evidence | 7 | 0.7000 | 8 | 0.8000 |
| confirmation_only | 4 | 0.4000 | 5 | 0.5000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 1.0000 | 29 | 2.9000 |
| evidence_ignored | 10 | 1.0000 | 39 | 3.9000 |
| judgment_without_evidence | 2 | 0.2000 | 3 | 0.3000 |
| dead_end_update | 1 | 0.1000 | 1 | 0.1000 |
| unresolved_contradiction | 2 | 0.2000 | 2 | 0.2000 |
| hypothesis_to_commitment_shortcut | 4 | 0.4000 | 5 | 0.5000 |
| test_without_evidence | 2 | 0.2000 | 2 | 0.2000 |
| no_belief_revision | 4 | 0.4000 | 4 | 0.4000 |
| orphan_evidence | 7 | 0.7000 | 8 | 0.8000 |
| confirmation_only | 4 | 0.4000 | 5 | 0.5000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 36 | 3.6000 |
| evidence_handling | 9 | 0.9000 | 52 | 5.2000 |
| experimental_strategy | 5 | 0.5000 | 10 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 36 | 3.6000 |
| evidence_handling | 10 | 1.0000 | 52 | 5.2000 |
| experimental_strategy | 5 | 0.5000 | 10 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 16 | 1.6000 |
| evidence_handling | 1 | 0.1000 | 1 | 0.1000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/wetlab/level2

- Traces: 10 | Total messages: 250 | Mean messages/trace: 25.00

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.3064 | 1.3080 |
| n_H | 0.1748 | 0.1760 |
| n_T | 0.3531 | 0.3600 |
| n_E | 0.4720 | 0.4760 |
| n_J | 0.2343 | 0.2280 |
| n_U | 0.0313 | 0.0280 |
| n_C | 0.0409 | 0.0400 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9000 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.0667 |
| orphan_evidence_rate | 0.2205 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.2000 |
| scientificness_score | 0.3260 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 5 | 0.5000 | 5 | 0.5000 |
| abductive | 10 | 1.0000 | 10 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 9 | 0.9000 | 38 | 3.8000 |
| evidence_ignored | 10 | 1.0000 | 59 | 5.9000 |
| judgment_without_evidence | 1 | 0.1000 | 2 | 0.2000 |
| dead_end_update | 1 | 0.1000 | 1 | 0.1000 |
| unresolved_contradiction | 3 | 0.3000 | 8 | 0.8000 |
| hypothesis_to_commitment_shortcut | 4 | 0.4000 | 9 | 0.9000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 5 | 0.5000 | 5 | 0.5000 |
| orphan_evidence | 4 | 0.4000 | 7 | 0.7000 |
| confirmation_only | 4 | 0.4000 | 9 | 0.9000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 9 | 0.9000 | 38 | 3.8000 |
| evidence_ignored | 10 | 1.0000 | 59 | 5.9000 |
| judgment_without_evidence | 1 | 0.1000 | 2 | 0.2000 |
| dead_end_update | 1 | 0.1000 | 1 | 0.1000 |
| unresolved_contradiction | 3 | 0.3000 | 8 | 0.8000 |
| hypothesis_to_commitment_shortcut | 4 | 0.4000 | 9 | 0.9000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 5 | 0.5000 | 5 | 0.5000 |
| orphan_evidence | 4 | 0.4000 | 7 | 0.7000 |
| confirmation_only | 4 | 0.4000 | 9 | 0.9000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 55 | 5.5000 |
| evidence_handling | 10 | 1.0000 | 68 | 6.8000 |
| experimental_strategy | 6 | 0.6000 | 15 | 1.5000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 55 | 5.5000 |
| evidence_handling | 10 | 1.0000 | 68 | 6.8000 |
| experimental_strategy | 6 | 0.6000 | 15 | 1.5000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 15 | 1.5000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/wetlab/level3

- Traces: 10 | Total messages: 236 | Mean messages/trace: 23.60

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1766 | 1.1864 |
| n_H | 0.1567 | 0.1610 |
| n_T | 0.2977 | 0.3051 |
| n_E | 0.4436 | 0.4449 |
| n_J | 0.2172 | 0.2161 |
| n_U | 0.0184 | 0.0169 |
| n_C | 0.0431 | 0.0424 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8800 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.2500 |
| orphan_evidence_rate | 0.2713 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.5000 |
| scientificness_score | 0.3311 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 4 | 0.4000 | 4 | 0.4000 |
| abductive | 10 | 1.0000 | 10 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 1.0000 | 33 | 3.3000 |
| evidence_ignored | 9 | 0.9000 | 52 | 5.2000 |
| judgment_without_evidence | 1 | 0.1000 | 2 | 0.2000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 1 | 0.1000 | 1 | 0.1000 |
| hypothesis_to_commitment_shortcut | 5 | 0.5000 | 9 | 0.9000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 6 | 0.6000 | 6 | 0.6000 |
| orphan_evidence | 6 | 0.6000 | 10 | 1.0000 |
| confirmation_only | 5 | 0.5000 | 10 | 1.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 1.0000 | 33 | 3.3000 |
| evidence_ignored | 10 | 1.0000 | 52 | 5.2000 |
| judgment_without_evidence | 1 | 0.1000 | 2 | 0.2000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 3 | 0.3000 | 4 | 0.4000 |
| hypothesis_to_commitment_shortcut | 5 | 0.5000 | 9 | 0.9000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 6 | 0.6000 | 6 | 0.6000 |
| orphan_evidence | 6 | 0.6000 | 10 | 1.0000 |
| confirmation_only | 5 | 0.5000 | 10 | 1.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 44 | 4.4000 |
| evidence_handling | 10 | 1.0000 | 64 | 6.4000 |
| experimental_strategy | 7 | 0.7000 | 15 | 1.5000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 47 | 4.7000 |
| evidence_handling | 10 | 1.0000 | 64 | 6.4000 |
| experimental_strategy | 7 | 0.7000 | 15 | 1.5000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 14 | 1.4000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

## By model

### claude_sonnet_45

- Traces: 154 | Total messages: 4712 | Mean messages/trace: 30.60

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2069 | 1.2638 |
| n_H | 0.1240 | 0.1384 |
| n_T | 0.2788 | 0.2978 |
| n_E | 0.4765 | 0.4921 |
| n_J | 0.2470 | 0.2651 |
| n_U | 0.0308 | 0.0335 |
| n_C | 0.0498 | 0.0369 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8662 |
| loop_density | 0.0001 |
| update_grounding_rate | 0.3836 |
| orphan_evidence_rate | 0.1725 |
| refute_neglect_rate | 0.1876 |
| hypothesis_switch_without_eval_rate | 0.5279 |
| scientificness_score | 0.4359 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 2 | 0.0130 | 2 | 0.0130 |
| ml_make_it_work | 5 | 0.0325 | 5 | 0.0325 |
| exploratory_to_confirmatory | 3 | 0.0195 | 3 | 0.0195 |
| bayesian_belief_updating | 59 | 0.3831 | 59 | 0.3831 |
| abductive | 84 | 0.5455 | 84 | 0.5455 |
| triangulation | 1 | 0.0065 | 1 | 0.0065 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 2 | 0.0130 | 2 | 0.0130 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 109 | 0.7078 | 498 | 3.2338 |
| evidence_ignored | 96 | 0.6234 | 929 | 6.0325 |
| judgment_without_evidence | 42 | 0.2727 | 88 | 0.5714 |
| dead_end_update | 26 | 0.1688 | 31 | 0.2013 |
| unresolved_contradiction | 25 | 0.1623 | 96 | 0.6234 |
| hypothesis_to_commitment_shortcut | 26 | 0.1688 | 37 | 0.2403 |
| test_without_evidence | 3 | 0.0195 | 48 | 0.3117 |
| no_belief_revision | 65 | 0.4221 | 65 | 0.4221 |
| orphan_evidence | 39 | 0.2532 | 80 | 0.5195 |
| confirmation_only | 17 | 0.1104 | 19 | 0.1234 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 112 | 0.7273 | 498 | 3.2338 |
| evidence_ignored | 141 | 0.9156 | 929 | 6.0325 |
| judgment_without_evidence | 42 | 0.2727 | 88 | 0.5714 |
| dead_end_update | 26 | 0.1688 | 31 | 0.2013 |
| unresolved_contradiction | 54 | 0.3506 | 135 | 0.8766 |
| hypothesis_to_commitment_shortcut | 26 | 0.1688 | 37 | 0.2403 |
| test_without_evidence | 23 | 0.1494 | 48 | 0.3117 |
| no_belief_revision | 65 | 0.4221 | 65 | 0.4221 |
| orphan_evidence | 39 | 0.2532 | 80 | 0.5195 |
| confirmation_only | 17 | 0.1104 | 19 | 0.1234 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 115 | 0.7468 | 613 | 3.9805 |
| evidence_handling | 123 | 0.7987 | 1145 | 7.4351 |
| experimental_strategy | 99 | 0.6429 | 133 | 0.8636 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 117 | 0.7597 | 652 | 4.2338 |
| evidence_handling | 142 | 0.9221 | 1145 | 7.4351 |
| experimental_strategy | 99 | 0.6429 | 133 | 0.8636 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 99 | 0.6429 | 145 | 0.9416 |
| evidence_handling | 4 | 0.0260 | 4 | 0.0260 |
| experimental_strategy | 7 | 0.0455 | 7 | 0.0455 |

### gpt_4o

- Traces: 163 | Total messages: 3723 | Mean messages/trace: 22.84

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.0925 | 1.1397 |
| n_H | 0.0869 | 0.0843 |
| n_T | 0.3009 | 0.3411 |
| n_E | 0.4329 | 0.4674 |
| n_J | 0.1996 | 0.1918 |
| n_U | 0.0116 | 0.0140 |
| n_C | 0.0606 | 0.0411 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7816 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.3659 |
| orphan_evidence_rate | 0.1101 |
| refute_neglect_rate | 0.0936 |
| hypothesis_switch_without_eval_rate | 0.5983 |
| scientificness_score | 0.4199 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 2 | 0.0123 | 2 | 0.0123 |
| ml_make_it_work | 4 | 0.0245 | 4 | 0.0245 |
| exploratory_to_confirmatory | 15 | 0.0920 | 15 | 0.0920 |
| bayesian_belief_updating | 32 | 0.1963 | 32 | 0.1963 |
| abductive | 103 | 0.6319 | 103 | 0.6319 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 2 | 0.0123 | 2 | 0.0123 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 88 | 0.5399 | 200 | 1.2270 |
| evidence_ignored | 120 | 0.7362 | 854 | 5.2393 |
| judgment_without_evidence | 11 | 0.0675 | 15 | 0.0920 |
| dead_end_update | 8 | 0.0491 | 9 | 0.0552 |
| unresolved_contradiction | 25 | 0.1534 | 57 | 0.3497 |
| hypothesis_to_commitment_shortcut | 34 | 0.2086 | 44 | 0.2699 |
| test_without_evidence | 0 | 0.0000 | 22 | 0.1350 |
| no_belief_revision | 120 | 0.7362 | 120 | 0.7362 |
| orphan_evidence | 20 | 0.1227 | 29 | 0.1779 |
| confirmation_only | 35 | 0.2147 | 46 | 0.2822 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 93 | 0.5706 | 200 | 1.2270 |
| evidence_ignored | 136 | 0.8344 | 854 | 5.2393 |
| judgment_without_evidence | 11 | 0.0675 | 15 | 0.0920 |
| dead_end_update | 8 | 0.0491 | 9 | 0.0552 |
| unresolved_contradiction | 34 | 0.2086 | 61 | 0.3742 |
| hypothesis_to_commitment_shortcut | 34 | 0.2086 | 44 | 0.2699 |
| test_without_evidence | 15 | 0.0920 | 22 | 0.1350 |
| no_belief_revision | 120 | 0.7362 | 120 | 0.7362 |
| orphan_evidence | 20 | 0.1227 | 29 | 0.1779 |
| confirmation_only | 35 | 0.2147 | 46 | 0.2822 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 100 | 0.6135 | 303 | 1.8589 |
| evidence_handling | 123 | 0.7546 | 920 | 5.6442 |
| experimental_strategy | 130 | 0.7975 | 173 | 1.0613 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 103 | 0.6319 | 307 | 1.8834 |
| evidence_handling | 137 | 0.8405 | 920 | 5.6442 |
| experimental_strategy | 130 | 0.7975 | 173 | 1.0613 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 107 | 0.6564 | 137 | 0.8405 |
| evidence_handling | 15 | 0.0920 | 15 | 0.0920 |
| experimental_strategy | 5 | 0.0307 | 6 | 0.0368 |

## By env

### catalyst

- Traces: 30 | Total messages: 656 | Mean messages/trace: 21.87

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1257 | 1.0991 |
| n_H | 0.0100 | 0.0091 |
| n_T | 0.4487 | 0.4284 |
| n_E | 0.5158 | 0.5152 |
| n_J | 0.0772 | 0.0777 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0740 | 0.0686 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.6333 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.1685 |
| refute_neglect_rate | N/A |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | N/A |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 3 | 0.1000 | 3 | 0.1000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 5 | 0.1667 | 5 | 0.1667 |
| evidence_ignored | 29 | 0.9667 | 234 | 7.8000 |
| judgment_without_evidence | 1 | 0.0333 | 1 | 0.0333 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 2 | 0.0667 | 2 | 0.0667 |
| test_without_evidence | 0 | 0.0000 | 1 | 0.0333 |
| no_belief_revision | 30 | 1.0000 | 30 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 2 | 0.0667 | 2 | 0.0667 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 5 | 0.1667 | 5 | 0.1667 |
| evidence_ignored | 30 | 1.0000 | 234 | 7.8000 |
| judgment_without_evidence | 1 | 0.0333 | 1 | 0.0333 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 2 | 0.0667 | 2 | 0.0667 |
| test_without_evidence | 1 | 0.0333 | 1 | 0.0333 |
| no_belief_revision | 30 | 1.0000 | 30 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 2 | 0.0667 | 2 | 0.0667 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.1667 | 7 | 0.2333 |
| evidence_handling | 30 | 1.0000 | 236 | 7.8667 |
| experimental_strategy | 30 | 1.0000 | 32 | 1.0667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.1667 | 7 | 0.2333 |
| evidence_handling | 30 | 1.0000 | 236 | 7.8667 |
| experimental_strategy | 30 | 1.0000 | 32 | 1.0667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.1000 | 3 | 0.1000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### md

- Traces: 35 | Total messages: 1161 | Mean messages/trace: 33.17

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1942 | 1.1240 |
| n_H | 0.1031 | 0.0904 |
| n_T | 0.3796 | 0.3342 |
| n_E | 0.5201 | 0.5125 |
| n_J | 0.1531 | 0.1525 |
| n_U | 0.0114 | 0.0121 |
| n_C | 0.0270 | 0.0224 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8629 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.3571 |
| orphan_evidence_rate | 0.3297 |
| refute_neglect_rate | 0.2497 |
| hypothesis_switch_without_eval_rate | 0.8571 |
| scientificness_score | 0.3821 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 2 | 0.0571 | 2 | 0.0571 |
| ml_make_it_work | 1 | 0.0286 | 1 | 0.0286 |
| exploratory_to_confirmatory | 9 | 0.2571 | 9 | 0.2571 |
| bayesian_belief_updating | 9 | 0.2571 | 9 | 0.2571 |
| abductive | 19 | 0.5429 | 19 | 0.5429 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 29 | 0.8286 | 66 | 1.8857 |
| evidence_ignored | 34 | 0.9714 | 357 | 10.2000 |
| judgment_without_evidence | 4 | 0.1143 | 5 | 0.1429 |
| dead_end_update | 3 | 0.0857 | 3 | 0.0857 |
| unresolved_contradiction | 10 | 0.2857 | 27 | 0.7714 |
| hypothesis_to_commitment_shortcut | 4 | 0.1143 | 4 | 0.1143 |
| test_without_evidence | 3 | 0.0857 | 36 | 1.0286 |
| no_belief_revision | 24 | 0.6857 | 24 | 0.6857 |
| orphan_evidence | 7 | 0.2000 | 10 | 0.2857 |
| confirmation_only | 4 | 0.1143 | 4 | 0.1143 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 31 | 0.8857 | 66 | 1.8857 |
| evidence_ignored | 35 | 1.0000 | 357 | 10.2000 |
| judgment_without_evidence | 4 | 0.1143 | 5 | 0.1429 |
| dead_end_update | 3 | 0.0857 | 3 | 0.0857 |
| unresolved_contradiction | 15 | 0.4286 | 32 | 0.9143 |
| hypothesis_to_commitment_shortcut | 4 | 0.1143 | 4 | 0.1143 |
| test_without_evidence | 13 | 0.3714 | 36 | 1.0286 |
| no_belief_revision | 24 | 0.6857 | 24 | 0.6857 |
| orphan_evidence | 7 | 0.2000 | 10 | 0.2857 |
| confirmation_only | 4 | 0.1143 | 4 | 0.1143 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 32 | 0.9143 | 97 | 2.7714 |
| evidence_handling | 35 | 1.0000 | 408 | 11.6571 |
| experimental_strategy | 28 | 0.8000 | 31 | 0.8857 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 33 | 0.9429 | 102 | 2.9143 |
| evidence_handling | 35 | 1.0000 | 408 | 11.6571 |
| experimental_strategy | 28 | 0.8000 | 31 | 0.8857 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 22 | 0.6286 | 30 | 0.8571 |
| evidence_handling | 9 | 0.2571 | 9 | 0.2571 |
| experimental_strategy | 1 | 0.0286 | 1 | 0.0286 |

### ml

- Traces: 15 | Total messages: 233 | Mean messages/trace: 15.53

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.0862 | 1.1030 |
| n_H | 0.0000 | 0.0000 |
| n_T | 0.4433 | 0.4506 |
| n_E | 0.4493 | 0.4549 |
| n_J | 0.1049 | 0.1073 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0887 | 0.0901 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.6000 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.0778 |
| refute_neglect_rate | N/A |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | N/A |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 0 | 0.0000 | 0 | 0.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 0 | 0.0000 | 0 | 0.0000 |
| evidence_ignored | 14 | 0.9333 | 67 | 4.4667 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 15 | 1.0000 | 15 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 0 | 0.0000 | 0 | 0.0000 |
| evidence_ignored | 14 | 0.9333 | 67 | 4.4667 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 15 | 1.0000 | 15 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 14 | 0.9333 | 67 | 4.4667 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 14 | 0.9333 | 67 | 4.4667 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### retrosynthesis

- Traces: 89 | Total messages: 2521 | Mean messages/trace: 28.33

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1125 | 1.1995 |
| n_H | 0.0743 | 0.0770 |
| n_T | 0.3536 | 0.3939 |
| n_E | 0.4451 | 0.4935 |
| n_J | 0.1648 | 0.1797 |
| n_U | 0.0195 | 0.0238 |
| n_C | 0.0552 | 0.0317 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8494 |
| loop_density | 0.0001 |
| update_grounding_rate | 0.6164 |
| orphan_evidence_rate | 0.0842 |
| refute_neglect_rate | 0.2321 |
| hypothesis_switch_without_eval_rate | 0.9122 |
| scientificness_score | 0.4800 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 2 | 0.0225 | 2 | 0.0225 |
| ml_make_it_work | 8 | 0.0899 | 8 | 0.0899 |
| exploratory_to_confirmatory | 6 | 0.0674 | 6 | 0.0674 |
| bayesian_belief_updating | 16 | 0.1798 | 16 | 0.1798 |
| abductive | 21 | 0.2360 | 21 | 0.2360 |
| triangulation | 1 | 0.0112 | 1 | 0.0112 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 4 | 0.0449 | 4 | 0.0449 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 42 | 0.4719 | 103 | 1.1573 |
| evidence_ignored | 74 | 0.8315 | 605 | 6.7978 |
| judgment_without_evidence | 7 | 0.0787 | 7 | 0.0787 |
| dead_end_update | 14 | 0.1573 | 15 | 0.1685 |
| unresolved_contradiction | 20 | 0.2247 | 55 | 0.6180 |
| hypothesis_to_commitment_shortcut | 12 | 0.1348 | 12 | 0.1348 |
| test_without_evidence | 0 | 0.0000 | 24 | 0.2697 |
| no_belief_revision | 49 | 0.5506 | 49 | 0.5506 |
| orphan_evidence | 3 | 0.0337 | 3 | 0.0337 |
| confirmation_only | 11 | 0.1236 | 11 | 0.1236 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 46 | 0.5169 | 103 | 1.1573 |
| evidence_ignored | 77 | 0.8652 | 605 | 6.7978 |
| judgment_without_evidence | 7 | 0.0787 | 7 | 0.0787 |
| dead_end_update | 14 | 0.1573 | 15 | 0.1685 |
| unresolved_contradiction | 27 | 0.3034 | 64 | 0.7191 |
| hypothesis_to_commitment_shortcut | 12 | 0.1348 | 12 | 0.1348 |
| test_without_evidence | 15 | 0.1685 | 24 | 0.2697 |
| no_belief_revision | 49 | 0.5506 | 49 | 0.5506 |
| orphan_evidence | 3 | 0.0337 | 3 | 0.0337 |
| confirmation_only | 11 | 0.1236 | 11 | 0.1236 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 50 | 0.5618 | 169 | 1.8989 |
| evidence_handling | 74 | 0.8315 | 639 | 7.1798 |
| experimental_strategy | 65 | 0.7303 | 76 | 0.8539 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 53 | 0.5955 | 178 | 2.0000 |
| evidence_handling | 78 | 0.8764 | 639 | 7.1798 |
| experimental_strategy | 65 | 0.7303 | 76 | 0.8539 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 35 | 0.3933 | 39 | 0.4382 |
| evidence_handling | 7 | 0.0787 | 7 | 0.0787 |
| experimental_strategy | 11 | 0.1236 | 12 | 0.1348 |

### spectra

- Traces: 88 | Total messages: 1578 | Mean messages/trace: 17.93

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.0858 | 1.1679 |
| n_H | 0.1416 | 0.1648 |
| n_T | 0.0954 | 0.1096 |
| n_E | 0.4133 | 0.4411 |
| n_J | 0.3389 | 0.3599 |
| n_U | 0.0270 | 0.0330 |
| n_C | 0.0697 | 0.0596 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7977 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.3144 |
| orphan_evidence_rate | 0.0619 |
| refute_neglect_rate | 0.0964 |
| hypothesis_switch_without_eval_rate | 0.3801 |
| scientificness_score | 0.4436 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.0114 | 1 | 0.0114 |
| bayesian_belief_updating | 28 | 0.3182 | 28 | 0.3182 |
| abductive | 84 | 0.9545 | 84 | 0.9545 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 62 | 0.7045 | 151 | 1.7159 |
| evidence_ignored | 22 | 0.2500 | 147 | 1.6705 |
| judgment_without_evidence | 18 | 0.2045 | 48 | 0.5455 |
| dead_end_update | 5 | 0.0568 | 5 | 0.0568 |
| unresolved_contradiction | 9 | 0.1023 | 31 | 0.3523 |
| hypothesis_to_commitment_shortcut | 22 | 0.2500 | 23 | 0.2614 |
| test_without_evidence | 0 | 0.0000 | 2 | 0.0227 |
| no_belief_revision | 51 | 0.5795 | 51 | 0.5795 |
| orphan_evidence | 4 | 0.0455 | 6 | 0.0682 |
| confirmation_only | 19 | 0.2159 | 20 | 0.2273 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 64 | 0.7273 | 151 | 1.7159 |
| evidence_ignored | 61 | 0.6932 | 147 | 1.6705 |
| judgment_without_evidence | 18 | 0.2045 | 48 | 0.5455 |
| dead_end_update | 5 | 0.0568 | 5 | 0.0568 |
| unresolved_contradiction | 22 | 0.2500 | 44 | 0.5000 |
| hypothesis_to_commitment_shortcut | 22 | 0.2500 | 23 | 0.2614 |
| test_without_evidence | 2 | 0.0227 | 2 | 0.0227 |
| no_belief_revision | 51 | 0.5795 | 51 | 0.5795 |
| orphan_evidence | 4 | 0.0455 | 6 | 0.0682 |
| confirmation_only | 19 | 0.2159 | 20 | 0.2273 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 68 | 0.7727 | 202 | 2.2955 |
| evidence_handling | 34 | 0.3864 | 203 | 2.3068 |
| experimental_strategy | 59 | 0.6705 | 79 | 0.8977 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 69 | 0.7841 | 215 | 2.4432 |
| evidence_handling | 62 | 0.7045 | 203 | 2.3068 |
| experimental_strategy | 59 | 0.6705 | 79 | 0.8977 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 86 | 0.9773 | 112 | 1.2727 |
| evidence_handling | 1 | 0.0114 | 1 | 0.0114 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### wetlab

- Traces: 60 | Total messages: 2286 | Mean messages/trace: 38.10

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2920 | 1.3333 |
| n_H | 0.1713 | 0.1754 |
| n_T | 0.3120 | 0.3206 |
| n_E | 0.4591 | 0.4724 |
| n_J | 0.2807 | 0.3014 |
| n_U | 0.0355 | 0.0367 |
| n_C | 0.0335 | 0.0267 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9467 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.1159 |
| orphan_evidence_rate | 0.2301 |
| refute_neglect_rate | 0.0454 |
| hypothesis_switch_without_eval_rate | 0.2171 |
| scientificness_score | 0.3552 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 2 | 0.0333 | 2 | 0.0333 |
| bayesian_belief_updating | 38 | 0.6333 | 38 | 0.6333 |
| abductive | 60 | 1.0000 | 60 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 59 | 0.9833 | 373 | 6.2167 |
| evidence_ignored | 43 | 0.7167 | 373 | 6.2167 |
| judgment_without_evidence | 23 | 0.3833 | 42 | 0.7000 |
| dead_end_update | 12 | 0.2000 | 17 | 0.2833 |
| unresolved_contradiction | 11 | 0.1833 | 40 | 0.6667 |
| hypothesis_to_commitment_shortcut | 20 | 0.3333 | 40 | 0.6667 |
| test_without_evidence | 0 | 0.0000 | 7 | 0.1167 |
| no_belief_revision | 16 | 0.2667 | 16 | 0.2667 |
| orphan_evidence | 45 | 0.7500 | 90 | 1.5000 |
| confirmation_only | 16 | 0.2667 | 28 | 0.4667 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 59 | 0.9833 | 373 | 6.2167 |
| evidence_ignored | 60 | 1.0000 | 373 | 6.2167 |
| judgment_without_evidence | 23 | 0.3833 | 42 | 0.7000 |
| dead_end_update | 12 | 0.2000 | 17 | 0.2833 |
| unresolved_contradiction | 24 | 0.4000 | 56 | 0.9333 |
| hypothesis_to_commitment_shortcut | 20 | 0.3333 | 40 | 0.6667 |
| test_without_evidence | 7 | 0.1167 | 7 | 0.1167 |
| no_belief_revision | 16 | 0.2667 | 16 | 0.2667 |
| orphan_evidence | 45 | 0.7500 | 90 | 1.5000 |
| confirmation_only | 16 | 0.2667 | 28 | 0.4667 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 60 | 1.0000 | 441 | 7.3500 |
| evidence_handling | 59 | 0.9833 | 512 | 8.5333 |
| experimental_strategy | 32 | 0.5333 | 73 | 1.2167 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 60 | 1.0000 | 457 | 7.6167 |
| evidence_handling | 60 | 1.0000 | 512 | 8.5333 |
| experimental_strategy | 32 | 0.5333 | 73 | 1.2167 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 60 | 1.0000 | 98 | 1.6333 |
| evidence_handling | 2 | 0.0333 | 2 | 0.0333 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

## By level

### level1

- Traces: 20 | Total messages: 536 | Mean messages/trace: 26.80

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2622 | 1.2966 |
| n_H | 0.1686 | 0.1716 |
| n_T | 0.2816 | 0.2817 |
| n_E | 0.4367 | 0.4478 |
| n_J | 0.2925 | 0.3116 |
| n_U | 0.0424 | 0.0466 |
| n_C | 0.0405 | 0.0373 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9500 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.1019 |
| orphan_evidence_rate | 0.2403 |
| refute_neglect_rate | 0.0500 |
| hypothesis_switch_without_eval_rate | 0.1354 |
| scientificness_score | 0.3738 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.0500 | 1 | 0.0500 |
| bayesian_belief_updating | 10 | 0.5000 | 10 | 0.5000 |
| abductive | 20 | 1.0000 | 20 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 20 | 1.0000 | 85 | 4.2500 |
| evidence_ignored | 12 | 0.6000 | 77 | 3.8500 |
| judgment_without_evidence | 10 | 0.5000 | 17 | 0.8500 |
| dead_end_update | 3 | 0.1500 | 3 | 0.1500 |
| unresolved_contradiction | 3 | 0.1500 | 6 | 0.3000 |
| hypothesis_to_commitment_shortcut | 5 | 0.2500 | 6 | 0.3000 |
| test_without_evidence | 0 | 0.0000 | 3 | 0.1500 |
| no_belief_revision | 5 | 0.2500 | 5 | 0.2500 |
| orphan_evidence | 16 | 0.8000 | 25 | 1.2500 |
| confirmation_only | 5 | 0.2500 | 6 | 0.3000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 20 | 1.0000 | 85 | 4.2500 |
| evidence_ignored | 20 | 1.0000 | 77 | 3.8500 |
| judgment_without_evidence | 10 | 0.5000 | 17 | 0.8500 |
| dead_end_update | 3 | 0.1500 | 3 | 0.1500 |
| unresolved_contradiction | 5 | 0.2500 | 7 | 0.3500 |
| hypothesis_to_commitment_shortcut | 5 | 0.2500 | 6 | 0.3000 |
| test_without_evidence | 3 | 0.1500 | 3 | 0.1500 |
| no_belief_revision | 5 | 0.2500 | 5 | 0.2500 |
| orphan_evidence | 16 | 0.8000 | 25 | 1.2500 |
| confirmation_only | 5 | 0.2500 | 6 | 0.3000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 1.0000 | 97 | 4.8500 |
| evidence_handling | 19 | 0.9500 | 122 | 6.1000 |
| experimental_strategy | 9 | 0.4500 | 14 | 0.7000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 1.0000 | 98 | 4.9000 |
| evidence_handling | 20 | 1.0000 | 122 | 6.1000 |
| experimental_strategy | 9 | 0.4500 | 14 | 0.7000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 1.0000 | 30 | 1.5000 |
| evidence_handling | 1 | 0.0500 | 1 | 0.0500 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### level2

- Traces: 20 | Total messages: 812 | Mean messages/trace: 40.60

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.3329 | 1.3547 |
| n_H | 0.1742 | 0.1736 |
| n_T | 0.3413 | 0.3436 |
| n_E | 0.4729 | 0.4778 |
| n_J | 0.2765 | 0.2968 |
| n_U | 0.0371 | 0.0369 |
| n_C | 0.0310 | 0.0259 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9500 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.0458 |
| orphan_evidence_rate | 0.2245 |
| refute_neglect_rate | 0.0167 |
| hypothesis_switch_without_eval_rate | 0.1824 |
| scientificness_score | 0.3367 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 15 | 0.7500 | 15 | 0.7500 |
| abductive | 20 | 1.0000 | 20 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 19 | 0.9500 | 130 | 6.5000 |
| evidence_ignored | 16 | 0.8000 | 137 | 6.8500 |
| judgment_without_evidence | 4 | 0.2000 | 9 | 0.4500 |
| dead_end_update | 5 | 0.2500 | 6 | 0.3000 |
| unresolved_contradiction | 3 | 0.1500 | 16 | 0.8000 |
| hypothesis_to_commitment_shortcut | 5 | 0.2500 | 12 | 0.6000 |
| test_without_evidence | 0 | 0.0000 | 3 | 0.1500 |
| no_belief_revision | 5 | 0.2500 | 5 | 0.2500 |
| orphan_evidence | 14 | 0.7000 | 31 | 1.5500 |
| confirmation_only | 4 | 0.2000 | 9 | 0.4500 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 19 | 0.9500 | 130 | 6.5000 |
| evidence_ignored | 20 | 1.0000 | 137 | 6.8500 |
| judgment_without_evidence | 4 | 0.2000 | 9 | 0.4500 |
| dead_end_update | 5 | 0.2500 | 6 | 0.3000 |
| unresolved_contradiction | 9 | 0.4500 | 21 | 1.0500 |
| hypothesis_to_commitment_shortcut | 5 | 0.2500 | 12 | 0.6000 |
| test_without_evidence | 3 | 0.1500 | 3 | 0.1500 |
| no_belief_revision | 5 | 0.2500 | 5 | 0.2500 |
| orphan_evidence | 14 | 0.7000 | 31 | 1.5500 |
| confirmation_only | 4 | 0.2000 | 9 | 0.4500 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 1.0000 | 155 | 7.7500 |
| evidence_handling | 20 | 1.0000 | 180 | 9.0000 |
| experimental_strategy | 10 | 0.5000 | 23 | 1.1500 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 1.0000 | 160 | 8.0000 |
| evidence_handling | 20 | 1.0000 | 180 | 9.0000 |
| experimental_strategy | 10 | 0.5000 | 23 | 1.1500 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 1.0000 | 35 | 1.7500 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### level3

- Traces: 20 | Total messages: 938 | Mean messages/trace: 46.90

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2809 | 1.3358 |
| n_H | 0.1711 | 0.1791 |
| n_T | 0.3131 | 0.3230 |
| n_E | 0.4676 | 0.4819 |
| n_J | 0.2732 | 0.2996 |
| n_U | 0.0269 | 0.0309 |
| n_C | 0.0289 | 0.0213 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9400 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.2000 |
| orphan_evidence_rate | 0.2253 |
| refute_neglect_rate | 0.0694 |
| hypothesis_switch_without_eval_rate | 0.3333 |
| scientificness_score | 0.3551 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.0500 | 1 | 0.0500 |
| bayesian_belief_updating | 13 | 0.6500 | 13 | 0.6500 |
| abductive | 20 | 1.0000 | 20 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 20 | 1.0000 | 158 | 7.9000 |
| evidence_ignored | 15 | 0.7500 | 159 | 7.9500 |
| judgment_without_evidence | 9 | 0.4500 | 16 | 0.8000 |
| dead_end_update | 4 | 0.2000 | 8 | 0.4000 |
| unresolved_contradiction | 5 | 0.2500 | 18 | 0.9000 |
| hypothesis_to_commitment_shortcut | 10 | 0.5000 | 22 | 1.1000 |
| test_without_evidence | 0 | 0.0000 | 1 | 0.0500 |
| no_belief_revision | 6 | 0.3000 | 6 | 0.3000 |
| orphan_evidence | 15 | 0.7500 | 34 | 1.7000 |
| confirmation_only | 7 | 0.3500 | 13 | 0.6500 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 20 | 1.0000 | 158 | 7.9000 |
| evidence_ignored | 20 | 1.0000 | 159 | 7.9500 |
| judgment_without_evidence | 9 | 0.4500 | 16 | 0.8000 |
| dead_end_update | 4 | 0.2000 | 8 | 0.4000 |
| unresolved_contradiction | 10 | 0.5000 | 28 | 1.4000 |
| hypothesis_to_commitment_shortcut | 10 | 0.5000 | 22 | 1.1000 |
| test_without_evidence | 1 | 0.0500 | 1 | 0.0500 |
| no_belief_revision | 6 | 0.3000 | 6 | 0.3000 |
| orphan_evidence | 15 | 0.7500 | 34 | 1.7000 |
| confirmation_only | 7 | 0.3500 | 13 | 0.6500 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 1.0000 | 189 | 9.4500 |
| evidence_handling | 20 | 1.0000 | 210 | 10.5000 |
| experimental_strategy | 13 | 0.6500 | 36 | 1.8000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 1.0000 | 199 | 9.9500 |
| evidence_handling | 20 | 1.0000 | 210 | 10.5000 |
| experimental_strategy | 13 | 0.6500 | 36 | 1.8000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 1.0000 | 33 | 1.6500 |
| evidence_handling | 1 | 0.0500 | 1 | 0.0500 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### level_1

- Traces: 139 | Total messages: 2786 | Mean messages/trace: 20.04

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.0820 | 1.0976 |
| n_H | 0.0712 | 0.0739 |
| n_T | 0.2969 | 0.2976 |
| n_E | 0.4426 | 0.4655 |
| n_J | 0.1897 | 0.1910 |
| n_U | 0.0121 | 0.0140 |
| n_C | 0.0694 | 0.0556 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7338 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.2893 |
| orphan_evidence_rate | 0.1157 |
| refute_neglect_rate | 0.1975 |
| hypothesis_switch_without_eval_rate | 0.7083 |
| scientificness_score | 0.3964 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 2 | 0.0144 | 2 | 0.0144 |
| exploratory_to_confirmatory | 8 | 0.0576 | 8 | 0.0576 |
| bayesian_belief_updating | 22 | 0.1583 | 22 | 0.1583 |
| abductive | 65 | 0.4676 | 65 | 0.4676 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 58 | 0.4173 | 117 | 0.8417 |
| evidence_ignored | 91 | 0.6547 | 622 | 4.4748 |
| judgment_without_evidence | 11 | 0.0791 | 24 | 0.1727 |
| dead_end_update | 4 | 0.0288 | 4 | 0.0288 |
| unresolved_contradiction | 15 | 0.1079 | 42 | 0.3022 |
| hypothesis_to_commitment_shortcut | 16 | 0.1151 | 17 | 0.1223 |
| test_without_evidence | 3 | 0.0216 | 32 | 0.2302 |
| no_belief_revision | 111 | 0.7986 | 111 | 0.7986 |
| orphan_evidence | 6 | 0.0432 | 10 | 0.0719 |
| confirmation_only | 13 | 0.0935 | 14 | 0.1007 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 61 | 0.4388 | 117 | 0.8417 |
| evidence_ignored | 112 | 0.8058 | 622 | 4.4748 |
| judgment_without_evidence | 11 | 0.0791 | 24 | 0.1727 |
| dead_end_update | 4 | 0.0288 | 4 | 0.0288 |
| unresolved_contradiction | 23 | 0.1655 | 47 | 0.3381 |
| hypothesis_to_commitment_shortcut | 16 | 0.1151 | 17 | 0.1223 |
| test_without_evidence | 10 | 0.0719 | 32 | 0.2302 |
| no_belief_revision | 111 | 0.7986 | 111 | 0.7986 |
| orphan_evidence | 6 | 0.0432 | 10 | 0.0719 |
| confirmation_only | 13 | 0.0935 | 14 | 0.1007 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 65 | 0.4676 | 173 | 1.2446 |
| evidence_handling | 99 | 0.7122 | 688 | 4.9496 |
| experimental_strategy | 117 | 0.8417 | 132 | 0.9496 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 67 | 0.4820 | 178 | 1.2806 |
| evidence_handling | 113 | 0.8129 | 688 | 4.9496 |
| experimental_strategy | 117 | 0.8417 | 132 | 0.9496 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 68 | 0.4892 | 87 | 0.6259 |
| evidence_handling | 8 | 0.0576 | 8 | 0.0576 |
| experimental_strategy | 2 | 0.0144 | 2 | 0.0144 |

### level_2

- Traces: 89 | Total messages: 2088 | Mean messages/trace: 23.46

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1249 | 1.2055 |
| n_H | 0.1209 | 0.1279 |
| n_T | 0.2263 | 0.2797 |
| n_E | 0.4493 | 0.4866 |
| n_J | 0.2480 | 0.2409 |
| n_U | 0.0225 | 0.0273 |
| n_C | 0.0580 | 0.0431 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8382 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.5688 |
| orphan_evidence_rate | 0.1250 |
| refute_neglect_rate | 0.2113 |
| hypothesis_switch_without_eval_rate | 0.5658 |
| scientificness_score | 0.4795 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 3 | 0.0337 | 3 | 0.0337 |
| ml_make_it_work | 5 | 0.0562 | 5 | 0.0562 |
| exploratory_to_confirmatory | 7 | 0.0787 | 7 | 0.0787 |
| bayesian_belief_updating | 22 | 0.2472 | 22 | 0.2472 |
| abductive | 60 | 0.6742 | 60 | 0.6742 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 3 | 0.0337 | 3 | 0.0337 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 61 | 0.6854 | 159 | 1.7865 |
| evidence_ignored | 53 | 0.5955 | 428 | 4.8090 |
| judgment_without_evidence | 13 | 0.1461 | 31 | 0.3483 |
| dead_end_update | 9 | 0.1011 | 9 | 0.1011 |
| unresolved_contradiction | 11 | 0.1236 | 39 | 0.4382 |
| hypothesis_to_commitment_shortcut | 21 | 0.2360 | 21 | 0.2360 |
| test_without_evidence | 0 | 0.0000 | 16 | 0.1798 |
| no_belief_revision | 50 | 0.5618 | 50 | 0.5618 |
| orphan_evidence | 8 | 0.0899 | 9 | 0.1011 |
| confirmation_only | 20 | 0.2247 | 20 | 0.2247 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 64 | 0.7191 | 159 | 1.7865 |
| evidence_ignored | 76 | 0.8539 | 428 | 4.8090 |
| judgment_without_evidence | 13 | 0.1461 | 31 | 0.3483 |
| dead_end_update | 9 | 0.1011 | 9 | 0.1011 |
| unresolved_contradiction | 25 | 0.2809 | 60 | 0.6742 |
| hypothesis_to_commitment_shortcut | 21 | 0.2360 | 21 | 0.2360 |
| test_without_evidence | 12 | 0.1348 | 16 | 0.1798 |
| no_belief_revision | 50 | 0.5618 | 50 | 0.5618 |
| orphan_evidence | 8 | 0.0899 | 9 | 0.1011 |
| confirmation_only | 20 | 0.2247 | 20 | 0.2247 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 66 | 0.7416 | 218 | 2.4494 |
| evidence_handling | 59 | 0.6629 | 484 | 5.4382 |
| experimental_strategy | 62 | 0.6966 | 80 | 0.8989 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 68 | 0.7640 | 239 | 2.6854 |
| evidence_handling | 77 | 0.8652 | 484 | 5.4382 |
| experimental_strategy | 62 | 0.6966 | 80 | 0.8989 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 68 | 0.7640 | 85 | 0.9551 |
| evidence_handling | 7 | 0.0787 | 7 | 0.0787 |
| experimental_strategy | 7 | 0.0787 | 8 | 0.0899 |

### level_3

- Traces: 29 | Total messages: 1275 | Mean messages/trace: 43.97

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2382 | 1.2353 |
| n_H | 0.0801 | 0.0722 |
| n_T | 0.4086 | 0.4133 |
| n_E | 0.5134 | 0.5224 |
| n_J | 0.1827 | 0.1875 |
| n_U | 0.0282 | 0.0235 |
| n_C | 0.0251 | 0.0165 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9448 |
| loop_density | 0.0004 |
| update_grounding_rate | 0.5934 |
| orphan_evidence_rate | 0.1208 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.7759 |
| scientificness_score | 0.5239 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 1 | 0.0345 | 1 | 0.0345 |
| ml_make_it_work | 2 | 0.0690 | 2 | 0.0690 |
| exploratory_to_confirmatory | 1 | 0.0345 | 1 | 0.0345 |
| bayesian_belief_updating | 9 | 0.3103 | 9 | 0.3103 |
| abductive | 2 | 0.0690 | 2 | 0.0690 |
| triangulation | 1 | 0.0345 | 1 | 0.0345 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 1 | 0.0345 | 1 | 0.0345 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 19 | 0.6552 | 49 | 1.6897 |
| evidence_ignored | 29 | 1.0000 | 360 | 12.4138 |
| judgment_without_evidence | 6 | 0.2069 | 6 | 0.2069 |
| dead_end_update | 9 | 0.3103 | 10 | 0.3448 |
| unresolved_contradiction | 13 | 0.4483 | 32 | 1.1034 |
| hypothesis_to_commitment_shortcut | 3 | 0.1034 | 3 | 0.1034 |
| test_without_evidence | 0 | 0.0000 | 15 | 0.5172 |
| no_belief_revision | 8 | 0.2759 | 8 | 0.2759 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 3 | 0.1034 | 3 | 0.1034 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 21 | 0.7241 | 49 | 1.6897 |
| evidence_ignored | 29 | 1.0000 | 360 | 12.4138 |
| judgment_without_evidence | 6 | 0.2069 | 6 | 0.2069 |
| dead_end_update | 9 | 0.3103 | 10 | 0.3448 |
| unresolved_contradiction | 16 | 0.5517 | 33 | 1.1379 |
| hypothesis_to_commitment_shortcut | 3 | 0.1034 | 3 | 0.1034 |
| test_without_evidence | 9 | 0.3103 | 15 | 0.5172 |
| no_belief_revision | 8 | 0.2759 | 8 | 0.2759 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 3 | 0.1034 | 3 | 0.1034 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 24 | 0.8276 | 84 | 2.8966 |
| evidence_handling | 29 | 1.0000 | 381 | 13.1379 |
| experimental_strategy | 18 | 0.6207 | 21 | 0.7241 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 25 | 0.8621 | 85 | 2.9310 |
| evidence_handling | 29 | 1.0000 | 381 | 13.1379 |
| experimental_strategy | 18 | 0.6207 | 21 | 0.7241 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 0.3448 | 12 | 0.4138 |
| evidence_handling | 2 | 0.0690 | 2 | 0.0690 |
| experimental_strategy | 3 | 0.1034 | 3 | 0.1034 |

## Overall

### overall

- Traces: 317 | Total messages: 8435 | Mean messages/trace: 26.61

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1481 | 1.2090 |
| n_H | 0.1049 | 0.1145 |
| n_T | 0.2901 | 0.3169 |
| n_E | 0.4541 | 0.4812 |
| n_J | 0.2226 | 0.2327 |
| n_U | 0.0209 | 0.0249 |
| n_C | 0.0553 | 0.0388 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8227 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.3749 |
| orphan_evidence_rate | 0.1404 |
| refute_neglect_rate | 0.1445 |
| hypothesis_switch_without_eval_rate | 0.5602 |
| scientificness_score | 0.4281 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 4 | 0.0126 | 4 | 0.0126 |
| ml_make_it_work | 9 | 0.0284 | 9 | 0.0284 |
| exploratory_to_confirmatory | 18 | 0.0568 | 18 | 0.0568 |
| bayesian_belief_updating | 91 | 0.2871 | 91 | 0.2871 |
| abductive | 187 | 0.5899 | 187 | 0.5899 |
| triangulation | 1 | 0.0032 | 1 | 0.0032 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 4 | 0.0126 | 4 | 0.0126 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 197 | 0.6215 | 698 | 2.2019 |
| evidence_ignored | 216 | 0.6814 | 1783 | 5.6246 |
| judgment_without_evidence | 53 | 0.1672 | 103 | 0.3249 |
| dead_end_update | 34 | 0.1073 | 40 | 0.1262 |
| unresolved_contradiction | 50 | 0.1577 | 153 | 0.4826 |
| hypothesis_to_commitment_shortcut | 60 | 0.1893 | 81 | 0.2555 |
| test_without_evidence | 3 | 0.0095 | 70 | 0.2208 |
| no_belief_revision | 185 | 0.5836 | 185 | 0.5836 |
| orphan_evidence | 59 | 0.1861 | 109 | 0.3438 |
| confirmation_only | 52 | 0.1640 | 65 | 0.2050 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 205 | 0.6467 | 698 | 2.2019 |
| evidence_ignored | 277 | 0.8738 | 1783 | 5.6246 |
| judgment_without_evidence | 53 | 0.1672 | 103 | 0.3249 |
| dead_end_update | 34 | 0.1073 | 40 | 0.1262 |
| unresolved_contradiction | 88 | 0.2776 | 196 | 0.6183 |
| hypothesis_to_commitment_shortcut | 60 | 0.1893 | 81 | 0.2555 |
| test_without_evidence | 38 | 0.1199 | 70 | 0.2208 |
| no_belief_revision | 185 | 0.5836 | 185 | 0.5836 |
| orphan_evidence | 59 | 0.1861 | 109 | 0.3438 |
| confirmation_only | 52 | 0.1640 | 65 | 0.2050 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 215 | 0.6782 | 916 | 2.8896 |
| evidence_handling | 246 | 0.7760 | 2065 | 6.5142 |
| experimental_strategy | 229 | 0.7224 | 306 | 0.9653 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 220 | 0.6940 | 959 | 3.0252 |
| evidence_handling | 279 | 0.8801 | 2065 | 6.5142 |
| experimental_strategy | 229 | 0.7224 | 306 | 0.9653 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 206 | 0.6498 | 282 | 0.8896 |
| evidence_handling | 19 | 0.0599 | 19 | 0.0599 |
| experimental_strategy | 12 | 0.0379 | 13 | 0.0410 |
