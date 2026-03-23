# Reasoning annotation analysis

- Node probability definition: node_count / number_of_messages_in_original_trace

## By model + env + level

### claude_sonnet_45/afm/level_1

- Traces: 5 | Total messages: 145 | Mean messages/trace: 29.00

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 0.9896 | 1.0000 |
| n_H | 0.0331 | 0.0345 |
| n_T | 0.3052 | 0.3103 |
| n_E | 0.4731 | 0.4759 |
| n_J | 0.1435 | 0.1448 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0347 | 0.0345 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7600 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.2331 |
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
| abductive | 4 | 0.8000 | 4 | 0.8000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.6000 | 4 | 0.8000 |
| evidence_ignored | 5 | 1.0000 | 34 | 6.8000 |
| judgment_without_evidence | 1 | 0.2000 | 1 | 0.2000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 5 | 1.0000 | 5 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.6000 | 4 | 0.8000 |
| evidence_ignored | 5 | 1.0000 | 34 | 6.8000 |
| judgment_without_evidence | 1 | 0.2000 | 1 | 0.2000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 5 | 1.0000 | 5 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 4 | 0.8000 |
| evidence_handling | 5 | 1.0000 | 35 | 7.0000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 4 | 0.8000 |
| evidence_handling | 5 | 1.0000 | 35 | 7.0000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 4 | 0.8000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/afm/level_2

- Traces: 2 | Total messages: 64 | Mean messages/trace: 32.00

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.0345 | 1.0312 |
| n_H | 0.0803 | 0.0781 |
| n_T | 0.3123 | 0.3125 |
| n_E | 0.5187 | 0.5156 |
| n_J | 0.0916 | 0.0938 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0315 | 0.0312 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8000 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.1838 |
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
| triangulation | 1 | 0.5000 | 1 | 0.5000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 2 | 1.0000 | 3 | 1.5000 |
| evidence_ignored | 2 | 1.0000 | 18 | 9.0000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 2 | 1.0000 | 2 | 1.0000 |
| orphan_evidence | 1 | 0.5000 | 1 | 0.5000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 2 | 1.0000 | 3 | 1.5000 |
| evidence_ignored | 2 | 1.0000 | 18 | 9.0000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 2 | 1.0000 | 2 | 1.0000 |
| orphan_evidence | 1 | 0.5000 | 1 | 0.5000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 1.0000 | 3 | 1.5000 |
| evidence_handling | 2 | 1.0000 | 19 | 9.5000 |
| experimental_strategy | 2 | 1.0000 | 2 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 1.0000 | 3 | 1.5000 |
| evidence_handling | 2 | 1.0000 | 19 | 9.5000 |
| experimental_strategy | 2 | 1.0000 | 2 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 1 | 0.5000 | 1 | 0.5000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/afm/level_3

- Traces: 1 | Total messages: 43 | Mean messages/trace: 43.00

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2326 | 1.2326 |
| n_H | 0.0698 | 0.0698 |
| n_T | 0.4419 | 0.4419 |
| n_E | 0.5349 | 0.5349 |
| n_J | 0.1860 | 0.1860 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0000 | 0.0000 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8000 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.1739 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | N/A |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 1 | 1.0000 | 1 | 1.0000 |
| abductive | 1 | 1.0000 | 1 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 1 | 1.0000 | 3 | 3.0000 |
| evidence_ignored | 1 | 1.0000 | 7 | 7.0000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 1 | 1.0000 |
| no_belief_revision | 1 | 1.0000 | 1 | 1.0000 |
| orphan_evidence | 1 | 1.0000 | 1 | 1.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 1 | 1.0000 | 3 | 3.0000 |
| evidence_ignored | 1 | 1.0000 | 7 | 7.0000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 1 | 1.0000 | 1 | 1.0000 |
| no_belief_revision | 1 | 1.0000 | 1 | 1.0000 |
| orphan_evidence | 1 | 1.0000 | 1 | 1.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 1.0000 | 3 | 3.0000 |
| evidence_handling | 1 | 1.0000 | 9 | 9.0000 |
| experimental_strategy | 1 | 1.0000 | 1 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 1.0000 | 3 | 3.0000 |
| evidence_handling | 1 | 1.0000 | 9 | 9.0000 |
| experimental_strategy | 1 | 1.0000 | 1 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 1.0000 | 2 | 2.0000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/afm/level_4

- Traces: 2 | Total messages: 86 | Mean messages/trace: 43.00

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.3140 | 1.3140 |
| n_H | 0.0698 | 0.0698 |
| n_T | 0.5698 | 0.5698 |
| n_E | 0.5698 | 0.5698 |
| n_J | 0.1047 | 0.1047 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0000 | 0.0000 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8000 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.3442 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | N/A |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 1 | 0.5000 | 1 | 0.5000 |
| abductive | 1 | 0.5000 | 1 | 0.5000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 2 | 1.0000 | 5 | 2.5000 |
| evidence_ignored | 2 | 1.0000 | 31 | 15.5000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 4 | 2.0000 |
| no_belief_revision | 2 | 1.0000 | 2 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 2 | 1.0000 | 5 | 2.5000 |
| evidence_ignored | 2 | 1.0000 | 31 | 15.5000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 2 | 1.0000 | 4 | 2.0000 |
| no_belief_revision | 2 | 1.0000 | 2 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 1.0000 | 5 | 2.5000 |
| evidence_handling | 2 | 1.0000 | 35 | 17.5000 |
| experimental_strategy | 2 | 1.0000 | 2 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 1.0000 | 5 | 2.5000 |
| evidence_handling | 2 | 1.0000 | 35 | 17.5000 |
| experimental_strategy | 2 | 1.0000 | 2 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 0.5000 | 2 | 1.0000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

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

### claude_sonnet_45/ml/level_1

- Traces: 15 | Total messages: 276 | Mean messages/trace: 18.40

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2532 | 1.2464 |
| n_H | 0.0148 | 0.0145 |
| n_T | 0.4292 | 0.4275 |
| n_E | 0.5212 | 0.5181 |
| n_J | 0.1879 | 0.1884 |
| n_U | 0.0035 | 0.0036 |
| n_C | 0.0965 | 0.0942 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.6667 |
| loop_density | 0.0000 |
| update_grounding_rate | 1.0000 |
| orphan_evidence_rate | 0.0588 |
| refute_neglect_rate | N/A |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | 0.5500 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 2 | 0.1333 | 2 | 0.1333 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 1 | 0.0667 | 1 | 0.0667 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.2000 | 3 | 0.2000 |
| evidence_ignored | 15 | 1.0000 | 69 | 4.6000 |
| judgment_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 4 | 0.2667 |
| no_belief_revision | 14 | 0.9333 | 14 | 0.9333 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.2000 | 3 | 0.2000 |
| evidence_ignored | 15 | 1.0000 | 69 | 4.6000 |
| judgment_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 3 | 0.2000 | 4 | 0.2667 |
| no_belief_revision | 14 | 0.9333 | 14 | 0.9333 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.2000 | 3 | 0.2000 |
| evidence_handling | 15 | 1.0000 | 74 | 4.9333 |
| experimental_strategy | 14 | 0.9333 | 14 | 0.9333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.2000 | 3 | 0.2000 |
| evidence_handling | 15 | 1.0000 | 74 | 4.9333 |
| experimental_strategy | 14 | 0.9333 | 14 | 0.9333 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.1333 | 2 | 0.1333 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 1 | 0.0667 | 1 | 0.0667 |

### claude_sonnet_45/resistor/level_1

- Traces: 30 | Total messages: 442 | Mean messages/trace: 14.73

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2112 | 1.1561 |
| n_H | 0.1998 | 0.2262 |
| n_T | 0.1783 | 0.1765 |
| n_E | 0.3396 | 0.3190 |
| n_J | 0.3457 | 0.3303 |
| n_U | 0.0265 | 0.0385 |
| n_C | 0.1213 | 0.0656 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8733 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.1410 |
| orphan_evidence_rate | 0.0377 |
| refute_neglect_rate | 0.0264 |
| hypothesis_switch_without_eval_rate | 0.7490 |
| scientificness_score | 0.4408 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 6 | 0.2000 | 6 | 0.2000 |
| ml_make_it_work | 2 | 0.0667 | 2 | 0.0667 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 11 | 0.3667 | 11 | 0.3667 |
| abductive | 20 | 0.6667 | 20 | 0.6667 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 9 | 0.3000 | 47 | 1.5667 |
| evidence_ignored | 11 | 0.3667 | 37 | 1.2333 |
| judgment_without_evidence | 10 | 0.3333 | 34 | 1.1333 |
| dead_end_update | 2 | 0.0667 | 3 | 0.1000 |
| unresolved_contradiction | 2 | 0.0667 | 15 | 0.5000 |
| hypothesis_to_commitment_shortcut | 1 | 0.0333 | 1 | 0.0333 |
| test_without_evidence | 1 | 0.0333 | 4 | 0.1333 |
| no_belief_revision | 17 | 0.5667 | 17 | 0.5667 |
| orphan_evidence | 1 | 0.0333 | 1 | 0.0333 |
| confirmation_only | 3 | 0.1000 | 4 | 0.1333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 10 | 0.3333 | 47 | 1.5667 |
| evidence_ignored | 16 | 0.5333 | 37 | 1.2333 |
| judgment_without_evidence | 10 | 0.3333 | 34 | 1.1333 |
| dead_end_update | 2 | 0.0667 | 3 | 0.1000 |
| unresolved_contradiction | 9 | 0.3000 | 22 | 0.7333 |
| hypothesis_to_commitment_shortcut | 1 | 0.0333 | 1 | 0.0333 |
| test_without_evidence | 4 | 0.1333 | 4 | 0.1333 |
| no_belief_revision | 17 | 0.5667 | 17 | 0.5667 |
| orphan_evidence | 1 | 0.0333 | 1 | 0.0333 |
| confirmation_only | 3 | 0.1000 | 4 | 0.1333 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 12 | 0.4000 | 66 | 2.2000 |
| evidence_handling | 18 | 0.6000 | 76 | 2.5333 |
| experimental_strategy | 20 | 0.6667 | 21 | 0.7000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.4333 | 73 | 2.4333 |
| evidence_handling | 19 | 0.6333 | 76 | 2.5333 |
| experimental_strategy | 20 | 0.6667 | 21 | 0.7000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 24 | 0.8000 | 37 | 1.2333 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 2 | 0.0667 | 2 | 0.0667 |

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

### claude_sonnet_45/wetlab/level_1

- Traces: 15 | Total messages: 521 | Mean messages/trace: 34.73

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.3735 | 1.3954 |
| n_H | 0.1753 | 0.1766 |
| n_T | 0.2696 | 0.2802 |
| n_E | 0.4818 | 0.4837 |
| n_J | 0.3634 | 0.3724 |
| n_U | 0.0532 | 0.0537 |
| n_C | 0.0302 | 0.0288 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9733 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.2619 |
| orphan_evidence_rate | 0.1534 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.1597 |
| scientificness_score | 0.4326 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 7 | 0.4667 | 7 | 0.4667 |
| abductive | 15 | 1.0000 | 15 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 86 | 5.7333 |
| evidence_ignored | 6 | 0.4000 | 53 | 3.5333 |
| judgment_without_evidence | 8 | 0.5333 | 11 | 0.7333 |
| dead_end_update | 5 | 0.3333 | 5 | 0.3333 |
| unresolved_contradiction | 5 | 0.3333 | 8 | 0.5333 |
| hypothesis_to_commitment_shortcut | 4 | 0.2667 | 6 | 0.4000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 1 | 0.0667 | 1 | 0.0667 |
| orphan_evidence | 10 | 0.6667 | 19 | 1.2667 |
| confirmation_only | 4 | 0.2667 | 5 | 0.3333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 86 | 5.7333 |
| evidence_ignored | 15 | 1.0000 | 53 | 3.5333 |
| judgment_without_evidence | 8 | 0.5333 | 11 | 0.7333 |
| dead_end_update | 5 | 0.3333 | 5 | 0.3333 |
| unresolved_contradiction | 6 | 0.4000 | 8 | 0.5333 |
| hypothesis_to_commitment_shortcut | 4 | 0.2667 | 6 | 0.4000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 1 | 0.0667 | 1 | 0.0667 |
| orphan_evidence | 10 | 0.6667 | 19 | 1.2667 |
| confirmation_only | 4 | 0.2667 | 5 | 0.3333 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 99 | 6.6000 |
| evidence_handling | 13 | 0.8667 | 83 | 5.5333 |
| experimental_strategy | 7 | 0.4667 | 12 | 0.8000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 99 | 6.6000 |
| evidence_handling | 15 | 1.0000 | 83 | 5.5333 |
| experimental_strategy | 7 | 0.4667 | 12 | 0.8000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 22 | 1.4667 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/wetlab/level_2

- Traces: 15 | Total messages: 975 | Mean messages/trace: 65.00

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.3947 | 1.4072 |
| n_H | 0.1762 | 0.1785 |
| n_T | 0.3289 | 0.3323 |
| n_E | 0.5013 | 0.5067 |
| n_J | 0.3275 | 0.3323 |
| n_U | 0.0467 | 0.0451 |
| n_C | 0.0141 | 0.0123 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 1.0000 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.1133 |
| orphan_evidence_rate | 0.1978 |
| refute_neglect_rate | 0.0208 |
| hypothesis_switch_without_eval_rate | 0.3361 |
| scientificness_score | 0.3863 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.0667 | 1 | 0.0667 |
| bayesian_belief_updating | 8 | 0.5333 | 8 | 0.5333 |
| abductive | 15 | 1.0000 | 15 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 165 | 11.0000 |
| evidence_ignored | 11 | 0.7333 | 157 | 10.4667 |
| judgment_without_evidence | 10 | 0.6667 | 24 | 1.6000 |
| dead_end_update | 10 | 0.6667 | 15 | 1.0000 |
| unresolved_contradiction | 5 | 0.3333 | 23 | 1.5333 |
| hypothesis_to_commitment_shortcut | 2 | 0.1333 | 4 | 0.2667 |
| test_without_evidence | 0 | 0.0000 | 2 | 0.1333 |
| no_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| orphan_evidence | 15 | 1.0000 | 41 | 2.7333 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 165 | 11.0000 |
| evidence_ignored | 15 | 1.0000 | 157 | 10.4667 |
| judgment_without_evidence | 10 | 0.6667 | 24 | 1.6000 |
| dead_end_update | 10 | 0.6667 | 15 | 1.0000 |
| unresolved_contradiction | 8 | 0.5333 | 24 | 1.6000 |
| hypothesis_to_commitment_shortcut | 2 | 0.1333 | 4 | 0.2667 |
| test_without_evidence | 2 | 0.1333 | 2 | 0.1333 |
| no_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| orphan_evidence | 15 | 1.0000 | 41 | 2.7333 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 188 | 12.5333 |
| evidence_handling | 15 | 1.0000 | 224 | 14.9333 |
| experimental_strategy | 10 | 0.6667 | 19 | 1.2667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 189 | 12.6000 |
| evidence_handling | 15 | 1.0000 | 224 | 14.9333 |
| experimental_strategy | 10 | 0.6667 | 19 | 1.2667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 23 | 1.5333 |
| evidence_handling | 1 | 0.0667 | 1 | 0.0667 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/wetlab/level_3

- Traces: 15 | Total messages: 1099 | Mean messages/trace: 73.27

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.4257 | 1.4313 |
| n_H | 0.1737 | 0.1747 |
| n_T | 0.3466 | 0.3530 |
| n_E | 0.5079 | 0.5123 |
| n_J | 0.3415 | 0.3385 |
| n_U | 0.0420 | 0.0400 |
| n_C | 0.0139 | 0.0127 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 1.0000 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.0500 |
| orphan_evidence_rate | 0.2231 |
| refute_neglect_rate | 0.0952 |
| hypothesis_switch_without_eval_rate | 0.3503 |
| scientificness_score | 0.3631 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 1 | 0.0667 | 1 | 0.0667 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.0667 | 1 | 0.0667 |
| bayesian_belief_updating | 14 | 0.9333 | 14 | 0.9333 |
| abductive | 15 | 1.0000 | 15 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 177 | 11.8000 |
| evidence_ignored | 11 | 0.7333 | 177 | 11.8000 |
| judgment_without_evidence | 7 | 0.4667 | 13 | 0.8667 |
| dead_end_update | 14 | 0.9333 | 22 | 1.4667 |
| unresolved_contradiction | 6 | 0.4000 | 28 | 1.8667 |
| hypothesis_to_commitment_shortcut | 4 | 0.2667 | 8 | 0.5333 |
| test_without_evidence | 0 | 0.0000 | 5 | 0.3333 |
| no_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| orphan_evidence | 15 | 1.0000 | 40 | 2.6667 |
| confirmation_only | 2 | 0.1333 | 3 | 0.2000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 177 | 11.8000 |
| evidence_ignored | 15 | 1.0000 | 177 | 11.8000 |
| judgment_without_evidence | 7 | 0.4667 | 13 | 0.8667 |
| dead_end_update | 14 | 0.9333 | 22 | 1.4667 |
| unresolved_contradiction | 9 | 0.6000 | 28 | 1.8667 |
| hypothesis_to_commitment_shortcut | 4 | 0.2667 | 8 | 0.5333 |
| test_without_evidence | 4 | 0.2667 | 5 | 0.3333 |
| no_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| orphan_evidence | 15 | 1.0000 | 40 | 2.6667 |
| confirmation_only | 2 | 0.1333 | 3 | 0.2000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 208 | 13.8667 |
| evidence_handling | 15 | 1.0000 | 235 | 15.6667 |
| experimental_strategy | 14 | 0.9333 | 30 | 2.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 208 | 13.8667 |
| evidence_handling | 15 | 1.0000 | 235 | 15.6667 |
| experimental_strategy | 14 | 0.9333 | 30 | 2.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 30 | 2.0000 |
| evidence_handling | 1 | 0.0667 | 1 | 0.0667 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/afm/level_1

- Traces: 5 | Total messages: 145 | Mean messages/trace: 29.00

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1451 | 1.1172 |
| n_H | 0.0337 | 0.0345 |
| n_T | 0.3637 | 0.3310 |
| n_E | 0.5479 | 0.5517 |
| n_J | 0.1667 | 0.1724 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0331 | 0.0276 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7600 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.1256 |
| refute_neglect_rate | 0.1667 |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | N/A |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 4 | 0.8000 | 4 | 0.8000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.6000 | 3 | 0.6000 |
| evidence_ignored | 4 | 0.8000 | 29 | 5.8000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 3 | 0.6000 | 9 | 1.8000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 5 | 1.0000 | 5 | 1.0000 |
| orphan_evidence | 1 | 0.2000 | 1 | 0.2000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.6000 | 3 | 0.6000 |
| evidence_ignored | 5 | 1.0000 | 29 | 5.8000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 3 | 0.6000 | 9 | 1.8000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 5 | 1.0000 | 5 | 1.0000 |
| orphan_evidence | 1 | 0.2000 | 1 | 0.2000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 12 | 2.4000 |
| evidence_handling | 4 | 0.8000 | 30 | 6.0000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 12 | 2.4000 |
| evidence_handling | 5 | 1.0000 | 30 | 6.0000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 4 | 0.8000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/afm/level_2

- Traces: 5 | Total messages: 163 | Mean messages/trace: 32.60

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1051 | 1.1534 |
| n_H | 0.0829 | 0.0920 |
| n_T | 0.2612 | 0.2945 |
| n_E | 0.4906 | 0.4908 |
| n_J | 0.1985 | 0.2086 |
| n_U | 0.0381 | 0.0368 |
| n_C | 0.0339 | 0.0307 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 1.0000 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.8000 |
| orphan_evidence_rate | 0.1595 |
| refute_neglect_rate | 0.0500 |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | 0.5051 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 2 | 0.4000 | 2 | 0.4000 |
| abductive | 3 | 0.6000 | 3 | 0.6000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 2 | 0.4000 | 4 | 0.8000 |
| evidence_ignored | 4 | 0.8000 | 24 | 4.8000 |
| judgment_without_evidence | 2 | 0.4000 | 2 | 0.4000 |
| dead_end_update | 2 | 0.4000 | 3 | 0.6000 |
| unresolved_contradiction | 1 | 0.2000 | 3 | 0.6000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 1 | 0.2000 |
| no_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.6000 | 4 | 0.8000 |
| evidence_ignored | 5 | 1.0000 | 24 | 4.8000 |
| judgment_without_evidence | 2 | 0.4000 | 2 | 0.4000 |
| dead_end_update | 2 | 0.4000 | 3 | 0.6000 |
| unresolved_contradiction | 2 | 0.4000 | 3 | 0.6000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 1 | 0.2000 | 1 | 0.2000 |
| no_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.4000 | 7 | 1.4000 |
| evidence_handling | 4 | 0.8000 | 27 | 5.4000 |
| experimental_strategy | 2 | 0.4000 | 3 | 0.6000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 7 | 1.4000 |
| evidence_handling | 5 | 1.0000 | 27 | 5.4000 |
| experimental_strategy | 2 | 0.4000 | 3 | 0.6000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 5 | 1.0000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/afm/level_3

- Traces: 5 | Total messages: 117 | Mean messages/trace: 23.40

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1158 | 1.1111 |
| n_H | 0.0249 | 0.0342 |
| n_T | 0.2945 | 0.2906 |
| n_E | 0.5344 | 0.5214 |
| n_J | 0.2107 | 0.2137 |
| n_U | 0.0054 | 0.0085 |
| n_C | 0.0459 | 0.0427 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7200 |
| loop_density | 0.0000 |
| update_grounding_rate | 1.0000 |
| orphan_evidence_rate | 0.0499 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 1.0000 |
| scientificness_score | 0.6882 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 1 | 0.2000 | 1 | 0.2000 |
| exploratory_to_confirmatory | 1 | 0.2000 | 1 | 0.2000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 2 | 0.4000 | 2 | 0.4000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 1 | 0.2000 | 1 | 0.2000 |
| evidence_ignored | 1 | 0.2000 | 14 | 2.8000 |
| judgment_without_evidence | 1 | 0.2000 | 1 | 0.2000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 2 | 0.4000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 4 | 0.8000 | 4 | 0.8000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 1 | 0.2000 | 1 | 0.2000 |
| evidence_ignored | 5 | 1.0000 | 14 | 2.8000 |
| judgment_without_evidence | 1 | 0.2000 | 1 | 0.2000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 1 | 0.2000 | 2 | 0.4000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| no_belief_revision | 4 | 0.8000 | 4 | 0.8000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 0.2000 | 3 | 0.6000 |
| evidence_handling | 2 | 0.4000 | 15 | 3.0000 |
| experimental_strategy | 4 | 0.8000 | 4 | 0.8000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 0.2000 | 3 | 0.6000 |
| evidence_handling | 5 | 1.0000 | 15 | 3.0000 |
| experimental_strategy | 4 | 0.8000 | 4 | 0.8000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.4000 | 2 | 0.4000 |
| evidence_handling | 1 | 0.2000 | 1 | 0.2000 |
| experimental_strategy | 1 | 0.2000 | 1 | 0.2000 |

### gpt_4o/afm/level_4

- Traces: 5 | Total messages: 133 | Mean messages/trace: 26.60

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 0.9137 | 0.8947 |
| n_H | 0.0414 | 0.0376 |
| n_T | 0.2169 | 0.2180 |
| n_E | 0.4424 | 0.4286 |
| n_J | 0.1721 | 0.1805 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0408 | 0.0301 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7600 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.1383 |
| refute_neglect_rate | 0.5000 |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | N/A |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 4 | 0.8000 | 4 | 0.8000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 2 | 0.4000 | 2 | 0.4000 |
| evidence_ignored | 3 | 0.6000 | 15 | 3.0000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 1 | 0.2000 | 2 | 0.4000 |
| hypothesis_to_commitment_shortcut | 2 | 0.4000 | 2 | 0.4000 |
| test_without_evidence | 0 | 0.0000 | 2 | 0.4000 |
| no_belief_revision | 5 | 1.0000 | 5 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 2 | 0.4000 | 2 | 0.4000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 2 | 0.4000 | 2 | 0.4000 |
| evidence_ignored | 5 | 1.0000 | 15 | 3.0000 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 1 | 0.2000 | 2 | 0.4000 |
| hypothesis_to_commitment_shortcut | 2 | 0.4000 | 2 | 0.4000 |
| test_without_evidence | 1 | 0.2000 | 2 | 0.4000 |
| no_belief_revision | 5 | 1.0000 | 5 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 2 | 0.4000 | 2 | 0.4000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 6 | 1.2000 |
| evidence_handling | 3 | 0.6000 | 17 | 3.4000 |
| experimental_strategy | 5 | 1.0000 | 7 | 1.4000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 6 | 1.2000 |
| evidence_handling | 5 | 1.0000 | 17 | 3.4000 |
| experimental_strategy | 5 | 1.0000 | 7 | 1.4000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 4 | 0.8000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
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

### gpt_4o/resistor/level_1

- Traces: 29 | Total messages: 1021 | Mean messages/trace: 35.21

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.4509 | 1.5083 |
| n_H | 0.1539 | 0.1469 |
| n_T | 0.3948 | 0.4182 |
| n_E | 0.5253 | 0.5524 |
| n_J | 0.3342 | 0.3595 |
| n_U | 0.0207 | 0.0206 |
| n_C | 0.0221 | 0.0108 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9034 |
| loop_density | 0.0005 |
| update_grounding_rate | 0.2941 |
| orphan_evidence_rate | 0.1144 |
| refute_neglect_rate | 0.0692 |
| hypothesis_switch_without_eval_rate | 0.6250 |
| scientificness_score | 0.4053 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 7 | 0.2414 | 7 | 0.2414 |
| bayesian_belief_updating | 19 | 0.6552 | 19 | 0.6552 |
| abductive | 25 | 0.8621 | 25 | 0.8621 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 1 | 0.0345 | 1 | 0.0345 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 19 | 0.6552 | 70 | 2.4138 |
| evidence_ignored | 13 | 0.4483 | 167 | 5.7586 |
| judgment_without_evidence | 9 | 0.3103 | 9 | 0.3103 |
| dead_end_update | 10 | 0.3448 | 11 | 0.3793 |
| unresolved_contradiction | 8 | 0.2759 | 59 | 2.0345 |
| hypothesis_to_commitment_shortcut | 2 | 0.0690 | 3 | 0.1034 |
| test_without_evidence | 0 | 0.0000 | 4 | 0.1379 |
| no_belief_revision | 12 | 0.4138 | 12 | 0.4138 |
| orphan_evidence | 8 | 0.2759 | 10 | 0.3448 |
| confirmation_only | 2 | 0.0690 | 3 | 0.1034 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 22 | 0.7586 | 70 | 2.4138 |
| evidence_ignored | 28 | 0.9655 | 167 | 5.7586 |
| judgment_without_evidence | 9 | 0.3103 | 9 | 0.3103 |
| dead_end_update | 10 | 0.3448 | 11 | 0.3793 |
| unresolved_contradiction | 20 | 0.6897 | 64 | 2.2069 |
| hypothesis_to_commitment_shortcut | 2 | 0.0690 | 3 | 0.1034 |
| test_without_evidence | 4 | 0.1379 | 4 | 0.1379 |
| no_belief_revision | 12 | 0.4138 | 12 | 0.4138 |
| orphan_evidence | 8 | 0.2759 | 10 | 0.3448 |
| confirmation_only | 2 | 0.0690 | 3 | 0.1034 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 0.6897 | 132 | 4.5517 |
| evidence_handling | 20 | 0.6897 | 190 | 6.5517 |
| experimental_strategy | 22 | 0.7586 | 26 | 0.8966 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 25 | 0.8621 | 137 | 4.7241 |
| evidence_handling | 28 | 0.9655 | 190 | 6.5517 |
| experimental_strategy | 22 | 0.7586 | 26 | 0.8966 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 29 | 1.0000 | 44 | 1.5172 |
| evidence_handling | 7 | 0.2414 | 7 | 0.2414 |
| experimental_strategy | 1 | 0.0345 | 1 | 0.0345 |

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

### gpt_4o/wetlab/level_1

- Traces: 15 | Total messages: 315 | Mean messages/trace: 21.00

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1752 | 1.1841 |
| n_H | 0.1672 | 0.1651 |
| n_T | 0.2812 | 0.2889 |
| n_E | 0.4276 | 0.4286 |
| n_J | 0.2262 | 0.2286 |
| n_U | 0.0240 | 0.0254 |
| n_C | 0.0489 | 0.0476 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9067 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.1250 |
| orphan_evidence_rate | 0.3334 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.0000 |
| scientificness_score | 0.3616 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 1 | 0.0667 | 1 | 0.0667 |
| bayesian_belief_updating | 9 | 0.6000 | 9 | 0.6000 |
| abductive | 15 | 1.0000 | 15 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 50 | 3.3333 |
| evidence_ignored | 15 | 1.0000 | 60 | 4.0000 |
| judgment_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| dead_end_update | 1 | 0.0667 | 1 | 0.0667 |
| unresolved_contradiction | 2 | 0.1333 | 5 | 0.3333 |
| hypothesis_to_commitment_shortcut | 11 | 0.7333 | 14 | 0.9333 |
| test_without_evidence | 0 | 0.0000 | 1 | 0.0667 |
| no_belief_revision | 7 | 0.4667 | 7 | 0.4667 |
| orphan_evidence | 13 | 0.8667 | 18 | 1.2000 |
| confirmation_only | 11 | 0.7333 | 14 | 0.9333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 50 | 3.3333 |
| evidence_ignored | 15 | 1.0000 | 60 | 4.0000 |
| judgment_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| dead_end_update | 1 | 0.0667 | 1 | 0.0667 |
| unresolved_contradiction | 5 | 0.3333 | 5 | 0.3333 |
| hypothesis_to_commitment_shortcut | 11 | 0.7333 | 14 | 0.9333 |
| test_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| no_belief_revision | 7 | 0.4667 | 7 | 0.4667 |
| orphan_evidence | 13 | 0.8667 | 18 | 1.2000 |
| confirmation_only | 11 | 0.7333 | 14 | 0.9333 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 69 | 4.6000 |
| evidence_handling | 15 | 1.0000 | 80 | 5.3333 |
| experimental_strategy | 11 | 0.7333 | 22 | 1.4667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 69 | 4.6000 |
| evidence_handling | 15 | 1.0000 | 80 | 5.3333 |
| experimental_strategy | 11 | 0.7333 | 22 | 1.4667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 24 | 1.6000 |
| evidence_handling | 1 | 0.0667 | 1 | 0.0667 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/wetlab/level_2

- Traces: 15 | Total messages: 351 | Mean messages/trace: 23.40

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1617 | 1.1652 |
| n_H | 0.1654 | 0.1652 |
| n_T | 0.2687 | 0.2735 |
| n_E | 0.4174 | 0.4217 |
| n_J | 0.2279 | 0.2251 |
| n_U | 0.0388 | 0.0370 |
| n_C | 0.0437 | 0.0427 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9333 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.2000 |
| orphan_evidence_rate | 0.2192 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.0312 |
| scientificness_score | 0.3836 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 9 | 0.6000 | 9 | 0.6000 |
| abductive | 15 | 1.0000 | 15 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 54 | 3.6000 |
| evidence_ignored | 14 | 0.9333 | 68 | 4.5333 |
| judgment_without_evidence | 5 | 0.3333 | 6 | 0.4000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 1 | 0.0667 | 2 | 0.1333 |
| hypothesis_to_commitment_shortcut | 6 | 0.4000 | 12 | 0.8000 |
| test_without_evidence | 0 | 0.0000 | 3 | 0.2000 |
| no_belief_revision | 5 | 0.3333 | 5 | 0.3333 |
| orphan_evidence | 14 | 0.9333 | 23 | 1.5333 |
| confirmation_only | 6 | 0.4000 | 11 | 0.7333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 54 | 3.6000 |
| evidence_ignored | 15 | 1.0000 | 68 | 4.5333 |
| judgment_without_evidence | 5 | 0.3333 | 6 | 0.4000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 2 | 0.1333 | 4 | 0.2667 |
| hypothesis_to_commitment_shortcut | 6 | 0.4000 | 12 | 0.8000 |
| test_without_evidence | 3 | 0.2000 | 3 | 0.2000 |
| no_belief_revision | 5 | 0.3333 | 5 | 0.3333 |
| orphan_evidence | 14 | 0.9333 | 23 | 1.5333 |
| confirmation_only | 6 | 0.4000 | 11 | 0.7333 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 67 | 4.4667 |
| evidence_handling | 15 | 1.0000 | 100 | 6.6667 |
| experimental_strategy | 8 | 0.5333 | 17 | 1.1333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 69 | 4.6000 |
| evidence_handling | 15 | 1.0000 | 100 | 6.6667 |
| experimental_strategy | 8 | 0.5333 | 17 | 1.1333 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 24 | 1.6000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_4o/wetlab/level_3

- Traces: 15 | Total messages: 377 | Mean messages/trace: 25.13

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2056 | 1.2202 |
| n_H | 0.1609 | 0.1645 |
| n_T | 0.3101 | 0.3156 |
| n_E | 0.4470 | 0.4509 |
| n_J | 0.2267 | 0.2308 |
| n_U | 0.0202 | 0.0186 |
| n_C | 0.0407 | 0.0398 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8933 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.4286 |
| orphan_evidence_rate | 0.2075 |
| refute_neglect_rate | 0.0000 |
| hypothesis_switch_without_eval_rate | 0.0000 |
| scientificness_score | 0.4521 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 2 | 0.1333 | 2 | 0.1333 |
| bayesian_belief_updating | 7 | 0.4667 | 7 | 0.4667 |
| abductive | 15 | 1.0000 | 15 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 14 | 0.9333 | 46 | 3.0667 |
| evidence_ignored | 13 | 0.8667 | 71 | 4.7333 |
| judgment_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 6 | 0.4000 | 12 | 0.8000 |
| hypothesis_to_commitment_shortcut | 4 | 0.2667 | 4 | 0.2667 |
| test_without_evidence | 0 | 0.0000 | 2 | 0.1333 |
| no_belief_revision | 8 | 0.5333 | 8 | 0.5333 |
| orphan_evidence | 11 | 0.7333 | 19 | 1.2667 |
| confirmation_only | 5 | 0.3333 | 8 | 0.5333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 15 | 1.0000 | 46 | 3.0667 |
| evidence_ignored | 15 | 1.0000 | 71 | 4.7333 |
| judgment_without_evidence | 1 | 0.0667 | 1 | 0.0667 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 6 | 0.4000 | 12 | 0.8000 |
| hypothesis_to_commitment_shortcut | 4 | 0.2667 | 4 | 0.2667 |
| test_without_evidence | 2 | 0.1333 | 2 | 0.1333 |
| no_belief_revision | 8 | 0.5333 | 8 | 0.5333 |
| orphan_evidence | 11 | 0.7333 | 19 | 1.2667 |
| confirmation_only | 5 | 0.3333 | 8 | 0.5333 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 66 | 4.4000 |
| evidence_handling | 15 | 1.0000 | 93 | 6.2000 |
| experimental_strategy | 10 | 0.6667 | 12 | 0.8000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 66 | 4.4000 |
| evidence_handling | 15 | 1.0000 | 93 | 6.2000 |
| experimental_strategy | 10 | 0.6667 | 12 | 0.8000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 22 | 1.4667 |
| evidence_handling | 2 | 0.1333 | 2 | 0.1333 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

## By model

### claude_sonnet_45

- Traces: 224 | Total messages: 6769 | Mean messages/trace: 30.22

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2213 | 1.2789 |
| n_H | 0.1266 | 0.1399 |
| n_T | 0.2831 | 0.3076 |
| n_E | 0.4660 | 0.4893 |
| n_J | 0.2582 | 0.2726 |
| n_U | 0.0284 | 0.0332 |
| n_C | 0.0589 | 0.0363 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8580 |
| loop_density | 0.0001 |
| update_grounding_rate | 0.3776 |
| orphan_evidence_rate | 0.1501 |
| refute_neglect_rate | 0.1438 |
| hypothesis_switch_without_eval_rate | 0.5617 |
| scientificness_score | 0.4451 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 9 | 0.0402 | 9 | 0.0402 |
| ml_make_it_work | 7 | 0.0312 | 7 | 0.0312 |
| exploratory_to_confirmatory | 4 | 0.0179 | 4 | 0.0179 |
| bayesian_belief_updating | 78 | 0.3482 | 78 | 0.3482 |
| abductive | 127 | 0.5670 | 127 | 0.5670 |
| triangulation | 2 | 0.0089 | 2 | 0.0089 |
| preregistered | 1 | 0.0045 | 1 | 0.0045 |
| active_learning | 2 | 0.0089 | 2 | 0.0089 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 144 | 0.6429 | 718 | 3.2054 |
| evidence_ignored | 145 | 0.6473 | 1289 | 5.7545 |
| judgment_without_evidence | 60 | 0.2679 | 137 | 0.6116 |
| dead_end_update | 47 | 0.2098 | 61 | 0.2723 |
| unresolved_contradiction | 37 | 0.1652 | 141 | 0.6295 |
| hypothesis_to_commitment_shortcut | 30 | 0.1339 | 39 | 0.1741 |
| test_without_evidence | 4 | 0.0179 | 63 | 0.2812 |
| no_belief_revision | 106 | 0.4732 | 106 | 0.4732 |
| orphan_evidence | 54 | 0.2411 | 118 | 0.5268 |
| confirmation_only | 23 | 0.1027 | 27 | 0.1205 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 148 | 0.6607 | 718 | 3.2054 |
| evidence_ignored | 197 | 0.8795 | 1289 | 5.7545 |
| judgment_without_evidence | 60 | 0.2679 | 137 | 0.6116 |
| dead_end_update | 47 | 0.2098 | 61 | 0.2723 |
| unresolved_contradiction | 70 | 0.3125 | 175 | 0.7812 |
| hypothesis_to_commitment_shortcut | 30 | 0.1339 | 39 | 0.1741 |
| test_without_evidence | 34 | 0.1518 | 63 | 0.2812 |
| no_belief_revision | 106 | 0.4732 | 106 | 0.4732 |
| orphan_evidence | 54 | 0.2411 | 118 | 0.5268 |
| confirmation_only | 23 | 0.1027 | 27 | 0.1205 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 153 | 0.6830 | 886 | 3.9554 |
| evidence_handling | 179 | 0.7991 | 1607 | 7.1741 |
| experimental_strategy | 160 | 0.7143 | 206 | 0.9196 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 156 | 0.6964 | 920 | 4.1071 |
| evidence_handling | 201 | 0.8973 | 1607 | 7.1741 |
| experimental_strategy | 160 | 0.7143 | 206 | 0.9196 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 146 | 0.6518 | 214 | 0.9554 |
| evidence_handling | 6 | 0.0268 | 6 | 0.0268 |
| experimental_strategy | 10 | 0.0446 | 10 | 0.0446 |

### gpt_4o

- Traces: 227 | Total messages: 5653 | Mean messages/trace: 24.90

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1383 | 1.1990 |
| n_H | 0.0971 | 0.0973 |
| n_T | 0.3064 | 0.3426 |
| n_E | 0.4501 | 0.4828 |
| n_J | 0.2181 | 0.2254 |
| n_U | 0.0140 | 0.0161 |
| n_C | 0.0527 | 0.0348 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8097 |
| loop_density | 0.0001 |
| update_grounding_rate | 0.3975 |
| orphan_evidence_rate | 0.1198 |
| refute_neglect_rate | 0.0859 |
| hypothesis_switch_without_eval_rate | 0.5171 |
| scientificness_score | 0.4355 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 2 | 0.0088 | 2 | 0.0088 |
| ml_make_it_work | 5 | 0.0220 | 5 | 0.0220 |
| exploratory_to_confirmatory | 25 | 0.1101 | 25 | 0.1101 |
| bayesian_belief_updating | 63 | 0.2775 | 63 | 0.2775 |
| abductive | 156 | 0.6872 | 156 | 0.6872 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 3 | 0.0132 | 3 | 0.0132 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 130 | 0.5727 | 330 | 1.4537 |
| evidence_ignored | 159 | 0.7004 | 1152 | 5.0749 |
| judgment_without_evidence | 26 | 0.1145 | 28 | 0.1233 |
| dead_end_update | 19 | 0.0837 | 22 | 0.0969 |
| unresolved_contradiction | 42 | 0.1850 | 140 | 0.6167 |
| hypothesis_to_commitment_shortcut | 46 | 0.2026 | 56 | 0.2467 |
| test_without_evidence | 0 | 0.0000 | 33 | 0.1454 |
| no_belief_revision | 151 | 0.6652 | 151 | 0.6652 |
| orphan_evidence | 50 | 0.2203 | 75 | 0.3304 |
| confirmation_only | 48 | 0.2115 | 60 | 0.2643 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 140 | 0.6167 | 330 | 1.4537 |
| evidence_ignored | 199 | 0.8767 | 1152 | 5.0749 |
| judgment_without_evidence | 26 | 0.1145 | 28 | 0.1233 |
| dead_end_update | 19 | 0.0837 | 22 | 0.0969 |
| unresolved_contradiction | 66 | 0.2907 | 148 | 0.6520 |
| hypothesis_to_commitment_shortcut | 46 | 0.2026 | 56 | 0.2467 |
| test_without_evidence | 25 | 0.1101 | 33 | 0.1454 |
| no_belief_revision | 151 | 0.6652 | 151 | 0.6652 |
| orphan_evidence | 50 | 0.2203 | 75 | 0.3304 |
| confirmation_only | 48 | 0.2115 | 60 | 0.2643 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 145 | 0.6388 | 530 | 2.3348 |
| evidence_handling | 172 | 0.7577 | 1288 | 5.6740 |
| experimental_strategy | 179 | 0.7885 | 229 | 1.0088 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 154 | 0.6784 | 538 | 2.3700 |
| evidence_handling | 200 | 0.8811 | 1288 | 5.6740 |
| experimental_strategy | 179 | 0.7885 | 229 | 1.0088 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 166 | 0.7313 | 221 | 0.9736 |
| evidence_handling | 25 | 0.1101 | 25 | 0.1101 |
| experimental_strategy | 7 | 0.0308 | 8 | 0.0352 |

## By env

### afm

- Traces: 30 | Total messages: 896 | Mean messages/trace: 29.87

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.0759 | 1.0893 |
| n_H | 0.0483 | 0.0536 |
| n_T | 0.3138 | 0.3259 |
| n_E | 0.5051 | 0.5045 |
| n_J | 0.1679 | 0.1696 |
| n_U | 0.0072 | 0.0078 |
| n_C | 0.0335 | 0.0279 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8000 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.9000 |
| orphan_evidence_rate | 0.1587 |
| refute_neglect_rate | 0.1558 |
| hypothesis_switch_without_eval_rate | 1.0000 |
| scientificness_score | 0.5967 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 1 | 0.0333 | 1 | 0.0333 |
| exploratory_to_confirmatory | 1 | 0.0333 | 1 | 0.0333 |
| bayesian_belief_updating | 4 | 0.1333 | 4 | 0.1333 |
| abductive | 19 | 0.6333 | 19 | 0.6333 |
| triangulation | 1 | 0.0333 | 1 | 0.0333 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 16 | 0.5333 | 25 | 0.8333 |
| evidence_ignored | 22 | 0.7333 | 172 | 5.7333 |
| judgment_without_evidence | 4 | 0.1333 | 4 | 0.1333 |
| dead_end_update | 2 | 0.0667 | 3 | 0.1000 |
| unresolved_contradiction | 5 | 0.1667 | 16 | 0.5333 |
| hypothesis_to_commitment_shortcut | 2 | 0.0667 | 2 | 0.0667 |
| test_without_evidence | 0 | 0.0000 | 8 | 0.2667 |
| no_belief_revision | 24 | 0.8000 | 24 | 0.8000 |
| orphan_evidence | 3 | 0.1000 | 3 | 0.1000 |
| confirmation_only | 2 | 0.0667 | 2 | 0.0667 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 17 | 0.5667 | 25 | 0.8333 |
| evidence_ignored | 30 | 1.0000 | 172 | 5.7333 |
| judgment_without_evidence | 4 | 0.1333 | 4 | 0.1333 |
| dead_end_update | 2 | 0.0667 | 3 | 0.1000 |
| unresolved_contradiction | 7 | 0.2333 | 16 | 0.5333 |
| hypothesis_to_commitment_shortcut | 2 | 0.0667 | 2 | 0.0667 |
| test_without_evidence | 5 | 0.1667 | 8 | 0.2667 |
| no_belief_revision | 24 | 0.8000 | 24 | 0.8000 |
| orphan_evidence | 3 | 0.1000 | 3 | 0.1000 |
| confirmation_only | 2 | 0.0667 | 2 | 0.0667 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 18 | 0.6000 | 43 | 1.4333 |
| evidence_handling | 23 | 0.7667 | 187 | 6.2333 |
| experimental_strategy | 26 | 0.8667 | 29 | 0.9667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 19 | 0.6333 | 43 | 1.4333 |
| evidence_handling | 30 | 1.0000 | 187 | 6.2333 |
| experimental_strategy | 26 | 0.8667 | 29 | 0.9667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 21 | 0.7000 | 23 | 0.7667 |
| evidence_handling | 2 | 0.0667 | 2 | 0.0667 |
| experimental_strategy | 1 | 0.0333 | 1 | 0.0333 |

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

- Traces: 30 | Total messages: 509 | Mean messages/trace: 16.97

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1697 | 1.1807 |
| n_H | 0.0074 | 0.0079 |
| n_T | 0.4363 | 0.4381 |
| n_E | 0.4853 | 0.4892 |
| n_J | 0.1464 | 0.1513 |
| n_U | 0.0018 | 0.0020 |
| n_C | 0.0926 | 0.0923 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.6333 |
| loop_density | 0.0000 |
| update_grounding_rate | 1.0000 |
| orphan_evidence_rate | 0.0683 |
| refute_neglect_rate | N/A |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | 0.5500 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 0 | 0.0000 | 0 | 0.0000 |
| abductive | 2 | 0.0667 | 2 | 0.0667 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 1 | 0.0333 | 1 | 0.0333 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.1000 | 3 | 0.1000 |
| evidence_ignored | 29 | 0.9667 | 136 | 4.5333 |
| judgment_without_evidence | 1 | 0.0333 | 1 | 0.0333 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 0 | 0.0000 | 4 | 0.1333 |
| no_belief_revision | 29 | 0.9667 | 29 | 0.9667 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 3 | 0.1000 | 3 | 0.1000 |
| evidence_ignored | 29 | 0.9667 | 136 | 4.5333 |
| judgment_without_evidence | 1 | 0.0333 | 1 | 0.0333 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_to_commitment_shortcut | 0 | 0.0000 | 0 | 0.0000 |
| test_without_evidence | 3 | 0.1000 | 4 | 0.1333 |
| no_belief_revision | 29 | 0.9667 | 29 | 0.9667 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.1000 | 3 | 0.1000 |
| evidence_handling | 29 | 0.9667 | 141 | 4.7000 |
| experimental_strategy | 29 | 0.9667 | 29 | 0.9667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.1000 | 3 | 0.1000 |
| evidence_handling | 29 | 0.9667 | 141 | 4.7000 |
| experimental_strategy | 29 | 0.9667 | 29 | 0.9667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.0667 | 2 | 0.0667 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 1 | 0.0333 | 1 | 0.0333 |

### resistor

- Traces: 59 | Total messages: 1463 | Mean messages/trace: 24.80

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.3290 | 1.4019 |
| n_H | 0.1772 | 0.1709 |
| n_T | 0.2847 | 0.3452 |
| n_E | 0.4309 | 0.4819 |
| n_J | 0.3400 | 0.3506 |
| n_U | 0.0237 | 0.0260 |
| n_C | 0.0725 | 0.0273 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8881 |
| loop_density | 0.0002 |
| update_grounding_rate | 0.2163 |
| orphan_evidence_rate | 0.0754 |
| refute_neglect_rate | 0.0474 |
| hypothesis_switch_without_eval_rate | 0.6881 |
| scientificness_score | 0.4234 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 6 | 0.1017 | 6 | 0.1017 |
| ml_make_it_work | 2 | 0.0339 | 2 | 0.0339 |
| exploratory_to_confirmatory | 7 | 0.1186 | 7 | 0.1186 |
| bayesian_belief_updating | 30 | 0.5085 | 30 | 0.5085 |
| abductive | 45 | 0.7627 | 45 | 0.7627 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 1 | 0.0169 | 1 | 0.0169 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 28 | 0.4746 | 117 | 1.9831 |
| evidence_ignored | 24 | 0.4068 | 204 | 3.4576 |
| judgment_without_evidence | 19 | 0.3220 | 43 | 0.7288 |
| dead_end_update | 12 | 0.2034 | 14 | 0.2373 |
| unresolved_contradiction | 10 | 0.1695 | 74 | 1.2542 |
| hypothesis_to_commitment_shortcut | 3 | 0.0508 | 4 | 0.0678 |
| test_without_evidence | 1 | 0.0169 | 8 | 0.1356 |
| no_belief_revision | 29 | 0.4915 | 29 | 0.4915 |
| orphan_evidence | 9 | 0.1525 | 11 | 0.1864 |
| confirmation_only | 5 | 0.0847 | 7 | 0.1186 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 32 | 0.5424 | 117 | 1.9831 |
| evidence_ignored | 44 | 0.7458 | 204 | 3.4576 |
| judgment_without_evidence | 19 | 0.3220 | 43 | 0.7288 |
| dead_end_update | 12 | 0.2034 | 14 | 0.2373 |
| unresolved_contradiction | 29 | 0.4915 | 86 | 1.4576 |
| hypothesis_to_commitment_shortcut | 3 | 0.0508 | 4 | 0.0678 |
| test_without_evidence | 8 | 0.1356 | 8 | 0.1356 |
| no_belief_revision | 29 | 0.4915 | 29 | 0.4915 |
| orphan_evidence | 9 | 0.1525 | 11 | 0.1864 |
| confirmation_only | 5 | 0.0847 | 7 | 0.1186 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 32 | 0.5424 | 198 | 3.3559 |
| evidence_handling | 38 | 0.6441 | 266 | 4.5085 |
| experimental_strategy | 42 | 0.7119 | 47 | 0.7966 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 38 | 0.6441 | 210 | 3.5593 |
| evidence_handling | 47 | 0.7966 | 266 | 4.5085 |
| experimental_strategy | 42 | 0.7119 | 47 | 0.7966 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 53 | 0.8983 | 81 | 1.3729 |
| evidence_handling | 7 | 0.1186 | 7 | 0.1186 |
| experimental_strategy | 3 | 0.0508 | 3 | 0.0508 |

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

- Traces: 90 | Total messages: 3638 | Mean messages/trace: 40.42

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2894 | 1.3507 |
| n_H | 0.1698 | 0.1732 |
| n_T | 0.3008 | 0.3200 |
| n_E | 0.4638 | 0.4843 |
| n_J | 0.2856 | 0.3101 |
| n_U | 0.0375 | 0.0396 |
| n_C | 0.0319 | 0.0236 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9511 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.1965 |
| orphan_evidence_rate | 0.2224 |
| refute_neglect_rate | 0.0193 |
| hypothesis_switch_without_eval_rate | 0.1462 |
| scientificness_score | 0.3965 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 1 | 0.0111 | 1 | 0.0111 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 5 | 0.0556 | 5 | 0.0556 |
| bayesian_belief_updating | 54 | 0.6000 | 54 | 0.6000 |
| abductive | 90 | 1.0000 | 90 | 1.0000 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 89 | 0.9889 | 578 | 6.4222 |
| evidence_ignored | 70 | 0.7778 | 586 | 6.5111 |
| judgment_without_evidence | 32 | 0.3556 | 56 | 0.6222 |
| dead_end_update | 30 | 0.3333 | 43 | 0.4778 |
| unresolved_contradiction | 25 | 0.2778 | 78 | 0.8667 |
| hypothesis_to_commitment_shortcut | 31 | 0.3444 | 48 | 0.5333 |
| test_without_evidence | 0 | 0.0000 | 13 | 0.1444 |
| no_belief_revision | 21 | 0.2333 | 21 | 0.2333 |
| orphan_evidence | 78 | 0.8667 | 160 | 1.7778 |
| confirmation_only | 28 | 0.3111 | 41 | 0.4556 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 90 | 1.0000 | 578 | 6.4222 |
| evidence_ignored | 90 | 1.0000 | 586 | 6.5111 |
| judgment_without_evidence | 32 | 0.3556 | 56 | 0.6222 |
| dead_end_update | 30 | 0.3333 | 43 | 0.4778 |
| unresolved_contradiction | 36 | 0.4000 | 81 | 0.9000 |
| hypothesis_to_commitment_shortcut | 31 | 0.3444 | 48 | 0.5333 |
| test_without_evidence | 12 | 0.1333 | 13 | 0.1444 |
| no_belief_revision | 21 | 0.2333 | 21 | 0.2333 |
| orphan_evidence | 78 | 0.8667 | 160 | 1.7778 |
| confirmation_only | 28 | 0.3111 | 41 | 0.4556 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 90 | 1.0000 | 697 | 7.7444 |
| evidence_handling | 88 | 0.9778 | 815 | 9.0556 |
| experimental_strategy | 60 | 0.6667 | 112 | 1.2444 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 90 | 1.0000 | 700 | 7.7778 |
| evidence_handling | 90 | 1.0000 | 815 | 9.0556 |
| experimental_strategy | 60 | 0.6667 | 112 | 1.2444 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 90 | 1.0000 | 145 | 1.6111 |
| evidence_handling | 5 | 0.0556 | 5 | 0.0556 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

## By level

### level_1

- Traces: 253 | Total messages: 5651 | Mean messages/trace: 22.34

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1720 | 1.2139 |
| n_H | 0.1030 | 0.1087 |
| n_T | 0.3008 | 0.3153 |
| n_E | 0.4487 | 0.4744 |
| n_J | 0.2357 | 0.2493 |
| n_U | 0.0170 | 0.0202 |
| n_C | 0.0668 | 0.0460 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7913 |
| loop_density | 0.0001 |
| update_grounding_rate | 0.3069 |
| orphan_evidence_rate | 0.1206 |
| refute_neglect_rate | 0.1112 |
| hypothesis_switch_without_eval_rate | 0.5987 |
| scientificness_score | 0.4162 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 6 | 0.0237 | 6 | 0.0237 |
| ml_make_it_work | 4 | 0.0158 | 4 | 0.0158 |
| exploratory_to_confirmatory | 16 | 0.0632 | 16 | 0.0632 |
| bayesian_belief_updating | 68 | 0.2688 | 68 | 0.2688 |
| abductive | 150 | 0.5929 | 150 | 0.5929 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 1 | 0.0040 | 1 | 0.0040 |
| active_learning | 1 | 0.0040 | 1 | 0.0040 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 125 | 0.4941 | 380 | 1.5020 |
| evidence_ignored | 160 | 0.6324 | 1071 | 4.2332 |
| judgment_without_evidence | 41 | 0.1621 | 81 | 0.3202 |
| dead_end_update | 22 | 0.0870 | 24 | 0.0949 |
| unresolved_contradiction | 35 | 0.1383 | 138 | 0.5455 |
| hypothesis_to_commitment_shortcut | 34 | 0.1344 | 41 | 0.1621 |
| test_without_evidence | 4 | 0.0158 | 45 | 0.1779 |
| no_belief_revision | 172 | 0.6798 | 172 | 0.6798 |
| orphan_evidence | 39 | 0.1542 | 59 | 0.2332 |
| confirmation_only | 33 | 0.1304 | 40 | 0.1581 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 132 | 0.5217 | 380 | 1.5020 |
| evidence_ignored | 211 | 0.8340 | 1071 | 4.2332 |
| judgment_without_evidence | 41 | 0.1621 | 81 | 0.3202 |
| dead_end_update | 22 | 0.0870 | 24 | 0.0949 |
| unresolved_contradiction | 66 | 0.2609 | 155 | 0.6126 |
| hypothesis_to_commitment_shortcut | 34 | 0.1344 | 41 | 0.1621 |
| test_without_evidence | 22 | 0.0870 | 45 | 0.1779 |
| no_belief_revision | 172 | 0.6798 | 172 | 0.6798 |
| orphan_evidence | 39 | 0.1542 | 59 | 0.2332 |
| confirmation_only | 33 | 0.1304 | 40 | 0.1581 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 137 | 0.5415 | 558 | 2.2055 |
| evidence_handling | 189 | 0.7470 | 1256 | 4.9644 |
| experimental_strategy | 201 | 0.7945 | 237 | 0.9368 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 145 | 0.5731 | 575 | 2.2727 |
| evidence_handling | 215 | 0.8498 | 1256 | 4.9644 |
| experimental_strategy | 201 | 0.7945 | 237 | 0.9368 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 161 | 0.6364 | 224 | 0.8854 |
| evidence_handling | 16 | 0.0632 | 16 | 0.0632 |
| experimental_strategy | 6 | 0.0237 | 6 | 0.0237 |

### level_2

- Traces: 126 | Total messages: 3641 | Mean messages/trace: 28.90

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1592 | 1.2502 |
| n_H | 0.1306 | 0.1425 |
| n_T | 0.2463 | 0.2944 |
| n_E | 0.4544 | 0.4864 |
| n_J | 0.2506 | 0.2598 |
| n_U | 0.0276 | 0.0330 |
| n_C | 0.0497 | 0.0341 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8746 |
| loop_density | 0.0000 |
| update_grounding_rate | 0.4784 |
| orphan_evidence_rate | 0.1472 |
| refute_neglect_rate | 0.1562 |
| hypothesis_switch_without_eval_rate | 0.4555 |
| scientificness_score | 0.4576 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 3 | 0.0238 | 3 | 0.0238 |
| ml_make_it_work | 5 | 0.0397 | 5 | 0.0397 |
| exploratory_to_confirmatory | 8 | 0.0635 | 8 | 0.0635 |
| bayesian_belief_updating | 41 | 0.3254 | 41 | 0.3254 |
| abductive | 93 | 0.7381 | 93 | 0.7381 |
| triangulation | 1 | 0.0079 | 1 | 0.0079 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 3 | 0.0238 | 3 | 0.0238 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 95 | 0.7540 | 385 | 3.0556 |
| evidence_ignored | 84 | 0.6667 | 695 | 5.5159 |
| judgment_without_evidence | 30 | 0.2381 | 63 | 0.5000 |
| dead_end_update | 21 | 0.1667 | 27 | 0.2143 |
| unresolved_contradiction | 18 | 0.1429 | 67 | 0.5317 |
| hypothesis_to_commitment_shortcut | 29 | 0.2302 | 37 | 0.2937 |
| test_without_evidence | 0 | 0.0000 | 22 | 0.1746 |
| no_belief_revision | 57 | 0.4524 | 57 | 0.4524 |
| orphan_evidence | 38 | 0.3016 | 74 | 0.5873 |
| confirmation_only | 26 | 0.2063 | 31 | 0.2460 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 99 | 0.7857 | 385 | 3.0556 |
| evidence_ignored | 113 | 0.8968 | 695 | 5.5159 |
| judgment_without_evidence | 30 | 0.2381 | 63 | 0.5000 |
| dead_end_update | 21 | 0.1667 | 27 | 0.2143 |
| unresolved_contradiction | 37 | 0.2937 | 91 | 0.7222 |
| hypothesis_to_commitment_shortcut | 29 | 0.2302 | 37 | 0.2937 |
| test_without_evidence | 18 | 0.1429 | 22 | 0.1746 |
| no_belief_revision | 57 | 0.4524 | 57 | 0.4524 |
| orphan_evidence | 38 | 0.3016 | 74 | 0.5873 |
| confirmation_only | 26 | 0.2063 | 31 | 0.2460 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 100 | 0.7937 | 483 | 3.8333 |
| evidence_handling | 95 | 0.7540 | 854 | 6.7778 |
| experimental_strategy | 84 | 0.6667 | 121 | 0.9603 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 103 | 0.8175 | 507 | 4.0238 |
| evidence_handling | 114 | 0.9048 | 854 | 6.7778 |
| experimental_strategy | 84 | 0.6667 | 121 | 0.9603 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 103 | 0.8175 | 137 | 1.0873 |
| evidence_handling | 9 | 0.0714 | 9 | 0.0714 |
| experimental_strategy | 7 | 0.0556 | 8 | 0.0635 |

### level_3

- Traces: 65 | Total messages: 2911 | Mean messages/trace: 44.78

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.2644 | 1.3023 |
| n_H | 0.1159 | 0.1213 |
| n_T | 0.3633 | 0.3734 |
| n_E | 0.4987 | 0.5094 |
| n_J | 0.2317 | 0.2511 |
| n_U | 0.0274 | 0.0282 |
| n_C | 0.0273 | 0.0189 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.9262 |
| loop_density | 0.0002 |
| update_grounding_rate | 0.4592 |
| orphan_evidence_rate | 0.1598 |
| refute_neglect_rate | 0.0220 |
| hypothesis_switch_without_eval_rate | 0.5118 |
| scientificness_score | 0.4822 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 2 | 0.0308 | 2 | 0.0308 |
| ml_make_it_work | 3 | 0.0462 | 3 | 0.0462 |
| exploratory_to_confirmatory | 5 | 0.0769 | 5 | 0.0769 |
| bayesian_belief_updating | 31 | 0.4769 | 31 | 0.4769 |
| abductive | 35 | 0.5385 | 35 | 0.5385 |
| triangulation | 1 | 0.0154 | 1 | 0.0154 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 1 | 0.0154 | 1 | 0.0154 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 50 | 0.7692 | 276 | 4.2462 |
| evidence_ignored | 55 | 0.8462 | 629 | 9.6769 |
| judgment_without_evidence | 15 | 0.2308 | 21 | 0.3231 |
| dead_end_update | 23 | 0.3538 | 32 | 0.4923 |
| unresolved_contradiction | 25 | 0.3846 | 74 | 1.1385 |
| hypothesis_to_commitment_shortcut | 11 | 0.1692 | 15 | 0.2308 |
| test_without_evidence | 0 | 0.0000 | 23 | 0.3538 |
| no_belief_revision | 21 | 0.3231 | 21 | 0.3231 |
| orphan_evidence | 27 | 0.4154 | 60 | 0.9231 |
| confirmation_only | 10 | 0.1538 | 14 | 0.2154 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 53 | 0.8154 | 276 | 4.2462 |
| evidence_ignored | 65 | 1.0000 | 629 | 9.6769 |
| judgment_without_evidence | 15 | 0.2308 | 21 | 0.3231 |
| dead_end_update | 23 | 0.3538 | 32 | 0.4923 |
| unresolved_contradiction | 32 | 0.4923 | 75 | 1.1538 |
| hypothesis_to_commitment_shortcut | 11 | 0.1692 | 15 | 0.2308 |
| test_without_evidence | 16 | 0.2462 | 23 | 0.3538 |
| no_belief_revision | 21 | 0.3231 | 21 | 0.3231 |
| orphan_evidence | 27 | 0.4154 | 60 | 0.9231 |
| confirmation_only | 10 | 0.1538 | 14 | 0.2154 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 56 | 0.8615 | 364 | 5.6000 |
| evidence_handling | 62 | 0.9538 | 733 | 11.2769 |
| experimental_strategy | 47 | 0.7231 | 68 | 1.0462 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 57 | 0.8769 | 365 | 5.6154 |
| evidence_handling | 65 | 1.0000 | 733 | 11.2769 |
| experimental_strategy | 47 | 0.7231 | 68 | 1.0462 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 43 | 0.6615 | 68 | 1.0462 |
| evidence_handling | 6 | 0.0923 | 6 | 0.0923 |
| experimental_strategy | 4 | 0.0615 | 4 | 0.0615 |

### level_4

- Traces: 7 | Total messages: 219 | Mean messages/trace: 31.29

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.0281 | 1.0594 |
| n_H | 0.0495 | 0.0502 |
| n_T | 0.3177 | 0.3562 |
| n_E | 0.4788 | 0.4840 |
| n_J | 0.1529 | 0.1507 |
| n_U | 0.0000 | 0.0000 |
| n_C | 0.0292 | 0.0183 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.7714 |
| loop_density | 0.0000 |
| update_grounding_rate | N/A |
| orphan_evidence_rate | 0.1971 |
| refute_neglect_rate | 0.3571 |
| hypothesis_switch_without_eval_rate | N/A |
| scientificness_score | N/A |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 0 | 0.0000 | 0 | 0.0000 |
| ml_make_it_work | 0 | 0.0000 | 0 | 0.0000 |
| exploratory_to_confirmatory | 0 | 0.0000 | 0 | 0.0000 |
| bayesian_belief_updating | 1 | 0.1429 | 1 | 0.1429 |
| abductive | 5 | 0.7143 | 5 | 0.7143 |
| triangulation | 0 | 0.0000 | 0 | 0.0000 |
| preregistered | 0 | 0.0000 | 0 | 0.0000 |
| active_learning | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 4 | 0.5714 | 7 | 1.0000 |
| evidence_ignored | 5 | 0.7143 | 46 | 6.5714 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 1 | 0.1429 | 2 | 0.2857 |
| hypothesis_to_commitment_shortcut | 2 | 0.2857 | 2 | 0.2857 |
| test_without_evidence | 0 | 0.0000 | 6 | 0.8571 |
| no_belief_revision | 7 | 1.0000 | 7 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 2 | 0.2857 | 2 | 0.2857 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 4 | 0.5714 | 7 | 1.0000 |
| evidence_ignored | 7 | 1.0000 | 46 | 6.5714 |
| judgment_without_evidence | 0 | 0.0000 | 0 | 0.0000 |
| dead_end_update | 0 | 0.0000 | 0 | 0.0000 |
| unresolved_contradiction | 1 | 0.1429 | 2 | 0.2857 |
| hypothesis_to_commitment_shortcut | 2 | 0.2857 | 2 | 0.2857 |
| test_without_evidence | 3 | 0.4286 | 6 | 0.8571 |
| no_belief_revision | 7 | 1.0000 | 7 | 1.0000 |
| orphan_evidence | 0 | 0.0000 | 0 | 0.0000 |
| confirmation_only | 2 | 0.2857 | 2 | 0.2857 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.7143 | 11 | 1.5714 |
| evidence_handling | 5 | 0.7143 | 52 | 7.4286 |
| experimental_strategy | 7 | 1.0000 | 9 | 1.2857 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.7143 | 11 | 1.5714 |
| evidence_handling | 7 | 1.0000 | 52 | 7.4286 |
| experimental_strategy | 7 | 1.0000 | 9 | 1.2857 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.7143 | 6 | 0.8571 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

## Overall

### overall

- Traces: 451 | Total messages: 12422 | Mean messages/trace: 27.54

#### Node probability per message

| field | mean/trace | pooled |
| --- | ---: | ---: |
| nodes_total | 1.1795 | 1.2426 |
| n_H | 0.1117 | 0.1205 |
| n_T | 0.2949 | 0.3235 |
| n_E | 0.4580 | 0.4863 |
| n_J | 0.2380 | 0.2511 |
| n_U | 0.0212 | 0.0254 |
| n_C | 0.0557 | 0.0357 |

#### Metric means

| metric | mean |
| --- | ---: |
| workflow_completeness | 0.8337 |
| loop_density | 0.0001 |
| update_grounding_rate | 0.3872 |
| orphan_evidence_rate | 0.1348 |
| refute_neglect_rate | 0.1152 |
| hypothesis_switch_without_eval_rate | 0.5405 |
| scientificness_score | 0.4404 |

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| popperian_falsification | 11 | 0.0244 | 11 | 0.0244 |
| ml_make_it_work | 12 | 0.0266 | 12 | 0.0266 |
| exploratory_to_confirmatory | 29 | 0.0643 | 29 | 0.0643 |
| bayesian_belief_updating | 141 | 0.3126 | 141 | 0.3126 |
| abductive | 283 | 0.6275 | 283 | 0.6275 |
| triangulation | 2 | 0.0044 | 2 | 0.0044 |
| preregistered | 1 | 0.0022 | 1 | 0.0022 |
| active_learning | 5 | 0.0111 | 5 | 0.0111 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 274 | 0.6075 | 1048 | 2.3237 |
| evidence_ignored | 304 | 0.6741 | 2441 | 5.4124 |
| judgment_without_evidence | 86 | 0.1907 | 165 | 0.3659 |
| dead_end_update | 66 | 0.1463 | 83 | 0.1840 |
| unresolved_contradiction | 79 | 0.1752 | 281 | 0.6231 |
| hypothesis_to_commitment_shortcut | 76 | 0.1685 | 95 | 0.2106 |
| test_without_evidence | 4 | 0.0089 | 96 | 0.2129 |
| no_belief_revision | 257 | 0.5698 | 257 | 0.5698 |
| orphan_evidence | 104 | 0.2306 | 193 | 0.4279 |
| confirmation_only | 71 | 0.1574 | 87 | 0.1929 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_hypothesis | 288 | 0.6386 | 1048 | 2.3237 |
| evidence_ignored | 396 | 0.8780 | 2441 | 5.4124 |
| judgment_without_evidence | 86 | 0.1907 | 165 | 0.3659 |
| dead_end_update | 66 | 0.1463 | 83 | 0.1840 |
| unresolved_contradiction | 136 | 0.3016 | 323 | 0.7162 |
| hypothesis_to_commitment_shortcut | 76 | 0.1685 | 95 | 0.2106 |
| test_without_evidence | 59 | 0.1308 | 96 | 0.2129 |
| no_belief_revision | 257 | 0.5698 | 257 | 0.5698 |
| orphan_evidence | 104 | 0.2306 | 193 | 0.4279 |
| confirmation_only | 71 | 0.1574 | 87 | 0.1929 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 298 | 0.6608 | 1416 | 3.1397 |
| evidence_handling | 351 | 0.7783 | 2895 | 6.4191 |
| experimental_strategy | 339 | 0.7517 | 435 | 0.9645 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 310 | 0.6874 | 1458 | 3.2328 |
| evidence_handling | 401 | 0.8891 | 2895 | 6.4191 |
| experimental_strategy | 339 | 0.7517 | 435 | 0.9645 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 312 | 0.6918 | 435 | 0.9645 |
| evidence_handling | 31 | 0.0687 | 31 | 0.0687 |
| experimental_strategy | 17 | 0.0377 | 18 | 0.0399 |
