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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 4 | 0.8000 | 4 | 0.8000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 4 | 0.8000 |
| evidence_non_uptake | 5 | 1.0000 | 34 | 6.8000 |
| unsupported_judgment | 1 | 0.2000 | 1 | 0.2000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 4 | 0.8000 |
| evidence_non_uptake | 5 | 1.0000 | 34 | 6.8000 |
| unsupported_judgment | 1 | 0.2000 | 1 | 0.2000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| convergent_multi_test_evidence | 1 | 0.5000 | 1 | 0.5000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 1.0000 | 3 | 1.5000 |
| evidence_non_uptake | 2 | 1.0000 | 18 | 9.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 2 | 1.0000 | 2 | 1.0000 |
| disconnected_evidence | 1 | 0.5000 | 1 | 0.5000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 1.0000 | 3 | 1.5000 |
| evidence_non_uptake | 2 | 1.0000 | 18 | 9.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 2 | 1.0000 | 2 | 1.0000 |
| disconnected_evidence | 1 | 0.5000 | 1 | 0.5000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 1 | 1.0000 | 1 | 1.0000 |
| evidence_led_hypothesis_generation | 1 | 1.0000 | 1 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 1 | 1.0000 | 3 | 3.0000 |
| evidence_non_uptake | 1 | 1.0000 | 7 | 7.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 1.0000 | 1 | 1.0000 |
| fixed_belief_trace | 1 | 1.0000 | 1 | 1.0000 |
| disconnected_evidence | 1 | 1.0000 | 1 | 1.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 1 | 1.0000 | 3 | 3.0000 |
| evidence_non_uptake | 1 | 1.0000 | 7 | 7.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 1.0000 | 1 | 1.0000 |
| fixed_belief_trace | 1 | 1.0000 | 1 | 1.0000 |
| disconnected_evidence | 1 | 1.0000 | 1 | 1.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 1 | 0.5000 | 1 | 0.5000 |
| evidence_led_hypothesis_generation | 1 | 0.5000 | 1 | 0.5000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 1.0000 | 5 | 2.5000 |
| evidence_non_uptake | 2 | 1.0000 | 31 | 15.5000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 1.0000 | 4 | 2.0000 |
| fixed_belief_trace | 2 | 1.0000 | 2 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 1.0000 | 5 | 2.5000 |
| evidence_non_uptake | 2 | 1.0000 | 31 | 15.5000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 1.0000 | 4 | 2.0000 |
| fixed_belief_trace | 2 | 1.0000 | 2 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 1 | 0.0667 | 1 | 0.0667 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.2000 | 3 | 0.2000 |
| evidence_non_uptake | 15 | 1.0000 | 121 | 8.0667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.2000 | 3 | 0.2000 |
| evidence_non_uptake | 15 | 1.0000 | 121 | 8.0667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0667 | 1 | 0.0667 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 1 | 0.1000 | 1 | 0.1000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 1 | 0.1000 | 1 | 0.1000 |
| evidence_led_hypothesis_generation | 1 | 0.1000 | 1 | 0.1000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.9000 | 15 | 1.5000 |
| evidence_non_uptake | 10 | 1.0000 | 89 | 8.9000 |
| unsupported_judgment | 1 | 0.1000 | 1 | 0.1000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 4 | 0.4000 | 9 | 0.9000 |
| premature_commitment | 1 | 0.1000 | 1 | 0.1000 |
| uninformative_test | 5 | 0.5000 | 27 | 2.7000 |
| fixed_belief_trace | 9 | 0.9000 | 9 | 0.9000 |
| disconnected_evidence | 3 | 0.3000 | 5 | 0.5000 |
| one_sided_confirmation | 1 | 0.1000 | 1 | 0.1000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.9000 | 15 | 1.5000 |
| evidence_non_uptake | 10 | 1.0000 | 89 | 8.9000 |
| unsupported_judgment | 1 | 0.1000 | 1 | 0.1000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 4 | 0.4000 | 13 | 1.3000 |
| premature_commitment | 1 | 0.1000 | 1 | 0.1000 |
| uninformative_test | 5 | 0.5000 | 27 | 2.7000 |
| fixed_belief_trace | 9 | 0.9000 | 9 | 0.9000 |
| disconnected_evidence | 3 | 0.3000 | 5 | 0.5000 |
| one_sided_confirmation | 1 | 0.1000 | 1 | 0.1000 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 1 | 0.1000 | 1 | 0.1000 |
| evidence_led_hypothesis_generation | 4 | 0.4000 | 4 | 0.4000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.9000 | 20 | 2.0000 |
| evidence_non_uptake | 10 | 1.0000 | 120 | 12.0000 |
| unsupported_judgment | 2 | 0.2000 | 2 | 0.2000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.2000 | 4 | 0.4000 |
| premature_commitment | 3 | 0.3000 | 3 | 0.3000 |
| uninformative_test | 4 | 0.4000 | 5 | 0.5000 |
| fixed_belief_trace | 8 | 0.8000 | 8 | 0.8000 |
| disconnected_evidence | 2 | 0.2000 | 2 | 0.2000 |
| one_sided_confirmation | 3 | 0.3000 | 3 | 0.3000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.9000 | 20 | 2.0000 |
| evidence_non_uptake | 10 | 1.0000 | 120 | 12.0000 |
| unsupported_judgment | 2 | 0.2000 | 2 | 0.2000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.2000 | 4 | 0.4000 |
| premature_commitment | 3 | 0.3000 | 3 | 0.3000 |
| uninformative_test | 4 | 0.4000 | 5 | 0.5000 |
| fixed_belief_trace | 8 | 0.8000 | 8 | 0.8000 |
| disconnected_evidence | 2 | 0.2000 | 2 | 0.2000 |
| one_sided_confirmation | 3 | 0.3000 | 3 | 0.3000 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 2 | 0.1333 | 2 | 0.1333 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 1 | 0.0667 | 1 | 0.0667 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.2000 | 3 | 0.2000 |
| evidence_non_uptake | 15 | 1.0000 | 69 | 4.6000 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 3 | 0.2000 | 4 | 0.2667 |
| fixed_belief_trace | 14 | 0.9333 | 14 | 0.9333 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.2000 | 3 | 0.2000 |
| evidence_non_uptake | 15 | 1.0000 | 69 | 4.6000 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 3 | 0.2000 | 4 | 0.2667 |
| fixed_belief_trace | 14 | 0.9333 | 14 | 0.9333 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

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
| refutation_driven_belief_revision | 6 | 0.2000 | 6 | 0.2000 |
| fixed_hypothesis_test_tuning | 2 | 0.0667 | 2 | 0.0667 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 11 | 0.3667 | 11 | 0.3667 |
| evidence_led_hypothesis_generation | 20 | 0.6667 | 20 | 0.6667 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 10 | 0.3333 | 47 | 1.5667 |
| evidence_non_uptake | 16 | 0.5333 | 37 | 1.2333 |
| unsupported_judgment | 10 | 0.3333 | 34 | 1.1333 |
| stalled_revision | 2 | 0.0667 | 3 | 0.1000 |
| contradiction_without_repair | 8 | 0.2667 | 15 | 0.5000 |
| premature_commitment | 1 | 0.0333 | 1 | 0.0333 |
| uninformative_test | 4 | 0.1333 | 4 | 0.1333 |
| fixed_belief_trace | 17 | 0.5667 | 17 | 0.5667 |
| disconnected_evidence | 1 | 0.0333 | 1 | 0.0333 |
| one_sided_confirmation | 3 | 0.1000 | 4 | 0.1333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 10 | 0.3333 | 47 | 1.5667 |
| evidence_non_uptake | 16 | 0.5333 | 37 | 1.2333 |
| unsupported_judgment | 10 | 0.3333 | 34 | 1.1333 |
| stalled_revision | 2 | 0.0667 | 3 | 0.1000 |
| contradiction_without_repair | 9 | 0.3000 | 22 | 0.7333 |
| premature_commitment | 1 | 0.0333 | 1 | 0.0333 |
| uninformative_test | 4 | 0.1333 | 4 | 0.1333 |
| fixed_belief_trace | 17 | 0.5667 | 17 | 0.5667 |
| disconnected_evidence | 1 | 0.0333 | 1 | 0.0333 |
| one_sided_confirmation | 3 | 0.1000 | 4 | 0.1333 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.4333 | 66 | 2.2000 |
| evidence_handling | 19 | 0.6333 | 76 | 2.5333 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 1 | 0.0667 | 1 | 0.0667 |
| explore_then_test_transition | 1 | 0.0667 | 1 | 0.0667 |
| hypothesis_reranking | 2 | 0.1333 | 2 | 0.1333 |
| evidence_led_hypothesis_generation | 5 | 0.3333 | 5 | 0.3333 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.1333 | 3 | 0.2000 |
| evidence_non_uptake | 9 | 0.6000 | 28 | 1.8667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 1 | 0.0667 | 1 | 0.0667 |
| contradiction_without_repair | 2 | 0.1333 | 7 | 0.4667 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 11 | 0.7333 | 11 | 0.7333 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.1333 | 3 | 0.2000 |
| evidence_non_uptake | 9 | 0.6000 | 28 | 1.8667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 1 | 0.0667 | 1 | 0.0667 |
| contradiction_without_repair | 2 | 0.1333 | 7 | 0.4667 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 11 | 0.7333 | 11 | 0.7333 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.2000 | 10 | 0.6667 |
| evidence_handling | 9 | 0.6000 | 28 | 1.8667 |
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
| refutation_driven_belief_revision | 1 | 0.0667 | 1 | 0.0667 |
| fixed_hypothesis_test_tuning | 3 | 0.2000 | 3 | 0.2000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 4 | 0.2667 | 4 | 0.2667 |
| evidence_led_hypothesis_generation | 1 | 0.0667 | 1 | 0.0667 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 10 | 0.6667 | 33 | 2.2000 |
| evidence_non_uptake | 15 | 1.0000 | 121 | 8.0667 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 4 | 0.2667 | 4 | 0.2667 |
| contradiction_without_repair | 5 | 0.3333 | 10 | 0.6667 |
| premature_commitment | 3 | 0.2000 | 3 | 0.2000 |
| uninformative_test | 3 | 0.2000 | 5 | 0.3333 |
| fixed_belief_trace | 6 | 0.4000 | 6 | 0.4000 |
| disconnected_evidence | 2 | 0.1333 | 2 | 0.1333 |
| one_sided_confirmation | 2 | 0.1333 | 2 | 0.1333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 10 | 0.6667 | 33 | 2.2000 |
| evidence_non_uptake | 15 | 1.0000 | 121 | 8.0667 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 4 | 0.2667 | 4 | 0.2667 |
| contradiction_without_repair | 7 | 0.4667 | 18 | 1.2000 |
| premature_commitment | 3 | 0.2000 | 3 | 0.2000 |
| uninformative_test | 3 | 0.2000 | 5 | 0.3333 |
| fixed_belief_trace | 6 | 0.4000 | 6 | 0.4000 |
| disconnected_evidence | 2 | 0.1333 | 2 | 0.1333 |
| one_sided_confirmation | 2 | 0.1333 | 2 | 0.1333 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 11 | 0.7333 | 45 | 3.0000 |
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
| refutation_driven_belief_revision | 1 | 0.0667 | 1 | 0.0667 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.0667 | 1 | 0.0667 |
| hypothesis_reranking | 5 | 0.3333 | 5 | 0.3333 |
| evidence_led_hypothesis_generation | 1 | 0.0667 | 1 | 0.0667 |
| convergent_multi_test_evidence | 1 | 0.0667 | 1 | 0.0667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 12 | 0.8000 | 33 | 2.2000 |
| evidence_non_uptake | 15 | 1.0000 | 129 | 8.6000 |
| unsupported_judgment | 2 | 0.1333 | 2 | 0.1333 |
| stalled_revision | 6 | 0.4000 | 6 | 0.4000 |
| contradiction_without_repair | 6 | 0.4000 | 14 | 0.9333 |
| premature_commitment | 3 | 0.2000 | 3 | 0.2000 |
| uninformative_test | 5 | 0.3333 | 5 | 0.3333 |
| fixed_belief_trace | 3 | 0.2000 | 3 | 0.2000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 12 | 0.8000 | 33 | 2.2000 |
| evidence_non_uptake | 15 | 1.0000 | 129 | 8.6000 |
| unsupported_judgment | 2 | 0.1333 | 2 | 0.1333 |
| stalled_revision | 6 | 0.4000 | 6 | 0.4000 |
| contradiction_without_repair | 7 | 0.4667 | 15 | 1.0000 |
| premature_commitment | 3 | 0.2000 | 3 | 0.2000 |
| uninformative_test | 5 | 0.3333 | 5 | 0.3333 |
| fixed_belief_trace | 3 | 0.2000 | 3 | 0.2000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 3 | 0.2000 | 3 | 0.2000 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 11 | 0.5000 | 11 | 0.5000 |
| evidence_led_hypothesis_generation | 21 | 0.9545 | 21 | 0.9545 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 19 | 0.8636 | 54 | 2.4545 |
| evidence_non_uptake | 17 | 0.7727 | 44 | 2.0000 |
| unsupported_judgment | 7 | 0.3182 | 19 | 0.8636 |
| stalled_revision | 2 | 0.0909 | 2 | 0.0909 |
| contradiction_without_repair | 7 | 0.3182 | 11 | 0.5000 |
| premature_commitment | 4 | 0.1818 | 5 | 0.2273 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 7 | 0.3182 | 7 | 0.3182 |
| disconnected_evidence | 2 | 0.0909 | 4 | 0.1818 |
| one_sided_confirmation | 1 | 0.0455 | 2 | 0.0909 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 19 | 0.8636 | 54 | 2.4545 |
| evidence_non_uptake | 17 | 0.7727 | 44 | 2.0000 |
| unsupported_judgment | 7 | 0.3182 | 19 | 0.8636 |
| stalled_revision | 2 | 0.0909 | 2 | 0.0909 |
| contradiction_without_repair | 7 | 0.3182 | 12 | 0.5455 |
| premature_commitment | 4 | 0.1818 | 5 | 0.2273 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 7 | 0.3182 | 7 | 0.3182 |
| disconnected_evidence | 2 | 0.0909 | 4 | 0.1818 |
| one_sided_confirmation | 1 | 0.0455 | 2 | 0.0909 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 20 | 0.9091 | 67 | 3.0455 |
| evidence_handling | 17 | 0.7727 | 67 | 3.0455 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 12 | 0.5455 | 12 | 0.5455 |
| evidence_led_hypothesis_generation | 20 | 0.9091 | 20 | 0.9091 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 18 | 0.8182 | 64 | 2.9091 |
| evidence_non_uptake | 20 | 0.9091 | 54 | 2.4545 |
| unsupported_judgment | 10 | 0.4545 | 28 | 1.2727 |
| stalled_revision | 3 | 0.1364 | 3 | 0.1364 |
| contradiction_without_repair | 7 | 0.3182 | 12 | 0.5455 |
| premature_commitment | 3 | 0.1364 | 3 | 0.1364 |
| uninformative_test | 1 | 0.0455 | 1 | 0.0455 |
| fixed_belief_trace | 5 | 0.2273 | 5 | 0.2273 |
| disconnected_evidence | 2 | 0.0909 | 2 | 0.0909 |
| one_sided_confirmation | 3 | 0.1364 | 3 | 0.1364 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 18 | 0.8182 | 64 | 2.9091 |
| evidence_non_uptake | 20 | 0.9091 | 54 | 2.4545 |
| unsupported_judgment | 10 | 0.4545 | 28 | 1.2727 |
| stalled_revision | 3 | 0.1364 | 3 | 0.1364 |
| contradiction_without_repair | 9 | 0.4091 | 24 | 1.0909 |
| premature_commitment | 3 | 0.1364 | 3 | 0.1364 |
| uninformative_test | 1 | 0.0455 | 1 | 0.0455 |
| fixed_belief_trace | 5 | 0.2273 | 5 | 0.2273 |
| disconnected_evidence | 2 | 0.0909 | 2 | 0.0909 |
| one_sided_confirmation | 3 | 0.1364 | 3 | 0.1364 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 19 | 0.8636 | 79 | 3.5909 |
| evidence_handling | 21 | 0.9545 | 85 | 3.8636 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 7 | 0.4667 | 7 | 0.4667 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 86 | 5.7333 |
| evidence_non_uptake | 15 | 1.0000 | 53 | 3.5333 |
| unsupported_judgment | 8 | 0.5333 | 11 | 0.7333 |
| stalled_revision | 5 | 0.3333 | 5 | 0.3333 |
| contradiction_without_repair | 6 | 0.4000 | 8 | 0.5333 |
| premature_commitment | 4 | 0.2667 | 6 | 0.4000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 1 | 0.0667 | 1 | 0.0667 |
| disconnected_evidence | 10 | 0.6667 | 19 | 1.2667 |
| one_sided_confirmation | 4 | 0.2667 | 5 | 0.3333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 86 | 5.7333 |
| evidence_non_uptake | 15 | 1.0000 | 53 | 3.5333 |
| unsupported_judgment | 8 | 0.5333 | 11 | 0.7333 |
| stalled_revision | 5 | 0.3333 | 5 | 0.3333 |
| contradiction_without_repair | 6 | 0.4000 | 8 | 0.5333 |
| premature_commitment | 4 | 0.2667 | 6 | 0.4000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 1 | 0.0667 | 1 | 0.0667 |
| disconnected_evidence | 10 | 0.6667 | 19 | 1.2667 |
| one_sided_confirmation | 4 | 0.2667 | 5 | 0.3333 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 99 | 6.6000 |
| evidence_handling | 15 | 1.0000 | 83 | 5.5333 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.0667 | 1 | 0.0667 |
| hypothesis_reranking | 8 | 0.5333 | 8 | 0.5333 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 165 | 11.0000 |
| evidence_non_uptake | 15 | 1.0000 | 157 | 10.4667 |
| unsupported_judgment | 10 | 0.6667 | 24 | 1.6000 |
| stalled_revision | 10 | 0.6667 | 15 | 1.0000 |
| contradiction_without_repair | 8 | 0.5333 | 23 | 1.5333 |
| premature_commitment | 2 | 0.1333 | 4 | 0.2667 |
| uninformative_test | 2 | 0.1333 | 2 | 0.1333 |
| fixed_belief_trace | 0 | 0.0000 | 0 | 0.0000 |
| disconnected_evidence | 15 | 1.0000 | 41 | 2.7333 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 165 | 11.0000 |
| evidence_non_uptake | 15 | 1.0000 | 157 | 10.4667 |
| unsupported_judgment | 10 | 0.6667 | 24 | 1.6000 |
| stalled_revision | 10 | 0.6667 | 15 | 1.0000 |
| contradiction_without_repair | 8 | 0.5333 | 24 | 1.6000 |
| premature_commitment | 2 | 0.1333 | 4 | 0.2667 |
| uninformative_test | 2 | 0.1333 | 2 | 0.1333 |
| fixed_belief_trace | 0 | 0.0000 | 0 | 0.0000 |
| disconnected_evidence | 15 | 1.0000 | 41 | 2.7333 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

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
| refutation_driven_belief_revision | 1 | 0.0667 | 1 | 0.0667 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.0667 | 1 | 0.0667 |
| hypothesis_reranking | 14 | 0.9333 | 14 | 0.9333 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 177 | 11.8000 |
| evidence_non_uptake | 15 | 1.0000 | 177 | 11.8000 |
| unsupported_judgment | 7 | 0.4667 | 13 | 0.8667 |
| stalled_revision | 14 | 0.9333 | 22 | 1.4667 |
| contradiction_without_repair | 9 | 0.6000 | 28 | 1.8667 |
| premature_commitment | 4 | 0.2667 | 8 | 0.5333 |
| uninformative_test | 4 | 0.2667 | 5 | 0.3333 |
| fixed_belief_trace | 0 | 0.0000 | 0 | 0.0000 |
| disconnected_evidence | 15 | 1.0000 | 40 | 2.6667 |
| one_sided_confirmation | 2 | 0.1333 | 3 | 0.2000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 177 | 11.8000 |
| evidence_non_uptake | 15 | 1.0000 | 177 | 11.8000 |
| unsupported_judgment | 7 | 0.4667 | 13 | 0.8667 |
| stalled_revision | 14 | 0.9333 | 22 | 1.4667 |
| contradiction_without_repair | 9 | 0.6000 | 28 | 1.8667 |
| premature_commitment | 4 | 0.2667 | 8 | 0.5333 |
| uninformative_test | 4 | 0.2667 | 5 | 0.3333 |
| fixed_belief_trace | 0 | 0.0000 | 0 | 0.0000 |
| disconnected_evidence | 15 | 1.0000 | 40 | 2.6667 |
| one_sided_confirmation | 2 | 0.1333 | 3 | 0.2000 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 4 | 0.8000 | 4 | 0.8000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 29 | 5.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 3 | 0.6000 | 9 | 1.8000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 1 | 0.2000 | 1 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 29 | 5.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 3 | 0.6000 | 9 | 1.8000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 1 | 0.2000 | 1 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 12 | 2.4000 |
| evidence_handling | 5 | 1.0000 | 30 | 6.0000 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 2 | 0.4000 | 2 | 0.4000 |
| evidence_led_hypothesis_generation | 3 | 0.6000 | 3 | 0.6000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 4 | 0.8000 |
| evidence_non_uptake | 5 | 1.0000 | 24 | 4.8000 |
| unsupported_judgment | 2 | 0.4000 | 2 | 0.4000 |
| stalled_revision | 2 | 0.4000 | 3 | 0.6000 |
| contradiction_without_repair | 2 | 0.4000 | 3 | 0.6000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 0.2000 | 1 | 0.2000 |
| fixed_belief_trace | 0 | 0.0000 | 0 | 0.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 4 | 0.8000 |
| evidence_non_uptake | 5 | 1.0000 | 24 | 4.8000 |
| unsupported_judgment | 2 | 0.4000 | 2 | 0.4000 |
| stalled_revision | 2 | 0.4000 | 3 | 0.6000 |
| contradiction_without_repair | 2 | 0.4000 | 3 | 0.6000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 0.2000 | 1 | 0.2000 |
| fixed_belief_trace | 0 | 0.0000 | 0 | 0.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 7 | 1.4000 |
| evidence_handling | 5 | 1.0000 | 27 | 5.4000 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 1 | 0.2000 | 1 | 0.2000 |
| explore_then_test_transition | 1 | 0.2000 | 1 | 0.2000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 2 | 0.4000 | 2 | 0.4000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 1 | 0.2000 | 1 | 0.2000 |
| evidence_non_uptake | 5 | 1.0000 | 14 | 2.8000 |
| unsupported_judgment | 1 | 0.2000 | 1 | 0.2000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2000 | 2 | 0.4000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 4 | 0.8000 | 4 | 0.8000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 1 | 0.2000 | 1 | 0.2000 |
| evidence_non_uptake | 5 | 1.0000 | 14 | 2.8000 |
| unsupported_judgment | 1 | 0.2000 | 1 | 0.2000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2000 | 2 | 0.4000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 4 | 0.8000 | 4 | 0.8000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 0.2000 | 3 | 0.6000 |
| evidence_handling | 5 | 1.0000 | 15 | 3.0000 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 4 | 0.8000 | 4 | 0.8000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.4000 | 2 | 0.4000 |
| evidence_non_uptake | 5 | 1.0000 | 15 | 3.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2000 | 2 | 0.4000 |
| premature_commitment | 2 | 0.4000 | 2 | 0.4000 |
| uninformative_test | 1 | 0.2000 | 2 | 0.4000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 2 | 0.4000 | 2 | 0.4000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.4000 | 2 | 0.4000 |
| evidence_non_uptake | 5 | 1.0000 | 15 | 3.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2000 | 2 | 0.4000 |
| premature_commitment | 2 | 0.4000 | 2 | 0.4000 |
| uninformative_test | 1 | 0.2000 | 2 | 0.4000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 2 | 0.4000 | 2 | 0.4000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 6 | 1.2000 |
| evidence_handling | 5 | 1.0000 | 17 | 3.4000 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 2 | 0.1333 | 2 | 0.1333 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.1333 | 2 | 0.1333 |
| evidence_non_uptake | 15 | 1.0000 | 113 | 7.5333 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 1 | 0.0667 | 1 | 0.0667 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.1333 | 2 | 0.1333 |
| evidence_non_uptake | 15 | 1.0000 | 113 | 7.5333 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 1 | 0.0667 | 1 | 0.0667 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0667 | 1 | 0.0667 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 5 | 0.5000 | 5 | 0.5000 |
| hypothesis_reranking | 4 | 0.4000 | 4 | 0.4000 |
| evidence_led_hypothesis_generation | 9 | 0.9000 | 9 | 0.9000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 8 | 0.8000 | 19 | 1.9000 |
| evidence_non_uptake | 10 | 1.0000 | 95 | 9.5000 |
| unsupported_judgment | 1 | 0.1000 | 2 | 0.2000 |
| stalled_revision | 1 | 0.1000 | 1 | 0.1000 |
| contradiction_without_repair | 6 | 0.6000 | 11 | 1.1000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.2000 | 2 | 0.2000 |
| fixed_belief_trace | 6 | 0.6000 | 6 | 0.6000 |
| disconnected_evidence | 1 | 0.1000 | 1 | 0.1000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 8 | 0.8000 | 19 | 1.9000 |
| evidence_non_uptake | 10 | 1.0000 | 95 | 9.5000 |
| unsupported_judgment | 1 | 0.1000 | 2 | 0.2000 |
| stalled_revision | 1 | 0.1000 | 1 | 0.1000 |
| contradiction_without_repair | 6 | 0.6000 | 11 | 1.1000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.2000 | 2 | 0.2000 |
| fixed_belief_trace | 6 | 0.6000 | 6 | 0.6000 |
| disconnected_evidence | 1 | 0.1000 | 1 | 0.1000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

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
| refutation_driven_belief_revision | 2 | 0.4000 | 2 | 0.4000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 4 | 0.8000 | 4 | 0.8000 |
| hypothesis_reranking | 3 | 0.6000 | 3 | 0.6000 |
| evidence_led_hypothesis_generation | 5 | 1.0000 | 5 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 1.0000 | 12 | 2.4000 |
| evidence_non_uptake | 5 | 1.0000 | 53 | 10.6000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 2 | 0.4000 | 2 | 0.4000 |
| contradiction_without_repair | 3 | 0.6000 | 3 | 0.6000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.4000 | 2 | 0.4000 |
| fixed_belief_trace | 1 | 0.2000 | 1 | 0.2000 |
| disconnected_evidence | 1 | 0.2000 | 2 | 0.4000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 1.0000 | 12 | 2.4000 |
| evidence_non_uptake | 5 | 1.0000 | 53 | 10.6000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 2 | 0.4000 | 2 | 0.4000 |
| contradiction_without_repair | 3 | 0.6000 | 4 | 0.8000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.4000 | 2 | 0.4000 |
| fixed_belief_trace | 1 | 0.2000 | 1 | 0.2000 |
| disconnected_evidence | 1 | 0.2000 | 2 | 0.4000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 15 | 3.0000 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 14 | 0.9333 | 67 | 4.4667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 14 | 0.9333 | 67 | 4.4667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 7 | 0.2414 | 7 | 0.2414 |
| hypothesis_reranking | 19 | 0.6552 | 19 | 0.6552 |
| evidence_led_hypothesis_generation | 25 | 0.8621 | 25 | 0.8621 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 1 | 0.0345 | 1 | 0.0345 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 22 | 0.7586 | 70 | 2.4138 |
| evidence_non_uptake | 28 | 0.9655 | 167 | 5.7586 |
| unsupported_judgment | 9 | 0.3103 | 9 | 0.3103 |
| stalled_revision | 10 | 0.3448 | 11 | 0.3793 |
| contradiction_without_repair | 17 | 0.5862 | 59 | 2.0345 |
| premature_commitment | 2 | 0.0690 | 3 | 0.1034 |
| uninformative_test | 4 | 0.1379 | 4 | 0.1379 |
| fixed_belief_trace | 12 | 0.4138 | 12 | 0.4138 |
| disconnected_evidence | 8 | 0.2759 | 10 | 0.3448 |
| one_sided_confirmation | 2 | 0.0690 | 3 | 0.1034 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 22 | 0.7586 | 70 | 2.4138 |
| evidence_non_uptake | 28 | 0.9655 | 167 | 5.7586 |
| unsupported_judgment | 9 | 0.3103 | 9 | 0.3103 |
| stalled_revision | 10 | 0.3448 | 11 | 0.3793 |
| contradiction_without_repair | 20 | 0.6897 | 64 | 2.2069 |
| premature_commitment | 2 | 0.0690 | 3 | 0.1034 |
| uninformative_test | 4 | 0.1379 | 4 | 0.1379 |
| fixed_belief_trace | 12 | 0.4138 | 12 | 0.4138 |
| disconnected_evidence | 8 | 0.2759 | 10 | 0.3448 |
| one_sided_confirmation | 2 | 0.0690 | 3 | 0.1034 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 24 | 0.8276 | 132 | 4.5517 |
| evidence_handling | 28 | 0.9655 | 190 | 6.5517 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.0667 | 1 | 0.0667 |
| hypothesis_reranking | 1 | 0.0667 | 1 | 0.0667 |
| evidence_led_hypothesis_generation | 5 | 0.3333 | 5 | 0.3333 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 6 | 0.4000 | 6 | 0.4000 |
| evidence_non_uptake | 10 | 0.6667 | 39 | 2.6000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 3 | 0.2000 | 3 | 0.2000 |
| uninformative_test | 1 | 0.0667 | 1 | 0.0667 |
| fixed_belief_trace | 14 | 0.9333 | 14 | 0.9333 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 6 | 0.4000 | 6 | 0.4000 |
| evidence_non_uptake | 10 | 0.6667 | 39 | 2.6000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 3 | 0.2000 | 3 | 0.2000 |
| uninformative_test | 1 | 0.0667 | 1 | 0.0667 |
| fixed_belief_trace | 14 | 0.9333 | 14 | 0.9333 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 6 | 0.4000 | 9 | 0.6000 |
| evidence_handling | 11 | 0.7333 | 40 | 2.6667 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 2 | 0.1333 | 2 | 0.1333 |
| explore_then_test_transition | 3 | 0.2000 | 3 | 0.2000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 8 | 0.5333 | 8 | 0.5333 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 2 | 0.1333 | 2 | 0.1333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 7 | 0.4667 | 12 | 0.8000 |
| evidence_non_uptake | 14 | 0.9333 | 57 | 3.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.1333 | 6 | 0.4000 |
| premature_commitment | 2 | 0.1333 | 2 | 0.1333 |
| uninformative_test | 2 | 0.1333 | 3 | 0.2000 |
| fixed_belief_trace | 10 | 0.6667 | 10 | 0.6667 |
| disconnected_evidence | 1 | 0.0667 | 1 | 0.0667 |
| one_sided_confirmation | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 7 | 0.4667 | 12 | 0.8000 |
| evidence_non_uptake | 14 | 0.9333 | 57 | 3.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.1333 | 6 | 0.4000 |
| premature_commitment | 2 | 0.1333 | 2 | 0.1333 |
| uninformative_test | 2 | 0.1333 | 3 | 0.2000 |
| fixed_belief_trace | 10 | 0.6667 | 10 | 0.6667 |
| disconnected_evidence | 1 | 0.0667 | 1 | 0.0667 |
| one_sided_confirmation | 3 | 0.2000 | 3 | 0.2000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.5333 | 21 | 1.4000 |
| evidence_handling | 14 | 0.9333 | 61 | 4.0667 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 2 | 0.1429 | 2 | 0.1429 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 4 | 0.2857 | 4 | 0.2857 |
| evidence_led_hypothesis_generation | 1 | 0.0714 | 1 | 0.0714 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.6429 | 16 | 1.1429 |
| evidence_non_uptake | 14 | 1.0000 | 231 | 16.5000 |
| unsupported_judgment | 4 | 0.2857 | 4 | 0.2857 |
| stalled_revision | 3 | 0.2143 | 4 | 0.2857 |
| contradiction_without_repair | 9 | 0.6429 | 18 | 1.2857 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 4 | 0.2857 | 10 | 0.7143 |
| fixed_belief_trace | 5 | 0.3571 | 5 | 0.3571 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.6429 | 16 | 1.1429 |
| evidence_non_uptake | 14 | 1.0000 | 231 | 16.5000 |
| unsupported_judgment | 4 | 0.2857 | 4 | 0.2857 |
| stalled_revision | 3 | 0.2143 | 4 | 0.2857 |
| contradiction_without_repair | 9 | 0.6429 | 18 | 1.2857 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 4 | 0.2857 | 10 | 0.7143 |
| fixed_belief_trace | 5 | 0.3571 | 5 | 0.3571 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 12 | 0.8571 | 34 | 2.4286 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.0455 | 1 | 0.0455 |
| hypothesis_reranking | 3 | 0.1364 | 3 | 0.1364 |
| evidence_led_hypothesis_generation | 21 | 0.9545 | 21 | 0.9545 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 12 | 0.5455 | 15 | 0.6818 |
| evidence_non_uptake | 12 | 0.5455 | 26 | 1.1818 |
| unsupported_judgment | 1 | 0.0455 | 1 | 0.0455 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 4 | 0.1818 | 4 | 0.1818 |
| premature_commitment | 5 | 0.2273 | 5 | 0.2273 |
| uninformative_test | 1 | 0.0455 | 1 | 0.0455 |
| fixed_belief_trace | 19 | 0.8636 | 19 | 0.8636 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 6 | 0.2727 | 6 | 0.2727 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 12 | 0.5455 | 15 | 0.6818 |
| evidence_non_uptake | 12 | 0.5455 | 26 | 1.1818 |
| unsupported_judgment | 1 | 0.0455 | 1 | 0.0455 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 4 | 0.1818 | 4 | 0.1818 |
| premature_commitment | 5 | 0.2273 | 5 | 0.2273 |
| uninformative_test | 1 | 0.0455 | 1 | 0.0455 |
| fixed_belief_trace | 19 | 0.8636 | 19 | 0.8636 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 6 | 0.2727 | 6 | 0.2727 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 14 | 0.6364 | 25 | 1.1364 |
| evidence_handling | 12 | 0.5455 | 28 | 1.2727 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 2 | 0.0909 | 2 | 0.0909 |
| evidence_led_hypothesis_generation | 22 | 1.0000 | 22 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 0.6818 | 18 | 0.8182 |
| evidence_non_uptake | 12 | 0.5455 | 23 | 1.0455 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.0909 | 4 | 0.1818 |
| premature_commitment | 10 | 0.4545 | 10 | 0.4545 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 20 | 0.9091 | 20 | 0.9091 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 9 | 0.4091 | 9 | 0.4091 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 0.6818 | 18 | 0.8182 |
| evidence_non_uptake | 12 | 0.5455 | 23 | 1.0455 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.0909 | 4 | 0.1818 |
| premature_commitment | 10 | 0.4545 | 10 | 0.4545 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 20 | 0.9091 | 20 | 0.9091 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 9 | 0.4091 | 9 | 0.4091 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 16 | 0.7273 | 31 | 1.4091 |
| evidence_handling | 12 | 0.5455 | 23 | 1.0455 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.0667 | 1 | 0.0667 |
| hypothesis_reranking | 9 | 0.6000 | 9 | 0.6000 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 50 | 3.3333 |
| evidence_non_uptake | 15 | 1.0000 | 60 | 4.0000 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 1 | 0.0667 | 1 | 0.0667 |
| contradiction_without_repair | 5 | 0.3333 | 5 | 0.3333 |
| premature_commitment | 11 | 0.7333 | 14 | 0.9333 |
| uninformative_test | 1 | 0.0667 | 1 | 0.0667 |
| fixed_belief_trace | 7 | 0.4667 | 7 | 0.4667 |
| disconnected_evidence | 13 | 0.8667 | 18 | 1.2000 |
| one_sided_confirmation | 11 | 0.7333 | 14 | 0.9333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 50 | 3.3333 |
| evidence_non_uptake | 15 | 1.0000 | 60 | 4.0000 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 1 | 0.0667 | 1 | 0.0667 |
| contradiction_without_repair | 5 | 0.3333 | 5 | 0.3333 |
| premature_commitment | 11 | 0.7333 | 14 | 0.9333 |
| uninformative_test | 1 | 0.0667 | 1 | 0.0667 |
| fixed_belief_trace | 7 | 0.4667 | 7 | 0.4667 |
| disconnected_evidence | 13 | 0.8667 | 18 | 1.2000 |
| one_sided_confirmation | 11 | 0.7333 | 14 | 0.9333 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 9 | 0.6000 | 9 | 0.6000 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 54 | 3.6000 |
| evidence_non_uptake | 15 | 1.0000 | 68 | 4.5333 |
| unsupported_judgment | 5 | 0.3333 | 6 | 0.4000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.0667 | 2 | 0.1333 |
| premature_commitment | 6 | 0.4000 | 12 | 0.8000 |
| uninformative_test | 3 | 0.2000 | 3 | 0.2000 |
| fixed_belief_trace | 5 | 0.3333 | 5 | 0.3333 |
| disconnected_evidence | 14 | 0.9333 | 23 | 1.5333 |
| one_sided_confirmation | 6 | 0.4000 | 11 | 0.7333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 54 | 3.6000 |
| evidence_non_uptake | 15 | 1.0000 | 68 | 4.5333 |
| unsupported_judgment | 5 | 0.3333 | 6 | 0.4000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.1333 | 4 | 0.2667 |
| premature_commitment | 6 | 0.4000 | 12 | 0.8000 |
| uninformative_test | 3 | 0.2000 | 3 | 0.2000 |
| fixed_belief_trace | 5 | 0.3333 | 5 | 0.3333 |
| disconnected_evidence | 14 | 0.9333 | 23 | 1.5333 |
| one_sided_confirmation | 6 | 0.4000 | 11 | 0.7333 |

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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 2 | 0.1333 | 2 | 0.1333 |
| hypothesis_reranking | 7 | 0.4667 | 7 | 0.4667 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 46 | 3.0667 |
| evidence_non_uptake | 15 | 1.0000 | 71 | 4.7333 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 6 | 0.4000 | 12 | 0.8000 |
| premature_commitment | 4 | 0.2667 | 4 | 0.2667 |
| uninformative_test | 2 | 0.1333 | 2 | 0.1333 |
| fixed_belief_trace | 8 | 0.5333 | 8 | 0.5333 |
| disconnected_evidence | 11 | 0.7333 | 19 | 1.2667 |
| one_sided_confirmation | 5 | 0.3333 | 8 | 0.5333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 46 | 3.0667 |
| evidence_non_uptake | 15 | 1.0000 | 71 | 4.7333 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 6 | 0.4000 | 12 | 0.8000 |
| premature_commitment | 4 | 0.2667 | 4 | 0.2667 |
| uninformative_test | 2 | 0.1333 | 2 | 0.1333 |
| fixed_belief_trace | 8 | 0.5333 | 8 | 0.5333 |
| disconnected_evidence | 11 | 0.7333 | 19 | 1.2667 |
| one_sided_confirmation | 5 | 0.3333 | 8 | 0.5333 |

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
| refutation_driven_belief_revision | 9 | 0.0402 | 9 | 0.0402 |
| fixed_hypothesis_test_tuning | 7 | 0.0312 | 7 | 0.0312 |
| explore_then_test_transition | 4 | 0.0179 | 4 | 0.0179 |
| hypothesis_reranking | 78 | 0.3482 | 78 | 0.3482 |
| evidence_led_hypothesis_generation | 127 | 0.5670 | 127 | 0.5670 |
| convergent_multi_test_evidence | 2 | 0.0089 | 2 | 0.0089 |
| precommitted_test_plan | 1 | 0.0045 | 1 | 0.0045 |
| evidence_guided_test_redesign | 2 | 0.0089 | 2 | 0.0089 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 148 | 0.6607 | 718 | 3.2054 |
| evidence_non_uptake | 197 | 0.8795 | 1289 | 5.7545 |
| unsupported_judgment | 60 | 0.2679 | 137 | 0.6116 |
| stalled_revision | 47 | 0.2098 | 61 | 0.2723 |
| contradiction_without_repair | 64 | 0.2857 | 141 | 0.6295 |
| premature_commitment | 30 | 0.1339 | 39 | 0.1741 |
| uninformative_test | 34 | 0.1518 | 63 | 0.2812 |
| fixed_belief_trace | 106 | 0.4732 | 106 | 0.4732 |
| disconnected_evidence | 54 | 0.2411 | 118 | 0.5268 |
| one_sided_confirmation | 23 | 0.1027 | 27 | 0.1205 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 148 | 0.6607 | 718 | 3.2054 |
| evidence_non_uptake | 197 | 0.8795 | 1289 | 5.7545 |
| unsupported_judgment | 60 | 0.2679 | 137 | 0.6116 |
| stalled_revision | 47 | 0.2098 | 61 | 0.2723 |
| contradiction_without_repair | 70 | 0.3125 | 175 | 0.7812 |
| premature_commitment | 30 | 0.1339 | 39 | 0.1741 |
| uninformative_test | 34 | 0.1518 | 63 | 0.2812 |
| fixed_belief_trace | 106 | 0.4732 | 106 | 0.4732 |
| disconnected_evidence | 54 | 0.2411 | 118 | 0.5268 |
| one_sided_confirmation | 23 | 0.1027 | 27 | 0.1205 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 156 | 0.6964 | 886 | 3.9554 |
| evidence_handling | 201 | 0.8973 | 1607 | 7.1741 |
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
| refutation_driven_belief_revision | 2 | 0.0088 | 2 | 0.0088 |
| fixed_hypothesis_test_tuning | 5 | 0.0220 | 5 | 0.0220 |
| explore_then_test_transition | 25 | 0.1101 | 25 | 0.1101 |
| hypothesis_reranking | 63 | 0.2775 | 63 | 0.2775 |
| evidence_led_hypothesis_generation | 156 | 0.6872 | 156 | 0.6872 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 3 | 0.0132 | 3 | 0.0132 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 140 | 0.6167 | 330 | 1.4537 |
| evidence_non_uptake | 199 | 0.8767 | 1152 | 5.0749 |
| unsupported_judgment | 26 | 0.1145 | 28 | 0.1233 |
| stalled_revision | 19 | 0.0837 | 22 | 0.0969 |
| contradiction_without_repair | 62 | 0.2731 | 140 | 0.6167 |
| premature_commitment | 46 | 0.2026 | 56 | 0.2467 |
| uninformative_test | 25 | 0.1101 | 33 | 0.1454 |
| fixed_belief_trace | 151 | 0.6652 | 151 | 0.6652 |
| disconnected_evidence | 50 | 0.2203 | 75 | 0.3304 |
| one_sided_confirmation | 48 | 0.2115 | 60 | 0.2643 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 140 | 0.6167 | 330 | 1.4537 |
| evidence_non_uptake | 199 | 0.8767 | 1152 | 5.0749 |
| unsupported_judgment | 26 | 0.1145 | 28 | 0.1233 |
| stalled_revision | 19 | 0.0837 | 22 | 0.0969 |
| contradiction_without_repair | 66 | 0.2907 | 148 | 0.6520 |
| premature_commitment | 46 | 0.2026 | 56 | 0.2467 |
| uninformative_test | 25 | 0.1101 | 33 | 0.1454 |
| fixed_belief_trace | 151 | 0.6652 | 151 | 0.6652 |
| disconnected_evidence | 50 | 0.2203 | 75 | 0.3304 |
| one_sided_confirmation | 48 | 0.2115 | 60 | 0.2643 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 153 | 0.6740 | 530 | 2.3348 |
| evidence_handling | 200 | 0.8811 | 1288 | 5.6740 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 1 | 0.0333 | 1 | 0.0333 |
| explore_then_test_transition | 1 | 0.0333 | 1 | 0.0333 |
| hypothesis_reranking | 4 | 0.1333 | 4 | 0.1333 |
| evidence_led_hypothesis_generation | 19 | 0.6333 | 19 | 0.6333 |
| convergent_multi_test_evidence | 1 | 0.0333 | 1 | 0.0333 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 17 | 0.5667 | 25 | 0.8333 |
| evidence_non_uptake | 30 | 1.0000 | 172 | 5.7333 |
| unsupported_judgment | 4 | 0.1333 | 4 | 0.1333 |
| stalled_revision | 2 | 0.0667 | 3 | 0.1000 |
| contradiction_without_repair | 7 | 0.2333 | 16 | 0.5333 |
| premature_commitment | 2 | 0.0667 | 2 | 0.0667 |
| uninformative_test | 5 | 0.1667 | 8 | 0.2667 |
| fixed_belief_trace | 24 | 0.8000 | 24 | 0.8000 |
| disconnected_evidence | 3 | 0.1000 | 3 | 0.1000 |
| one_sided_confirmation | 2 | 0.0667 | 2 | 0.0667 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 17 | 0.5667 | 25 | 0.8333 |
| evidence_non_uptake | 30 | 1.0000 | 172 | 5.7333 |
| unsupported_judgment | 4 | 0.1333 | 4 | 0.1333 |
| stalled_revision | 2 | 0.0667 | 3 | 0.1000 |
| contradiction_without_repair | 7 | 0.2333 | 16 | 0.5333 |
| premature_commitment | 2 | 0.0667 | 2 | 0.0667 |
| uninformative_test | 5 | 0.1667 | 8 | 0.2667 |
| fixed_belief_trace | 24 | 0.8000 | 24 | 0.8000 |
| disconnected_evidence | 3 | 0.1000 | 3 | 0.1000 |
| one_sided_confirmation | 2 | 0.0667 | 2 | 0.0667 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 19 | 0.6333 | 43 | 1.4333 |
| evidence_handling | 30 | 1.0000 | 187 | 6.2333 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 3 | 0.1000 | 3 | 0.1000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 0.1667 | 5 | 0.1667 |
| evidence_non_uptake | 30 | 1.0000 | 234 | 7.8000 |
| unsupported_judgment | 1 | 0.0333 | 1 | 0.0333 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 2 | 0.0667 | 2 | 0.0667 |
| uninformative_test | 1 | 0.0333 | 1 | 0.0333 |
| fixed_belief_trace | 30 | 1.0000 | 30 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 2 | 0.0667 | 2 | 0.0667 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 0.1667 | 5 | 0.1667 |
| evidence_non_uptake | 30 | 1.0000 | 234 | 7.8000 |
| unsupported_judgment | 1 | 0.0333 | 1 | 0.0333 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 2 | 0.0667 | 2 | 0.0667 |
| uninformative_test | 1 | 0.0333 | 1 | 0.0333 |
| fixed_belief_trace | 30 | 1.0000 | 30 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 2 | 0.0667 | 2 | 0.0667 |

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
| refutation_driven_belief_revision | 2 | 0.0571 | 2 | 0.0571 |
| fixed_hypothesis_test_tuning | 1 | 0.0286 | 1 | 0.0286 |
| explore_then_test_transition | 9 | 0.2571 | 9 | 0.2571 |
| hypothesis_reranking | 9 | 0.2571 | 9 | 0.2571 |
| evidence_led_hypothesis_generation | 19 | 0.5429 | 19 | 0.5429 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 31 | 0.8857 | 66 | 1.8857 |
| evidence_non_uptake | 35 | 1.0000 | 357 | 10.2000 |
| unsupported_judgment | 4 | 0.1143 | 5 | 0.1429 |
| stalled_revision | 3 | 0.0857 | 3 | 0.0857 |
| contradiction_without_repair | 15 | 0.4286 | 27 | 0.7714 |
| premature_commitment | 4 | 0.1143 | 4 | 0.1143 |
| uninformative_test | 13 | 0.3714 | 36 | 1.0286 |
| fixed_belief_trace | 24 | 0.6857 | 24 | 0.6857 |
| disconnected_evidence | 7 | 0.2000 | 10 | 0.2857 |
| one_sided_confirmation | 4 | 0.1143 | 4 | 0.1143 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 31 | 0.8857 | 66 | 1.8857 |
| evidence_non_uptake | 35 | 1.0000 | 357 | 10.2000 |
| unsupported_judgment | 4 | 0.1143 | 5 | 0.1429 |
| stalled_revision | 3 | 0.0857 | 3 | 0.0857 |
| contradiction_without_repair | 15 | 0.4286 | 32 | 0.9143 |
| premature_commitment | 4 | 0.1143 | 4 | 0.1143 |
| uninformative_test | 13 | 0.3714 | 36 | 1.0286 |
| fixed_belief_trace | 24 | 0.6857 | 24 | 0.6857 |
| disconnected_evidence | 7 | 0.2000 | 10 | 0.2857 |
| one_sided_confirmation | 4 | 0.1143 | 4 | 0.1143 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 33 | 0.9429 | 97 | 2.7714 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 2 | 0.0667 | 2 | 0.0667 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 1 | 0.0333 | 1 | 0.0333 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.1000 | 3 | 0.1000 |
| evidence_non_uptake | 29 | 0.9667 | 136 | 4.5333 |
| unsupported_judgment | 1 | 0.0333 | 1 | 0.0333 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 3 | 0.1000 | 4 | 0.1333 |
| fixed_belief_trace | 29 | 0.9667 | 29 | 0.9667 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.1000 | 3 | 0.1000 |
| evidence_non_uptake | 29 | 0.9667 | 136 | 4.5333 |
| unsupported_judgment | 1 | 0.0333 | 1 | 0.0333 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 3 | 0.1000 | 4 | 0.1333 |
| fixed_belief_trace | 29 | 0.9667 | 29 | 0.9667 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |

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
| refutation_driven_belief_revision | 6 | 0.1017 | 6 | 0.1017 |
| fixed_hypothesis_test_tuning | 2 | 0.0339 | 2 | 0.0339 |
| explore_then_test_transition | 7 | 0.1186 | 7 | 0.1186 |
| hypothesis_reranking | 30 | 0.5085 | 30 | 0.5085 |
| evidence_led_hypothesis_generation | 45 | 0.7627 | 45 | 0.7627 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 1 | 0.0169 | 1 | 0.0169 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 32 | 0.5424 | 117 | 1.9831 |
| evidence_non_uptake | 44 | 0.7458 | 204 | 3.4576 |
| unsupported_judgment | 19 | 0.3220 | 43 | 0.7288 |
| stalled_revision | 12 | 0.2034 | 14 | 0.2373 |
| contradiction_without_repair | 25 | 0.4237 | 74 | 1.2542 |
| premature_commitment | 3 | 0.0508 | 4 | 0.0678 |
| uninformative_test | 8 | 0.1356 | 8 | 0.1356 |
| fixed_belief_trace | 29 | 0.4915 | 29 | 0.4915 |
| disconnected_evidence | 9 | 0.1525 | 11 | 0.1864 |
| one_sided_confirmation | 5 | 0.0847 | 7 | 0.1186 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 32 | 0.5424 | 117 | 1.9831 |
| evidence_non_uptake | 44 | 0.7458 | 204 | 3.4576 |
| unsupported_judgment | 19 | 0.3220 | 43 | 0.7288 |
| stalled_revision | 12 | 0.2034 | 14 | 0.2373 |
| contradiction_without_repair | 29 | 0.4915 | 86 | 1.4576 |
| premature_commitment | 3 | 0.0508 | 4 | 0.0678 |
| uninformative_test | 8 | 0.1356 | 8 | 0.1356 |
| fixed_belief_trace | 29 | 0.4915 | 29 | 0.4915 |
| disconnected_evidence | 9 | 0.1525 | 11 | 0.1864 |
| one_sided_confirmation | 5 | 0.0847 | 7 | 0.1186 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 37 | 0.6271 | 198 | 3.3559 |
| evidence_handling | 47 | 0.7966 | 266 | 4.5085 |
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
| refutation_driven_belief_revision | 2 | 0.0225 | 2 | 0.0225 |
| fixed_hypothesis_test_tuning | 8 | 0.0899 | 8 | 0.0899 |
| explore_then_test_transition | 6 | 0.0674 | 6 | 0.0674 |
| hypothesis_reranking | 16 | 0.1798 | 16 | 0.1798 |
| evidence_led_hypothesis_generation | 21 | 0.2360 | 21 | 0.2360 |
| convergent_multi_test_evidence | 1 | 0.0112 | 1 | 0.0112 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 4 | 0.0449 | 4 | 0.0449 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 46 | 0.5169 | 103 | 1.1573 |
| evidence_non_uptake | 77 | 0.8652 | 605 | 6.7978 |
| unsupported_judgment | 7 | 0.0787 | 7 | 0.0787 |
| stalled_revision | 14 | 0.1573 | 15 | 0.1685 |
| contradiction_without_repair | 24 | 0.2697 | 55 | 0.6180 |
| premature_commitment | 12 | 0.1348 | 12 | 0.1348 |
| uninformative_test | 15 | 0.1685 | 24 | 0.2697 |
| fixed_belief_trace | 49 | 0.5506 | 49 | 0.5506 |
| disconnected_evidence | 3 | 0.0337 | 3 | 0.0337 |
| one_sided_confirmation | 11 | 0.1236 | 11 | 0.1236 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 46 | 0.5169 | 103 | 1.1573 |
| evidence_non_uptake | 77 | 0.8652 | 605 | 6.7978 |
| unsupported_judgment | 7 | 0.0787 | 7 | 0.0787 |
| stalled_revision | 14 | 0.1573 | 15 | 0.1685 |
| contradiction_without_repair | 27 | 0.3034 | 64 | 0.7191 |
| premature_commitment | 12 | 0.1348 | 12 | 0.1348 |
| uninformative_test | 15 | 0.1685 | 24 | 0.2697 |
| fixed_belief_trace | 49 | 0.5506 | 49 | 0.5506 |
| disconnected_evidence | 3 | 0.0337 | 3 | 0.0337 |
| one_sided_confirmation | 11 | 0.1236 | 11 | 0.1236 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 53 | 0.5955 | 169 | 1.8989 |
| evidence_handling | 78 | 0.8764 | 639 | 7.1798 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.0114 | 1 | 0.0114 |
| hypothesis_reranking | 28 | 0.3182 | 28 | 0.3182 |
| evidence_led_hypothesis_generation | 84 | 0.9545 | 84 | 0.9545 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 64 | 0.7273 | 151 | 1.7159 |
| evidence_non_uptake | 61 | 0.6932 | 147 | 1.6705 |
| unsupported_judgment | 18 | 0.2045 | 48 | 0.5455 |
| stalled_revision | 5 | 0.0568 | 5 | 0.0568 |
| contradiction_without_repair | 20 | 0.2273 | 31 | 0.3523 |
| premature_commitment | 22 | 0.2500 | 23 | 0.2614 |
| uninformative_test | 2 | 0.0227 | 2 | 0.0227 |
| fixed_belief_trace | 51 | 0.5795 | 51 | 0.5795 |
| disconnected_evidence | 4 | 0.0455 | 6 | 0.0682 |
| one_sided_confirmation | 19 | 0.2159 | 20 | 0.2273 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 64 | 0.7273 | 151 | 1.7159 |
| evidence_non_uptake | 61 | 0.6932 | 147 | 1.6705 |
| unsupported_judgment | 18 | 0.2045 | 48 | 0.5455 |
| stalled_revision | 5 | 0.0568 | 5 | 0.0568 |
| contradiction_without_repair | 22 | 0.2500 | 44 | 0.5000 |
| premature_commitment | 22 | 0.2500 | 23 | 0.2614 |
| uninformative_test | 2 | 0.0227 | 2 | 0.0227 |
| fixed_belief_trace | 51 | 0.5795 | 51 | 0.5795 |
| disconnected_evidence | 4 | 0.0455 | 6 | 0.0682 |
| one_sided_confirmation | 19 | 0.2159 | 20 | 0.2273 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 69 | 0.7841 | 202 | 2.2955 |
| evidence_handling | 62 | 0.7045 | 203 | 2.3068 |
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
| refutation_driven_belief_revision | 1 | 0.0111 | 1 | 0.0111 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 5 | 0.0556 | 5 | 0.0556 |
| hypothesis_reranking | 54 | 0.6000 | 54 | 0.6000 |
| evidence_led_hypothesis_generation | 90 | 1.0000 | 90 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 90 | 1.0000 | 578 | 6.4222 |
| evidence_non_uptake | 90 | 1.0000 | 586 | 6.5111 |
| unsupported_judgment | 32 | 0.3556 | 56 | 0.6222 |
| stalled_revision | 30 | 0.3333 | 43 | 0.4778 |
| contradiction_without_repair | 35 | 0.3889 | 78 | 0.8667 |
| premature_commitment | 31 | 0.3444 | 48 | 0.5333 |
| uninformative_test | 12 | 0.1333 | 13 | 0.1444 |
| fixed_belief_trace | 21 | 0.2333 | 21 | 0.2333 |
| disconnected_evidence | 78 | 0.8667 | 160 | 1.7778 |
| one_sided_confirmation | 28 | 0.3111 | 41 | 0.4556 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 90 | 1.0000 | 578 | 6.4222 |
| evidence_non_uptake | 90 | 1.0000 | 586 | 6.5111 |
| unsupported_judgment | 32 | 0.3556 | 56 | 0.6222 |
| stalled_revision | 30 | 0.3333 | 43 | 0.4778 |
| contradiction_without_repair | 36 | 0.4000 | 81 | 0.9000 |
| premature_commitment | 31 | 0.3444 | 48 | 0.5333 |
| uninformative_test | 12 | 0.1333 | 13 | 0.1444 |
| fixed_belief_trace | 21 | 0.2333 | 21 | 0.2333 |
| disconnected_evidence | 78 | 0.8667 | 160 | 1.7778 |
| one_sided_confirmation | 28 | 0.3111 | 41 | 0.4556 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 90 | 1.0000 | 697 | 7.7444 |
| evidence_handling | 90 | 1.0000 | 815 | 9.0556 |
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
| refutation_driven_belief_revision | 6 | 0.0237 | 6 | 0.0237 |
| fixed_hypothesis_test_tuning | 4 | 0.0158 | 4 | 0.0158 |
| explore_then_test_transition | 16 | 0.0632 | 16 | 0.0632 |
| hypothesis_reranking | 68 | 0.2688 | 68 | 0.2688 |
| evidence_led_hypothesis_generation | 150 | 0.5929 | 150 | 0.5929 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 1 | 0.0040 | 1 | 0.0040 |
| evidence_guided_test_redesign | 1 | 0.0040 | 1 | 0.0040 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 132 | 0.5217 | 380 | 1.5020 |
| evidence_non_uptake | 211 | 0.8340 | 1071 | 4.2332 |
| unsupported_judgment | 41 | 0.1621 | 81 | 0.3202 |
| stalled_revision | 22 | 0.0870 | 24 | 0.0949 |
| contradiction_without_repair | 62 | 0.2451 | 138 | 0.5455 |
| premature_commitment | 34 | 0.1344 | 41 | 0.1621 |
| uninformative_test | 22 | 0.0870 | 45 | 0.1779 |
| fixed_belief_trace | 172 | 0.6798 | 172 | 0.6798 |
| disconnected_evidence | 39 | 0.1542 | 59 | 0.2332 |
| one_sided_confirmation | 33 | 0.1304 | 40 | 0.1581 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 132 | 0.5217 | 380 | 1.5020 |
| evidence_non_uptake | 211 | 0.8340 | 1071 | 4.2332 |
| unsupported_judgment | 41 | 0.1621 | 81 | 0.3202 |
| stalled_revision | 22 | 0.0870 | 24 | 0.0949 |
| contradiction_without_repair | 66 | 0.2609 | 155 | 0.6126 |
| premature_commitment | 34 | 0.1344 | 41 | 0.1621 |
| uninformative_test | 22 | 0.0870 | 45 | 0.1779 |
| fixed_belief_trace | 172 | 0.6798 | 172 | 0.6798 |
| disconnected_evidence | 39 | 0.1542 | 59 | 0.2332 |
| one_sided_confirmation | 33 | 0.1304 | 40 | 0.1581 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 144 | 0.5692 | 558 | 2.2055 |
| evidence_handling | 215 | 0.8498 | 1256 | 4.9644 |
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
| refutation_driven_belief_revision | 3 | 0.0238 | 3 | 0.0238 |
| fixed_hypothesis_test_tuning | 5 | 0.0397 | 5 | 0.0397 |
| explore_then_test_transition | 8 | 0.0635 | 8 | 0.0635 |
| hypothesis_reranking | 41 | 0.3254 | 41 | 0.3254 |
| evidence_led_hypothesis_generation | 93 | 0.7381 | 93 | 0.7381 |
| convergent_multi_test_evidence | 1 | 0.0079 | 1 | 0.0079 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 3 | 0.0238 | 3 | 0.0238 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 99 | 0.7857 | 385 | 3.0556 |
| evidence_non_uptake | 113 | 0.8968 | 695 | 5.5159 |
| unsupported_judgment | 30 | 0.2381 | 63 | 0.5000 |
| stalled_revision | 21 | 0.1667 | 27 | 0.2143 |
| contradiction_without_repair | 32 | 0.2540 | 67 | 0.5317 |
| premature_commitment | 29 | 0.2302 | 37 | 0.2937 |
| uninformative_test | 18 | 0.1429 | 22 | 0.1746 |
| fixed_belief_trace | 57 | 0.4524 | 57 | 0.4524 |
| disconnected_evidence | 38 | 0.3016 | 74 | 0.5873 |
| one_sided_confirmation | 26 | 0.2063 | 31 | 0.2460 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 99 | 0.7857 | 385 | 3.0556 |
| evidence_non_uptake | 113 | 0.8968 | 695 | 5.5159 |
| unsupported_judgment | 30 | 0.2381 | 63 | 0.5000 |
| stalled_revision | 21 | 0.1667 | 27 | 0.2143 |
| contradiction_without_repair | 37 | 0.2937 | 91 | 0.7222 |
| premature_commitment | 29 | 0.2302 | 37 | 0.2937 |
| uninformative_test | 18 | 0.1429 | 22 | 0.1746 |
| fixed_belief_trace | 57 | 0.4524 | 57 | 0.4524 |
| disconnected_evidence | 38 | 0.3016 | 74 | 0.5873 |
| one_sided_confirmation | 26 | 0.2063 | 31 | 0.2460 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 103 | 0.8175 | 483 | 3.8333 |
| evidence_handling | 114 | 0.9048 | 854 | 6.7778 |
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
| refutation_driven_belief_revision | 2 | 0.0308 | 2 | 0.0308 |
| fixed_hypothesis_test_tuning | 3 | 0.0462 | 3 | 0.0462 |
| explore_then_test_transition | 5 | 0.0769 | 5 | 0.0769 |
| hypothesis_reranking | 31 | 0.4769 | 31 | 0.4769 |
| evidence_led_hypothesis_generation | 35 | 0.5385 | 35 | 0.5385 |
| convergent_multi_test_evidence | 1 | 0.0154 | 1 | 0.0154 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 1 | 0.0154 | 1 | 0.0154 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 53 | 0.8154 | 276 | 4.2462 |
| evidence_non_uptake | 65 | 1.0000 | 629 | 9.6769 |
| unsupported_judgment | 15 | 0.2308 | 21 | 0.3231 |
| stalled_revision | 23 | 0.3538 | 32 | 0.4923 |
| contradiction_without_repair | 31 | 0.4769 | 74 | 1.1385 |
| premature_commitment | 11 | 0.1692 | 15 | 0.2308 |
| uninformative_test | 16 | 0.2462 | 23 | 0.3538 |
| fixed_belief_trace | 21 | 0.3231 | 21 | 0.3231 |
| disconnected_evidence | 27 | 0.4154 | 60 | 0.9231 |
| one_sided_confirmation | 10 | 0.1538 | 14 | 0.2154 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 53 | 0.8154 | 276 | 4.2462 |
| evidence_non_uptake | 65 | 1.0000 | 629 | 9.6769 |
| unsupported_judgment | 15 | 0.2308 | 21 | 0.3231 |
| stalled_revision | 23 | 0.3538 | 32 | 0.4923 |
| contradiction_without_repair | 32 | 0.4923 | 75 | 1.1538 |
| premature_commitment | 11 | 0.1692 | 15 | 0.2308 |
| uninformative_test | 16 | 0.2462 | 23 | 0.3538 |
| fixed_belief_trace | 21 | 0.3231 | 21 | 0.3231 |
| disconnected_evidence | 27 | 0.4154 | 60 | 0.9231 |
| one_sided_confirmation | 10 | 0.1538 | 14 | 0.2154 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 57 | 0.8769 | 364 | 5.6000 |
| evidence_handling | 65 | 1.0000 | 733 | 11.2769 |
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
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 1 | 0.1429 | 1 | 0.1429 |
| evidence_led_hypothesis_generation | 5 | 0.7143 | 5 | 0.7143 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 4 | 0.5714 | 7 | 1.0000 |
| evidence_non_uptake | 7 | 1.0000 | 46 | 6.5714 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.1429 | 2 | 0.2857 |
| premature_commitment | 2 | 0.2857 | 2 | 0.2857 |
| uninformative_test | 3 | 0.4286 | 6 | 0.8571 |
| fixed_belief_trace | 7 | 1.0000 | 7 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 2 | 0.2857 | 2 | 0.2857 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 4 | 0.5714 | 7 | 1.0000 |
| evidence_non_uptake | 7 | 1.0000 | 46 | 6.5714 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.1429 | 2 | 0.2857 |
| premature_commitment | 2 | 0.2857 | 2 | 0.2857 |
| uninformative_test | 3 | 0.4286 | 6 | 0.8571 |
| fixed_belief_trace | 7 | 1.0000 | 7 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 2 | 0.2857 | 2 | 0.2857 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.7143 | 11 | 1.5714 |
| evidence_handling | 7 | 1.0000 | 52 | 7.4286 |
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
| refutation_driven_belief_revision | 11 | 0.0244 | 11 | 0.0244 |
| fixed_hypothesis_test_tuning | 12 | 0.0266 | 12 | 0.0266 |
| explore_then_test_transition | 29 | 0.0643 | 29 | 0.0643 |
| hypothesis_reranking | 141 | 0.3126 | 141 | 0.3126 |
| evidence_led_hypothesis_generation | 283 | 0.6275 | 283 | 0.6275 |
| convergent_multi_test_evidence | 2 | 0.0044 | 2 | 0.0044 |
| precommitted_test_plan | 1 | 0.0022 | 1 | 0.0022 |
| evidence_guided_test_redesign | 5 | 0.0111 | 5 | 0.0111 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 288 | 0.6386 | 1048 | 2.3237 |
| evidence_non_uptake | 396 | 0.8780 | 2441 | 5.4124 |
| unsupported_judgment | 86 | 0.1907 | 165 | 0.3659 |
| stalled_revision | 66 | 0.1463 | 83 | 0.1840 |
| contradiction_without_repair | 126 | 0.2794 | 281 | 0.6231 |
| premature_commitment | 76 | 0.1685 | 95 | 0.2106 |
| uninformative_test | 59 | 0.1308 | 96 | 0.2129 |
| fixed_belief_trace | 257 | 0.5698 | 257 | 0.5698 |
| disconnected_evidence | 104 | 0.2306 | 193 | 0.4279 |
| one_sided_confirmation | 71 | 0.1574 | 87 | 0.1929 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 288 | 0.6386 | 1048 | 2.3237 |
| evidence_non_uptake | 396 | 0.8780 | 2441 | 5.4124 |
| unsupported_judgment | 86 | 0.1907 | 165 | 0.3659 |
| stalled_revision | 66 | 0.1463 | 83 | 0.1840 |
| contradiction_without_repair | 136 | 0.3016 | 323 | 0.7162 |
| premature_commitment | 76 | 0.1685 | 95 | 0.2106 |
| uninformative_test | 59 | 0.1308 | 96 | 0.2129 |
| fixed_belief_trace | 257 | 0.5698 | 257 | 0.5698 |
| disconnected_evidence | 104 | 0.2306 | 193 | 0.4279 |
| one_sided_confirmation | 71 | 0.1574 | 87 | 0.1929 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 309 | 0.6851 | 1416 | 3.1397 |
| evidence_handling | 401 | 0.8891 | 2895 | 6.4191 |
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
