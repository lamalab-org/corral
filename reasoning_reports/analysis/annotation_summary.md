# Reasoning annotation analysis

## By model + env + level

### claude_sonnet_45/afm/level_1

- Traces: 5

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 3 | 0.6000 | 3 | 0.6000 |
| explore_then_test_transition | 3 | 0.6000 | 3 | 0.6000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 5 | 1.0000 | 5 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 2 | 0.4000 | 2 | 0.4000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.4000 | 2 | 0.4000 |
| evidence_non_uptake | 5 | 1.0000 | 35 | 7.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.4000 | 4 | 0.8000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 1 | 0.2000 | 1 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.4000 | 2 | 0.4000 |
| evidence_non_uptake | 5 | 1.0000 | 35 | 7.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.4000 | 4 | 0.8000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 1 | 0.2000 | 1 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.4000 | 2 | 0.4000 |
| evidence_handling | 5 | 1.0000 | 40 | 8.0000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.4000 | 2 | 0.4000 |
| evidence_handling | 5 | 1.0000 | 40 | 8.0000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 5 | 1.0000 |
| evidence_handling | 3 | 0.6000 | 3 | 0.6000 |
| experimental_strategy | 4 | 0.8000 | 5 | 1.0000 |

### claude_sonnet_45/afm/level_2

- Traces: 5

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 4 | 0.8000 | 4 | 0.8000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.2000 | 1 | 0.2000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 1 | 0.2000 | 1 | 0.2000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 1 | 0.2000 | 1 | 0.2000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 5 | 1.0000 |
| evidence_non_uptake | 5 | 1.0000 | 37 | 7.4000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.4000 | 2 | 0.4000 |
| fixed_belief_trace | 1 | 0.2000 | 1 | 0.2000 |
| disconnected_evidence | 1 | 0.2000 | 1 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 5 | 1.0000 |
| evidence_non_uptake | 5 | 1.0000 | 37 | 7.4000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.4000 | 2 | 0.4000 |
| fixed_belief_trace | 1 | 0.2000 | 1 | 0.2000 |
| disconnected_evidence | 1 | 0.2000 | 1 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 5 | 1.0000 |
| evidence_handling | 5 | 1.0000 | 40 | 8.0000 |
| experimental_strategy | 1 | 0.2000 | 1 | 0.2000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 5 | 1.0000 |
| evidence_handling | 5 | 1.0000 | 40 | 8.0000 |
| experimental_strategy | 1 | 0.2000 | 1 | 0.2000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 5 | 1.0000 |
| evidence_handling | 1 | 0.2000 | 1 | 0.2000 |
| experimental_strategy | 1 | 0.2000 | 1 | 0.2000 |

### claude_sonnet_45/afm/level_3

- Traces: 2

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 2 | 1.0000 | 2 | 1.0000 |
| hypothesis_reranking | 1 | 0.5000 | 1 | 0.5000 |
| evidence_led_hypothesis_generation | 2 | 1.0000 | 2 | 1.0000 |
| convergent_multi_test_evidence | 1 | 0.5000 | 1 | 0.5000 |
| evidence_guided_test_redesign | 2 | 1.0000 | 2 | 1.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 1.0000 | 3 | 1.5000 |
| evidence_non_uptake | 2 | 1.0000 | 25 | 12.5000 |
| unsupported_judgment | 1 | 0.5000 | 1 | 0.5000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 2 | 1.0000 | 2 | 1.0000 |
| disconnected_evidence | 1 | 0.5000 | 2 | 1.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 1.0000 | 3 | 1.5000 |
| evidence_non_uptake | 2 | 1.0000 | 25 | 12.5000 |
| unsupported_judgment | 1 | 0.5000 | 1 | 0.5000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 2 | 1.0000 | 2 | 1.0000 |
| disconnected_evidence | 1 | 0.5000 | 2 | 1.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 1.0000 | 3 | 1.5000 |
| evidence_handling | 2 | 1.0000 | 28 | 14.0000 |
| experimental_strategy | 2 | 1.0000 | 2 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 1.0000 | 3 | 1.5000 |
| evidence_handling | 2 | 1.0000 | 28 | 14.0000 |
| experimental_strategy | 2 | 1.0000 | 2 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 1.0000 | 3 | 1.5000 |
| evidence_handling | 2 | 1.0000 | 3 | 1.5000 |
| experimental_strategy | 2 | 1.0000 | 2 | 1.0000 |

### claude_sonnet_45/afm/level_4

- Traces: 4

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 3 | 0.7500 | 3 | 0.7500 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 3 | 0.7500 | 3 | 0.7500 |
| convergent_multi_test_evidence | 1 | 0.2500 | 1 | 0.2500 |
| evidence_guided_test_redesign | 2 | 0.5000 | 2 | 0.5000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.7500 | 4 | 1.0000 |
| evidence_non_uptake | 4 | 1.0000 | 37 | 9.2500 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2500 | 1 | 0.2500 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 3 | 0.7500 | 16 | 4.0000 |
| fixed_belief_trace | 4 | 1.0000 | 4 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.7500 | 4 | 1.0000 |
| evidence_non_uptake | 4 | 1.0000 | 37 | 9.2500 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2500 | 1 | 0.2500 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 3 | 0.7500 | 16 | 4.0000 |
| fixed_belief_trace | 4 | 1.0000 | 4 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.7500 | 5 | 1.2500 |
| evidence_handling | 4 | 1.0000 | 53 | 13.2500 |
| experimental_strategy | 4 | 1.0000 | 4 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.7500 | 5 | 1.2500 |
| evidence_handling | 4 | 1.0000 | 53 | 13.2500 |
| experimental_strategy | 4 | 1.0000 | 4 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.7500 | 3 | 0.7500 |
| evidence_handling | 4 | 1.0000 | 4 | 1.0000 |
| experimental_strategy | 2 | 0.5000 | 2 | 0.5000 |

### claude_sonnet_45/catalyst/level_1

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 15 | 1.0000 | 95 | 6.3333 |
| unsupported_judgment | 3 | 0.2000 | 3 | 0.2000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 3 | 0.2000 | 3 | 0.2000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 15 | 1.0000 | 95 | 6.3333 |
| unsupported_judgment | 3 | 0.2000 | 3 | 0.2000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 3 | 0.2000 | 3 | 0.2000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 15 | 1.0000 | 101 | 6.7333 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 15 | 1.0000 | 101 | 6.7333 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### claude_sonnet_45/md/level_1

- Traces: 10

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 2 | 0.2000 | 2 | 0.2000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 4 | 0.4000 | 4 | 0.4000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 5 | 0.5000 | 5 | 0.5000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 1 | 0.1000 | 1 | 0.1000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 8 | 0.8000 | 13 | 1.3000 |
| evidence_non_uptake | 10 | 1.0000 | 53 | 5.3000 |
| unsupported_judgment | 2 | 0.2000 | 3 | 0.3000 |
| stalled_revision | 2 | 0.2000 | 3 | 0.3000 |
| contradiction_without_repair | 4 | 0.4000 | 11 | 1.1000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 8 | 0.8000 | 28 | 2.8000 |
| fixed_belief_trace | 7 | 0.7000 | 7 | 0.7000 |
| disconnected_evidence | 2 | 0.2000 | 6 | 0.6000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 3 | 0.3000 | 3 | 0.3000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 8 | 0.8000 | 13 | 1.3000 |
| evidence_non_uptake | 10 | 1.0000 | 53 | 5.3000 |
| unsupported_judgment | 2 | 0.2000 | 3 | 0.3000 |
| stalled_revision | 2 | 0.2000 | 3 | 0.3000 |
| contradiction_without_repair | 4 | 0.4000 | 12 | 1.2000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 8 | 0.8000 | 28 | 2.8000 |
| fixed_belief_trace | 7 | 0.7000 | 7 | 0.7000 |
| disconnected_evidence | 2 | 0.2000 | 6 | 0.6000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 1 | 0.1000 | 1 | 0.1000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.8000 | 24 | 2.4000 |
| evidence_handling | 10 | 1.0000 | 90 | 9.0000 |
| experimental_strategy | 9 | 0.9000 | 13 | 1.3000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.8000 | 25 | 2.5000 |
| evidence_handling | 10 | 1.0000 | 90 | 9.0000 |
| experimental_strategy | 10 | 1.0000 | 11 | 1.1000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 6 | 0.6000 | 7 | 0.7000 |
| evidence_handling | 4 | 0.4000 | 4 | 0.4000 |
| experimental_strategy | 1 | 0.1000 | 1 | 0.1000 |

### claude_sonnet_45/md/level_2

- Traces: 10

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 2 | 0.2000 | 2 | 0.2000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 5 | 0.5000 | 5 | 0.5000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 8 | 0.8000 | 8 | 0.8000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 2 | 0.2000 | 2 | 0.2000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.9000 | 15 | 1.5000 |
| evidence_non_uptake | 9 | 0.9000 | 66 | 6.6000 |
| unsupported_judgment | 5 | 0.5000 | 5 | 0.5000 |
| stalled_revision | 2 | 0.2000 | 2 | 0.2000 |
| contradiction_without_repair | 1 | 0.1000 | 1 | 0.1000 |
| premature_commitment | 2 | 0.2000 | 2 | 0.2000 |
| uninformative_test | 7 | 0.7000 | 26 | 2.6000 |
| fixed_belief_trace | 7 | 0.7000 | 7 | 0.7000 |
| disconnected_evidence | 6 | 0.6000 | 16 | 1.6000 |
| one_sided_confirmation | 1 | 0.1000 | 1 | 0.1000 |
| precommitted_test_plan | 1 | 0.1000 | 1 | 0.1000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.9000 | 15 | 1.5000 |
| evidence_non_uptake | 9 | 0.9000 | 66 | 6.6000 |
| unsupported_judgment | 5 | 0.5000 | 5 | 0.5000 |
| stalled_revision | 2 | 0.2000 | 2 | 0.2000 |
| contradiction_without_repair | 1 | 0.1000 | 1 | 0.1000 |
| premature_commitment | 2 | 0.2000 | 2 | 0.2000 |
| uninformative_test | 7 | 0.7000 | 26 | 2.6000 |
| fixed_belief_trace | 7 | 0.7000 | 7 | 0.7000 |
| disconnected_evidence | 6 | 0.6000 | 16 | 1.6000 |
| one_sided_confirmation | 1 | 0.1000 | 1 | 0.1000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.9000 | 17 | 1.7000 |
| evidence_handling | 10 | 1.0000 | 113 | 11.3000 |
| experimental_strategy | 9 | 0.9000 | 12 | 1.2000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.9000 | 17 | 1.7000 |
| evidence_handling | 10 | 1.0000 | 113 | 11.3000 |
| experimental_strategy | 9 | 0.9000 | 11 | 1.1000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.8000 | 10 | 1.0000 |
| evidence_handling | 5 | 0.5000 | 5 | 0.5000 |
| experimental_strategy | 2 | 0.2000 | 2 | 0.2000 |

### claude_sonnet_45/ml/level_1

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 1 | 0.0667 | 1 | 0.0667 |
| explore_then_test_transition | 1 | 0.0667 | 1 | 0.0667 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 1 | 0.0667 | 1 | 0.0667 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 2 | 0.1333 | 2 | 0.1333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 15 | 1.0000 | 56 | 3.7333 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.1333 | 6 | 0.4000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 2 | 0.1333 | 3 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 15 | 1.0000 | 56 | 3.7333 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.1333 | 6 | 0.4000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 2 | 0.1333 | 3 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 15 | 1.0000 | 65 | 4.3333 |
| experimental_strategy | 15 | 1.0000 | 16 | 1.0667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 15 | 1.0000 | 65 | 4.3333 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 0.0667 | 1 | 0.0667 |
| evidence_handling | 1 | 0.0667 | 1 | 0.0667 |
| experimental_strategy | 2 | 0.1333 | 3 | 0.2000 |

### claude_sonnet_45/resistor/level_1

- Traces: 30

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 12 | 0.4000 | 12 | 0.4000 |
| fixed_hypothesis_test_tuning | 5 | 0.1667 | 5 | 0.1667 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 5 | 0.1667 | 5 | 0.1667 |
| evidence_led_hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| convergent_multi_test_evidence | 2 | 0.0667 | 2 | 0.0667 |
| evidence_guided_test_redesign | 11 | 0.3667 | 11 | 0.3667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 8 | 0.2667 | 28 | 0.9333 |
| evidence_non_uptake | 7 | 0.2333 | 16 | 0.5333 |
| unsupported_judgment | 5 | 0.1667 | 6 | 0.2000 |
| stalled_revision | 5 | 0.1667 | 13 | 0.4333 |
| contradiction_without_repair | 10 | 0.3333 | 18 | 0.6000 |
| premature_commitment | 1 | 0.0333 | 1 | 0.0333 |
| uninformative_test | 5 | 0.1667 | 9 | 0.3000 |
| fixed_belief_trace | 18 | 0.6000 | 18 | 0.6000 |
| disconnected_evidence | 5 | 0.1667 | 10 | 0.3333 |
| one_sided_confirmation | 1 | 0.0333 | 1 | 0.0333 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 8 | 0.2667 | 28 | 0.9333 |
| evidence_non_uptake | 7 | 0.2333 | 16 | 0.5333 |
| unsupported_judgment | 5 | 0.1667 | 6 | 0.2000 |
| stalled_revision | 5 | 0.1667 | 13 | 0.4333 |
| contradiction_without_repair | 10 | 0.3333 | 26 | 0.8667 |
| premature_commitment | 1 | 0.0333 | 1 | 0.0333 |
| uninformative_test | 5 | 0.1667 | 9 | 0.3000 |
| fixed_belief_trace | 18 | 0.6000 | 18 | 0.6000 |
| disconnected_evidence | 5 | 0.1667 | 10 | 0.3333 |
| one_sided_confirmation | 1 | 0.0333 | 1 | 0.0333 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.4333 | 47 | 1.5667 |
| evidence_handling | 9 | 0.3000 | 41 | 1.3667 |
| experimental_strategy | 23 | 0.7667 | 32 | 1.0667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.4333 | 55 | 1.8333 |
| evidence_handling | 9 | 0.3000 | 41 | 1.3667 |
| experimental_strategy | 23 | 0.7667 | 32 | 1.0667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.4333 | 17 | 0.5667 |
| evidence_handling | 2 | 0.0667 | 2 | 0.0667 |
| experimental_strategy | 12 | 0.4000 | 16 | 0.5333 |

### claude_sonnet_45/retrosynthesis/level_1

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 11 | 0.7333 | 11 | 0.7333 |
| explore_then_test_transition | 13 | 0.8667 | 13 | 0.8667 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 13 | 0.8667 | 13 | 0.8667 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 7 | 0.4667 | 7 | 0.4667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.1333 | 2 | 0.1333 |
| evidence_non_uptake | 6 | 0.4000 | 16 | 1.0667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.0667 | 2 | 0.1333 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 1 | 0.0667 | 2 | 0.1333 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 2 | 0.1333 | 8 | 0.5333 |
| one_sided_confirmation | 1 | 0.0667 | 1 | 0.0667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.1333 | 2 | 0.1333 |
| evidence_non_uptake | 6 | 0.4000 | 16 | 1.0667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.0667 | 2 | 0.1333 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 1 | 0.0667 | 2 | 0.1333 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 2 | 0.1333 | 8 | 0.5333 |
| one_sided_confirmation | 1 | 0.0667 | 1 | 0.0667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.1333 | 5 | 0.3333 |
| evidence_handling | 6 | 0.4000 | 26 | 1.7333 |
| experimental_strategy | 15 | 1.0000 | 16 | 1.0667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 2 | 0.1333 | 5 | 0.3333 |
| evidence_handling | 6 | 0.4000 | 26 | 1.7333 |
| experimental_strategy | 15 | 1.0000 | 16 | 1.0667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.8667 | 13 | 0.8667 |
| evidence_handling | 13 | 0.8667 | 13 | 0.8667 |
| experimental_strategy | 13 | 0.8667 | 18 | 1.2000 |

### claude_sonnet_45/retrosynthesis/level_2

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 4 | 0.2667 | 4 | 0.2667 |
| fixed_hypothesis_test_tuning | 4 | 0.2667 | 4 | 0.2667 |
| explore_then_test_transition | 4 | 0.2667 | 4 | 0.2667 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 4 | 0.2667 | 4 | 0.2667 |
| convergent_multi_test_evidence | 5 | 0.3333 | 5 | 0.3333 |
| evidence_guided_test_redesign | 8 | 0.5333 | 8 | 0.5333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 6 | 0.4000 | 9 | 0.6000 |
| evidence_non_uptake | 14 | 0.9333 | 109 | 7.2667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.1333 | 2 | 0.1333 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 6 | 0.4000 | 19 | 1.2667 |
| fixed_belief_trace | 11 | 0.7333 | 11 | 0.7333 |
| disconnected_evidence | 9 | 0.6000 | 15 | 1.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 6 | 0.4000 | 9 | 0.6000 |
| evidence_non_uptake | 14 | 0.9333 | 109 | 7.2667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 3 | 0.2000 | 5 | 0.3333 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 6 | 0.4000 | 19 | 1.2667 |
| fixed_belief_trace | 11 | 0.7333 | 11 | 0.7333 |
| disconnected_evidence | 9 | 0.6000 | 15 | 1.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.5333 | 11 | 0.7333 |
| evidence_handling | 14 | 0.9333 | 143 | 9.5333 |
| experimental_strategy | 11 | 0.7333 | 11 | 0.7333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.6000 | 14 | 0.9333 |
| evidence_handling | 14 | 0.9333 | 143 | 9.5333 |
| experimental_strategy | 11 | 0.7333 | 11 | 0.7333 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 6 | 0.4000 | 8 | 0.5333 |
| evidence_handling | 8 | 0.5333 | 9 | 0.6000 |
| experimental_strategy | 8 | 0.5333 | 12 | 0.8000 |

### claude_sonnet_45/retrosynthesis/level_3

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 1 | 0.0667 | 1 | 0.0667 |
| fixed_hypothesis_test_tuning | 1 | 0.0667 | 1 | 0.0667 |
| explore_then_test_transition | 13 | 0.8667 | 13 | 0.8667 |
| hypothesis_reranking | 2 | 0.1333 | 2 | 0.1333 |
| evidence_led_hypothesis_generation | 13 | 0.8667 | 13 | 0.8667 |
| convergent_multi_test_evidence | 1 | 0.0667 | 1 | 0.0667 |
| evidence_guided_test_redesign | 6 | 0.4000 | 6 | 0.4000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 6 | 0.4000 | 12 | 0.8000 |
| evidence_non_uptake | 15 | 1.0000 | 109 | 7.2667 |
| unsupported_judgment | 1 | 0.0667 | 2 | 0.1333 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 3 | 0.2000 | 3 | 0.2000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 8 | 0.5333 | 12 | 0.8000 |
| fixed_belief_trace | 14 | 0.9333 | 14 | 0.9333 |
| disconnected_evidence | 6 | 0.4000 | 7 | 0.4667 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 6 | 0.4000 | 12 | 0.8000 |
| evidence_non_uptake | 15 | 1.0000 | 109 | 7.2667 |
| unsupported_judgment | 1 | 0.0667 | 2 | 0.1333 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 3 | 0.2000 | 3 | 0.2000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 8 | 0.5333 | 12 | 0.8000 |
| fixed_belief_trace | 14 | 0.9333 | 14 | 0.9333 |
| disconnected_evidence | 6 | 0.4000 | 7 | 0.4667 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 7 | 0.4667 | 15 | 1.0000 |
| evidence_handling | 15 | 1.0000 | 130 | 8.6667 |
| experimental_strategy | 14 | 0.9333 | 14 | 0.9333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 7 | 0.4667 | 15 | 1.0000 |
| evidence_handling | 15 | 1.0000 | 130 | 8.6667 |
| experimental_strategy | 14 | 0.9333 | 15 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 14 | 0.9333 | 16 | 1.0667 |
| evidence_handling | 13 | 0.8667 | 14 | 0.9333 |
| experimental_strategy | 7 | 0.4667 | 7 | 0.4667 |

### claude_sonnet_45/spectra/level_1

- Traces: 23

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 15 | 0.6522 | 15 | 0.6522 |
| fixed_hypothesis_test_tuning | 4 | 0.1739 | 4 | 0.1739 |
| explore_then_test_transition | 22 | 0.9565 | 22 | 0.9565 |
| hypothesis_reranking | 7 | 0.3043 | 7 | 0.3043 |
| evidence_led_hypothesis_generation | 23 | 1.0000 | 23 | 1.0000 |
| convergent_multi_test_evidence | 10 | 0.4348 | 10 | 0.4348 |
| evidence_guided_test_redesign | 11 | 0.4783 | 11 | 0.4783 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 10 | 0.4348 | 24 | 1.0435 |
| evidence_non_uptake | 17 | 0.7391 | 48 | 2.0870 |
| unsupported_judgment | 4 | 0.1739 | 6 | 0.2609 |
| stalled_revision | 3 | 0.1304 | 3 | 0.1304 |
| contradiction_without_repair | 11 | 0.4783 | 24 | 1.0435 |
| premature_commitment | 1 | 0.0435 | 1 | 0.0435 |
| uninformative_test | 8 | 0.3478 | 14 | 0.6087 |
| fixed_belief_trace | 7 | 0.3043 | 7 | 0.3043 |
| disconnected_evidence | 3 | 0.1304 | 3 | 0.1304 |
| one_sided_confirmation | 2 | 0.0870 | 2 | 0.0870 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 10 | 0.4348 | 24 | 1.0435 |
| evidence_non_uptake | 17 | 0.7391 | 48 | 2.0870 |
| unsupported_judgment | 4 | 0.1739 | 6 | 0.2609 |
| stalled_revision | 3 | 0.1304 | 3 | 0.1304 |
| contradiction_without_repair | 11 | 0.4783 | 27 | 1.1739 |
| premature_commitment | 1 | 0.0435 | 1 | 0.0435 |
| uninformative_test | 8 | 0.3478 | 14 | 0.6087 |
| fixed_belief_trace | 7 | 0.3043 | 7 | 0.3043 |
| disconnected_evidence | 3 | 0.1304 | 3 | 0.1304 |
| one_sided_confirmation | 2 | 0.0870 | 2 | 0.0870 |
| precommitted_test_plan | 1 | 0.0435 | 1 | 0.0435 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 0.6522 | 50 | 2.1739 |
| evidence_handling | 18 | 0.7826 | 71 | 3.0870 |
| experimental_strategy | 10 | 0.4348 | 11 | 0.4783 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 0.6522 | 53 | 2.3043 |
| evidence_handling | 18 | 0.7826 | 71 | 3.0870 |
| experimental_strategy | 11 | 0.4783 | 12 | 0.5217 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 23 | 1.0000 | 45 | 1.9565 |
| evidence_handling | 22 | 0.9565 | 32 | 1.3913 |
| experimental_strategy | 15 | 0.6522 | 15 | 0.6522 |

### claude_sonnet_45/spectra/level_2

- Traces: 23

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 17 | 0.7391 | 17 | 0.7391 |
| fixed_hypothesis_test_tuning | 2 | 0.0870 | 2 | 0.0870 |
| explore_then_test_transition | 23 | 1.0000 | 23 | 1.0000 |
| hypothesis_reranking | 11 | 0.4783 | 11 | 0.4783 |
| evidence_led_hypothesis_generation | 23 | 1.0000 | 23 | 1.0000 |
| convergent_multi_test_evidence | 8 | 0.3478 | 8 | 0.3478 |
| evidence_guided_test_redesign | 15 | 0.6522 | 15 | 0.6522 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.3913 | 17 | 0.7391 |
| evidence_non_uptake | 19 | 0.8261 | 46 | 2.0000 |
| unsupported_judgment | 5 | 0.2174 | 6 | 0.2609 |
| stalled_revision | 3 | 0.1304 | 4 | 0.1739 |
| contradiction_without_repair | 8 | 0.3478 | 10 | 0.4348 |
| premature_commitment | 1 | 0.0435 | 1 | 0.0435 |
| uninformative_test | 8 | 0.3478 | 10 | 0.4348 |
| fixed_belief_trace | 6 | 0.2609 | 6 | 0.2609 |
| disconnected_evidence | 2 | 0.0870 | 2 | 0.0870 |
| one_sided_confirmation | 2 | 0.0870 | 2 | 0.0870 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.3913 | 17 | 0.7391 |
| evidence_non_uptake | 19 | 0.8261 | 46 | 2.0000 |
| unsupported_judgment | 5 | 0.2174 | 6 | 0.2609 |
| stalled_revision | 3 | 0.1304 | 4 | 0.1739 |
| contradiction_without_repair | 9 | 0.3913 | 17 | 0.7391 |
| premature_commitment | 1 | 0.0435 | 1 | 0.0435 |
| uninformative_test | 8 | 0.3478 | 10 | 0.4348 |
| fixed_belief_trace | 6 | 0.2609 | 6 | 0.2609 |
| disconnected_evidence | 2 | 0.0870 | 2 | 0.0870 |
| one_sided_confirmation | 2 | 0.0870 | 2 | 0.0870 |
| precommitted_test_plan | 1 | 0.0435 | 1 | 0.0435 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.5652 | 29 | 1.2609 |
| evidence_handling | 19 | 0.8261 | 64 | 2.7826 |
| experimental_strategy | 9 | 0.3913 | 11 | 0.4783 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 14 | 0.6087 | 36 | 1.5652 |
| evidence_handling | 19 | 0.8261 | 64 | 2.7826 |
| experimental_strategy | 10 | 0.4348 | 12 | 0.5217 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 23 | 1.0000 | 51 | 2.2174 |
| evidence_handling | 23 | 1.0000 | 31 | 1.3478 |
| experimental_strategy | 16 | 0.6957 | 17 | 0.7391 |

### claude_sonnet_45/wetlab/level_1

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 13 | 0.8667 | 13 | 0.8667 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 14 | 0.9333 | 14 | 0.9333 |
| hypothesis_reranking | 4 | 0.2667 | 4 | 0.2667 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 14 | 0.9333 | 14 | 0.9333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 14 | 0.9333 | 48 | 3.2000 |
| evidence_non_uptake | 15 | 1.0000 | 77 | 5.1333 |
| unsupported_judgment | 4 | 0.2667 | 6 | 0.4000 |
| stalled_revision | 10 | 0.6667 | 13 | 0.8667 |
| contradiction_without_repair | 4 | 0.2667 | 7 | 0.4667 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 10 | 0.6667 | 24 | 1.6000 |
| fixed_belief_trace | 1 | 0.0667 | 1 | 0.0667 |
| disconnected_evidence | 7 | 0.4667 | 15 | 1.0000 |
| one_sided_confirmation | 1 | 0.0667 | 1 | 0.0667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 14 | 0.9333 | 48 | 3.2000 |
| evidence_non_uptake | 15 | 1.0000 | 77 | 5.1333 |
| unsupported_judgment | 4 | 0.2667 | 6 | 0.4000 |
| stalled_revision | 10 | 0.6667 | 13 | 0.8667 |
| contradiction_without_repair | 5 | 0.3333 | 8 | 0.5333 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 10 | 0.6667 | 24 | 1.6000 |
| fixed_belief_trace | 1 | 0.0667 | 1 | 0.0667 |
| disconnected_evidence | 7 | 0.4667 | 15 | 1.0000 |
| one_sided_confirmation | 1 | 0.0667 | 1 | 0.0667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 14 | 0.9333 | 56 | 3.7333 |
| evidence_handling | 15 | 1.0000 | 122 | 8.1333 |
| experimental_strategy | 11 | 0.7333 | 15 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 14 | 0.9333 | 57 | 3.8000 |
| evidence_handling | 15 | 1.0000 | 122 | 8.1333 |
| experimental_strategy | 11 | 0.7333 | 15 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 32 | 2.1333 |
| evidence_handling | 14 | 0.9333 | 14 | 0.9333 |
| experimental_strategy | 14 | 0.9333 | 14 | 0.9333 |

### claude_sonnet_45/wetlab/level_2

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 13 | 0.8667 | 13 | 0.8667 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 13 | 0.8667 | 13 | 0.8667 |
| hypothesis_reranking | 5 | 0.3333 | 5 | 0.3333 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 2 | 0.1333 | 2 | 0.1333 |
| evidence_guided_test_redesign | 12 | 0.8000 | 12 | 0.8000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 79 | 5.2667 |
| evidence_non_uptake | 15 | 1.0000 | 193 | 12.8667 |
| unsupported_judgment | 3 | 0.2000 | 3 | 0.2000 |
| stalled_revision | 10 | 0.6667 | 19 | 1.2667 |
| contradiction_without_repair | 4 | 0.2667 | 8 | 0.5333 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 11 | 0.7333 | 33 | 2.2000 |
| fixed_belief_trace | 0 | 0.0000 | 0 | 0.0000 |
| disconnected_evidence | 12 | 0.8000 | 18 | 1.2000 |
| one_sided_confirmation | 1 | 0.0667 | 1 | 0.0667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 79 | 5.2667 |
| evidence_non_uptake | 15 | 1.0000 | 193 | 12.8667 |
| unsupported_judgment | 3 | 0.2000 | 3 | 0.2000 |
| stalled_revision | 10 | 0.6667 | 19 | 1.2667 |
| contradiction_without_repair | 4 | 0.2667 | 10 | 0.6667 |
| premature_commitment | 1 | 0.0667 | 1 | 0.0667 |
| uninformative_test | 11 | 0.7333 | 33 | 2.2000 |
| fixed_belief_trace | 0 | 0.0000 | 0 | 0.0000 |
| disconnected_evidence | 12 | 0.8000 | 18 | 1.2000 |
| one_sided_confirmation | 1 | 0.0667 | 1 | 0.0667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 88 | 5.8667 |
| evidence_handling | 15 | 1.0000 | 247 | 16.4667 |
| experimental_strategy | 11 | 0.7333 | 20 | 1.3333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 90 | 6.0000 |
| evidence_handling | 15 | 1.0000 | 247 | 16.4667 |
| experimental_strategy | 11 | 0.7333 | 20 | 1.3333 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 33 | 2.2000 |
| evidence_handling | 13 | 0.8667 | 15 | 1.0000 |
| experimental_strategy | 12 | 0.8000 | 12 | 0.8000 |

### claude_sonnet_45/wetlab/level_3

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 11 | 0.7333 | 11 | 0.7333 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 15 | 1.0000 | 15 | 1.0000 |
| hypothesis_reranking | 4 | 0.2667 | 4 | 0.2667 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 1 | 0.0667 | 1 | 0.0667 |
| evidence_guided_test_redesign | 13 | 0.8667 | 13 | 0.8667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 81 | 5.4000 |
| evidence_non_uptake | 15 | 1.0000 | 209 | 13.9333 |
| unsupported_judgment | 4 | 0.2667 | 4 | 0.2667 |
| stalled_revision | 9 | 0.6000 | 13 | 0.8667 |
| contradiction_without_repair | 5 | 0.3333 | 9 | 0.6000 |
| premature_commitment | 3 | 0.2000 | 3 | 0.2000 |
| uninformative_test | 12 | 0.8000 | 45 | 3.0000 |
| fixed_belief_trace | 4 | 0.2667 | 4 | 0.2667 |
| disconnected_evidence | 10 | 0.6667 | 17 | 1.1333 |
| one_sided_confirmation | 4 | 0.2667 | 4 | 0.2667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 81 | 5.4000 |
| evidence_non_uptake | 15 | 1.0000 | 209 | 13.9333 |
| unsupported_judgment | 4 | 0.2667 | 4 | 0.2667 |
| stalled_revision | 9 | 0.6000 | 13 | 0.8667 |
| contradiction_without_repair | 5 | 0.3333 | 10 | 0.6667 |
| premature_commitment | 3 | 0.2000 | 3 | 0.2000 |
| uninformative_test | 12 | 0.8000 | 45 | 3.0000 |
| fixed_belief_trace | 4 | 0.2667 | 4 | 0.2667 |
| disconnected_evidence | 10 | 0.6667 | 17 | 1.1333 |
| one_sided_confirmation | 4 | 0.2667 | 4 | 0.2667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 94 | 6.2667 |
| evidence_handling | 15 | 1.0000 | 275 | 18.3333 |
| experimental_strategy | 13 | 0.8667 | 20 | 1.3333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 95 | 6.3333 |
| evidence_handling | 15 | 1.0000 | 275 | 18.3333 |
| experimental_strategy | 13 | 0.8667 | 20 | 1.3333 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 30 | 2.0000 |
| evidence_handling | 15 | 1.0000 | 16 | 1.0667 |
| experimental_strategy | 13 | 0.8667 | 13 | 0.8667 |

### gpt_4o/afm/level_1

- Traces: 5

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 2 | 0.4000 | 2 | 0.4000 |
| explore_then_test_transition | 3 | 0.6000 | 3 | 0.6000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 3 | 0.6000 | 3 | 0.6000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 4 | 0.8000 | 4 | 0.8000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 1 | 0.2000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 15 | 3.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2000 | 4 | 0.8000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 0.2000 | 23 | 4.6000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 1 | 0.2000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 15 | 3.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2000 | 4 | 0.8000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 0.2000 | 23 | 4.6000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 0.2000 | 7 | 1.4000 |
| evidence_handling | 5 | 1.0000 | 38 | 7.6000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 0.2000 | 7 | 1.4000 |
| evidence_handling | 5 | 1.0000 | 38 | 7.6000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 3 | 0.6000 |
| evidence_handling | 3 | 0.6000 | 3 | 0.6000 |
| experimental_strategy | 4 | 0.8000 | 6 | 1.2000 |

### gpt_4o/afm/level_2

- Traces: 5

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 2 | 0.4000 | 2 | 0.4000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 5 | 1.0000 | 5 | 1.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 5 | 1.0000 | 5 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 5 | 1.0000 | 5 | 1.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 1 | 0.2000 | 1 | 0.2000 |
| evidence_non_uptake | 5 | 1.0000 | 21 | 4.2000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.4000 | 4 | 0.8000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 0.2000 | 2 | 0.4000 |
| fixed_belief_trace | 3 | 0.6000 | 3 | 0.6000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 1 | 0.2000 | 1 | 0.2000 |
| evidence_non_uptake | 5 | 1.0000 | 21 | 4.2000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.4000 | 7 | 1.4000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 0.2000 | 2 | 0.4000 |
| fixed_belief_trace | 3 | 0.6000 | 3 | 0.6000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 5 | 1.0000 |
| evidence_handling | 5 | 1.0000 | 23 | 4.6000 |
| experimental_strategy | 3 | 0.6000 | 3 | 0.6000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 8 | 1.6000 |
| evidence_handling | 5 | 1.0000 | 23 | 4.6000 |
| experimental_strategy | 3 | 0.6000 | 3 | 0.6000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 7 | 1.4000 |
| evidence_handling | 5 | 1.0000 | 5 | 1.0000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

### gpt_4o/afm/level_3

- Traces: 5

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 2 | 0.4000 | 2 | 0.4000 |
| explore_then_test_transition | 5 | 1.0000 | 5 | 1.0000 |
| hypothesis_reranking | 1 | 0.2000 | 1 | 0.2000 |
| evidence_led_hypothesis_generation | 5 | 1.0000 | 5 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 4 | 0.8000 | 4 | 0.8000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 14 | 2.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 3 | 0.6000 | 3 | 0.6000 |
| uninformative_test | 2 | 0.4000 | 4 | 0.8000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 3 | 0.6000 | 3 | 0.6000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 14 | 2.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 3 | 0.6000 | 3 | 0.6000 |
| uninformative_test | 2 | 0.4000 | 4 | 0.8000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 3 | 0.6000 | 3 | 0.6000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 6 | 1.2000 |
| evidence_handling | 5 | 1.0000 | 18 | 3.6000 |
| experimental_strategy | 5 | 1.0000 | 8 | 1.6000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 6 | 1.2000 |
| evidence_handling | 5 | 1.0000 | 18 | 3.6000 |
| experimental_strategy | 5 | 1.0000 | 8 | 1.6000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 6 | 1.2000 |
| evidence_handling | 5 | 1.0000 | 5 | 1.0000 |
| experimental_strategy | 5 | 1.0000 | 6 | 1.2000 |

### gpt_4o/afm/level_4

- Traces: 5

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 1 | 0.2000 | 1 | 0.2000 |
| explore_then_test_transition | 2 | 0.4000 | 2 | 0.4000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 4 | 0.8000 | 4 | 0.8000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 3 | 0.6000 | 3 | 0.6000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 15 | 3.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 2 | 0.4000 | 2 | 0.4000 |
| uninformative_test | 3 | 0.6000 | 25 | 5.0000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 2 | 0.4000 | 2 | 0.4000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 15 | 3.0000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 2 | 0.4000 | 2 | 0.4000 |
| uninformative_test | 3 | 0.6000 | 25 | 5.0000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 2 | 0.4000 | 2 | 0.4000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 5 | 1.0000 |
| evidence_handling | 5 | 1.0000 | 40 | 8.0000 |
| experimental_strategy | 5 | 1.0000 | 7 | 1.4000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 5 | 1.0000 |
| evidence_handling | 5 | 1.0000 | 40 | 8.0000 |
| experimental_strategy | 5 | 1.0000 | 7 | 1.4000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 4 | 0.8000 |
| evidence_handling | 2 | 0.4000 | 2 | 0.4000 |
| experimental_strategy | 3 | 0.6000 | 4 | 0.8000 |

### gpt_4o/catalyst/level_1

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 1 | 0.0667 | 1 | 0.0667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 15 | 1.0000 | 87 | 5.8000 |
| unsupported_judgment | 2 | 0.1333 | 2 | 0.1333 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.1333 | 46 | 3.0667 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 15 | 1.0000 | 87 | 5.8000 |
| unsupported_judgment | 2 | 0.1333 | 2 | 0.1333 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.1333 | 46 | 3.0667 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 15 | 1.0000 | 135 | 9.0000 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 15 | 1.0000 | 135 | 9.0000 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 1 | 0.0667 | 1 | 0.0667 |

### gpt_4o/md/level_1

- Traces: 10

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 5 | 0.5000 | 5 | 0.5000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 7 | 0.7000 | 7 | 0.7000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 10 | 1.0000 | 10 | 1.0000 |
| convergent_multi_test_evidence | 1 | 0.1000 | 1 | 0.1000 |
| evidence_guided_test_redesign | 6 | 0.6000 | 6 | 0.6000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.9000 | 19 | 1.9000 |
| evidence_non_uptake | 10 | 1.0000 | 41 | 4.1000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 4 | 0.4000 | 5 | 0.5000 |
| contradiction_without_repair | 4 | 0.4000 | 5 | 0.5000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 6 | 0.6000 | 8 | 0.8000 |
| fixed_belief_trace | 4 | 0.4000 | 4 | 0.4000 |
| disconnected_evidence | 6 | 0.6000 | 7 | 0.7000 |
| one_sided_confirmation | 1 | 0.1000 | 1 | 0.1000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 9 | 0.9000 | 19 | 1.9000 |
| evidence_non_uptake | 10 | 1.0000 | 41 | 4.1000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 4 | 0.4000 | 5 | 0.5000 |
| contradiction_without_repair | 5 | 0.5000 | 8 | 0.8000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 6 | 0.6000 | 8 | 0.8000 |
| fixed_belief_trace | 4 | 0.4000 | 4 | 0.4000 |
| disconnected_evidence | 6 | 0.6000 | 7 | 0.7000 |
| one_sided_confirmation | 1 | 0.1000 | 1 | 0.1000 |
| precommitted_test_plan | 1 | 0.1000 | 1 | 0.1000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 25 | 2.5000 |
| evidence_handling | 10 | 1.0000 | 56 | 5.6000 |
| experimental_strategy | 8 | 0.8000 | 9 | 0.9000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 28 | 2.8000 |
| evidence_handling | 10 | 1.0000 | 56 | 5.6000 |
| experimental_strategy | 9 | 0.9000 | 10 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 1.0000 | 15 | 1.5000 |
| evidence_handling | 7 | 0.7000 | 8 | 0.8000 |
| experimental_strategy | 6 | 0.6000 | 6 | 0.6000 |

### gpt_4o/md/level_2

- Traces: 5

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 1 | 0.2000 | 1 | 0.2000 |
| explore_then_test_transition | 5 | 1.0000 | 5 | 1.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 5 | 1.0000 | 5 | 1.0000 |
| convergent_multi_test_evidence | 1 | 0.2000 | 1 | 0.2000 |
| evidence_guided_test_redesign | 5 | 1.0000 | 5 | 1.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 4 | 0.8000 |
| evidence_non_uptake | 5 | 1.0000 | 29 | 5.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 3 | 0.6000 | 6 | 1.2000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.4000 | 3 | 0.6000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 2 | 0.4000 | 3 | 0.6000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 4 | 0.8000 |
| evidence_non_uptake | 5 | 1.0000 | 29 | 5.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 3 | 0.6000 | 6 | 1.2000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.4000 | 3 | 0.6000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 2 | 0.4000 | 3 | 0.6000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 10 | 2.0000 |
| evidence_handling | 5 | 1.0000 | 35 | 7.0000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 10 | 2.0000 |
| evidence_handling | 5 | 1.0000 | 35 | 7.0000 |
| experimental_strategy | 5 | 1.0000 | 5 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 5 | 1.0000 |
| evidence_handling | 5 | 1.0000 | 6 | 1.2000 |
| experimental_strategy | 5 | 1.0000 | 6 | 1.2000 |

### gpt_4o/ml/level_1

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 3 | 0.2000 | 3 | 0.2000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 5 | 0.3333 | 5 | 0.3333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 13 | 0.8667 | 55 | 3.6667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 13 | 0.8667 | 55 | 3.6667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 13 | 0.8667 | 55 | 3.6667 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 13 | 0.8667 | 55 | 3.6667 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 5 | 0.3333 | 8 | 0.5333 |

### gpt_4o/resistor/level_1

- Traces: 30

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 12 | 0.4000 | 12 | 0.4000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 22 | 0.7333 | 22 | 0.7333 |
| hypothesis_reranking | 9 | 0.3000 | 9 | 0.3000 |
| evidence_led_hypothesis_generation | 27 | 0.9000 | 27 | 0.9000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 26 | 0.8667 | 26 | 0.8667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 13 | 0.4333 | 22 | 0.7333 |
| evidence_non_uptake | 25 | 0.8333 | 73 | 2.4333 |
| unsupported_judgment | 12 | 0.4000 | 19 | 0.6333 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 14 | 0.4667 | 50 | 1.6667 |
| premature_commitment | 2 | 0.0667 | 2 | 0.0667 |
| uninformative_test | 15 | 0.5000 | 42 | 1.4000 |
| fixed_belief_trace | 18 | 0.6000 | 18 | 0.6000 |
| disconnected_evidence | 10 | 0.3333 | 24 | 0.8000 |
| one_sided_confirmation | 3 | 0.1000 | 3 | 0.1000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 13 | 0.4333 | 22 | 0.7333 |
| evidence_non_uptake | 25 | 0.8333 | 73 | 2.4333 |
| unsupported_judgment | 12 | 0.4000 | 19 | 0.6333 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 14 | 0.4667 | 56 | 1.8667 |
| premature_commitment | 2 | 0.0667 | 2 | 0.0667 |
| uninformative_test | 15 | 0.5000 | 42 | 1.4000 |
| fixed_belief_trace | 18 | 0.6000 | 18 | 0.6000 |
| disconnected_evidence | 10 | 0.3333 | 24 | 0.8000 |
| one_sided_confirmation | 3 | 0.1000 | 3 | 0.1000 |
| precommitted_test_plan | 9 | 0.3000 | 21 | 0.7000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 23 | 0.7667 | 75 | 2.5000 |
| evidence_handling | 27 | 0.9000 | 158 | 5.2667 |
| experimental_strategy | 19 | 0.6333 | 20 | 0.6667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 23 | 0.7667 | 81 | 2.7000 |
| evidence_handling | 27 | 0.9000 | 158 | 5.2667 |
| experimental_strategy | 24 | 0.8000 | 41 | 1.3667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 29 | 0.9667 | 48 | 1.6000 |
| evidence_handling | 22 | 0.7333 | 22 | 0.7333 |
| experimental_strategy | 26 | 0.8667 | 26 | 0.8667 |

### gpt_4o/retrosynthesis/level_1

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 2 | 0.1333 | 2 | 0.1333 |
| fixed_hypothesis_test_tuning | 3 | 0.2000 | 3 | 0.2000 |
| explore_then_test_transition | 6 | 0.4000 | 6 | 0.4000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 8 | 0.5333 | 8 | 0.5333 |
| convergent_multi_test_evidence | 1 | 0.0667 | 1 | 0.0667 |
| evidence_guided_test_redesign | 8 | 0.5333 | 8 | 0.5333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.1333 | 2 | 0.1333 |
| evidence_non_uptake | 7 | 0.4667 | 26 | 1.7333 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.0667 | 1 | 0.0667 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.1333 | 9 | 0.6000 |
| fixed_belief_trace | 13 | 0.8667 | 13 | 0.8667 |
| disconnected_evidence | 1 | 0.0667 | 1 | 0.0667 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.1333 | 2 | 0.1333 |
| evidence_non_uptake | 7 | 0.4667 | 26 | 1.7333 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.0667 | 1 | 0.0667 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.1333 | 9 | 0.6000 |
| fixed_belief_trace | 13 | 0.8667 | 13 | 0.8667 |
| disconnected_evidence | 1 | 0.0667 | 1 | 0.0667 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.2000 | 3 | 0.2000 |
| evidence_handling | 8 | 0.5333 | 36 | 2.4000 |
| experimental_strategy | 13 | 0.8667 | 13 | 0.8667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.2000 | 3 | 0.2000 |
| evidence_handling | 8 | 0.5333 | 36 | 2.4000 |
| experimental_strategy | 13 | 0.8667 | 13 | 0.8667 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.5333 | 10 | 0.6667 |
| evidence_handling | 6 | 0.4000 | 7 | 0.4667 |
| experimental_strategy | 10 | 0.6667 | 11 | 0.7333 |

### gpt_4o/retrosynthesis/level_2

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 6 | 0.4000 | 6 | 0.4000 |
| explore_then_test_transition | 10 | 0.6667 | 10 | 0.6667 |
| hypothesis_reranking | 1 | 0.0667 | 1 | 0.0667 |
| evidence_led_hypothesis_generation | 12 | 0.8000 | 12 | 0.8000 |
| convergent_multi_test_evidence | 1 | 0.0667 | 1 | 0.0667 |
| evidence_guided_test_redesign | 11 | 0.7333 | 11 | 0.7333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 0.3333 | 8 | 0.5333 |
| evidence_non_uptake | 12 | 0.8000 | 40 | 2.6667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 4 | 0.2667 | 6 | 0.4000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 2 | 0.1333 | 3 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 0.3333 | 8 | 0.5333 |
| evidence_non_uptake | 12 | 0.8000 | 40 | 2.6667 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 4 | 0.2667 | 6 | 0.4000 |
| fixed_belief_trace | 15 | 1.0000 | 15 | 1.0000 |
| disconnected_evidence | 2 | 0.1333 | 3 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.3333 | 8 | 0.5333 |
| evidence_handling | 12 | 0.8000 | 49 | 3.2667 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.3333 | 8 | 0.5333 |
| evidence_handling | 12 | 0.8000 | 49 | 3.2667 |
| experimental_strategy | 15 | 1.0000 | 15 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 12 | 0.8000 | 13 | 0.8667 |
| evidence_handling | 10 | 0.6667 | 11 | 0.7333 |
| experimental_strategy | 12 | 0.8000 | 17 | 1.1333 |

### gpt_4o/retrosynthesis/level_3

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 3 | 0.2000 | 3 | 0.2000 |
| fixed_hypothesis_test_tuning | 5 | 0.3333 | 5 | 0.3333 |
| explore_then_test_transition | 11 | 0.7333 | 11 | 0.7333 |
| hypothesis_reranking | 2 | 0.1333 | 2 | 0.1333 |
| evidence_led_hypothesis_generation | 11 | 0.7333 | 11 | 0.7333 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 11 | 0.7333 | 11 | 0.7333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 0.3333 | 12 | 0.8000 |
| evidence_non_uptake | 14 | 0.9333 | 180 | 12.0000 |
| unsupported_judgment | 5 | 0.3333 | 6 | 0.4000 |
| stalled_revision | 1 | 0.0667 | 1 | 0.0667 |
| contradiction_without_repair | 8 | 0.5333 | 27 | 1.8000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 12 | 0.8000 | 32 | 2.1333 |
| fixed_belief_trace | 12 | 0.8000 | 12 | 0.8000 |
| disconnected_evidence | 2 | 0.1333 | 3 | 0.2000 |
| one_sided_confirmation | 1 | 0.0667 | 3 | 0.2000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 0.3333 | 12 | 0.8000 |
| evidence_non_uptake | 14 | 0.9333 | 180 | 12.0000 |
| unsupported_judgment | 5 | 0.3333 | 6 | 0.4000 |
| stalled_revision | 1 | 0.0667 | 1 | 0.0667 |
| contradiction_without_repair | 8 | 0.5333 | 27 | 1.8000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 12 | 0.8000 | 32 | 2.1333 |
| fixed_belief_trace | 12 | 0.8000 | 12 | 0.8000 |
| disconnected_evidence | 2 | 0.1333 | 3 | 0.2000 |
| one_sided_confirmation | 1 | 0.0667 | 3 | 0.2000 |
| precommitted_test_plan | 2 | 0.1333 | 4 | 0.2667 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.6000 | 42 | 2.8000 |
| evidence_handling | 15 | 1.0000 | 221 | 14.7333 |
| experimental_strategy | 13 | 0.8667 | 13 | 0.8667 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.6000 | 42 | 2.8000 |
| evidence_handling | 15 | 1.0000 | 221 | 14.7333 |
| experimental_strategy | 14 | 0.9333 | 17 | 1.1333 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 13 | 0.8667 | 16 | 1.0667 |
| evidence_handling | 11 | 0.7333 | 11 | 0.7333 |
| experimental_strategy | 13 | 0.8667 | 16 | 1.0667 |

### gpt_4o/spectra/level_1

- Traces: 22

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 5 | 0.2273 | 5 | 0.2273 |
| fixed_hypothesis_test_tuning | 6 | 0.2727 | 6 | 0.2727 |
| explore_then_test_transition | 19 | 0.8636 | 19 | 0.8636 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 22 | 1.0000 | 22 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 12 | 0.5455 | 12 | 0.5455 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 10 | 0.4545 | 13 | 0.5909 |
| evidence_non_uptake | 6 | 0.2727 | 18 | 0.8182 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 2 | 0.0909 | 3 | 0.1364 |
| contradiction_without_repair | 2 | 0.0909 | 3 | 0.1364 |
| premature_commitment | 1 | 0.0455 | 1 | 0.0455 |
| uninformative_test | 2 | 0.0909 | 2 | 0.0909 |
| fixed_belief_trace | 17 | 0.7727 | 17 | 0.7727 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 2 | 0.0909 | 2 | 0.0909 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 10 | 0.4545 | 13 | 0.5909 |
| evidence_non_uptake | 6 | 0.2727 | 18 | 0.8182 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 2 | 0.0909 | 3 | 0.1364 |
| contradiction_without_repair | 2 | 0.0909 | 3 | 0.1364 |
| premature_commitment | 1 | 0.0455 | 1 | 0.0455 |
| uninformative_test | 2 | 0.0909 | 2 | 0.0909 |
| fixed_belief_trace | 17 | 0.7727 | 17 | 0.7727 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 2 | 0.0909 | 2 | 0.0909 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 11 | 0.5000 | 18 | 0.8182 |
| evidence_handling | 6 | 0.2727 | 20 | 0.9091 |
| experimental_strategy | 19 | 0.8636 | 21 | 0.9545 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 11 | 0.5000 | 18 | 0.8182 |
| evidence_handling | 6 | 0.2727 | 20 | 0.9091 |
| experimental_strategy | 19 | 0.8636 | 21 | 0.9545 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 22 | 1.0000 | 27 | 1.2273 |
| evidence_handling | 19 | 0.8636 | 19 | 0.8636 |
| experimental_strategy | 16 | 0.7273 | 18 | 0.8182 |

### gpt_4o/spectra/level_2

- Traces: 22

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 4 | 0.1818 | 4 | 0.1818 |
| fixed_hypothesis_test_tuning | 9 | 0.4091 | 9 | 0.4091 |
| explore_then_test_transition | 19 | 0.8636 | 19 | 0.8636 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 22 | 1.0000 | 22 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 14 | 0.6364 | 14 | 0.6364 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 0.2273 | 8 | 0.3636 |
| evidence_non_uptake | 4 | 0.1818 | 9 | 0.4091 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.0455 | 1 | 0.0455 |
| premature_commitment | 1 | 0.0455 | 1 | 0.0455 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 18 | 0.8182 | 18 | 0.8182 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0455 | 1 | 0.0455 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 0.2273 | 8 | 0.3636 |
| evidence_non_uptake | 4 | 0.1818 | 9 | 0.4091 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.0455 | 1 | 0.0455 |
| premature_commitment | 1 | 0.0455 | 1 | 0.0455 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 18 | 0.8182 | 18 | 0.8182 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0455 | 1 | 0.0455 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.2273 | 10 | 0.4545 |
| evidence_handling | 4 | 0.1818 | 9 | 0.4091 |
| experimental_strategy | 18 | 0.8182 | 19 | 0.8636 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 0.2273 | 10 | 0.4545 |
| evidence_handling | 4 | 0.1818 | 9 | 0.4091 |
| experimental_strategy | 18 | 0.8182 | 19 | 0.8636 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 22 | 1.0000 | 26 | 1.1818 |
| evidence_handling | 19 | 0.8636 | 19 | 0.8636 |
| experimental_strategy | 18 | 0.8182 | 23 | 1.0455 |

### gpt_4o/wetlab/level_1

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 6 | 0.4000 | 6 | 0.4000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 13 | 0.8667 | 13 | 0.8667 |
| hypothesis_reranking | 2 | 0.1333 | 2 | 0.1333 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 8 | 0.5333 | 8 | 0.5333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 25 | 1.6667 |
| evidence_non_uptake | 15 | 1.0000 | 69 | 4.6000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 5 | 0.3333 | 6 | 0.4000 |
| contradiction_without_repair | 5 | 0.3333 | 6 | 0.4000 |
| premature_commitment | 6 | 0.4000 | 6 | 0.4000 |
| uninformative_test | 10 | 0.6667 | 18 | 1.2000 |
| fixed_belief_trace | 9 | 0.6000 | 9 | 0.6000 |
| disconnected_evidence | 10 | 0.6667 | 17 | 1.1333 |
| one_sided_confirmation | 6 | 0.4000 | 6 | 0.4000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 15 | 1.0000 | 25 | 1.6667 |
| evidence_non_uptake | 15 | 1.0000 | 69 | 4.6000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 5 | 0.3333 | 6 | 0.4000 |
| contradiction_without_repair | 5 | 0.3333 | 7 | 0.4667 |
| premature_commitment | 6 | 0.4000 | 6 | 0.4000 |
| uninformative_test | 10 | 0.6667 | 18 | 1.2000 |
| fixed_belief_trace | 9 | 0.6000 | 9 | 0.6000 |
| disconnected_evidence | 10 | 0.6667 | 17 | 1.1333 |
| one_sided_confirmation | 6 | 0.4000 | 6 | 0.4000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 37 | 2.4667 |
| evidence_handling | 15 | 1.0000 | 104 | 6.9333 |
| experimental_strategy | 15 | 1.0000 | 21 | 1.4000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 38 | 2.5333 |
| evidence_handling | 15 | 1.0000 | 104 | 6.9333 |
| experimental_strategy | 15 | 1.0000 | 21 | 1.4000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 23 | 1.5333 |
| evidence_handling | 13 | 0.8667 | 13 | 0.8667 |
| experimental_strategy | 8 | 0.5333 | 8 | 0.5333 |

### gpt_4o/wetlab/level_2

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 5 | 0.3333 | 5 | 0.3333 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 14 | 0.9333 | 14 | 0.9333 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 10 | 0.6667 | 10 | 0.6667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 13 | 0.8667 | 28 | 1.8667 |
| evidence_non_uptake | 15 | 1.0000 | 62 | 4.1333 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 4 | 0.2667 | 4 | 0.2667 |
| contradiction_without_repair | 9 | 0.6000 | 14 | 0.9333 |
| premature_commitment | 3 | 0.2000 | 3 | 0.2000 |
| uninformative_test | 8 | 0.5333 | 10 | 0.6667 |
| fixed_belief_trace | 10 | 0.6667 | 10 | 0.6667 |
| disconnected_evidence | 7 | 0.4667 | 8 | 0.5333 |
| one_sided_confirmation | 4 | 0.2667 | 4 | 0.2667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 13 | 0.8667 | 28 | 1.8667 |
| evidence_non_uptake | 15 | 1.0000 | 62 | 4.1333 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 4 | 0.2667 | 4 | 0.2667 |
| contradiction_without_repair | 9 | 0.6000 | 14 | 0.9333 |
| premature_commitment | 3 | 0.2000 | 3 | 0.2000 |
| uninformative_test | 8 | 0.5333 | 10 | 0.6667 |
| fixed_belief_trace | 10 | 0.6667 | 10 | 0.6667 |
| disconnected_evidence | 7 | 0.4667 | 8 | 0.5333 |
| one_sided_confirmation | 4 | 0.2667 | 4 | 0.2667 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 46 | 3.0667 |
| evidence_handling | 15 | 1.0000 | 80 | 5.3333 |
| experimental_strategy | 14 | 0.9333 | 17 | 1.1333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 46 | 3.0667 |
| evidence_handling | 15 | 1.0000 | 80 | 5.3333 |
| experimental_strategy | 14 | 0.9333 | 17 | 1.1333 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 20 | 1.3333 |
| evidence_handling | 14 | 0.9333 | 14 | 0.9333 |
| experimental_strategy | 10 | 0.6667 | 10 | 0.6667 |

### gpt_4o/wetlab/level_3

- Traces: 15

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 5 | 0.3333 | 5 | 0.3333 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 15 | 1.0000 | 15 | 1.0000 |
| hypothesis_reranking | 1 | 0.0667 | 1 | 0.0667 |
| evidence_led_hypothesis_generation | 15 | 1.0000 | 15 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 8 | 0.5333 | 8 | 0.5333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 12 | 0.8000 | 28 | 1.8667 |
| evidence_non_uptake | 15 | 1.0000 | 75 | 5.0000 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 2 | 0.1333 | 2 | 0.1333 |
| contradiction_without_repair | 5 | 0.3333 | 9 | 0.6000 |
| premature_commitment | 5 | 0.3333 | 5 | 0.3333 |
| uninformative_test | 8 | 0.5333 | 11 | 0.7333 |
| fixed_belief_trace | 10 | 0.6667 | 10 | 0.6667 |
| disconnected_evidence | 6 | 0.4000 | 7 | 0.4667 |
| one_sided_confirmation | 6 | 0.4000 | 6 | 0.4000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 12 | 0.8000 | 28 | 1.8667 |
| evidence_non_uptake | 15 | 1.0000 | 75 | 5.0000 |
| unsupported_judgment | 1 | 0.0667 | 1 | 0.0667 |
| stalled_revision | 2 | 0.1333 | 2 | 0.1333 |
| contradiction_without_repair | 5 | 0.3333 | 10 | 0.6667 |
| premature_commitment | 5 | 0.3333 | 5 | 0.3333 |
| uninformative_test | 8 | 0.5333 | 11 | 0.7333 |
| fixed_belief_trace | 10 | 0.6667 | 10 | 0.6667 |
| disconnected_evidence | 6 | 0.4000 | 7 | 0.4667 |
| one_sided_confirmation | 6 | 0.4000 | 6 | 0.4000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 43 | 2.8667 |
| evidence_handling | 15 | 1.0000 | 94 | 6.2667 |
| experimental_strategy | 13 | 0.8667 | 17 | 1.1333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 44 | 2.9333 |
| evidence_handling | 15 | 1.0000 | 94 | 6.2667 |
| experimental_strategy | 13 | 0.8667 | 17 | 1.1333 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 15 | 1.0000 | 21 | 1.4000 |
| evidence_handling | 15 | 1.0000 | 15 | 1.0000 |
| experimental_strategy | 8 | 0.5333 | 8 | 0.5333 |

### gpt_oss_120b/afm/level_1

- Traces: 4

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 1 | 0.2500 | 1 | 0.2500 |
| explore_then_test_transition | 1 | 0.2500 | 1 | 0.2500 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 4 | 1.0000 | 4 | 1.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.7500 | 4 | 1.0000 |
| evidence_non_uptake | 2 | 0.5000 | 7 | 1.7500 |
| unsupported_judgment | 2 | 0.5000 | 2 | 0.5000 |
| stalled_revision | 1 | 0.2500 | 1 | 0.2500 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 3 | 0.7500 | 16 | 4.0000 |
| fixed_belief_trace | 3 | 0.7500 | 3 | 0.7500 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.7500 | 4 | 1.0000 |
| evidence_non_uptake | 2 | 0.5000 | 7 | 1.7500 |
| unsupported_judgment | 2 | 0.5000 | 2 | 0.5000 |
| stalled_revision | 1 | 0.2500 | 1 | 0.2500 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 3 | 0.7500 | 16 | 4.0000 |
| fixed_belief_trace | 3 | 0.7500 | 3 | 0.7500 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.7500 | 4 | 1.0000 |
| evidence_handling | 4 | 1.0000 | 25 | 6.2500 |
| experimental_strategy | 4 | 1.0000 | 4 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.7500 | 4 | 1.0000 |
| evidence_handling | 4 | 1.0000 | 25 | 6.2500 |
| experimental_strategy | 4 | 1.0000 | 4 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 1.0000 | 4 | 1.0000 |
| evidence_handling | 1 | 0.2500 | 1 | 0.2500 |
| experimental_strategy | 1 | 0.2500 | 1 | 0.2500 |

### gpt_oss_120b/afm/level_2

- Traces: 4

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 3 | 0.7500 | 3 | 0.7500 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.2500 | 1 | 0.2500 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 2 | 0.5000 | 2 | 0.5000 |
| convergent_multi_test_evidence | 3 | 0.7500 | 3 | 0.7500 |
| evidence_guided_test_redesign | 1 | 0.2500 | 1 | 0.2500 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.7500 | 6 | 1.5000 |
| evidence_non_uptake | 2 | 0.5000 | 4 | 1.0000 |
| unsupported_judgment | 2 | 0.5000 | 7 | 1.7500 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2500 | 1 | 0.2500 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 4 | 1.0000 | 37 | 9.2500 |
| fixed_belief_trace | 1 | 0.2500 | 1 | 0.2500 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 2 | 0.5000 | 2 | 0.5000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.7500 | 6 | 1.5000 |
| evidence_non_uptake | 2 | 0.5000 | 4 | 1.0000 |
| unsupported_judgment | 2 | 0.5000 | 7 | 1.7500 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.5000 | 7 | 1.7500 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 4 | 1.0000 | 37 | 9.2500 |
| fixed_belief_trace | 1 | 0.2500 | 1 | 0.2500 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 1 | 0.2500 | 2 | 0.5000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 1.0000 | 7 | 1.7500 |
| evidence_handling | 4 | 1.0000 | 48 | 12.0000 |
| experimental_strategy | 2 | 0.5000 | 3 | 0.7500 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 1.0000 | 13 | 3.2500 |
| evidence_handling | 4 | 1.0000 | 48 | 12.0000 |
| experimental_strategy | 2 | 0.5000 | 3 | 0.7500 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 1.0000 | 5 | 1.2500 |
| evidence_handling | 3 | 0.7500 | 4 | 1.0000 |
| experimental_strategy | 1 | 0.2500 | 1 | 0.2500 |

### gpt_oss_120b/afm/level_3

- Traces: 4

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 1 | 0.2500 | 1 | 0.2500 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 2 | 0.5000 | 2 | 0.5000 |
| hypothesis_reranking | 1 | 0.2500 | 1 | 0.2500 |
| evidence_led_hypothesis_generation | 3 | 0.7500 | 3 | 0.7500 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 2 | 0.5000 | 2 | 0.5000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 1 | 0.2500 | 1 | 0.2500 |
| evidence_non_uptake | 3 | 0.7500 | 6 | 1.5000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.5000 | 2 | 0.5000 |
| premature_commitment | 1 | 0.2500 | 1 | 0.2500 |
| uninformative_test | 2 | 0.5000 | 6 | 1.5000 |
| fixed_belief_trace | 3 | 0.7500 | 3 | 0.7500 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.2500 | 1 | 0.2500 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 1 | 0.2500 | 1 | 0.2500 |
| evidence_non_uptake | 3 | 0.7500 | 6 | 1.5000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.5000 | 2 | 0.5000 |
| premature_commitment | 1 | 0.2500 | 1 | 0.2500 |
| uninformative_test | 2 | 0.5000 | 6 | 1.5000 |
| fixed_belief_trace | 3 | 0.7500 | 3 | 0.7500 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.2500 | 1 | 0.2500 |
| precommitted_test_plan | 2 | 0.5000 | 3 | 0.7500 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.7500 | 4 | 1.0000 |
| evidence_handling | 3 | 0.7500 | 12 | 3.0000 |
| experimental_strategy | 3 | 0.7500 | 4 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.7500 | 4 | 1.0000 |
| evidence_handling | 3 | 0.7500 | 12 | 3.0000 |
| experimental_strategy | 4 | 1.0000 | 7 | 1.7500 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.7500 | 5 | 1.2500 |
| evidence_handling | 2 | 0.5000 | 2 | 0.5000 |
| experimental_strategy | 2 | 0.5000 | 2 | 0.5000 |

### gpt_oss_120b/afm/level_4

- Traces: 3

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 2 | 0.6667 | 2 | 0.6667 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 2 | 0.6667 | 2 | 0.6667 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 3 | 1.0000 | 3 | 1.0000 |
| convergent_multi_test_evidence | 1 | 0.3333 | 1 | 0.3333 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.6667 | 2 | 0.6667 |
| evidence_non_uptake | 2 | 0.6667 | 4 | 1.3333 |
| unsupported_judgment | 1 | 0.3333 | 1 | 0.3333 |
| stalled_revision | 1 | 0.3333 | 1 | 0.3333 |
| contradiction_without_repair | 1 | 0.3333 | 1 | 0.3333 |
| premature_commitment | 2 | 0.6667 | 2 | 0.6667 |
| uninformative_test | 3 | 1.0000 | 19 | 6.3333 |
| fixed_belief_trace | 1 | 0.3333 | 1 | 0.3333 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.3333 | 1 | 0.3333 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.6667 | 2 | 0.6667 |
| evidence_non_uptake | 2 | 0.6667 | 4 | 1.3333 |
| unsupported_judgment | 1 | 0.3333 | 1 | 0.3333 |
| stalled_revision | 1 | 0.3333 | 1 | 0.3333 |
| contradiction_without_repair | 1 | 0.3333 | 1 | 0.3333 |
| premature_commitment | 2 | 0.6667 | 2 | 0.6667 |
| uninformative_test | 3 | 1.0000 | 19 | 6.3333 |
| fixed_belief_trace | 1 | 0.3333 | 1 | 0.3333 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.3333 | 1 | 0.3333 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 1.0000 | 4 | 1.3333 |
| evidence_handling | 3 | 1.0000 | 24 | 8.0000 |
| experimental_strategy | 2 | 0.6667 | 4 | 1.3333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 1.0000 | 4 | 1.3333 |
| evidence_handling | 3 | 1.0000 | 24 | 8.0000 |
| experimental_strategy | 2 | 0.6667 | 4 | 1.3333 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 1.0000 | 5 | 1.6667 |
| evidence_handling | 2 | 0.6667 | 3 | 1.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_oss_120b/catalyst/level_1

- Traces: 23

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 1 | 0.0435 | 1 | 0.0435 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 2 | 0.0870 | 2 | 0.0870 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 12 | 0.5217 | 12 | 0.5217 |
| convergent_multi_test_evidence | 1 | 0.0435 | 1 | 0.0435 |
| evidence_guided_test_redesign | 3 | 0.1304 | 3 | 0.1304 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 16 | 0.6957 | 16 | 0.6957 |
| evidence_non_uptake | 10 | 0.4348 | 16 | 0.6957 |
| unsupported_judgment | 3 | 0.1304 | 5 | 0.2174 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 1 | 0.0435 | 1 | 0.0435 |
| uninformative_test | 3 | 0.1304 | 43 | 1.8696 |
| fixed_belief_trace | 22 | 0.9565 | 22 | 0.9565 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0435 | 1 | 0.0435 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 16 | 0.6957 | 16 | 0.6957 |
| evidence_non_uptake | 10 | 0.4348 | 16 | 0.6957 |
| unsupported_judgment | 3 | 0.1304 | 5 | 0.2174 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 1 | 0.0435 | 1 | 0.0435 |
| uninformative_test | 3 | 0.1304 | 43 | 1.8696 |
| fixed_belief_trace | 22 | 0.9565 | 22 | 0.9565 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0435 | 1 | 0.0435 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 16 | 0.6957 | 17 | 0.7391 |
| evidence_handling | 10 | 0.4348 | 64 | 2.7826 |
| experimental_strategy | 22 | 0.9565 | 23 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 16 | 0.6957 | 17 | 0.7391 |
| evidence_handling | 10 | 0.4348 | 64 | 2.7826 |
| experimental_strategy | 22 | 0.9565 | 23 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 12 | 0.5217 | 13 | 0.5652 |
| evidence_handling | 2 | 0.0870 | 3 | 0.1304 |
| experimental_strategy | 3 | 0.1304 | 3 | 0.1304 |

### gpt_oss_120b/md/level_1

- Traces: 13

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 7 | 0.5385 | 7 | 0.5385 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 11 | 0.8462 | 12 | 0.9231 |
| evidence_non_uptake | 4 | 0.3077 | 6 | 0.4615 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.0769 | 1 | 0.0769 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 13 | 1.0000 | 13 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 11 | 0.8462 | 12 | 0.9231 |
| evidence_non_uptake | 4 | 0.3077 | 6 | 0.4615 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.0769 | 1 | 0.0769 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 13 | 1.0000 | 13 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 11 | 0.8462 | 13 | 1.0000 |
| evidence_handling | 4 | 0.3077 | 6 | 0.4615 |
| experimental_strategy | 13 | 1.0000 | 13 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 11 | 0.8462 | 13 | 1.0000 |
| evidence_handling | 4 | 0.3077 | 6 | 0.4615 |
| experimental_strategy | 13 | 1.0000 | 13 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 7 | 0.5385 | 7 | 0.5385 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_oss_120b/md/level_2

- Traces: 10

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.1000 | 1 | 0.1000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 8 | 0.8000 | 8 | 0.8000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 1 | 0.1000 | 1 | 0.1000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 8 | 0.8000 | 9 | 0.9000 |
| evidence_non_uptake | 6 | 0.6000 | 8 | 0.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 0.1000 | 4 | 0.4000 |
| fixed_belief_trace | 10 | 1.0000 | 10 | 1.0000 |
| disconnected_evidence | 1 | 0.1000 | 1 | 0.1000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 8 | 0.8000 | 9 | 0.9000 |
| evidence_non_uptake | 6 | 0.6000 | 8 | 0.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 0.1000 | 4 | 0.4000 |
| fixed_belief_trace | 10 | 1.0000 | 10 | 1.0000 |
| disconnected_evidence | 1 | 0.1000 | 1 | 0.1000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.8000 | 9 | 0.9000 |
| evidence_handling | 6 | 0.6000 | 13 | 1.3000 |
| experimental_strategy | 10 | 1.0000 | 10 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.8000 | 9 | 0.9000 |
| evidence_handling | 6 | 0.6000 | 13 | 1.3000 |
| experimental_strategy | 10 | 1.0000 | 10 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.8000 | 8 | 0.8000 |
| evidence_handling | 1 | 0.1000 | 1 | 0.1000 |
| experimental_strategy | 1 | 0.1000 | 1 | 0.1000 |

### gpt_oss_120b/resistor/level_1

- Traces: 28

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 1 | 0.0357 | 1 | 0.0357 |
| evidence_led_hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 28 | 1.0000 | 39 | 1.3929 |
| evidence_non_uptake | 0 | 0.0000 | 0 | 0.0000 |
| unsupported_judgment | 22 | 0.7857 | 23 | 0.8214 |
| stalled_revision | 6 | 0.2143 | 7 | 0.2500 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 4 | 0.1429 | 4 | 0.1429 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 22 | 0.7857 | 22 | 0.7857 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 28 | 1.0000 | 39 | 1.3929 |
| evidence_non_uptake | 0 | 0.0000 | 0 | 0.0000 |
| unsupported_judgment | 22 | 0.7857 | 23 | 0.8214 |
| stalled_revision | 6 | 0.2143 | 7 | 0.2500 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 4 | 0.1429 | 4 | 0.1429 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 22 | 0.7857 | 22 | 0.7857 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 28 | 1.0000 | 39 | 1.3929 |
| evidence_handling | 22 | 0.7857 | 23 | 0.8214 |
| experimental_strategy | 28 | 1.0000 | 33 | 1.1786 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 28 | 1.0000 | 39 | 1.3929 |
| evidence_handling | 22 | 0.7857 | 23 | 0.8214 |
| experimental_strategy | 28 | 1.0000 | 33 | 1.1786 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 0.0357 | 1 | 0.0357 |
| evidence_handling | 0 | 0.0000 | 0 | 0.0000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

### gpt_oss_120b/retrosynthesis/level_1

- Traces: 8

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 1 | 0.1250 | 1 | 0.1250 |
| explore_then_test_transition | 1 | 0.1250 | 1 | 0.1250 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 3 | 0.3750 | 3 | 0.3750 |
| convergent_multi_test_evidence | 1 | 0.1250 | 1 | 0.1250 |
| evidence_guided_test_redesign | 3 | 0.3750 | 3 | 0.3750 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.3750 | 4 | 0.5000 |
| evidence_non_uptake | 1 | 0.1250 | 3 | 0.3750 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.1250 | 2 | 0.2500 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 0.1250 | 2 | 0.2500 |
| fixed_belief_trace | 8 | 1.0000 | 8 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.3750 | 4 | 0.5000 |
| evidence_non_uptake | 1 | 0.1250 | 3 | 0.3750 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.1250 | 2 | 0.2500 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 1 | 0.1250 | 2 | 0.2500 |
| fixed_belief_trace | 8 | 1.0000 | 8 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.3750 | 6 | 0.7500 |
| evidence_handling | 1 | 0.1250 | 5 | 0.6250 |
| experimental_strategy | 8 | 1.0000 | 8 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.3750 | 6 | 0.7500 |
| evidence_handling | 1 | 0.1250 | 5 | 0.6250 |
| experimental_strategy | 8 | 1.0000 | 8 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.3750 | 3 | 0.3750 |
| evidence_handling | 1 | 0.1250 | 2 | 0.2500 |
| experimental_strategy | 3 | 0.3750 | 4 | 0.5000 |

### gpt_oss_120b/retrosynthesis/level_2

- Traces: 9

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 2 | 0.2222 | 2 | 0.2222 |
| fixed_hypothesis_test_tuning | 1 | 0.1111 | 1 | 0.1111 |
| explore_then_test_transition | 0 | 0.0000 | 0 | 0.0000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 2 | 0.2222 | 2 | 0.2222 |
| convergent_multi_test_evidence | 1 | 0.1111 | 1 | 0.1111 |
| evidence_guided_test_redesign | 3 | 0.3333 | 3 | 0.3333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 4 | 0.4444 | 5 | 0.5556 |
| evidence_non_uptake | 4 | 0.4444 | 8 | 0.8889 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 2 | 0.2222 | 2 | 0.2222 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 1 | 0.1111 | 1 | 0.1111 |
| uninformative_test | 2 | 0.2222 | 8 | 0.8889 |
| fixed_belief_trace | 7 | 0.7778 | 7 | 0.7778 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.1111 | 1 | 0.1111 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 4 | 0.4444 | 5 | 0.5556 |
| evidence_non_uptake | 4 | 0.4444 | 8 | 0.8889 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 2 | 0.2222 | 2 | 0.2222 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 1 | 0.1111 | 1 | 0.1111 |
| uninformative_test | 2 | 0.2222 | 8 | 0.8889 |
| fixed_belief_trace | 7 | 0.7778 | 7 | 0.7778 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.1111 | 1 | 0.1111 |
| precommitted_test_plan | 1 | 0.1111 | 1 | 0.1111 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.4444 | 6 | 0.6667 |
| evidence_handling | 5 | 0.5556 | 16 | 1.7778 |
| experimental_strategy | 9 | 1.0000 | 10 | 1.1111 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.4444 | 6 | 0.6667 |
| evidence_handling | 5 | 0.5556 | 16 | 1.7778 |
| experimental_strategy | 9 | 1.0000 | 11 | 1.2222 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.4444 | 4 | 0.4444 |
| evidence_handling | 1 | 0.1111 | 1 | 0.1111 |
| experimental_strategy | 3 | 0.3333 | 4 | 0.4444 |

### gpt_oss_120b/retrosynthesis/level_3

- Traces: 7

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 2 | 0.2857 | 2 | 0.2857 |
| fixed_hypothesis_test_tuning | 2 | 0.2857 | 2 | 0.2857 |
| explore_then_test_transition | 2 | 0.2857 | 2 | 0.2857 |
| hypothesis_reranking | 1 | 0.1429 | 1 | 0.1429 |
| evidence_led_hypothesis_generation | 5 | 0.7143 | 5 | 0.7143 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 2 | 0.2857 | 2 | 0.2857 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 4 | 0.5714 | 4 | 0.5714 |
| evidence_non_uptake | 2 | 0.2857 | 5 | 0.7143 |
| unsupported_judgment | 1 | 0.1429 | 1 | 0.1429 |
| stalled_revision | 1 | 0.1429 | 1 | 0.1429 |
| contradiction_without_repair | 1 | 0.1429 | 1 | 0.1429 |
| premature_commitment | 1 | 0.1429 | 1 | 0.1429 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 5 | 0.7143 | 5 | 0.7143 |
| disconnected_evidence | 1 | 0.1429 | 1 | 0.1429 |
| one_sided_confirmation | 3 | 0.4286 | 3 | 0.4286 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 4 | 0.5714 | 4 | 0.5714 |
| evidence_non_uptake | 2 | 0.2857 | 5 | 0.7143 |
| unsupported_judgment | 1 | 0.1429 | 1 | 0.1429 |
| stalled_revision | 1 | 0.1429 | 1 | 0.1429 |
| contradiction_without_repair | 1 | 0.1429 | 1 | 0.1429 |
| premature_commitment | 1 | 0.1429 | 1 | 0.1429 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 5 | 0.7143 | 5 | 0.7143 |
| disconnected_evidence | 1 | 0.1429 | 1 | 0.1429 |
| one_sided_confirmation | 3 | 0.4286 | 3 | 0.4286 |
| precommitted_test_plan | 1 | 0.1429 | 1 | 0.1429 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 6 | 0.8571 | 8 | 1.1429 |
| evidence_handling | 2 | 0.2857 | 7 | 1.0000 |
| experimental_strategy | 6 | 0.8571 | 7 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 6 | 0.8571 | 8 | 1.1429 |
| evidence_handling | 2 | 0.2857 | 7 | 1.0000 |
| experimental_strategy | 7 | 1.0000 | 8 | 1.1429 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 7 | 1.0000 | 8 | 1.1429 |
| evidence_handling | 2 | 0.2857 | 2 | 0.2857 |
| experimental_strategy | 4 | 0.5714 | 4 | 0.5714 |

### gpt_oss_120b/spectra/level_1

- Traces: 16

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 3 | 0.1875 | 3 | 0.1875 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 3 | 0.1875 | 3 | 0.1875 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 11 | 0.6875 | 11 | 0.6875 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 3 | 0.1875 | 3 | 0.1875 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 10 | 0.6250 | 11 | 0.6875 |
| evidence_non_uptake | 2 | 0.1250 | 2 | 0.1250 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 1 | 0.0625 | 1 | 0.0625 |
| contradiction_without_repair | 1 | 0.0625 | 1 | 0.0625 |
| premature_commitment | 1 | 0.0625 | 1 | 0.0625 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 13 | 0.8125 | 13 | 0.8125 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0625 | 1 | 0.0625 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 10 | 0.6250 | 11 | 0.6875 |
| evidence_non_uptake | 2 | 0.1250 | 2 | 0.1250 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 1 | 0.0625 | 1 | 0.0625 |
| contradiction_without_repair | 1 | 0.0625 | 1 | 0.0625 |
| premature_commitment | 1 | 0.0625 | 1 | 0.0625 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 13 | 0.8125 | 13 | 0.8125 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0625 | 1 | 0.0625 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 0.6250 | 13 | 0.8125 |
| evidence_handling | 2 | 0.1250 | 2 | 0.1250 |
| experimental_strategy | 14 | 0.8750 | 15 | 0.9375 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 0.6250 | 13 | 0.8125 |
| evidence_handling | 2 | 0.1250 | 2 | 0.1250 |
| experimental_strategy | 14 | 0.8750 | 15 | 0.9375 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 11 | 0.6875 | 14 | 0.8750 |
| evidence_handling | 3 | 0.1875 | 3 | 0.1875 |
| experimental_strategy | 3 | 0.1875 | 3 | 0.1875 |

### gpt_oss_120b/spectra/level_2

- Traces: 14

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 1 | 0.0714 | 1 | 0.0714 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 2 | 0.1429 | 2 | 0.1429 |
| hypothesis_reranking | 1 | 0.0714 | 1 | 0.0714 |
| evidence_led_hypothesis_generation | 8 | 0.5714 | 8 | 0.5714 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 1 | 0.0714 | 1 | 0.0714 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 11 | 0.7857 | 12 | 0.8571 |
| evidence_non_uptake | 2 | 0.1429 | 5 | 0.3571 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 4 | 0.2857 | 4 | 0.2857 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 13 | 0.9286 | 13 | 0.9286 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 4 | 0.2857 | 4 | 0.2857 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 11 | 0.7857 | 12 | 0.8571 |
| evidence_non_uptake | 2 | 0.1429 | 5 | 0.3571 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 4 | 0.2857 | 4 | 0.2857 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 13 | 0.9286 | 13 | 0.9286 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 4 | 0.2857 | 4 | 0.2857 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 11 | 0.7857 | 16 | 1.1429 |
| evidence_handling | 2 | 0.1429 | 5 | 0.3571 |
| experimental_strategy | 13 | 0.9286 | 17 | 1.2143 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 11 | 0.7857 | 16 | 1.1429 |
| evidence_handling | 2 | 0.1429 | 5 | 0.3571 |
| experimental_strategy | 13 | 0.9286 | 17 | 1.2143 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 8 | 0.5714 | 10 | 0.7143 |
| evidence_handling | 2 | 0.1429 | 2 | 0.1429 |
| experimental_strategy | 1 | 0.0714 | 1 | 0.0714 |

### gpt_oss_120b/wetlab/level_1

- Traces: 5

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 3 | 0.6000 | 3 | 0.6000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 3 | 0.6000 | 3 | 0.6000 |
| hypothesis_reranking | 2 | 0.4000 | 2 | 0.4000 |
| evidence_led_hypothesis_generation | 5 | 1.0000 | 5 | 1.0000 |
| convergent_multi_test_evidence | 1 | 0.2000 | 1 | 0.2000 |
| evidence_guided_test_redesign | 3 | 0.6000 | 3 | 0.6000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 1.0000 | 14 | 2.8000 |
| evidence_non_uptake | 5 | 1.0000 | 28 | 5.6000 |
| unsupported_judgment | 2 | 0.4000 | 3 | 0.6000 |
| stalled_revision | 4 | 0.8000 | 6 | 1.2000 |
| contradiction_without_repair | 2 | 0.4000 | 2 | 0.4000 |
| premature_commitment | 1 | 0.2000 | 1 | 0.2000 |
| uninformative_test | 1 | 0.2000 | 1 | 0.2000 |
| fixed_belief_trace | 1 | 0.2000 | 1 | 0.2000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.2000 | 1 | 0.2000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 5 | 1.0000 | 14 | 2.8000 |
| evidence_non_uptake | 5 | 1.0000 | 28 | 5.6000 |
| unsupported_judgment | 2 | 0.4000 | 3 | 0.6000 |
| stalled_revision | 4 | 0.8000 | 6 | 1.2000 |
| contradiction_without_repair | 2 | 0.4000 | 2 | 0.4000 |
| premature_commitment | 1 | 0.2000 | 1 | 0.2000 |
| uninformative_test | 1 | 0.2000 | 1 | 0.2000 |
| fixed_belief_trace | 1 | 0.2000 | 1 | 0.2000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.2000 | 1 | 0.2000 |
| precommitted_test_plan | 1 | 0.2000 | 1 | 0.2000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 17 | 3.4000 |
| evidence_handling | 5 | 1.0000 | 32 | 6.4000 |
| experimental_strategy | 5 | 1.0000 | 8 | 1.6000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 17 | 3.4000 |
| evidence_handling | 5 | 1.0000 | 32 | 6.4000 |
| experimental_strategy | 5 | 1.0000 | 9 | 1.8000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 5 | 1.0000 | 10 | 2.0000 |
| evidence_handling | 3 | 0.6000 | 4 | 0.8000 |
| experimental_strategy | 3 | 0.6000 | 3 | 0.6000 |

### gpt_oss_120b/wetlab/level_2

- Traces: 5

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 2 | 0.4000 | 2 | 0.4000 |
| fixed_hypothesis_test_tuning | 1 | 0.2000 | 1 | 0.2000 |
| explore_then_test_transition | 4 | 0.8000 | 4 | 0.8000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 4 | 0.8000 | 4 | 0.8000 |
| convergent_multi_test_evidence | 1 | 0.2000 | 1 | 0.2000 |
| evidence_guided_test_redesign | 3 | 0.6000 | 3 | 0.6000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.4000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 23 | 4.6000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 2 | 0.4000 | 2 | 0.4000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.4000 | 2 | 0.4000 |
| fixed_belief_trace | 3 | 0.6000 | 3 | 0.6000 |
| disconnected_evidence | 1 | 0.2000 | 1 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 2 | 0.4000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 23 | 4.6000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 3 | 0.6000 | 3 | 0.6000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.4000 | 2 | 0.4000 |
| fixed_belief_trace | 3 | 0.6000 | 3 | 0.6000 |
| disconnected_evidence | 1 | 0.2000 | 1 | 0.2000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 5 | 1.0000 |
| evidence_handling | 5 | 1.0000 | 26 | 5.2000 |
| experimental_strategy | 3 | 0.6000 | 3 | 0.6000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 6 | 1.2000 |
| evidence_handling | 5 | 1.0000 | 26 | 5.2000 |
| experimental_strategy | 3 | 0.6000 | 3 | 0.6000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 4 | 0.8000 | 6 | 1.2000 |
| evidence_handling | 4 | 0.8000 | 5 | 1.0000 |
| experimental_strategy | 3 | 0.6000 | 4 | 0.8000 |

### gpt_oss_120b/wetlab/level_3

- Traces: 5

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 1 | 0.2000 | 1 | 0.2000 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 3 | 0.6000 | 3 | 0.6000 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 14 | 2.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2000 | 1 | 0.2000 |
| premature_commitment | 1 | 0.2000 | 1 | 0.2000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.2000 | 1 | 0.2000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 3 | 0.6000 | 3 | 0.6000 |
| evidence_non_uptake | 5 | 1.0000 | 14 | 2.8000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 1 | 0.2000 | 1 | 0.2000 |
| premature_commitment | 1 | 0.2000 | 1 | 0.2000 |
| uninformative_test | 0 | 0.0000 | 0 | 0.0000 |
| fixed_belief_trace | 5 | 1.0000 | 5 | 1.0000 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.2000 | 1 | 0.2000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 5 | 1.0000 |
| evidence_handling | 5 | 1.0000 | 14 | 2.8000 |
| experimental_strategy | 5 | 1.0000 | 6 | 1.2000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 5 | 1.0000 |
| evidence_handling | 5 | 1.0000 | 14 | 2.8000 |
| experimental_strategy | 5 | 1.0000 | 6 | 1.2000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 3 | 0.6000 | 3 | 0.6000 |
| evidence_handling | 1 | 0.2000 | 1 | 0.2000 |
| experimental_strategy | 0 | 0.0000 | 0 | 0.0000 |

## By model

### claude_sonnet_45

- Traces: 232

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 94 | 0.4052 | 94 | 0.4052 |
| fixed_hypothesis_test_tuning | 31 | 0.1336 | 31 | 0.1336 |
| explore_then_test_transition | 136 | 0.5862 | 136 | 0.5862 |
| hypothesis_reranking | 39 | 0.1681 | 39 | 0.1681 |
| evidence_led_hypothesis_generation | 146 | 0.6293 | 146 | 0.6293 |
| convergent_multi_test_evidence | 31 | 0.1336 | 31 | 0.1336 |
| evidence_guided_test_redesign | 109 | 0.4698 | 109 | 0.4698 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 112 | 0.4828 | 342 | 1.4741 |
| evidence_non_uptake | 188 | 0.8103 | 1227 | 5.2888 |
| unsupported_judgment | 37 | 0.1595 | 45 | 0.1940 |
| stalled_revision | 44 | 0.1897 | 70 | 0.3017 |
| contradiction_without_repair | 54 | 0.2328 | 96 | 0.4138 |
| premature_commitment | 11 | 0.0474 | 11 | 0.0474 |
| uninformative_test | 96 | 0.4138 | 253 | 1.0905 |
| fixed_belief_trace | 132 | 0.5690 | 132 | 0.5690 |
| disconnected_evidence | 69 | 0.2974 | 124 | 0.5345 |
| one_sided_confirmation | 13 | 0.0560 | 13 | 0.0560 |
| precommitted_test_plan | 5 | 0.0216 | 5 | 0.0216 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 112 | 0.4828 | 342 | 1.4741 |
| evidence_non_uptake | 188 | 0.8103 | 1227 | 5.2888 |
| unsupported_judgment | 37 | 0.1595 | 45 | 0.1940 |
| stalled_revision | 44 | 0.1897 | 70 | 0.3017 |
| contradiction_without_repair | 57 | 0.2457 | 122 | 0.5259 |
| premature_commitment | 11 | 0.0474 | 11 | 0.0474 |
| uninformative_test | 96 | 0.4138 | 253 | 1.0905 |
| fixed_belief_trace | 132 | 0.5690 | 132 | 0.5690 |
| disconnected_evidence | 69 | 0.2974 | 124 | 0.5345 |
| one_sided_confirmation | 13 | 0.0560 | 13 | 0.0560 |
| precommitted_test_plan | 4 | 0.0172 | 4 | 0.0172 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 129 | 0.5560 | 451 | 1.9440 |
| evidence_handling | 192 | 0.8276 | 1649 | 7.1078 |
| experimental_strategy | 177 | 0.7629 | 218 | 0.9397 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 131 | 0.5647 | 477 | 2.0560 |
| evidence_handling | 192 | 0.8276 | 1649 | 7.1078 |
| experimental_strategy | 180 | 0.7759 | 217 | 0.9353 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 166 | 0.7155 | 279 | 1.2026 |
| evidence_handling | 143 | 0.6164 | 167 | 0.7198 |
| experimental_strategy | 124 | 0.5345 | 140 | 0.6034 |

### gpt_4o

- Traces: 229

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 49 | 0.2140 | 49 | 0.2140 |
| fixed_hypothesis_test_tuning | 38 | 0.1659 | 38 | 0.1659 |
| explore_then_test_transition | 156 | 0.6812 | 156 | 0.6812 |
| hypothesis_reranking | 16 | 0.0699 | 16 | 0.0699 |
| evidence_led_hypothesis_generation | 179 | 0.7817 | 179 | 0.7817 |
| convergent_multi_test_evidence | 4 | 0.0175 | 4 | 0.0175 |
| evidence_guided_test_redesign | 141 | 0.6157 | 141 | 0.6157 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 100 | 0.4367 | 179 | 0.7817 |
| evidence_non_uptake | 176 | 0.7686 | 829 | 3.6201 |
| unsupported_judgment | 20 | 0.0873 | 28 | 0.1223 |
| stalled_revision | 18 | 0.0786 | 21 | 0.0917 |
| contradiction_without_repair | 55 | 0.2402 | 130 | 0.5677 |
| premature_commitment | 23 | 0.1004 | 23 | 0.1004 |
| uninformative_test | 78 | 0.3406 | 241 | 1.0524 |
| fixed_belief_trace | 179 | 0.7817 | 179 | 0.7817 |
| disconnected_evidence | 46 | 0.2009 | 73 | 0.3188 |
| one_sided_confirmation | 29 | 0.1266 | 31 | 0.1354 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 100 | 0.4367 | 179 | 0.7817 |
| evidence_non_uptake | 176 | 0.7686 | 829 | 3.6201 |
| unsupported_judgment | 20 | 0.0873 | 28 | 0.1223 |
| stalled_revision | 18 | 0.0786 | 21 | 0.0917 |
| contradiction_without_repair | 56 | 0.2445 | 144 | 0.6288 |
| premature_commitment | 23 | 0.1004 | 23 | 0.1004 |
| uninformative_test | 78 | 0.3406 | 241 | 1.0524 |
| fixed_belief_trace | 179 | 0.7817 | 179 | 0.7817 |
| disconnected_evidence | 46 | 0.2009 | 73 | 0.3188 |
| one_sided_confirmation | 29 | 0.1266 | 31 | 0.1354 |
| precommitted_test_plan | 12 | 0.0524 | 26 | 0.1135 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 125 | 0.5459 | 340 | 1.4847 |
| evidence_handling | 180 | 0.7860 | 1171 | 5.1135 |
| experimental_strategy | 200 | 0.8734 | 223 | 0.9738 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 125 | 0.5459 | 354 | 1.5459 |
| evidence_handling | 180 | 0.7860 | 1171 | 5.1135 |
| experimental_strategy | 207 | 0.9039 | 249 | 1.0873 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 183 | 0.7991 | 244 | 1.0655 |
| evidence_handling | 156 | 0.6812 | 160 | 0.6987 |
| experimental_strategy | 155 | 0.6769 | 179 | 0.7817 |

### gpt_oss_120b

- Traces: 158

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 20 | 0.1266 | 20 | 0.1266 |
| fixed_hypothesis_test_tuning | 6 | 0.0380 | 6 | 0.0380 |
| explore_then_test_transition | 25 | 0.1582 | 25 | 0.1582 |
| hypothesis_reranking | 6 | 0.0380 | 6 | 0.0380 |
| evidence_led_hypothesis_generation | 80 | 0.5063 | 80 | 0.5063 |
| convergent_multi_test_evidence | 9 | 0.0570 | 9 | 0.0570 |
| evidence_guided_test_redesign | 25 | 0.1582 | 25 | 0.1582 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 114 | 0.7215 | 145 | 0.9177 |
| evidence_non_uptake | 55 | 0.3481 | 139 | 0.8797 |
| unsupported_judgment | 33 | 0.2089 | 42 | 0.2658 |
| stalled_revision | 16 | 0.1013 | 19 | 0.1203 |
| contradiction_without_repair | 13 | 0.0823 | 14 | 0.0886 |
| premature_commitment | 17 | 0.1076 | 17 | 0.1076 |
| uninformative_test | 22 | 0.1392 | 138 | 0.8734 |
| fixed_belief_trace | 130 | 0.8228 | 130 | 0.8228 |
| disconnected_evidence | 3 | 0.0190 | 3 | 0.0190 |
| one_sided_confirmation | 14 | 0.0886 | 14 | 0.0886 |
| precommitted_test_plan | 2 | 0.0127 | 2 | 0.0127 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 114 | 0.7215 | 145 | 0.9177 |
| evidence_non_uptake | 55 | 0.3481 | 139 | 0.8797 |
| unsupported_judgment | 33 | 0.2089 | 42 | 0.2658 |
| stalled_revision | 16 | 0.1013 | 19 | 0.1203 |
| contradiction_without_repair | 15 | 0.0949 | 21 | 0.1329 |
| premature_commitment | 17 | 0.1076 | 17 | 0.1076 |
| uninformative_test | 22 | 0.1392 | 138 | 0.8734 |
| fixed_belief_trace | 130 | 0.8228 | 130 | 0.8228 |
| disconnected_evidence | 3 | 0.0190 | 3 | 0.0190 |
| one_sided_confirmation | 14 | 0.0886 | 14 | 0.0886 |
| precommitted_test_plan | 6 | 0.0380 | 8 | 0.0506 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 121 | 0.7658 | 173 | 1.0949 |
| evidence_handling | 83 | 0.5253 | 322 | 2.0380 |
| experimental_strategy | 147 | 0.9304 | 168 | 1.0633 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 121 | 0.7658 | 180 | 1.1392 |
| evidence_handling | 83 | 0.5253 | 322 | 2.0380 |
| experimental_strategy | 149 | 0.9430 | 174 | 1.1013 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 87 | 0.5506 | 106 | 0.6709 |
| evidence_handling | 28 | 0.1772 | 34 | 0.2152 |
| experimental_strategy | 28 | 0.1772 | 31 | 0.1962 |

## By env

### afm

- Traces: 51

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 12 | 0.2353 | 12 | 0.2353 |
| fixed_hypothesis_test_tuning | 9 | 0.1765 | 9 | 0.1765 |
| explore_then_test_transition | 30 | 0.5882 | 30 | 0.5882 |
| hypothesis_reranking | 3 | 0.0588 | 3 | 0.0588 |
| evidence_led_hypothesis_generation | 40 | 0.7843 | 40 | 0.7843 |
| convergent_multi_test_evidence | 6 | 0.1176 | 6 | 0.1176 |
| evidence_guided_test_redesign | 26 | 0.5098 | 26 | 0.5098 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 27 | 0.5294 | 37 | 0.7255 |
| evidence_non_uptake | 45 | 0.8824 | 220 | 4.3137 |
| unsupported_judgment | 6 | 0.1176 | 11 | 0.2157 |
| stalled_revision | 2 | 0.0392 | 2 | 0.0392 |
| contradiction_without_repair | 8 | 0.1569 | 13 | 0.2549 |
| premature_commitment | 8 | 0.1569 | 8 | 0.1569 |
| uninformative_test | 26 | 0.5098 | 154 | 3.0196 |
| fixed_belief_trace | 38 | 0.7451 | 38 | 0.7451 |
| disconnected_evidence | 3 | 0.0588 | 4 | 0.0784 |
| one_sided_confirmation | 7 | 0.1373 | 7 | 0.1373 |
| precommitted_test_plan | 2 | 0.0392 | 2 | 0.0392 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 27 | 0.5294 | 37 | 0.7255 |
| evidence_non_uptake | 45 | 0.8824 | 220 | 4.3137 |
| unsupported_judgment | 6 | 0.1176 | 11 | 0.2157 |
| stalled_revision | 2 | 0.0392 | 2 | 0.0392 |
| contradiction_without_repair | 9 | 0.1765 | 22 | 0.4314 |
| premature_commitment | 8 | 0.1569 | 8 | 0.1569 |
| uninformative_test | 26 | 0.5098 | 154 | 3.0196 |
| fixed_belief_trace | 38 | 0.7451 | 38 | 0.7451 |
| disconnected_evidence | 3 | 0.0588 | 4 | 0.0784 |
| one_sided_confirmation | 7 | 0.1373 | 7 | 0.1373 |
| precommitted_test_plan | 3 | 0.0588 | 5 | 0.0980 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 33 | 0.6471 | 57 | 1.1176 |
| evidence_handling | 50 | 0.9804 | 389 | 7.6275 |
| experimental_strategy | 41 | 0.8039 | 50 | 0.9804 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 33 | 0.6471 | 66 | 1.2941 |
| evidence_handling | 50 | 0.9804 | 389 | 7.6275 |
| experimental_strategy | 42 | 0.8235 | 53 | 1.0392 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 45 | 0.8824 | 55 | 1.0784 |
| evidence_handling | 33 | 0.6471 | 36 | 0.7059 |
| experimental_strategy | 30 | 0.5882 | 35 | 0.6863 |

### catalyst

- Traces: 53

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 1 | 0.0189 | 1 | 0.0189 |
| fixed_hypothesis_test_tuning | 0 | 0.0000 | 0 | 0.0000 |
| explore_then_test_transition | 2 | 0.0377 | 2 | 0.0377 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 12 | 0.2264 | 12 | 0.2264 |
| convergent_multi_test_evidence | 1 | 0.0189 | 1 | 0.0189 |
| evidence_guided_test_redesign | 4 | 0.0755 | 4 | 0.0755 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 16 | 0.3019 | 16 | 0.3019 |
| evidence_non_uptake | 40 | 0.7547 | 198 | 3.7358 |
| unsupported_judgment | 8 | 0.1509 | 10 | 0.1887 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 1 | 0.0189 | 1 | 0.0189 |
| uninformative_test | 8 | 0.1509 | 92 | 1.7358 |
| fixed_belief_trace | 52 | 0.9811 | 52 | 0.9811 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0189 | 1 | 0.0189 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 16 | 0.3019 | 16 | 0.3019 |
| evidence_non_uptake | 40 | 0.7547 | 198 | 3.7358 |
| unsupported_judgment | 8 | 0.1509 | 10 | 0.1887 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 1 | 0.0189 | 1 | 0.0189 |
| uninformative_test | 8 | 0.1509 | 92 | 1.7358 |
| fixed_belief_trace | 52 | 0.9811 | 52 | 0.9811 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 1 | 0.0189 | 1 | 0.0189 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 16 | 0.3019 | 17 | 0.3208 |
| evidence_handling | 40 | 0.7547 | 300 | 5.6604 |
| experimental_strategy | 52 | 0.9811 | 53 | 1.0000 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 16 | 0.3019 | 17 | 0.3208 |
| evidence_handling | 40 | 0.7547 | 300 | 5.6604 |
| experimental_strategy | 52 | 0.9811 | 53 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 12 | 0.2264 | 13 | 0.2453 |
| evidence_handling | 2 | 0.0377 | 3 | 0.0566 |
| experimental_strategy | 4 | 0.0755 | 4 | 0.0755 |

### md

- Traces: 58

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 9 | 0.1552 | 9 | 0.1552 |
| fixed_hypothesis_test_tuning | 1 | 0.0172 | 1 | 0.0172 |
| explore_then_test_transition | 22 | 0.3793 | 22 | 0.3793 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 43 | 0.7414 | 43 | 0.7414 |
| convergent_multi_test_evidence | 2 | 0.0345 | 2 | 0.0345 |
| evidence_guided_test_redesign | 15 | 0.2586 | 15 | 0.2586 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 48 | 0.8276 | 72 | 1.2414 |
| evidence_non_uptake | 44 | 0.7586 | 203 | 3.5000 |
| unsupported_judgment | 7 | 0.1207 | 8 | 0.1379 |
| stalled_revision | 8 | 0.1379 | 10 | 0.1724 |
| contradiction_without_repair | 13 | 0.2241 | 24 | 0.4138 |
| premature_commitment | 2 | 0.0345 | 2 | 0.0345 |
| uninformative_test | 24 | 0.4138 | 69 | 1.1897 |
| fixed_belief_trace | 46 | 0.7931 | 46 | 0.7931 |
| disconnected_evidence | 17 | 0.2931 | 33 | 0.5690 |
| one_sided_confirmation | 2 | 0.0345 | 2 | 0.0345 |
| precommitted_test_plan | 4 | 0.0690 | 4 | 0.0690 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 48 | 0.8276 | 72 | 1.2414 |
| evidence_non_uptake | 44 | 0.7586 | 203 | 3.5000 |
| unsupported_judgment | 7 | 0.1207 | 8 | 0.1379 |
| stalled_revision | 8 | 0.1379 | 10 | 0.1724 |
| contradiction_without_repair | 14 | 0.2414 | 28 | 0.4828 |
| premature_commitment | 2 | 0.0345 | 2 | 0.0345 |
| uninformative_test | 24 | 0.4138 | 69 | 1.1897 |
| fixed_belief_trace | 46 | 0.7931 | 46 | 0.7931 |
| disconnected_evidence | 17 | 0.2931 | 33 | 0.5690 |
| one_sided_confirmation | 2 | 0.0345 | 2 | 0.0345 |
| precommitted_test_plan | 2 | 0.0345 | 2 | 0.0345 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 50 | 0.8621 | 98 | 1.6897 |
| evidence_handling | 45 | 0.7759 | 313 | 5.3966 |
| experimental_strategy | 54 | 0.9310 | 62 | 1.0690 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 50 | 0.8621 | 102 | 1.7586 |
| evidence_handling | 45 | 0.7759 | 313 | 5.3966 |
| experimental_strategy | 56 | 0.9655 | 60 | 1.0345 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 44 | 0.7586 | 52 | 0.8966 |
| evidence_handling | 22 | 0.3793 | 24 | 0.4138 |
| experimental_strategy | 15 | 0.2586 | 16 | 0.2759 |

### ml

- Traces: 30

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 0 | 0.0000 | 0 | 0.0000 |
| fixed_hypothesis_test_tuning | 4 | 0.1333 | 4 | 0.1333 |
| explore_then_test_transition | 1 | 0.0333 | 1 | 0.0333 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 1 | 0.0333 | 1 | 0.0333 |
| convergent_multi_test_evidence | 0 | 0.0000 | 0 | 0.0000 |
| evidence_guided_test_redesign | 7 | 0.2333 | 7 | 0.2333 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 28 | 0.9333 | 111 | 3.7000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.0667 | 6 | 0.2000 |
| fixed_belief_trace | 30 | 1.0000 | 30 | 1.0000 |
| disconnected_evidence | 2 | 0.0667 | 3 | 0.1000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 1 | 0.0333 | 1 | 0.0333 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 0 | 0.0000 | 0 | 0.0000 |
| evidence_non_uptake | 28 | 0.9333 | 111 | 3.7000 |
| unsupported_judgment | 0 | 0.0000 | 0 | 0.0000 |
| stalled_revision | 0 | 0.0000 | 0 | 0.0000 |
| contradiction_without_repair | 0 | 0.0000 | 0 | 0.0000 |
| premature_commitment | 0 | 0.0000 | 0 | 0.0000 |
| uninformative_test | 2 | 0.0667 | 6 | 0.2000 |
| fixed_belief_trace | 30 | 1.0000 | 30 | 1.0000 |
| disconnected_evidence | 2 | 0.0667 | 3 | 0.1000 |
| one_sided_confirmation | 0 | 0.0000 | 0 | 0.0000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 28 | 0.9333 | 120 | 4.0000 |
| experimental_strategy | 30 | 1.0000 | 31 | 1.0333 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 0 | 0.0000 | 0 | 0.0000 |
| evidence_handling | 28 | 0.9333 | 120 | 4.0000 |
| experimental_strategy | 30 | 1.0000 | 30 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 1 | 0.0333 | 1 | 0.0333 |
| evidence_handling | 1 | 0.0333 | 1 | 0.0333 |
| experimental_strategy | 7 | 0.2333 | 11 | 0.3667 |

### resistor

- Traces: 88

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 24 | 0.2727 | 24 | 0.2727 |
| fixed_hypothesis_test_tuning | 5 | 0.0568 | 5 | 0.0568 |
| explore_then_test_transition | 22 | 0.2500 | 22 | 0.2500 |
| hypothesis_reranking | 15 | 0.1705 | 15 | 0.1705 |
| evidence_led_hypothesis_generation | 27 | 0.3068 | 27 | 0.3068 |
| convergent_multi_test_evidence | 2 | 0.0227 | 2 | 0.0227 |
| evidence_guided_test_redesign | 37 | 0.4205 | 37 | 0.4205 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 49 | 0.5568 | 89 | 1.0114 |
| evidence_non_uptake | 32 | 0.3636 | 89 | 1.0114 |
| unsupported_judgment | 39 | 0.4432 | 48 | 0.5455 |
| stalled_revision | 11 | 0.1250 | 20 | 0.2273 |
| contradiction_without_repair | 24 | 0.2727 | 68 | 0.7727 |
| premature_commitment | 7 | 0.0795 | 7 | 0.0795 |
| uninformative_test | 20 | 0.2273 | 51 | 0.5795 |
| fixed_belief_trace | 58 | 0.6591 | 58 | 0.6591 |
| disconnected_evidence | 15 | 0.1705 | 34 | 0.3864 |
| one_sided_confirmation | 4 | 0.0455 | 4 | 0.0455 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 49 | 0.5568 | 89 | 1.0114 |
| evidence_non_uptake | 32 | 0.3636 | 89 | 1.0114 |
| unsupported_judgment | 39 | 0.4432 | 48 | 0.5455 |
| stalled_revision | 11 | 0.1250 | 20 | 0.2273 |
| contradiction_without_repair | 24 | 0.2727 | 82 | 0.9318 |
| premature_commitment | 7 | 0.0795 | 7 | 0.0795 |
| uninformative_test | 20 | 0.2273 | 51 | 0.5795 |
| fixed_belief_trace | 58 | 0.6591 | 58 | 0.6591 |
| disconnected_evidence | 15 | 0.1705 | 34 | 0.3864 |
| one_sided_confirmation | 4 | 0.0455 | 4 | 0.0455 |
| precommitted_test_plan | 9 | 0.1023 | 21 | 0.2386 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 64 | 0.7273 | 161 | 1.8295 |
| evidence_handling | 58 | 0.6591 | 222 | 2.5227 |
| experimental_strategy | 70 | 0.7955 | 85 | 0.9659 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 64 | 0.7273 | 175 | 1.9886 |
| evidence_handling | 58 | 0.6591 | 222 | 2.5227 |
| experimental_strategy | 75 | 0.8523 | 106 | 1.2045 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 43 | 0.4886 | 66 | 0.7500 |
| evidence_handling | 24 | 0.2727 | 24 | 0.2727 |
| experimental_strategy | 38 | 0.4318 | 42 | 0.4773 |

### retrosynthesis

- Traces: 114

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 14 | 0.1228 | 14 | 0.1228 |
| fixed_hypothesis_test_tuning | 34 | 0.2982 | 34 | 0.2982 |
| explore_then_test_transition | 60 | 0.5263 | 60 | 0.5263 |
| hypothesis_reranking | 6 | 0.0526 | 6 | 0.0526 |
| evidence_led_hypothesis_generation | 71 | 0.6228 | 71 | 0.6228 |
| convergent_multi_test_evidence | 10 | 0.0877 | 10 | 0.0877 |
| evidence_guided_test_redesign | 59 | 0.5175 | 59 | 0.5175 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 37 | 0.3246 | 58 | 0.5088 |
| evidence_non_uptake | 75 | 0.6579 | 496 | 4.3509 |
| unsupported_judgment | 7 | 0.0614 | 9 | 0.0789 |
| stalled_revision | 4 | 0.0351 | 4 | 0.0351 |
| contradiction_without_repair | 17 | 0.1491 | 38 | 0.3333 |
| premature_commitment | 3 | 0.0263 | 3 | 0.0263 |
| uninformative_test | 36 | 0.3158 | 90 | 0.7895 |
| fixed_belief_trace | 100 | 0.8772 | 100 | 0.8772 |
| disconnected_evidence | 23 | 0.2018 | 38 | 0.3333 |
| one_sided_confirmation | 6 | 0.0526 | 8 | 0.0702 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 37 | 0.3246 | 58 | 0.5088 |
| evidence_non_uptake | 75 | 0.6579 | 496 | 4.3509 |
| unsupported_judgment | 7 | 0.0614 | 9 | 0.0789 |
| stalled_revision | 4 | 0.0351 | 4 | 0.0351 |
| contradiction_without_repair | 18 | 0.1579 | 41 | 0.3596 |
| premature_commitment | 3 | 0.0263 | 3 | 0.0263 |
| uninformative_test | 36 | 0.3158 | 90 | 0.7895 |
| fixed_belief_trace | 100 | 0.8772 | 100 | 0.8772 |
| disconnected_evidence | 23 | 0.2018 | 38 | 0.3333 |
| one_sided_confirmation | 6 | 0.0526 | 8 | 0.0702 |
| precommitted_test_plan | 5 | 0.0439 | 7 | 0.0614 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 47 | 0.4123 | 104 | 0.9123 |
| evidence_handling | 78 | 0.6842 | 633 | 5.5526 |
| experimental_strategy | 104 | 0.9123 | 107 | 0.9386 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 48 | 0.4211 | 107 | 0.9386 |
| evidence_handling | 78 | 0.6842 | 633 | 5.5526 |
| experimental_strategy | 106 | 0.9298 | 114 | 1.0000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 80 | 0.7018 | 91 | 0.7982 |
| evidence_handling | 65 | 0.5702 | 70 | 0.6140 |
| experimental_strategy | 73 | 0.6404 | 93 | 0.8158 |

### spectra

- Traces: 120

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 45 | 0.3750 | 45 | 0.3750 |
| fixed_hypothesis_test_tuning | 21 | 0.1750 | 21 | 0.1750 |
| explore_then_test_transition | 88 | 0.7333 | 88 | 0.7333 |
| hypothesis_reranking | 19 | 0.1583 | 19 | 0.1583 |
| evidence_led_hypothesis_generation | 109 | 0.9083 | 109 | 0.9083 |
| convergent_multi_test_evidence | 18 | 0.1500 | 18 | 0.1500 |
| evidence_guided_test_redesign | 56 | 0.4667 | 56 | 0.4667 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 55 | 0.4583 | 85 | 0.7083 |
| evidence_non_uptake | 50 | 0.4167 | 128 | 1.0667 |
| unsupported_judgment | 9 | 0.0750 | 12 | 0.1000 |
| stalled_revision | 9 | 0.0750 | 11 | 0.0917 |
| contradiction_without_repair | 23 | 0.1917 | 39 | 0.3250 |
| premature_commitment | 9 | 0.0750 | 9 | 0.0750 |
| uninformative_test | 18 | 0.1500 | 26 | 0.2167 |
| fixed_belief_trace | 74 | 0.6167 | 74 | 0.6167 |
| disconnected_evidence | 5 | 0.0417 | 5 | 0.0417 |
| one_sided_confirmation | 12 | 0.1000 | 12 | 0.1000 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 55 | 0.4583 | 85 | 0.7083 |
| evidence_non_uptake | 50 | 0.4167 | 128 | 1.0667 |
| unsupported_judgment | 9 | 0.0750 | 12 | 0.1000 |
| stalled_revision | 9 | 0.0750 | 11 | 0.0917 |
| contradiction_without_repair | 24 | 0.2000 | 49 | 0.4083 |
| premature_commitment | 9 | 0.0750 | 9 | 0.0750 |
| uninformative_test | 18 | 0.1500 | 26 | 0.2167 |
| fixed_belief_trace | 74 | 0.6167 | 74 | 0.6167 |
| disconnected_evidence | 5 | 0.0417 | 5 | 0.0417 |
| one_sided_confirmation | 12 | 0.1000 | 12 | 0.1000 |
| precommitted_test_plan | 2 | 0.0167 | 2 | 0.0167 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 65 | 0.5417 | 136 | 1.1333 |
| evidence_handling | 51 | 0.4250 | 171 | 1.4250 |
| experimental_strategy | 83 | 0.6917 | 94 | 0.7833 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 66 | 0.5500 | 146 | 1.2167 |
| evidence_handling | 51 | 0.4250 | 171 | 1.4250 |
| experimental_strategy | 85 | 0.7083 | 96 | 0.8000 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 109 | 0.9083 | 173 | 1.4417 |
| evidence_handling | 88 | 0.7333 | 106 | 0.8833 |
| experimental_strategy | 69 | 0.5750 | 77 | 0.6417 |

### wetlab

- Traces: 105

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 58 | 0.5524 | 58 | 0.5524 |
| fixed_hypothesis_test_tuning | 1 | 0.0095 | 1 | 0.0095 |
| explore_then_test_transition | 92 | 0.8762 | 92 | 0.8762 |
| hypothesis_reranking | 18 | 0.1714 | 18 | 0.1714 |
| evidence_led_hypothesis_generation | 102 | 0.9714 | 102 | 0.9714 |
| convergent_multi_test_evidence | 5 | 0.0476 | 5 | 0.0476 |
| evidence_guided_test_redesign | 71 | 0.6762 | 71 | 0.6762 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 94 | 0.8952 | 309 | 2.9429 |
| evidence_non_uptake | 105 | 1.0000 | 750 | 7.1429 |
| unsupported_judgment | 14 | 0.1333 | 17 | 0.1619 |
| stalled_revision | 44 | 0.4190 | 63 | 0.6000 |
| contradiction_without_repair | 37 | 0.3524 | 58 | 0.5524 |
| premature_commitment | 21 | 0.2000 | 21 | 0.2000 |
| uninformative_test | 62 | 0.5905 | 144 | 1.3714 |
| fixed_belief_trace | 43 | 0.4095 | 43 | 0.4095 |
| disconnected_evidence | 53 | 0.5048 | 83 | 0.7905 |
| one_sided_confirmation | 24 | 0.2286 | 24 | 0.2286 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 94 | 0.8952 | 309 | 2.9429 |
| evidence_non_uptake | 105 | 1.0000 | 750 | 7.1429 |
| unsupported_judgment | 14 | 0.1333 | 17 | 0.1619 |
| stalled_revision | 44 | 0.4190 | 63 | 0.6000 |
| contradiction_without_repair | 39 | 0.3714 | 65 | 0.6190 |
| premature_commitment | 21 | 0.2000 | 21 | 0.2000 |
| uninformative_test | 62 | 0.5905 | 144 | 1.3714 |
| fixed_belief_trace | 43 | 0.4095 | 43 | 0.4095 |
| disconnected_evidence | 53 | 0.5048 | 83 | 0.7905 |
| one_sided_confirmation | 24 | 0.2286 | 24 | 0.2286 |
| precommitted_test_plan | 1 | 0.0095 | 1 | 0.0095 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 100 | 0.9524 | 391 | 3.7238 |
| evidence_handling | 105 | 1.0000 | 994 | 9.4667 |
| experimental_strategy | 90 | 0.8571 | 127 | 1.2095 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 100 | 0.9524 | 398 | 3.7905 |
| evidence_handling | 105 | 1.0000 | 994 | 9.4667 |
| experimental_strategy | 90 | 0.8571 | 128 | 1.2190 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 102 | 0.9714 | 178 | 1.6952 |
| evidence_handling | 92 | 0.8762 | 97 | 0.9238 |
| experimental_strategy | 71 | 0.6762 | 72 | 0.6857 |

## By level

### level_1

- Traces: 352

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 79 | 0.2244 | 79 | 0.2244 |
| fixed_hypothesis_test_tuning | 40 | 0.1136 | 40 | 0.1136 |
| explore_then_test_transition | 137 | 0.3892 | 137 | 0.3892 |
| hypothesis_reranking | 30 | 0.0852 | 30 | 0.0852 |
| evidence_led_hypothesis_generation | 189 | 0.5369 | 189 | 0.5369 |
| convergent_multi_test_evidence | 17 | 0.0483 | 17 | 0.0483 |
| evidence_guided_test_redesign | 130 | 0.3693 | 130 | 0.3693 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 170 | 0.4830 | 301 | 0.8551 |
| evidence_non_uptake | 210 | 0.5966 | 842 | 2.3920 |
| unsupported_judgment | 61 | 0.1733 | 78 | 0.2216 |
| stalled_revision | 43 | 0.1222 | 61 | 0.1733 |
| contradiction_without_repair | 62 | 0.1761 | 137 | 0.3892 |
| premature_commitment | 20 | 0.0568 | 20 | 0.0568 |
| uninformative_test | 85 | 0.2415 | 300 | 0.8523 |
| fixed_belief_trace | 261 | 0.7415 | 261 | 0.7415 |
| disconnected_evidence | 49 | 0.1392 | 95 | 0.2699 |
| one_sided_confirmation | 20 | 0.0568 | 20 | 0.0568 |
| precommitted_test_plan | 4 | 0.0114 | 4 | 0.0114 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 170 | 0.4830 | 301 | 0.8551 |
| evidence_non_uptake | 210 | 0.5966 | 842 | 2.3920 |
| unsupported_judgment | 61 | 0.1733 | 78 | 0.2216 |
| stalled_revision | 43 | 0.1222 | 61 | 0.1733 |
| contradiction_without_repair | 64 | 0.1818 | 160 | 0.4545 |
| premature_commitment | 20 | 0.0568 | 20 | 0.0568 |
| uninformative_test | 85 | 0.2415 | 300 | 0.8523 |
| fixed_belief_trace | 261 | 0.7415 | 261 | 0.7415 |
| disconnected_evidence | 49 | 0.1392 | 95 | 0.2699 |
| one_sided_confirmation | 20 | 0.0568 | 20 | 0.0568 |
| precommitted_test_plan | 13 | 0.0369 | 25 | 0.0710 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 193 | 0.5483 | 458 | 1.3011 |
| evidence_handling | 240 | 0.6818 | 1315 | 3.7358 |
| experimental_strategy | 306 | 0.8693 | 346 | 0.9830 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 193 | 0.5483 | 481 | 1.3665 |
| evidence_handling | 240 | 0.6818 | 1315 | 3.7358 |
| experimental_strategy | 314 | 0.8920 | 367 | 1.0426 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 206 | 0.5852 | 298 | 0.8466 |
| evidence_handling | 139 | 0.3949 | 154 | 0.4375 |
| experimental_strategy | 150 | 0.4261 | 170 | 0.4830 |

### level_2

- Traces: 172

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 59 | 0.3430 | 59 | 0.3430 |
| fixed_hypothesis_test_tuning | 24 | 0.1395 | 24 | 0.1395 |
| explore_then_test_transition | 107 | 0.6221 | 107 | 0.6221 |
| hypothesis_reranking | 18 | 0.1047 | 18 | 0.1047 |
| evidence_led_hypothesis_generation | 134 | 0.7791 | 134 | 0.7791 |
| convergent_multi_test_evidence | 22 | 0.1279 | 22 | 0.1279 |
| evidence_guided_test_redesign | 92 | 0.5349 | 92 | 0.5349 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 97 | 0.5640 | 209 | 1.2151 |
| evidence_non_uptake | 122 | 0.7093 | 660 | 3.8372 |
| unsupported_judgment | 15 | 0.0872 | 21 | 0.1221 |
| stalled_revision | 21 | 0.1221 | 31 | 0.1802 |
| contradiction_without_repair | 33 | 0.1919 | 49 | 0.2849 |
| premature_commitment | 13 | 0.0756 | 13 | 0.0756 |
| uninformative_test | 58 | 0.3372 | 162 | 0.9419 |
| fixed_belief_trace | 110 | 0.6395 | 110 | 0.6395 |
| disconnected_evidence | 43 | 0.2500 | 68 | 0.3953 |
| one_sided_confirmation | 14 | 0.0814 | 14 | 0.0814 |
| precommitted_test_plan | 3 | 0.0174 | 3 | 0.0174 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 97 | 0.5640 | 209 | 1.2151 |
| evidence_non_uptake | 122 | 0.7093 | 660 | 3.8372 |
| unsupported_judgment | 15 | 0.0872 | 21 | 0.1221 |
| stalled_revision | 21 | 0.1221 | 31 | 0.1802 |
| contradiction_without_repair | 37 | 0.2151 | 71 | 0.4128 |
| premature_commitment | 13 | 0.0756 | 13 | 0.0756 |
| uninformative_test | 58 | 0.3372 | 162 | 0.9419 |
| fixed_belief_trace | 110 | 0.6395 | 110 | 0.6395 |
| disconnected_evidence | 43 | 0.2500 | 68 | 0.3953 |
| one_sided_confirmation | 14 | 0.0814 | 14 | 0.0814 |
| precommitted_test_plan | 3 | 0.0174 | 4 | 0.0233 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 110 | 0.6395 | 272 | 1.5814 |
| evidence_handling | 126 | 0.7326 | 911 | 5.2965 |
| experimental_strategy | 133 | 0.7733 | 157 | 0.9128 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 112 | 0.6512 | 294 | 1.7093 |
| evidence_handling | 126 | 0.7326 | 911 | 5.2965 |
| experimental_strategy | 134 | 0.7791 | 158 | 0.9186 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 143 | 0.8314 | 211 | 1.2267 |
| evidence_handling | 114 | 0.6628 | 129 | 0.7500 |
| experimental_strategy | 98 | 0.5698 | 116 | 0.6744 |

### level_3

- Traces: 83

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 23 | 0.2771 | 23 | 0.2771 |
| fixed_hypothesis_test_tuning | 10 | 0.1205 | 10 | 0.1205 |
| explore_then_test_transition | 66 | 0.7952 | 66 | 0.7952 |
| hypothesis_reranking | 13 | 0.1566 | 13 | 0.1566 |
| evidence_led_hypothesis_generation | 72 | 0.8675 | 72 | 0.8675 |
| convergent_multi_test_evidence | 3 | 0.0361 | 3 | 0.0361 |
| evidence_guided_test_redesign | 48 | 0.5783 | 48 | 0.5783 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 51 | 0.6145 | 147 | 1.7711 |
| evidence_non_uptake | 76 | 0.9157 | 637 | 7.6747 |
| unsupported_judgment | 13 | 0.1566 | 15 | 0.1807 |
| stalled_revision | 13 | 0.1566 | 17 | 0.2048 |
| contradiction_without_repair | 25 | 0.3012 | 52 | 0.6265 |
| premature_commitment | 14 | 0.1687 | 14 | 0.1687 |
| uninformative_test | 44 | 0.5301 | 110 | 1.3253 |
| fixed_belief_trace | 60 | 0.7229 | 60 | 0.7229 |
| disconnected_evidence | 26 | 0.3133 | 37 | 0.4458 |
| one_sided_confirmation | 19 | 0.2289 | 21 | 0.2530 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 51 | 0.6145 | 147 | 1.7711 |
| evidence_non_uptake | 76 | 0.9157 | 637 | 7.6747 |
| unsupported_judgment | 13 | 0.1566 | 15 | 0.1807 |
| stalled_revision | 13 | 0.1566 | 17 | 0.2048 |
| contradiction_without_repair | 25 | 0.3012 | 54 | 0.6506 |
| premature_commitment | 14 | 0.1687 | 14 | 0.1687 |
| uninformative_test | 44 | 0.5301 | 110 | 1.3253 |
| fixed_belief_trace | 60 | 0.7229 | 60 | 0.7229 |
| disconnected_evidence | 26 | 0.3133 | 37 | 0.4458 |
| one_sided_confirmation | 19 | 0.2289 | 21 | 0.2530 |
| precommitted_test_plan | 6 | 0.0723 | 9 | 0.1084 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 63 | 0.7590 | 220 | 2.6506 |
| evidence_handling | 77 | 0.9277 | 799 | 9.6265 |
| experimental_strategy | 74 | 0.8916 | 91 | 1.0964 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 63 | 0.7590 | 222 | 2.6747 |
| evidence_handling | 77 | 0.9277 | 799 | 9.6265 |
| experimental_strategy | 77 | 0.9277 | 100 | 1.2048 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 77 | 0.9277 | 108 | 1.3012 |
| evidence_handling | 66 | 0.7952 | 69 | 0.8313 |
| experimental_strategy | 54 | 0.6506 | 58 | 0.6988 |

### level_4

- Traces: 12

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 2 | 0.1667 | 2 | 0.1667 |
| fixed_hypothesis_test_tuning | 1 | 0.0833 | 1 | 0.0833 |
| explore_then_test_transition | 7 | 0.5833 | 7 | 0.5833 |
| hypothesis_reranking | 0 | 0.0000 | 0 | 0.0000 |
| evidence_led_hypothesis_generation | 10 | 0.8333 | 10 | 0.8333 |
| convergent_multi_test_evidence | 2 | 0.1667 | 2 | 0.1667 |
| evidence_guided_test_redesign | 5 | 0.4167 | 5 | 0.4167 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 8 | 0.6667 | 9 | 0.7500 |
| evidence_non_uptake | 11 | 0.9167 | 56 | 4.6667 |
| unsupported_judgment | 1 | 0.0833 | 1 | 0.0833 |
| stalled_revision | 1 | 0.0833 | 1 | 0.0833 |
| contradiction_without_repair | 2 | 0.1667 | 2 | 0.1667 |
| premature_commitment | 4 | 0.3333 | 4 | 0.3333 |
| uninformative_test | 9 | 0.7500 | 60 | 5.0000 |
| fixed_belief_trace | 10 | 0.8333 | 10 | 0.8333 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 3 | 0.2500 | 3 | 0.2500 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 8 | 0.6667 | 9 | 0.7500 |
| evidence_non_uptake | 11 | 0.9167 | 56 | 4.6667 |
| unsupported_judgment | 1 | 0.0833 | 1 | 0.0833 |
| stalled_revision | 1 | 0.0833 | 1 | 0.0833 |
| contradiction_without_repair | 2 | 0.1667 | 2 | 0.1667 |
| premature_commitment | 4 | 0.3333 | 4 | 0.3333 |
| uninformative_test | 9 | 0.7500 | 60 | 5.0000 |
| fixed_belief_trace | 10 | 0.8333 | 10 | 0.8333 |
| disconnected_evidence | 0 | 0.0000 | 0 | 0.0000 |
| one_sided_confirmation | 3 | 0.2500 | 3 | 0.2500 |
| precommitted_test_plan | 0 | 0.0000 | 0 | 0.0000 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.7500 | 14 | 1.1667 |
| evidence_handling | 12 | 1.0000 | 117 | 9.7500 |
| experimental_strategy | 11 | 0.9167 | 15 | 1.2500 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 9 | 0.7500 | 14 | 1.1667 |
| evidence_handling | 12 | 1.0000 | 117 | 9.7500 |
| experimental_strategy | 11 | 0.9167 | 15 | 1.2500 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 10 | 0.8333 | 12 | 1.0000 |
| evidence_handling | 8 | 0.6667 | 9 | 0.7500 |
| experimental_strategy | 5 | 0.4167 | 6 | 0.5000 |

## Overall

### overall

- Traces: 619

#### Global subgraph presence

| subgraph | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| refutation_driven_belief_revision | 163 | 0.2633 | 163 | 0.2633 |
| fixed_hypothesis_test_tuning | 75 | 0.1212 | 75 | 0.1212 |
| explore_then_test_transition | 317 | 0.5121 | 317 | 0.5121 |
| hypothesis_reranking | 61 | 0.0985 | 61 | 0.0985 |
| evidence_led_hypothesis_generation | 405 | 0.6543 | 405 | 0.6543 |
| convergent_multi_test_evidence | 44 | 0.0711 | 44 | 0.0711 |
| evidence_guided_test_redesign | 275 | 0.4443 | 275 | 0.4443 |

#### Anti-pattern presence (global)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 326 | 0.5267 | 666 | 1.0759 |
| evidence_non_uptake | 419 | 0.6769 | 2195 | 3.5460 |
| unsupported_judgment | 90 | 0.1454 | 115 | 0.1858 |
| stalled_revision | 78 | 0.1260 | 110 | 0.1777 |
| contradiction_without_repair | 122 | 0.1971 | 240 | 0.3877 |
| premature_commitment | 51 | 0.0824 | 51 | 0.0824 |
| uninformative_test | 196 | 0.3166 | 632 | 1.0210 |
| fixed_belief_trace | 441 | 0.7124 | 441 | 0.7124 |
| disconnected_evidence | 118 | 0.1906 | 200 | 0.3231 |
| one_sided_confirmation | 56 | 0.0905 | 58 | 0.0937 |
| precommitted_test_plan | 7 | 0.0113 | 7 | 0.0113 |

#### Anti-pattern presence (local)

| anti-pattern | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| untested_claim | 326 | 0.5267 | 666 | 1.0759 |
| evidence_non_uptake | 419 | 0.6769 | 2195 | 3.5460 |
| unsupported_judgment | 90 | 0.1454 | 115 | 0.1858 |
| stalled_revision | 78 | 0.1260 | 110 | 0.1777 |
| contradiction_without_repair | 128 | 0.2068 | 287 | 0.4637 |
| premature_commitment | 51 | 0.0824 | 51 | 0.0824 |
| uninformative_test | 196 | 0.3166 | 632 | 1.0210 |
| fixed_belief_trace | 441 | 0.7124 | 441 | 0.7124 |
| disconnected_evidence | 118 | 0.1906 | 200 | 0.3231 |
| one_sided_confirmation | 56 | 0.0905 | 58 | 0.0937 |
| precommitted_test_plan | 22 | 0.0355 | 38 | 0.0614 |

#### Anti-pattern family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 375 | 0.6058 | 964 | 1.5574 |
| evidence_handling | 455 | 0.7351 | 3142 | 5.0759 |
| experimental_strategy | 524 | 0.8465 | 609 | 0.9838 |

#### Anti-pattern family presence (local)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 377 | 0.6090 | 1011 | 1.6333 |
| evidence_handling | 455 | 0.7351 | 3142 | 5.0759 |
| experimental_strategy | 536 | 0.8659 | 640 | 1.0339 |

#### Subgraph family presence (global)

| family | count | fraction | raw_total | raw_mean |
| --- | ---: | ---: | ---: | ---: |
| hypothesis_generation | 436 | 0.7044 | 629 | 1.0162 |
| evidence_handling | 327 | 0.5283 | 361 | 0.5832 |
| experimental_strategy | 307 | 0.4960 | 350 | 0.5654 |
