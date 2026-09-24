# Level-specific blank submission templates

Each JSON bundle contains a blank `manifest` and linked JSON templates under
`files`. Level 1 bundles contain only the evidence requested for preparation;
the existing top-level bundles describe the full Level 2 workflow. The
environment selects the matching level, shows its manifest in the task prompt,
and writes its templates plus a short `README.md` into
`submission_examples/` before capturing the initial workspace.

| Task | Level 1 | Level 2 |
| --- | --- | --- |
| 1: Silicon diffusion | [task_1.json](level_1/task_1.json) | [task_1.json](task_1.json) |
| 2: Glass transition | [task_2.json](level_1/task_2.json) | [task_2.json](task_2.json) |
| 3: Silver fine-tuning | [task_3.json](level_1/task_3.json) | [task_3.json](task_3.json) |
| 4: Palladium phonons | [task_4.json](level_1/task_4.json) | [task_4.json](task_4.json) |
| 5: Strained silicon phonons | [task_5.json](level_1/task_5.json) | [task_5.json](task_5.json) |
| 6: Aluminum VDOS | [task_6.json](level_1/task_6.json) | [task_6.json](task_6.json) |
| 7: Thermal expansion | [task_7.json](level_1/task_7.json) | [task_7.json](task_7.json) |
| 8: Silicon regression | [task_8.json](level_1/task_8.json) | [task_8.json](task_8.json) |
| 9: Copper regression | [task_9.json](level_1/task_9.json) | [task_9.json](task_9.json) |
| 10: Aluminum heat capacity | [task_10.json](level_1/task_10.json) | [task_10.json](task_10.json) |

Measurements and selected values remain blank. Units, task-fixed settings, and
named example methods identify the format. Keep these bundles aligned with the
corresponding scorer when changing artifact fields; evaluator policies belong in the
[scoring documentation](../workflow_scoring/README.md).
