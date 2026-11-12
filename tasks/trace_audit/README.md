# Trace audit pilots

Both environments ask auditors to submit a short JSON summary and selected
findings. The shared issue definitions come from
`corral.epistemic_analysis.epistemic_prompts.TRACE_CHECK_DEFINITIONS`.

## 18-trace pilot

Nine incident traces and nine controls are presented in two conditions:

- Level 1 shows source-model folders.
- Level 2 masks source labels and uses neutral folders.

Run with `trace_audit_pilot` and `level` set to `1` or `2`.

## Resistor folder-label pilot

Six incidents and three controls are shown under model-name folders. The same
traces appear in three counterbalanced folder assignments (`a`, `b`, `c`).
Original transcript messages are preserved and may contain source clues.

Run with `trace_audit_identity_pilot` and `folder_assignment` set to `a`, `b`,
or `c`.

The agent-visible task prompts and JSON submission formats are defined in
`src/trace_audit/pilot.py` and `src/trace_audit/resistor_identity_pilot.py`.
Private manifests and adjudication data stay in the researcher workspace.
