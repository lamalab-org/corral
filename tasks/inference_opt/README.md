# Inference-time optimization environment

A teacher agent writes a Python policy that improves a frozen student model at
inference time: prompts, sampling, verification, anything that leaves the
weights alone.

Level 1 tasks have one student. Level 2 tasks run one policy on two students,
and both must pass. A task scores 1 when the policy answers correctly at least
half of the held-out test questions the student's zero-shot baseline gets
wrong, and 0 otherwise (`pass_rule` in the task JSON; `{"kind": "continuous"}`
scores the improvement instead).

Tasks use the `primitive` policy API (`ctx.student.generate` only). A task can
set `"policy_api": "enhanced"` to add `sample` and `batch`.

How the policy is isolated from the answers is described in `architecture.md`.

## Run

```bash
cd tasks/inference_opt
uv sync
uv run python -m inference_opt.env --level 1
uv run pytest tests -q
```

The policy is isolated only inside Corral's Docker trial. Outside it, the
policy runs as a plain subprocess and can read the labels on disk.

The jail tests in `tests/test_isolation.py` need that Docker boundary: run them
in the trial image as root with `CORRAL_PERMISSION_TESTS=1`.

Set `CORRAL_VLLM_URL`, or a per-student variable such as
`CORRAL_VLLM_URL_STUDENT_A`, before starting Corral. Labels ship in
`inference_opt/data/frozen/v1/private/`; `CORRAL_INFERENCE_LABELS_PATH` points
elsewhere.

## Questions and baselines

The frozen set has 60 questions each for `mmlu_pro`, `bbh`, `gpqa_diamond`,
`math`, `chembench` and `arc_challenge`, split 30 train / 30 test.
`arc_challenge` has baselines but no tasks.

Each task pins its zero-shot baselines to one deployment. Serve these models,
and re-run `scripts/measure_baselines.py --write-tasks` if you change one:

| student | model | env var |
| --- | --- | --- |
| `student_a` | `Qwen/Qwen3.5-9B` | `CORRAL_VLLM_URL_STUDENT_A` |
| `student_b` | `Qwen/Qwen3-8B` | `CORRAL_VLLM_URL_STUDENT_B` |
| `student_c` | `google/gemma-4-12B-it` | `CORRAL_VLLM_URL_STUDENT_C` |
| `student_d` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | `CORRAL_VLLM_URL_STUDENT_D` |

`student_a` and `student_b` appear in level 1 and in level 2 `*_ab` tasks;
`student_c` and `student_d` in level 2 `*_cd` tasks.

`scripts/generate_tasks.py` writes the task JSONs with zero baselines;
`measure_baselines.py --write-tasks` fills them in.

## Agent tools

Besides Corral's file tools for editing `policy/policy.py`:

| tool | purpose |
| --- | --- |
| `get_baseline` | The measured train baseline, by topic. |
| `reveal_train_questions` | Spend a reveal on labelled train questions. |
| `query_student` | Probe the student directly, from the probe budget. |
| `dry_run_policy` | Run the policy on two train questions. |
| `evaluate_candidate` | Run and record a full train experiment. |
| `inspect_failures` | Answers, logs and errors from an earlier run. |
| `compare_runs` | Recorded experiments and the best so far. |
| `get_budget` | What is left of each budget. |
| `submit_policy` | Run the policy once on the test split; the result stays hidden until scoring. |
