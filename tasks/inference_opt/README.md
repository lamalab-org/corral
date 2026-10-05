# Inference-time optimization environment

A teacher agent writes a Python policy that improves a frozen student model at inference time: prompts, sampling, verification, etc.

Level 1 tasks have one student.
Level 2 tasks run one policy on two students, and both must pass.
A task scores 1 when the policy answers correctly at least half of the held-out test questions the student's zero-shot baseline gets wrong, and 0 otherwise (`pass_rule` in the task JSON; `{"kind": "continuous"}` scores the improvement instead).

A policy defines `run(questions, ctx)`. `ctx.student.generate` takes one prompt, or a list of prompts that are sent together, and `ctx.answer` records answers; `inference_opt/guide/policy_api.md` is the full contract.

Every run is limited by its student-call budget, 8192 generated tokens per call and its wall clock (`run_time_limit_s`, default one hour).
`student_concurrency` in the task JSON (default 16) caps the requests one run has in flight; the policy does not set it.

How the policy is isolated from the answers is described in `architecture.md`.

## Run

```bash
cd tasks/inference_opt
uv sync
uv run python -m inference_opt.env --level 1
uv run pytest tests -q
```

The policy is isolated only inside Corral's Docker trial.
Outside it, the policy runs as a plain subprocess and can read the labels on disk.

`corral bench --sandbox docker --build-sandbox-image` rebuilds the trial image from this checkout.
To build the image by hand, from the repository root:

```bash
docker build -f docker/benchmark.Dockerfile --build-arg CORRAL_TASK=inference_opt \
  --build-arg CORRAL_EXTRAS=langfuse -t corral-inference_opt:langfuse .
```

Leave out `CORRAL_EXTRAS` and tag it `:latest` when Langfuse is not configured.

The jail tests in `tests/test_isolation.py` need that Docker boundary: run them in the trial image as root with `CORRAL_PERMISSION_TESTS=1`.

Set `CORRAL_VLLM_URL`, or a per-student variable such as `CORRAL_VLLM_URL_STUDENT_A`, before starting Corral.
Labels ship in `inference_opt/data/frozen/v1/private/`; `CORRAL_INFERENCE_LABELS_PATH` points elsewhere.

## Questions and baselines

The frozen set has 60 questions each for `mmlu_pro`, `bbh`, `gpqa_diamond`, `math`, `chembench` and `arc_challenge`, split 30 train / 30 test.
ChemBench leaves out its `chemical_preference` questions, and every ChemBench multiple-choice question asks for all correct options, as ChemBench's own harness does, so the agent is never told how many are right.
Each benchmark also has 30 example questions, chosen by `scripts/select_examples.py` with the same topic mix as train.
The agent reveals examples with their gold answers, and policies receive them as `ctx.examples`; they are never scored, so train answers are never shown.

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

`scripts/generate_tasks.py` writes the task JSONs with zero baselines; `measure_baselines.py --write-tasks` fills them in.
`measure_baselines.py --splits examples` records each student's zero-shot reply to the examples in `inference_opt/data/baselines/examples_v1.jsonl`, which the agent sees when it reveals them.

## Agent tools

Besides Corral's file tools for editing `policy/policy.py`:

| tool | purpose |
| --- | --- |
| `get_baseline` | The measured train baseline, by topic. |
| `reveal_examples` | Spend a reveal on labelled example questions, with each student's zero-shot reply. |
| `query_student` | Probe the student directly, from the probe budget. |
| `dry_run_policy` | Run the policy on two train questions. |
| `evaluate_candidate` | Run and record a full train experiment. |
| `inspect_failures` | Answers, logs and errors from an earlier run. |
| `compare_runs` | Recorded experiments and the best so far. |
| `get_budget` | What is left of each budget. |
| `submit_policy` | Run the policy once on the test split; the result stays hidden until scoring. A task allows one submission; the agent then ends the episode with Corral's `submit_answer`. |
