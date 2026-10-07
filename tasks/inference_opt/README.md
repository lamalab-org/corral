# Inference-time optimization environment

A teacher agent writes a Python policy that makes a frozen student model better at MATH Level 5 problems without changing its weights: through prompting, sampling, verification, answer selection and so on.
The policy defines `run(questions, ctx)`; `inference_opt/guide/policy_api.md` is its full contract, and `architecture.md` explains how agent code is kept away from the answers.

## Tasks

There are ten tasks per level, two for each of five MATH subjects.
A level 1 task has one student; a level 2 task runs the same policy on two students, and both must pass.

A task scores 1 if, on the 30 held-out test questions, every student reaches **77.02%** accuracy with the submitted policy, and 0 otherwise.
77.02% is the mean MATH score of the 50 models in a public leaderboard snapshot (`inference_opt/data/reference/math_leaderboard.csv`); with 30 questions it means 24 correct.
The rule is `pass_rule` in each task JSON.
`{"kind": "headroom", "min_closed": 0.5}` instead requires every student to close half the gap between its baseline and a perfect score, and `{"kind": "continuous"}` scores the improvement itself; `scripts/generate_tasks.py --pass-rule headroom` writes the former.

Every run is limited by its student-call budget, 8192 generated tokens per call (reasoning included) and its wall clock (`run_time_limit_s`, default one hour).
`student_concurrency` in the task JSON (default 16) caps the requests one run has in flight.

## Questions

The frozen set is `inference_opt/data/frozen/v2`, built by `scripts/select_math_levels.py` from the MATH test questions in bp:

| benchmark | MATH subject | train | test | examples |
| --- | --- | --- | --- | --- |
| `math_algebra` | algebra | 30 | 30 | 20 |
| `math_prealgebra` | prealgebra | 30 | 30 | 20 |
| `math_number_theory` | number theory | 30 | 30 | 20 |
| `math_counting_probability` | counting & probability | 30 | 30 | 20 |
| `math_geometry` | geometry | 30 | 30 | 20 |

- Every question is Level 5, MATH's hardest; the level comes from the original release, `EleutherAI/hendrycks_math`.
- Answers may be numbers or expressions. A question is kept only if Inspect's MATH scorer grades its own gold answer as correct, which leaves out matrices, coordinate pairs and intervals.
- The three splits are drawn from one pool and dealt by bp's IRT difficulty, so they are equally hard.
- Every question has `answer_format: "math"`, so nothing in its metadata says whether the answer is a number.
- Train runs are scored and their per-question results shown; their answers are not. Examples are revealed with their gold answers and never scored. Test questions are used once, by the submission.

`inference_opt/data/frozen/v1` is the earlier set of mixed benchmarks (MMLU-Pro, BBH, GPQA-Diamond, MATH, ChemBench, ARC-Challenge). It is kept on disk and not used by any task.

## Grading

Answers are graded exactly as Inspect's MATH eval grades them, with `expression_exact_match_sympy` from `inspect_evals`: the answer is read from a final `ANSWER: <answer>` line and compared with the gold answer using sympy.
A reply whose last line is not an `ANSWER:` line is wrong, even if it contains the right answer in `\boxed{}`.

The zero-shot baseline sends each question in Inspect's MATH prompt, word for word, at temperature 0, seed 0 and at most 8192 tokens.
Policies may prompt and sample however they like; only grading is fixed.

## Students

| student | model | served by |
| --- | --- | --- |
| `student_a` | Qwen 2.5 7B Instruct | `openrouter/qwen/qwen-2.5-7b-instruct@phala` |
| `student_b` | Llama 3.1 8B Instruct | `openrouter/meta-llama/llama-3.1-8b-instruct@coreweave` |
| `student_c` | Llama 3.2 3B Instruct | `openrouter/meta-llama/llama-3.2-3b-instruct@parasail` |
| `student_d` | Ministral 3 3B (2512) | `openrouter/mistralai/ministral-3b-2512@mistral` |

A model spec names where a student is served:

- `openrouter/<model>@<provider>` is sent to OpenRouter with `OPENROUTER_API_KEY`, and routed only to that provider, with fallbacks off.
  A provider can serve a different quantisation or build of the same model, so pinning it keeps the student the same across runs.
- `vllm/<served model>` is sent to a vLLM server at `CORRAL_VLLM_URL`, or at a per-student `CORRAL_VLLM_URL_STUDENT_A`, with `VLLM_API_KEY` if set.

Requests that hit a rate limit, a server error or a dropped connection are retried with backoff for about four minutes before the call fails.

Test baselines, which the pass rule does not use but which show how far each task has to go:

| student | algebra | prealgebra | number theory | counting & probability | geometry |
| --- | --- | --- | --- | --- | --- |
| `student_a` | 0.43 | 0.37 | 0.13 | 0.10 | 0.10 |
| `student_b` | 0.33 | 0.27 | 0.20 | 0.17 | 0.07 |
| `student_c` | 0.43 | 0.20 | 0.20 | 0.20 | 0.13 |
| `student_d` | 0.47 | 0.50 | 0.77 | 0.43 | 0.20 |

Each task is paired with students below the bar; see `LEVEL_1` and `LEVEL_2` in `scripts/generate_tasks.py`.
Qwen 2.5 often ends with `\boxed{}` instead of the requested `ANSWER:` line, so part of its headroom is following the format.

## Run

```bash
cd tasks/inference_opt
uv sync
uv run pytest tests -q
uv run python -m inference_opt.env --level 1
```

The policy is isolated only inside Corral's Docker trial.
Outside it, the policy runs as a plain subprocess and can read the labels on disk.

`corral bench --sandbox docker --build-sandbox-image` rebuilds the trial image from this checkout.
To build it by hand, from the repository root:

```bash
docker build -f docker/benchmark.Dockerfile --build-arg CORRAL_TASK=inference_opt \
  --build-arg CORRAL_EXTRAS=langfuse -t corral-inference_opt:langfuse .
```

Leave out `CORRAL_EXTRAS` and tag it `:latest` when Langfuse is not configured.
Pass `OPENROUTER_API_KEY` into the trial the same way as the other keys.
The jail tests in `tests/test_isolation.py` need that Docker boundary: run them in the trial image as root with `CORRAL_PERMISSION_TESTS=1`.

Labels ship in `inference_opt/data/frozen/v2/private/`; `CORRAL_INFERENCE_LABELS_PATH` points elsewhere.

## Rebuilding the set, baselines and tasks

```bash
uv run python scripts/select_math_levels.py --bp-root /path/to/bp
uv run python scripts/measure_baselines.py --model student_a=openrouter/... ... --runs-per-model 2
uv run python scripts/measure_baselines.py --model student_a=openrouter/... ... --runs-per-model 2 --splits examples
uv run python scripts/generate_tasks.py
```

`measure_baselines.py` writes `inference_opt/data/baselines/v2.json` (train and test) and `examples_v2.jsonl` (each student's reply to every example, shown to the agent when it reveals one).
`generate_tasks.py` reads the baselines and model specs from `v2.json`, so a task always matches what was measured.

| script | purpose |
| --- | --- |
| `select_math_levels.py` | Build the v2 set. |
| `measure_baselines.py` | Measure the zero-shot baselines. |
| `generate_tasks.py` | Write the level 1 and level 2 task JSONs. |
| `extract_private_runs.py` | Unpack the graded logs a finished trial kept out of the agent's reach. |
| `frozen_set.py` | Helpers shared by the dataset scripts. |
| `select_examples.py`, `drop_category.py`, `mark_chembench_multi.py` | How the v1 set was built and edited. |

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
