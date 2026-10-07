# Inference-time optimization environment

A teacher agent writes a Python policy that makes a frozen student model more accurate or more robust on maths problems without changing its weights: through prompting, sampling, verification, answer selection and so on.
The policy defines `run(questions, ctx)`; `inference_opt/guide/policy_api.md` is its full contract, and `architecture.md` explains how agent code is kept away from the answers.

## Tasks

There are ten tasks per level.
Every task pairs one benchmark with one student, and scores 1 if the student, with the submitted policy, matches a larger model of its own family run zero-shot, and 0 otherwise.
The larger model answers each test question once, with the same prompt, token limit and scorer as the student's own baseline.
The agent is told which model it has to match and that model's train score; test scores stay hidden.

**Level 1: accuracy on MATH Level 5.** The student must answer at least as many of the 30 test questions correctly as the larger model.

| task | student | larger model | student test | larger model test |
| --- | --- | --- | --- | --- |
| `math_algebra_b` | Llama 3.1 8B | Llama 3.1 70B | 0.33 | 0.53 |
| `math_algebra_d` | Ministral 3 3B | Ministral 3 14B | 0.47 | 0.70 |
| `math_prealgebra_b` | Llama 3.1 8B | Llama 3.1 70B | 0.27 | 0.63 |
| `math_prealgebra_d` | Ministral 3 3B | Ministral 3 14B | 0.50 | 0.73 |
| `math_number_theory_b` | Llama 3.1 8B | Llama 3.1 70B | 0.20 | 0.40 |
| `math_number_theory_a` | Qwen 2.5 7B | Qwen 2.5 72B | 0.13 | 0.73 |
| `math_counting_probability_b` | Llama 3.1 8B | Llama 3.1 70B | 0.17 | 0.33 |
| `math_counting_probability_a` | Qwen 2.5 7B | Qwen 2.5 72B | 0.10 | 0.27 |
| `math_geometry_d` | Ministral 3 3B | Ministral 3 14B | 0.20 | 0.37 |
| `math_geometry_a` | Qwen 2.5 7B | Qwen 2.5 72B | 0.10 | 0.27 |

**Level 2: robustness on GSM-Symbolic.** Each question is one instance of a template, the same problem with different names and numbers, and each template has five test instances.
Template consistency is the share of test templates whose five instances are all answered correctly.
The student must reach the larger model's template consistency, and its accuracy must not fall below its own zero-shot accuracy, so a policy cannot make some templates consistent by giving up on the rest.

| task | student | larger model | student test consistency | larger model test consistency |
| --- | --- | --- | --- | --- |
| `gsm_symbolic_b` | Llama 3.1 8B | Llama 3.1 70B | 0.60 | 0.70 |
| `gsm_p1_b` | Llama 3.1 8B | Llama 3.1 70B | 0.40 | 0.53 |
| `gsm_p2_b` | Llama 3.1 8B | Llama 3.1 70B | 0.10 | 0.60 |
| `gsm_symbolic_d` | Ministral 3 3B | Ministral 3 14B | 0.50 | 0.70 |
| `gsm_p1_d` | Ministral 3 3B | Ministral 3 14B | 0.47 | 0.67 |
| `gsm_p2_d` | Ministral 3 3B | Ministral 3 14B | 0.35 | 0.50 |
| `gsm_p2_a` | Qwen 2.5 7B | Qwen 2.5 72B | 0.35 | 0.60 |
| `gsm_p2_e` | Qwen 3.5 9B, thinking off | Qwen 3.5 27B, thinking off | 0.50 | 0.75 |
| `gsm_symbolic_c` | Llama 3.2 3B | Llama 3.1 70B | 0.33 | 0.70 |
| `gsm_p1_c` | Llama 3.2 3B | Llama 3.1 70B | 0.20 | 0.53 |

The pairs in both levels are those where the larger model leads by a clear margin on test (0.13 on MATH, 0.10 on GSM-Symbolic) and also leads on train.
A tie passes.
Llama 3.2 3B has no larger Llama 3.2 text model, so it is held to Llama 3.1 70B.

The rule is `pass_rule` in each task JSON: `{"kind": "reference", "metric": "accuracy" | "template_consistency", ...}`.
The scorer also accepts `threshold` (a fixed accuracy, such as the 77.02% mean of the 50 models in `inference_opt/data/reference/math_leaderboard.csv`) and `headroom` (close half the gap between the baseline and a perfect score).

Every run is limited by its student-call budget, 8192 generated tokens per call (reasoning included) and its wall clock (`run_time_limit_s`, default one hour).
`student_concurrency` in the task JSON caps the requests one run has in flight, as one number or one per student (default 16): 8 for Llama 3.2 3B and 16 for Qwen 3.5 9B, whose providers rate-limit bursts, and 32 for the others.

## Questions

The frozen set is `inference_opt/data/frozen/v2`.

**MATH**, built by `scripts/select_math_levels.py` from the MATH test questions in bp:

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

**GSM-Symbolic** ([apple-aiml-research/ml-gsm-symbolic](https://github.com/apple-aiml-research/ml-gsm-symbolic)), built by `scripts/select_gsm_symbolic.py`:

| benchmark | what it is | train | test | examples |
| --- | --- | --- | --- | --- |
| `gsm_symbolic` | GSM8K test questions as templates with varied names and numbers | 30 templates | 30 | 20 |
| `gsm_p1` | the same templates with one added clause | 30 | 30 | 20 |
| `gsm_p2` | 50 of them with two added clauses | 20 | 20 | 10 |

- Each template contributes 5 of its 50 instances, so a split holds 150 questions (100 for `gsm_p2`).
- Splits are by GSM8K question and the same in all three variants, so a test template never appears in train or the examples of any variant.
- `category` (the policy's `question.topic`) names the template.

Train runs are scored and their per-question results shown; their answers are not.
Examples are revealed with their gold answers and never scored.
Test questions are used once, by the submission.

`inference_opt/data/frozen/v1` is the earlier set of mixed benchmarks (MMLU-Pro, BBH, GPQA-Diamond, MATH, ChemBench, ARC-Challenge). It is kept on disk and not used by any task.

## Grading

**MATH** is graded exactly as Inspect's MATH eval grades it, with `expression_exact_match_sympy` from `inspect_evals`: the answer is read from a final `ANSWER: <answer>` line and compared with the gold answer using sympy.
A reply whose last line is not an `ANSWER:` line is wrong, even if it contains the right answer in `\boxed{}`.
The zero-shot baseline sends each question in Inspect's MATH prompt, word for word.

**GSM-Symbolic** is graded by the repository's own rule: commas are removed, the last number in the reply is the answer, and it is compared with the gold answer as a number.
The zero-shot baseline uses the repository's evaluation prompt without its eight worked examples: "As an expert problem solver, solve step by step the following mathematical question. Q: … A: Let's think step by step."

Baselines run at temperature 0, seed 0 and at most 8192 tokens.
Policies may prompt and sample however they like; only grading is fixed.

## Students and larger models

| student | model | served by | larger model, served by |
| --- | --- | --- | --- |
| `student_a` | Qwen 2.5 7B Instruct | `openrouter/qwen/qwen-2.5-7b-instruct@phala` | Qwen 2.5 72B Instruct, `@novita` |
| `student_b` | Llama 3.1 8B Instruct | `openrouter/meta-llama/llama-3.1-8b-instruct@coreweave` | Llama 3.1 70B Instruct, `@amazon-bedrock` |
| `student_c` | Llama 3.2 3B Instruct | `openrouter/meta-llama/llama-3.2-3b-instruct@parasail` | Llama 3.1 70B Instruct, `@amazon-bedrock` |
| `student_d` | Ministral 3 3B (2512) | `openrouter/mistralai/ministral-3b-2512@mistral` | Ministral 3 14B (2512), `@mistral` |
| `student_e` | Qwen 3.5 9B, thinking off | `openrouter/qwen/qwen3.5-9b@together` | Qwen 3.5 27B, thinking off, `@novita` |

The larger models are stored in `v2.json` as `reference_a`, `reference_b`, `reference_d` and `reference_e`.
They are only measured, never called during a task.

A model spec names where a student is served:

- `openrouter/<model>@<provider>` is sent to OpenRouter with `OPENROUTER_API_KEY`, and routed only to that provider, with fallbacks off.
  A provider can serve a different quantisation or build of the same model, so pinning it keeps the student the same across runs.
- `vllm/<served model>` is sent to a vLLM server at `CORRAL_VLLM_URL`, or at a per-student `CORRAL_VLLM_URL_STUDENT_A`, with `VLLM_API_KEY` if set.

`student_options` in a task JSON can turn a reasoning model's thinking off on every call, as for `student_e`: `{"student_e": {"reasoning": false}}`. A policy cannot turn it back on.
`measure_baselines.py --no-reasoning NAME` measures a model the same way.

Requests that hit a rate limit, a server error, a dropped connection, or a provider failing part-way through a reply are retried with backoff for about four minutes before the call fails.

Qwen 2.5 7B often ends with `\boxed{}` instead of the requested `ANSWER:` line, so part of its gap to Qwen 2.5 72B is following the format.

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
uv run python scripts/select_gsm_symbolic.py --source-dir /path/to/ml-gsm-symbolic/generated_data
uv run python scripts/measure_baselines.py --model student_a=openrouter/... ... --runs-per-model 2
uv run python scripts/measure_baselines.py --model reference_a=openrouter/... ... --runs-per-model 2
uv run python scripts/measure_baselines.py --model student_a=openrouter/... ... --runs-per-model 2 --splits examples
uv run python scripts/generate_tasks.py
```

`measure_baselines.py` writes `inference_opt/data/baselines/v2.json` (train and test) and `examples_v2.jsonl` (each student's reply to every example, shown to the agent when it reveals one).
Only students need the `--splits examples` run; larger models need train and test.
`generate_tasks.py` reads the baselines, the larger models' results and the model specs from `v2.json`, so a task always matches what was measured.

| script | purpose |
| --- | --- |
| `select_math_levels.py` | Build the MATH part of the v2 set. |
| `select_gsm_symbolic.py` | Build the GSM-Symbolic part of the v2 set. |
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
