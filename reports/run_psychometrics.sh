#!/usr/bin/env bash
#
# Run the psychometrics environment over both levels at once.
#
# Each level is one `corral bench` process with its own output directory,
# report and commit store, so the two never contend for the same SQLite file.
# Within a level, --max-parallel bounds concurrent task executions.
#
# Langfuse: `corral` loads ./.env itself, and tracing turns itself on as soon as
# LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are present. The langfuse SDK reads
# either LANGFUSE_BASE_URL or LANGFUSE_HOST, so the .env in this repo works
# unchanged. This needs the `langfuse` extra, which the uv invocation below adds.
#
# Sandbox: local by default, and deliberately. The REPL keeps its session in a
# local checkpointed worker when Docker enforcement is off, so these tasks do not
# need a container. Under --sandbox docker on this branch, Langfuse traces are
# silently dropped: LANGFUSE_* is not in the sandbox environment allowlist, and
# the container-side runner hard-codes LoggingObserver. PR #416 fixes both; until
# it lands, use local if you want traces.
#
# Override any setting from the environment, e.g.
#   MAX_PARALLEL=8 TRIALS=3 ./reports/run_psychometrics.sh
#   LEVELS="1" ./reports/run_psychometrics.sh        # one level only
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

MODEL="${MODEL:-gpt-5.6-sol}"
AGENT="${AGENT:-react}"
REASONING="${REASONING:-medium}"
# A reasoning model rejects any temperature but 1 while reasoning is active, and
# the react agent would otherwise send its own default of 0.7.
TEMPERATURE="${TEMPERATURE:-1}"
TRIALS="${TRIALS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
SANDBOX="${SANDBOX:-local}"
LEVELS="${LEVELS:-1 2}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/reports}"
# 1 adds each trial's agent messages and tool calls to report.json.
VERBOSE="${VERBOSE:-1}"
# Longest one PythonREPL call may run before it is stopped and the agent told.
REPL_TIMEOUT_MINUTES="${REPL_TIMEOUT_MINUTES:-15}"
# Seconds LiteLLM waits on a silent model stream before retrying it. Healthy
# calls here finish in under ~140s; LiteLLM's own default is 600s.
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
# Space-separated task ids to run instead of every task, e.g.
#   TASKS="psy_l1_t09_careless_responding psy_l2_t02_out_of_sample_generalization"
# Each id goes to its own level; a level with none of them is skipped.
TASKS="${TASKS:-}"

if [[ ! -f "$ROOT/.env" ]]; then
  echo "error: no .env at $ROOT/.env; Langfuse and OPENAI_API_KEY come from it" >&2
  exit 1
fi

# Fail before spending tokens if the key is missing from .env.
if ! grep -qE '^\s*OPENAI_API_KEY=.+' "$ROOT/.env"; then
  echo "error: OPENAI_API_KEY is not set in $ROOT/.env" >&2
  exit 1
fi
if ! grep -qE '^\s*LANGFUSE_SECRET_KEY=.+' "$ROOT/.env"; then
  echo "warning: LANGFUSE_SECRET_KEY missing from .env; tracing falls back to local logging" >&2
fi

# Without semopy every model-fitting scorer returns 0.0 and records no error, so
# a run in the wrong venv reports plausible zeros instead of failing. Refuse to
# start rather than produce numbers that look real.
if ! uv run --project tasks/psychometrics python -c "import semopy, factor_analyzer" 2>/dev/null; then
  echo "error: semopy/factor-analyzer missing from the psychometrics venv;" >&2
  echo "       the SEM scorers would silently score every task 0." >&2
  echo "       run: uv sync --project tasks/psychometrics" >&2
  exit 1
fi

if [[ "$SANDBOX" == "docker" ]]; then
  echo "warning: --sandbox docker drops Langfuse traces on this branch (see PR #416)" >&2
  if ! docker info >/dev/null 2>&1; then
    echo "error: --sandbox docker requested but the Docker daemon is not reachable" >&2
    exit 1
  fi
fi

echo "model=$MODEL agent=$AGENT reasoning=$REASONING temperature=$TEMPERATURE trials=$TRIALS"
echo "levels=$LEVELS max_parallel=$MAX_PARALLEL max_iterations=$MAX_ITERATIONS sandbox=$SANDBOX verbose=$VERBOSE"
echo "repl_timeout_minutes=$REPL_TIMEOUT_MINUTES request_timeout=${REQUEST_TIMEOUT}s"
[[ -n "$TASKS" ]] && echo "tasks=$TASKS"
echo "output=$OUT_ROOT/<level>/$STAMP"
echo

run_level() {
  local level="$1"
  local out="$OUT_ROOT/level_${level}/$STAMP"
  mkdir -p "$out"
  local verbose_flag=()
  [[ "$VERBOSE" == "1" ]] && verbose_flag=(--verbose)
  local task_flags=()
  local task
  for task in $TASKS; do
    [[ "$task" == psy_l${level}_* ]] && task_flags+=(--task "$task")
  done

  # --run-id names the run; --commit-file is per level so the two processes
  # never write the same SQLite database.
  # The psychometrics project venv, not the repo root one: the SEM scorers need
  # semopy and factor-analyzer, which only that venv has. Running from the root
  # venv still loads the task package, so every model-fitting scorer silently
  # returns 0 instead of failing. langfuse is a corral extra rather than a task
  # dependency, so it comes in with --with.
  uv run --project tasks/psychometrics --with "langfuse>=4,<5" corral bench \
    --environment psychometrics \
    --env-kwargs "{\"level\": $level, \"repl_timeout_minutes\": $REPL_TIMEOUT_MINUTES}" \
    --agent "$AGENT" \
    --model "$MODEL" \
    --agent-kwargs "{\"reasoning_effort\": \"$REASONING\", \"timeout\": $REQUEST_TIMEOUT}" \
    --temperature "$TEMPERATURE" \
    --trials "$TRIALS" \
    --max-parallel "$MAX_PARALLEL" \
    --max-iterations "$MAX_ITERATIONS" \
    --sandbox "$SANDBOX" \
    --run-id "psychometrics-l${level}-${STAMP}" \
    --output-dir "$out/state" \
    --report "$out/report.json" \
    --commit-file "$out/commits.sqlite3" \
    ${verbose_flag[@]+"${verbose_flag[@]}"} \
    ${task_flags[@]+"${task_flags[@]}"} \
    >"$out/run.log" 2>&1
}

# Two plain indexed arrays rather than one associative array: macOS still ships
# bash 3.2, where `declare -A` is a syntax error and a string subscript silently
# resolves to index 0.
started_levels=()
started_pids=()
for level in $LEVELS; do
  if [[ -n "$TASKS" && " $TASKS " != *" psy_l${level}_"* ]]; then
    echo "level $level: no selected tasks, skipped"
    continue
  fi
  run_level "$level" &
  started_pids+=("$!")
  started_levels+=("$level")
  echo "level $level started (pid $!) -> $OUT_ROOT/level_${level}/$STAMP/run.log"
done

status=0
index=0
while [ "$index" -lt "${#started_pids[@]}" ]; do
  level="${started_levels[$index]}"
  out="$OUT_ROOT/level_${level}/$STAMP"
  if wait "${started_pids[$index]}"; then
    echo "level $level: OK -> $out/report.json"
  else
    code=$?
    status=1
    echo "level $level: FAILED (exit $code); last lines of $out/run.log:" >&2
    tail -20 "$out/run.log" >&2 || true
  fi
  index=$((index + 1))
done

exit "$status"
