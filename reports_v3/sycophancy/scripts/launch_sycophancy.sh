#!/usr/bin/env bash
#
# Launch sycophancy experiment conditions headless.
# Each condition runs in its own directory under runs/{env}/{agent}/syco_{condition}/
#
# Reuses the same environment servers as the intervention experiment.
# Make sure servers are running before launching:
#   ../intervention/scripts/launch_sweep.sh --start-servers
#
# Usage:
#   # Launch misleading condition for all envs/agents
#   ./scripts/launch_sycophancy.sh --condition misleading
#
#   # Launch for specific env/agent
#   ./scripts/launch_sycophancy.sh --condition misleading --env spectra --agent react
#
#   # Launch all conditions
#   ./scripts/launch_sycophancy.sh --all
#
#   # Dry run
#   ./scripts/launch_sycophancy.sh --condition misleading --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCOPHANCY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER_SCRIPT="$SYCOPHANCY_ROOT/run_sycophancy.py"
PROJECT_ROOT="$(cd "$SYCOPHANCY_ROOT/../.." && pwd)"

# Environments in requested order
ALL_ENVS="spectra ml resistor wetlab retrosynthesis catalyst"
ALL_AGENTS="react toolcalling"
ALL_CONDITIONS="misleading neutral"

# ─── Helpers ─────────────────────────────────────────────────────────────────

timestamp() { date +"%Y%m%d_%H%M%S"; }

GIT_SHA=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")

run_completed() {
    local run_dir="$1"
    local count
    count=$(find "$run_dir" -maxdepth 1 -name '*_report.json' 2>/dev/null | wc -l)
    [ "$count" -gt 0 ]
}

launch_run() {
    local run_dir="$1"; shift
    local run_name="$1"; shift

    local ts
    ts=$(timestamp)
    local log_file="$run_dir/run_${ts}.log"
    local meta_file="$run_dir/run_${ts}_meta.json"

    mkdir -p "$run_dir"

    cat > "$meta_file" <<METAEOF
{"run_name": "$run_name", "start_time": "$ts", "git_sha": "$GIT_SHA", "command": "$*"}
METAEOF

    ln -sf "$(basename "$log_file")" "$run_dir/run.log"

    (
        cd "$run_dir"
        nohup "$@" > "$log_file" 2>&1 &
        echo $! > run.pid
    )
    echo "    log: $log_file"
}

# ─── Parse args ──────────────────────────────────────────────────────────────

ENV_FILTER=""
AGENT_FILTER=""
CONDITION_FILTER=""
DRY_RUN=false
TRIALS=""
MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --condition) CONDITION_FILTER="$2"; MODE="single"; shift 2 ;;
        --all) MODE="all"; shift ;;
        --env) ENV_FILTER="$2"; shift 2 ;;
        --agent) AGENT_FILTER="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "Usage: $0 MODE [options]"
    echo ""
    echo "Modes:"
    echo "  --condition NAME   Launch a specific condition (misleading, neutral)"
    echo "  --all              Launch all conditions"
    echo ""
    echo "Options:"
    echo "  --env NAME         Filter by environment"
    echo "  --agent NAME       Filter by agent type (react, toolcalling)"
    echo "  --trials N         Override trials per condition"
    echo "  --dry-run          Print commands without running"
    exit 1
fi

# ─── Determine conditions to run ─────────────────────────────────────────────

if [ "$MODE" = "single" ]; then
    CONDITIONS="$CONDITION_FILTER"
else
    CONDITIONS="$ALL_CONDITIONS"
fi

# ─── Launch ──────────────────────────────────────────────────────────────────

echo "=== Launching Sycophancy Experiments ==="
LAUNCHED=0

for ENV in $ALL_ENVS; do
    if [ -n "$ENV_FILTER" ] && [ "$ENV" != "$ENV_FILTER" ]; then continue; fi

    for AGENT in $ALL_AGENTS; do
        if [ -n "$AGENT_FILTER" ] && [ "$AGENT" != "$AGENT_FILTER" ]; then continue; fi

        for CONDITION in $CONDITIONS; do
            DIR_NAME="syco_${CONDITION}"
            RUN_DIR="$SYCOPHANCY_ROOT/runs/$ENV/$AGENT/$DIR_NAME"
            RUN_NAME="syco_${ENV}_${AGENT}_${CONDITION}"

            # Skip if already completed
            if run_completed "$RUN_DIR"; then
                echo "  $RUN_NAME: SKIPPED (report already exists in $RUN_DIR)"
                continue
            fi

            echo "  $RUN_NAME -> $RUN_DIR"
            LAUNCHED=$((LAUNCHED + 1))

            TRIALS_ARG=""
            if [ -n "$TRIALS" ]; then TRIALS_ARG="--trials $TRIALS"; fi

            if [ "$DRY_RUN" = true ]; then
                echo "    [DRY RUN] uv run python $RUNNER_SCRIPT --env $ENV --agent $AGENT --condition $CONDITION $TRIALS_ARG"
                continue
            fi

            launch_run "$RUN_DIR" "$RUN_NAME" \
                uv run python "$RUNNER_SCRIPT" \
                    --env "$ENV" \
                    --agent "$AGENT" \
                    --condition "$CONDITION" \
                    $TRIALS_ARG
        done
    done
done

echo ""
echo "Launched $LAUNCHED sycophancy runs."
echo "Monitor: find $SYCOPHANCY_ROOT/runs -name 'run_*.log' -path '*/syco_*' -exec tail -1 {} +"
echo "Reports: find $SYCOPHANCY_ROOT/runs -name 'syco_*_report.json'"
