#!/usr/bin/env bash
#
# Launch intervention experiment conditions headless.
# Each condition runs in its own directory under runs/{env}/{agent}/{condition}/
#
# Usage:
#   # Step 1: Launch baselines only (run these first)
#   ./scripts/launch_sweep.sh --baseline
#
#   # Step 2: After baselines finish, build trace registry
#   uv run python scripts/build_trace_registry.py
#
#   # Step 3: Launch intervention conditions
#   ./scripts/launch_sweep.sh --intervention
#
#   # Or launch everything for one env/agent
#   ./scripts/launch_sweep.sh --baseline --env spectra --agent react
#   ./scripts/launch_sweep.sh --intervention --env spectra --agent react
#
#   # Dry run
#   ./scripts/launch_sweep.sh --baseline --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVENTION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER_SCRIPT="$INTERVENTION_ROOT/run_intervention.py"
TASK_SELECTION="$INTERVENTION_ROOT/task_selection.json"
TRACE_REGISTRY="$INTERVENTION_ROOT/trace_registry.json"
PROJECT_ROOT="$(cd "$INTERVENTION_ROOT/../.." && pwd)"

# Defaults
ENV_FILTER=""
AGENT_FILTER=""
MAX_PARALLEL=0
DRY_RUN=false
MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline) MODE="baseline"; shift ;;
        --intervention) MODE="intervention"; shift ;;
        --env) ENV_FILTER="$2"; shift 2 ;;
        --agent) AGENT_FILTER="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "Usage: $0 --baseline|--intervention [options]"
    echo ""
    echo "  --baseline       Launch baseline (no intervention) runs"
    echo "  --intervention   Launch intervention runs (requires trace_registry.json)"
    echo "  --env NAME       Filter by environment"
    echo "  --agent NAME     Filter by agent type"
    echo "  --max-parallel N Limit concurrent runs"
    echo "  --dry-run        Print commands without running"
    exit 1
fi

# ─── Baseline mode ────────────────────────────────────────────────────────────
if [ "$MODE" = "baseline" ]; then
    if [ ! -f "$TASK_SELECTION" ]; then
        echo "ERROR: task_selection.json not found. Run first:"
        echo "  uv run python scripts/select_traces.py"
        exit 1
    fi

    echo "=== Launching Baselines ==="

    # Get env/agent combos from task_selection.json
    COMBOS=$(python3 -c "
import json
with open('$TASK_SELECTION') as f:
    sel = json.load(f)
for key, entry in sorted(sel.items()):
    if entry['task_ids']:
        print(f\"{entry['environment']} {entry['agent_type']}\")
")

    LAUNCHED=0
    while read -r ENV AGENT; do
        # Apply filters
        if [ -n "$ENV_FILTER" ] && [ "$ENV" != "$ENV_FILTER" ]; then continue; fi
        if [ -n "$AGENT_FILTER" ] && [ "$AGENT" != "$AGENT_FILTER" ]; then continue; fi

        RUN_DIR="$INTERVENTION_ROOT/runs/$ENV/$AGENT/baseline"
        RUN_NAME="${ENV}_${AGENT}_none"

        echo "  $RUN_NAME -> $RUN_DIR"

        if [ "$DRY_RUN" = true ]; then
            echo "    [DRY RUN] uv run python $RUNNER_SCRIPT --env $ENV --agent $AGENT --intervention none --task-selection $TASK_SELECTION"
            continue
        fi

        mkdir -p "$RUN_DIR"
        (
            cd "$RUN_DIR"
            nohup uv run python "$RUNNER_SCRIPT" \
                --env "$ENV" \
                --agent "$AGENT" \
                --intervention none \
                --task-selection "$TASK_SELECTION" \
                > run.log 2>&1 &
            echo $! > run.pid
        )
        LAUNCHED=$((LAUNCHED + 1))

        if [ $MAX_PARALLEL -gt 0 ] && [ $LAUNCHED -ge $MAX_PARALLEL ]; then
            echo "    Waiting for a slot (max $MAX_PARALLEL)..."
            wait -n 2>/dev/null || true
        fi
    done <<< "$COMBOS"

    echo ""
    echo "Launched $LAUNCHED baseline runs."
    echo "Monitor: find $INTERVENTION_ROOT/runs -name 'run.log' -path '*/baseline/*' -exec tail -1 {} +"
    echo ""
    echo "After baselines finish, run:"
    echo "  uv run python scripts/build_trace_registry.py"
    echo "  ./scripts/launch_sweep.sh --intervention"

# ─── Intervention mode ────────────────────────────────────────────────────────
elif [ "$MODE" = "intervention" ]; then
    if [ ! -f "$TRACE_REGISTRY" ]; then
        echo "ERROR: trace_registry.json not found. Run first:"
        echo "  uv run python scripts/build_trace_registry.py"
        exit 1
    fi

    echo "=== Launching Interventions ==="

    # Generate conditions from sweep_config (excludes baselines)
    FILTER_ARGS=""
    if [ -n "$ENV_FILTER" ]; then FILTER_ARGS="$FILTER_ARGS --env $ENV_FILTER"; fi
    if [ -n "$AGENT_FILTER" ]; then FILTER_ARGS="$FILTER_ARGS --agent $AGENT_FILTER"; fi

    CONDITIONS=$(cd "$PROJECT_ROOT" && uv run python "$SCRIPT_DIR/sweep_config.py" --json $FILTER_ARGS)

    LAUNCHED=0
    RUNNING=0

    echo "$CONDITIONS" | python3 -c "
import sys, json
conditions = json.load(sys.stdin)
for c in conditions:
    if c['intervention'] != 'none':
        print(f\"{c['env']} {c['agent']} {c['intervention']} {c['num_steps']} {c['dir_name']}\")
" | while read -r ENV AGENT INTERVENTION NUM_STEPS DIR_NAME; do

        RUN_DIR="$INTERVENTION_ROOT/runs/$ENV/$AGENT/$DIR_NAME"
        RUN_NAME="${ENV}_${AGENT}_${DIR_NAME}"

        echo "  $RUN_NAME -> $RUN_DIR"

        if [ "$DRY_RUN" = true ]; then
            echo "    [DRY RUN] uv run python $RUNNER_SCRIPT --env $ENV --agent $AGENT --intervention $INTERVENTION --num-steps $NUM_STEPS --trace-registry $TRACE_REGISTRY"
            continue
        fi

        mkdir -p "$RUN_DIR"
        (
            cd "$RUN_DIR"
            nohup uv run python "$RUNNER_SCRIPT" \
                --env "$ENV" \
                --agent "$AGENT" \
                --intervention "$INTERVENTION" \
                --num-steps "$NUM_STEPS" \
                --trace-registry "$TRACE_REGISTRY" \
                > run.log 2>&1 &
            echo $! > run.pid
        )
        LAUNCHED=$((LAUNCHED + 1))
        RUNNING=$((RUNNING + 1))

        if [ $MAX_PARALLEL -gt 0 ] && [ $RUNNING -ge $MAX_PARALLEL ]; then
            echo "    Waiting for a slot (max $MAX_PARALLEL)..."
            wait -n 2>/dev/null || true
            RUNNING=$((RUNNING - 1))
        fi
    done

    echo ""
    echo "Monitor: find $INTERVENTION_ROOT/runs -name 'run.log' -not -path '*/baseline/*' -exec tail -1 {} +"
    echo "Reports: find $INTERVENTION_ROOT/runs -name '*_report.json'"
fi
