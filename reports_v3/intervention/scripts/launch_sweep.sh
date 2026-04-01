#!/usr/bin/env bash
#
# Launch intervention experiment conditions headless.
# Each condition runs in its own directory under runs/{env}/{agent}/{condition}/
#
# The environment server is stateful — parallel agents on the same server
# would clash. So we start TWO server instances per environment (one per
# agent type) on different ports.
#
# Usage:
#   # Step 0: Set up environment venvs (one-time)
#   ./scripts/setup_envs.sh
#
#   # Step 0.5: Start environment servers
#   ./scripts/launch_sweep.sh --start-servers
#
#   # Step 1: Launch baselines only (run these first)
#   ./scripts/launch_sweep.sh --baseline
#
#   # Step 2: After baselines finish, build trace registry
#   uv run python scripts/build_trace_registry.py
#
#   # Step 3: Launch intervention conditions
#   ./scripts/launch_sweep.sh --intervention
#
#   # Stop servers when done
#   ./scripts/launch_sweep.sh --stop-servers
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
TASKS_DIR="$PROJECT_ROOT/tasks"
SERVER_DIR="$INTERVENTION_ROOT/servers"

# ─── Server config ────────────────────────────────────────────────────────────
# Each env has two ports (react / toolcalling) to allow full parallelism.
# Format: "venv_dir|module|extra_args"
declare -A ENV_SERVER_CFG
ENV_SERVER_CFG[spectra]="$TASKS_DIR/spectra_elucidation|spectra_elucidation.env|--level 2"
ENV_SERVER_CFG[resistor]="$TASKS_DIR/resistor_network|resistor_network.env|--mode single"
ENV_SERVER_CFG[wetlab]="$TASKS_DIR/wetlab|wetlab.env|--level 2"

# Ports: env -> "react_port toolcalling_port"
declare -A ENV_PORTS
ENV_PORTS[spectra]="8002 8012"
ENV_PORTS[resistor]="8001 8011"
ENV_PORTS[wetlab]="8003 8013"

ALL_ENVS="spectra resistor wetlab"
ALL_AGENTS="react toolcalling"

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
        --start-servers) MODE="start-servers"; shift ;;
        --stop-servers) MODE="stop-servers"; shift ;;
        --server-status) MODE="server-status"; shift ;;
        --env) ENV_FILTER="$2"; shift 2 ;;
        --agent) AGENT_FILTER="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "Usage: $0 MODE [options]"
    echo ""
    echo "Modes:"
    echo "  --start-servers  Start environment servers (2 per env, one per agent)"
    echo "  --stop-servers   Stop environment servers"
    echo "  --server-status  Check if servers are running"
    echo "  --baseline       Launch baseline (no intervention) runs"
    echo "  --intervention   Launch intervention runs (requires trace_registry.json)"
    echo ""
    echo "Options:"
    echo "  --env NAME       Filter by environment (spectra, resistor, wetlab)"
    echo "  --agent NAME     Filter by agent type (react, toolcalling)"
    echo "  --max-parallel N Limit concurrent runs"
    echo "  --dry-run        Print commands without running"
    exit 1
fi

# ─── Helper: get port for env/agent ──────────────────────────────────────────
get_port() {
    local env_name="$1" agent="$2"
    local ports=(${ENV_PORTS[$env_name]})
    if [ "$agent" = "react" ]; then
        echo "${ports[0]}"
    else
        echo "${ports[1]}"
    fi
}

# ─── Server management ───────────────────────────────────────────────────────

start_server() {
    local env_name="$1" agent="$2"
    local config="${ENV_SERVER_CFG[$env_name]}"
    IFS='|' read -r venv_dir module extra_args <<< "$config"
    local port
    port=$(get_port "$env_name" "$agent")

    local server_name="${env_name}_${agent}"
    local pid_file="$SERVER_DIR/${server_name}.pid"
    local log_file="$SERVER_DIR/${server_name}.log"

    # Check if already running
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "  $server_name: already running (PID $(cat "$pid_file")) on port $port"
        return 0
    fi

    local python="$venv_dir/.venv/bin/python"
    if [ ! -f "$python" ]; then
        echo "  ERROR: $env_name venv not found at $venv_dir/.venv"
        echo "  Run: ./scripts/setup_envs.sh $env_name"
        return 1
    fi

    echo "  $server_name: starting on port $port"

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] $python -m $module --port $port $extra_args"
        return 0
    fi

    mkdir -p "$SERVER_DIR"
    (
        cd "$venv_dir"
        nohup "$python" -m "$module" --port "$port" $extra_args \
            > "$log_file" 2>&1 &
        echo $! > "$pid_file"
    )
    echo "    PID: $(cat "$pid_file"), log: $log_file"
}

stop_server() {
    local env_name="$1" agent="$2"
    local server_name="${env_name}_${agent}"
    local pid_file="$SERVER_DIR/${server_name}.pid"

    if [ ! -f "$pid_file" ]; then
        echo "  $server_name: not running (no pid file)"
        return 0
    fi

    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        echo "  $server_name: stopping PID $pid"
        kill "$pid"
        rm -f "$pid_file"
    else
        echo "  $server_name: already stopped (stale pid file)"
        rm -f "$pid_file"
    fi
}

check_server() {
    local env_name="$1" agent="$2"
    local port
    port=$(get_port "$env_name" "$agent")
    local server_name="${env_name}_${agent}"
    local pid_file="$SERVER_DIR/${server_name}.pid"

    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "  $server_name: RUNNING (PID $(cat "$pid_file"), port $port)"
    else
        echo "  $server_name: STOPPED"
    fi
}

wait_for_servers() {
    echo "Waiting for servers to be ready..."
    for env_name in $ALL_ENVS; do
        if [ -n "$ENV_FILTER" ] && [ "$env_name" != "$ENV_FILTER" ]; then continue; fi
        for agent in $ALL_AGENTS; do
            if [ -n "$AGENT_FILTER" ] && [ "$agent" != "$AGENT_FILTER" ]; then continue; fi

            local port
            port=$(get_port "$env_name" "$agent")
            local server_name="${env_name}_${agent}"
            local attempts=0
            local max_attempts=30

            while [ $attempts -lt $max_attempts ]; do
                if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/tasks" 2>/dev/null | grep -q "^200"; then
                    echo "  $server_name: ready on port $port"
                    break
                fi
                attempts=$((attempts + 1))
                sleep 2
            done
            if [ $attempts -ge $max_attempts ]; then
                echo "  WARNING: $server_name may not be ready on port $port (timeout)"
            fi
        done
    done
}

# ─── Start servers ────────────────────────────────────────────────────────────
if [ "$MODE" = "start-servers" ]; then
    echo "=== Starting Environment Servers (2 per env) ==="
    for env_name in $ALL_ENVS; do
        if [ -n "$ENV_FILTER" ] && [ "$env_name" != "$ENV_FILTER" ]; then continue; fi
        for agent in $ALL_AGENTS; do
            if [ -n "$AGENT_FILTER" ] && [ "$agent" != "$AGENT_FILTER" ]; then continue; fi
            start_server "$env_name" "$agent"
        done
    done
    if [ "$DRY_RUN" = false ]; then
        wait_for_servers
    fi
    echo ""
    echo "Servers started. Check status: $0 --server-status"
    exit 0
fi

# ─── Stop servers ─────────────────────────────────────────────────────────────
if [ "$MODE" = "stop-servers" ]; then
    echo "=== Stopping Environment Servers ==="
    for env_name in $ALL_ENVS; do
        if [ -n "$ENV_FILTER" ] && [ "$env_name" != "$ENV_FILTER" ]; then continue; fi
        for agent in $ALL_AGENTS; do
            if [ -n "$AGENT_FILTER" ] && [ "$agent" != "$AGENT_FILTER" ]; then continue; fi
            stop_server "$env_name" "$agent"
        done
    done
    exit 0
fi

# ─── Server status ────────────────────────────────────────────────────────────
if [ "$MODE" = "server-status" ]; then
    echo "=== Server Status ==="
    for env_name in $ALL_ENVS; do
        for agent in $ALL_AGENTS; do
            check_server "$env_name" "$agent"
        done
    done
    exit 0
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
    COMBOS=$(uv run python -c "
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

    echo "$CONDITIONS" | uv run python -c "
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
