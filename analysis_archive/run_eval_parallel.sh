#!/bin/bash
# Parallel evaluation using GNU parallel or background jobs
#
# Usage:
#   # GNU parallel (for workstations)
#   bash run_eval_parallel.sh kfold 8
#   bash run_eval_parallel.sh group 8

set -e

MODE=$1  # kfold or group
MAX_JOBS=${2:-8}  # Max parallel jobs

if [[ "$MODE" == "kfold" ]]; then
    echo "Running K-fold CV with max $MAX_JOBS parallel jobs..."

    ENVIRONMENTS=("afm" "catalyst" "md" "ml" "resistor" "retro" "spectra")

    # Create output directory
    mkdir -p results/eval/kfold results/eval/logs

    # Export function for parallel
    export -f run_kfold_job

    run_kfold_job() {
        env=$1
        echo "Starting fold: $env"
        python evaluate_single_fold.py kfold \
            --holdout_env=$env \
            --output_file=results/eval/kfold/${env}.json \
            > results/eval/logs/kfold_${env}.log 2>&1
        echo "Completed fold: $env"
    }

    # Run in parallel using GNU parallel (if available)
    if command -v parallel &> /dev/null; then
        printf '%s\n' "${ENVIRONMENTS[@]}" | parallel -j $MAX_JOBS run_kfold_job {}
    else
        # Fallback to simple background jobs
        for env in "${ENVIRONMENTS[@]}"; do
            run_kfold_job $env &

            # Limit concurrent jobs
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 10
            done
        done
        wait
    fi

    echo "All K-fold jobs completed!"

    # Aggregate results
    python -c "
import json
import numpy as np
from pathlib import Path

envs = ['afm', 'catalyst', 'md', 'ml', 'resistor', 'retro', 'spectra']
results = []
for env in envs:
    with open(f'results/eval/kfold/{env}.json') as f:
        results.append(json.load(f))

valid_results = [r for r in results if r['metrics'] is not None]
if valid_results:
    all_metrics = [r['metrics'] for r in valid_results]
    summary = {
        metric: float(np.mean([m[metric] for m in all_metrics]))
        for metric in all_metrics[0]
    }
    summary['n_folds'] = len(valid_results)

    with open('results/eval/kfold_summary.json', 'w') as f:
        json.dump({'results': results, 'summary': summary}, f, indent=2)

    print('\n' + '=' * 80)
    print('K-FOLD CV SUMMARY')
    print('=' * 80)
    for metric, value in summary.items():
        if metric != 'n_folds':
            print(f'  {metric:20s}: {value:.4f}')
    print(f'  n_folds             : {summary[\"n_folds\"]}')
"

elif [[ "$MODE" == "group" ]]; then
    echo "Running Group LOO-CV with max $MAX_JOBS parallel jobs..."

    MODELS=("claude-4.5" "gpt-4o" "gpt-oss-120b")
    ENVIRONMENTS=("afm" "catalyst" "md" "ml" "resistor" "retro" "spectra")
    SCAFFOLDS=("react" "tool_calling")

    # Create output directory
    mkdir -p results/eval/group results/eval/logs

    # Generate all combinations
    GROUPS=()
    for model in "${MODELS[@]}"; do
        for env in "${ENVIRONMENTS[@]}"; do
            for scaffold in "${SCAFFOLDS[@]}"; do
                GROUPS+=("${model}_${env}_${scaffold}")
            done
        done
    done

    echo "Total groups: ${#GROUPS[@]}"

    export -f run_group_job

    run_group_job() {
        group=$1
        IFS='_' read -r model env scaffold <<< "$group"
        echo "Starting group: $group"
        python evaluate_single_fold.py group \
            --holdout_model=$model \
            --holdout_env=$env \
            --holdout_scaffold=$scaffold \
            --output_file=results/eval/group/${group}.json \
            > results/eval/logs/group_${group}.log 2>&1
        echo "Completed group: $group"
    }

    # Run in parallel
    if command -v parallel &> /dev/null; then
        printf '%s\n' "${GROUPS[@]}" | parallel -j $MAX_JOBS run_group_job {}
    else
        for group in "${GROUPS[@]}"; do
            run_group_job $group &

            # Limit concurrent jobs
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 10
            done
        done
        wait
    fi

    echo "All Group LOO jobs completed!"

    # Aggregate results
    python -c "
import json
import numpy as np
from pathlib import Path

groups = []
for model in ['claude-4.5', 'gpt-4o', 'gpt-oss-120b']:
    for env in ['afm', 'catalyst', 'md', 'ml', 'resistor', 'retro', 'spectra']:
        for scaffold in ['react', 'tool_calling']:
            groups.append(f'{model}_{env}_{scaffold}')

results = []
for group in groups:
    with open(f'results/eval/group/{group}.json') as f:
        results.append(json.load(f))

valid_results = [r for r in results if r['metrics'] is not None]
if valid_results:
    all_metrics = [r['metrics'] for r in valid_results]
    summary = {
        metric: float(np.mean([m[metric] for m in all_metrics]))
        for metric in all_metrics[0]
    }
    summary['n_groups'] = len(valid_results)

    with open('results/eval/group_summary.json', 'w') as f:
        json.dump({'results': results, 'summary': summary}, f, indent=2)

    print('\n' + '=' * 80)
    print('GROUP LOO-CV SUMMARY')
    print('=' * 80)
    for metric, value in summary.items():
        if metric != 'n_groups':
            print(f'  {metric:20s}: {value:.4f}')
    print(f'  n_groups            : {summary[\"n_groups\"]}')
"

else
    echo "Usage: $0 {kfold|group} [max_jobs]"
    exit 1
fi
