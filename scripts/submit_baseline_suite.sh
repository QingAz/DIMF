#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
SCRIPT_PATH="$PROJECT_ROOT/scripts/submit_baseline_quest.slurm"
CONFIG_PATH="${CONFIG_PATH:-configs/default.yaml}"
SEEDS="${SEEDS:-}"
ACCOUNT="${ACCOUNT:-p32954}"

SHORT_CPU_METHODS=(naive linear_regression random_forest lightgbm xgboost)
LONG_METHODS=(arima)
GPU_METHODS=(gru lstm tcn transformer)

mkdir -p "$PROJECT_ROOT/logs"

submit_job() {
    local method="$1"
    local device="$2"
    local export_vars="ALL,BASELINE_METHOD=$method,CONFIG_PATH=$CONFIG_PATH"
    shift
    shift
    local job_output
    if [ -n "$device" ]; then
        export_vars="${export_vars},DEVICE=${device}"
    fi
    if [ -n "$SEEDS" ]; then
        export_vars="${export_vars},SEEDS=${SEEDS}"
    fi
    job_output="$(sbatch --account="$ACCOUNT" --export="$export_vars" "$@" "$SCRIPT_PATH")"
    echo "$method -> $job_output"
}

echo "Submitting short CPU baselines..."
for method in "${SHORT_CPU_METHODS[@]}"; do
    submit_job "$method" "" \
        --job-name="b_${method}" \
        --partition=short \
        --cpus-per-task=8 \
        --mem=32G \
        --time=02:00:00
done

echo "Submitting long baseline jobs..."
for method in "${LONG_METHODS[@]}"; do
    submit_job "$method" "" \
        --job-name="b_${method}" \
        --partition=gengpu \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=08:00:00
done

echo "Submitting GPU baselines..."
for method in "${GPU_METHODS[@]}"; do
    submit_job "$method" "cuda" \
        --job-name="b_${method}" \
        --partition=gengpu \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=16:00:00
done
