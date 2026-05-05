#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=unified_hneg
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r42_unified_hardneg_sampling}"
RUNS="${RUNS:-old,seed134_e2}"
EPOCHS="${EPOCHS:-120}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-42}"
ZERO_LOSS_WEIGHT="${ZERO_LOSS_WEIGHT:-0.0}"
RELATIVE_FEATURES="${RELATIVE_FEATURES:-0}"
Q40_PRIOR_FEATURES="${Q40_PRIOR_FEATURES:-0}"
POS_LOSS_WEIGHT="${POS_LOSS_WEIGHT:-2.0}"
HARD_NEG_LOSS_WEIGHT="${HARD_NEG_LOSS_WEIGHT:-1.0}"
MEDIUM_NEG_LOSS_WEIGHT="${MEDIUM_NEG_LOSS_WEIGHT:-0.5}"
EASY_NEG_LOSS_WEIGHT="${EASY_NEG_LOSS_WEIGHT:-0.5}"
MAG_LOSS_WEIGHT="${MAG_LOSS_WEIGHT:-1.0}"
RATE_LOSS_WEIGHT="${RATE_LOSS_WEIGHT:-0.0}"
RATE_TARGET_SLACK="${RATE_TARGET_SLACK:-0.10}"
RANK_LOSS_WEIGHT="${RANK_LOSS_WEIGHT:-0.0}"
RANK_MARGIN="${RANK_MARGIN:-0.10}"
TEACHER_LOSS_WEIGHT="${TEACHER_LOSS_WEIGHT:-0.0}"
HARD_NEG_POSITIVE_FRACTION="${HARD_NEG_POSITIVE_FRACTION:-0.25}"
HARD_NEG_FRACTION="${HARD_NEG_FRACTION:-0.50}"
MEDIUM_NEG_FRACTION="${MEDIUM_NEG_FRACTION:-0.0}"
EASY_NEG_FRACTION="${EASY_NEG_FRACTION:-0.25}"
HARD_NEG_TOP_FRACTION="${HARD_NEG_TOP_FRACTION:-0.30}"
EASY_NEG_BOTTOM_FRACTION="${EASY_NEG_BOTTOM_FRACTION:-0.30}"
HARD_NEG_MAX_PER_POSITIVE="${HARD_NEG_MAX_PER_POSITIVE:-20.0}"
D_FLOORS="${D_FLOORS:-0.0,0.5,0.75,1.0,1.25}"
RANK_THRESHOLDS="${RANK_THRESHOLDS:-0.0}"
VAL_RECALL_MIN="${VAL_RECALL_MIN:-0.5}"
VAL_FAR_MAX="${VAL_FAR_MAX:-0.6}"
VAL_MAE_WEIGHT="${VAL_MAE_WEIGHT:-0.3}"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

module purge
module load py-torch/2.5.1-openmpi-gcc-12.4.0

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"
SYSTEM_PYTHON="$(which python)"

ensure_runtime_python() {
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        "$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"
    fi
    PYTHON_BIN="$VENV_DIR/bin/python"
}

ensure_python_deps() {
    ensure_runtime_python
    if "$PYTHON_BIN" -c "import numpy, pandas, sklearn, matplotlib, torch" >/dev/null 2>&1; then
        return 0
    fi
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install --upgrade --prefer-binary \
        numpy \
        pandas \
        scikit-learn \
        matplotlib
}

resolve_path() {
    "$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$1" "$PROJECT_ROOT"
}

ensure_runtime_python
ensure_python_deps

OUTPUT_RESOLVED="$(resolve_path "$OUTPUT_DIR")"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Runtime Python: ${PYTHON_BIN}"
echo "Output dir: ${OUTPUT_RESOLVED}"
echo "Runs: ${RUNS}"
echo "Epochs: ${EPOCHS}"
echo "Device: ${DEVICE}"
echo "Zero loss weight: ${ZERO_LOSS_WEIGHT}"
echo "Relative features: ${RELATIVE_FEATURES}"
echo "Q40 prior features: ${Q40_PRIOR_FEATURES}"
echo "Loss weights: pos=${POS_LOSS_WEIGHT} hard=${HARD_NEG_LOSS_WEIGHT} medium=${MEDIUM_NEG_LOSS_WEIGHT} easy=${EASY_NEG_LOSS_WEIGHT} mag=${MAG_LOSS_WEIGHT} rate=${RATE_LOSS_WEIGHT} rank=${RANK_LOSS_WEIGHT}"
echo "Sampling fractions: pos=${HARD_NEG_POSITIVE_FRACTION} hard=${HARD_NEG_FRACTION} medium=${MEDIUM_NEG_FRACTION} easy=${EASY_NEG_FRACTION}"
"$PYTHON_BIN" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

EXTRA_ARGS=()
if [ "${RELATIVE_FEATURES}" = "1" ]; then
    EXTRA_ARGS+=(--relative-features)
fi
if [ "${Q40_PRIOR_FEATURES}" = "1" ]; then
    EXTRA_ARGS+=(--q40-prior-features)
fi

srun "$PYTHON_BIN" -u scripts/run_unified_block_lag_scorer.py \
    --runs "$RUNS" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --zero-loss-weight "$ZERO_LOSS_WEIGHT" \
    --pos-loss-weight "$POS_LOSS_WEIGHT" \
    --hard-neg-loss-weight "$HARD_NEG_LOSS_WEIGHT" \
    --medium-neg-loss-weight "$MEDIUM_NEG_LOSS_WEIGHT" \
    --easy-neg-loss-weight "$EASY_NEG_LOSS_WEIGHT" \
    --mag-loss-weight "$MAG_LOSS_WEIGHT" \
    --rate-loss-weight "$RATE_LOSS_WEIGHT" \
    --rate-target-slack "$RATE_TARGET_SLACK" \
    --rank-loss-weight "$RANK_LOSS_WEIGHT" \
    --rank-margin "$RANK_MARGIN" \
    --teacher-loss-weight "$TEACHER_LOSS_WEIGHT" \
    --hard-negative-sampling \
    --hard-neg-positive-fraction "$HARD_NEG_POSITIVE_FRACTION" \
    --hard-neg-fraction "$HARD_NEG_FRACTION" \
    --medium-neg-fraction "$MEDIUM_NEG_FRACTION" \
    --easy-neg-fraction "$EASY_NEG_FRACTION" \
    --hard-neg-top-fraction "$HARD_NEG_TOP_FRACTION" \
    --easy-neg-bottom-fraction "$EASY_NEG_BOTTOM_FRACTION" \
    --hard-neg-max-per-positive "$HARD_NEG_MAX_PER_POSITIVE" \
    --d-floors "$D_FLOORS" \
    --rank-thresholds "$RANK_THRESHOLDS" \
    --val-recall-min "$VAL_RECALL_MIN" \
    --val-far-max "$VAL_FAR_MAX" \
    --val-mae-weight "$VAL_MAE_WEIGHT" \
    --output-dir "$OUTPUT_RESOLVED" \
    "${EXTRA_ARGS[@]}"

echo "Unified hard-negative summary:"
sed -n '1,20p' "$OUTPUT_RESOLVED/unified_vs_q40_eval_summary.csv"

echo "Job finished at: $(date)"
