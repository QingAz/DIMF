#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=plateau_seed
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
SEED="${SEED:-134}"
PREFIX="${PREFIX:-r32_seed${SEED}}"
DEVICE="${DEVICE:-cpu}"
ROW_EPOCHS="${ROW_EPOCHS:-2}"
MAG_EPOCHS="${MAG_EPOCHS:-2}"

TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-outputs/r15_candidate_generator_high_recall_smoke3/candidate_train_scored.csv}"
EVAL_CANDIDATES="${EVAL_CANDIDATES:-outputs/r15_candidate_generator_high_recall_smoke3/candidate_eval_scored.csv}"

ROW_DIR="outputs/${PREFIX}_row_localization_head"
MAG_DIR="outputs/${PREFIX}_conditional_magnitude_head"
VETO_DIR="outputs/${PREFIX}_light_veto_filter"
PLATEAU_DIR="outputs/${PREFIX}_weak_lag_plateau_probe"

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
    if "$PYTHON_BIN" -c "import numpy, pandas, sklearn, yaml, joblib, tqdm, matplotlib, torch" >/dev/null 2>&1; then
        return 0
    fi
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install --upgrade --prefer-binary \
        numpy \
        pandas \
        scikit-learn \
        pyyaml \
        tqdm \
        joblib \
        matplotlib
}

ensure_runtime_python
ensure_python_deps

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Seed: ${SEED}"
echo "Prefix: ${PREFIX}"
echo "Device: ${DEVICE}"
echo "Runtime Python: ${PYTHON_BIN}"
"$PYTHON_BIN" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "Training row localization head..."
srun "$PYTHON_BIN" -u scripts/run_row_localization_head.py \
    --train-series "$TRAIN_CANDIDATES" \
    --eval-series "$EVAL_CANDIDATES" \
    --output-dir "$ROW_DIR" \
    --head gru \
    --seed "$SEED" \
    --epochs "$ROW_EPOCHS" \
    --device "$DEVICE"

echo "Training conditional magnitude head..."
srun "$PYTHON_BIN" -u scripts/run_conditional_magnitude_head.py \
    --train-series "$ROW_DIR/row_localization_train_scored.csv" \
    --eval-series "$ROW_DIR/row_localization_eval_scored.csv" \
    --output-dir "$MAG_DIR" \
    --head gru \
    --seed "$SEED" \
    --epochs "$MAG_EPOCHS" \
    --device "$DEVICE"

echo "Applying light veto filter..."
"$PYTHON_BIN" -u scripts/run_light_veto_filter.py \
    --train-series "$MAG_DIR/conditional_magnitude_train_scored.csv" \
    --eval-series "$MAG_DIR/conditional_magnitude_eval_scored.csv" \
    --output-dir "$VETO_DIR"

echo "Running weak-lag plateau ablation..."
"$PYTHON_BIN" -u scripts/run_weak_lag_plateau_probe.py \
    --train-series "$VETO_DIR/light_veto_train_filtered.csv" \
    --eval-series "$VETO_DIR/light_veto_eval_filtered.csv" \
    --output-dir "$PLATEAU_DIR"

echo "Weak-lag plateau ablation:"
sed -n '1,10p' "$PLATEAU_DIR/weak_lag_plateau_ablation.csv"

echo "Best plateau segment:"
sed -n '1,10p' "$PLATEAU_DIR/weak_lag_plateau_best_segments.csv"

echo "Job finished at: $(date)"
