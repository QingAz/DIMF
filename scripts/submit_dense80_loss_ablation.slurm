#!/bin/bash
#SBATCH --account=e30514
#SBATCH --partition=gengpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=d80_lossab
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-transfer-eval"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
BASE_CONFIG="${BASE_CONFIG:-configs/multistage_localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_dense80.yaml}"
VARIANTS="${VARIANTS:-baseline_final,no_ent,no_tv,no_ent_tv,no_align,no_lag_sup,no_lag_posexp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dense80_loss_ablation_seed42}"
COMPARE_ROOT="${COMPARE_ROOT:-outputs/dense80_loss_ablation_compare_seed42}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

module purge
module load py-torch/2.5.1-openmpi-gcc-12.4.0

SYSTEM_PYTHON="$(which python)"
PYTHON_BIN=""

ensure_runtime_python() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
}

ensure_python_deps() {
  ensure_runtime_python
  if "$PYTHON_BIN" -c "import joblib, numpy, pandas, yaml, sklearn, matplotlib, six, torch" >/dev/null 2>&1; then
    return 0
  fi
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install --upgrade --prefer-binary joblib numpy pandas pyyaml scikit-learn matplotlib six
}

ensure_python_deps
PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Expected runtime python not found at $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/mpl-${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"

CMD=(
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_dense80_loss_ablation.py"
  --base-config "$BASE_CONFIG"
  --seed "$SEED"
  --device "$DEVICE"
  --python-bin "$PYTHON_BIN"
  --variants "$VARIANTS"
  --output-root "$OUTPUT_ROOT"
  --compare-root "$COMPARE_ROOT"
)

if [ "$SKIP_EXISTING" = "1" ]; then
  CMD+=(--skip-existing)
fi

echo "Starting dense80 loss ablation at $(date)"
echo "Seed: $SEED"
echo "Base config: $BASE_CONFIG"
echo "Variants: $VARIANTS"
echo "Output root: $OUTPUT_ROOT"
echo "Compare root: $COMPARE_ROOT"
"${CMD[@]}"
echo "Finished dense80 loss ablation at $(date)"
