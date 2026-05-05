#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=transfer_eval
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
mkdir -p "$PROJECT_ROOT/logs"
VENV_DIR="$PROJECT_ROOT/.venv-transfer-eval"

CONFIG="${CONFIG:?CONFIG is required}"
CSV_PATH="${CSV_PATH:?CSV_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
SPLITS="${SPLITS:?SPLITS is required}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-}"
SCALER_PATH="${SCALER_PATH:-}"

module purge
module load py-torch/2.5.1-openmpi-gcc-12.4.0

SYSTEM_PYTHON="$(which python)"

ensure_runtime_python() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
}

ensure_python_deps() {
  ensure_runtime_python
  if "$PYTHON_BIN" -c "import joblib, numpy, pandas, yaml, sklearn" >/dev/null 2>&1; then
    return 0
  fi
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install --upgrade --prefer-binary joblib numpy pandas pyyaml scikit-learn
}

ensure_python_deps

PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_transfer_checkpoint_eval.py" \
  --config "$CONFIG" \
  --csv-path "$CSV_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --splits "$SPLITS" \
  --device "$DEVICE" \
  ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
  ${SCALER_PATH:+--scaler-path "$SCALER_PATH"}
