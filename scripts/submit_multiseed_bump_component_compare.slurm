#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=ms_bump_cmp
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
mkdir -p "$PROJECT_ROOT/logs"
VENV_DIR="$PROJECT_ROOT/.venv-transfer-eval"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/r50_multiseed_bump_compare}"
CONFIGS="${CONFIGS:-configs/r10_detlocal_v3_seed42.yaml,configs/r10_detlocal_v3_seed134.yaml,configs/r10_detlocal_v3_seed321.yaml,configs/r10_detlocal_v3_seed456.yaml,configs/r10_detlocal_v3_seed712.yaml}"
BUMP_CSV="${BUMP_CSV:-data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_rawgap.csv}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

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
  if "$PYTHON_BIN" -c "import joblib, numpy, pandas, yaml, sklearn, matplotlib, six" >/dev/null 2>&1; then
    return 0
  fi
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install --upgrade --prefer-binary joblib numpy pandas pyyaml scikit-learn matplotlib six
}

ensure_python_deps

CMD=(
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_multiseed_bump_component_compare.py"
  --configs "$CONFIGS"
  --bump-csv "$BUMP_CSV"
  --output-root "$OUTPUT_ROOT"
  --python-bin "$PYTHON_BIN"
  --device "$DEVICE"
)

if [ "$SKIP_EXISTING" = "1" ]; then
  CMD+=(--skip-existing)
fi

MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1 "${CMD[@]}"
