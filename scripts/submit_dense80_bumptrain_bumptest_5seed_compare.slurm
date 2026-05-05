#!/bin/bash
#SBATCH --account=e30514
#SBATCH --partition=gengpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=dense80_bump5
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
mkdir -p "$PROJECT_ROOT/logs"
VENV_DIR="$PROJECT_ROOT/.venv-transfer-eval"
BASE_CONFIG="${BASE_CONFIG:-configs/multistage_localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_dense80.yaml}"
RUN_SUFFIX="${RUN_SUFFIX:-5seed}"
BUMP_CSV="${BUMP_CSV:-data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_dense80_rawgap.csv}"
COMPARE_OUTPUT_ROOT="${COMPARE_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/r52_dense80_bumptrain_bumptest_5seed_compare}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42 134 321 456 712}"
SKIP_TRAIN_IF_PRESENT="${SKIP_TRAIN_IF_PRESENT:-1}"
TARGET_POSITIVE_FRACTION="${TARGET_POSITIVE_FRACTION:-0.80}"
INPUT_FULL="${INPUT_FULL:-data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_dense40_full.csv}"
OUTPUT_FULL="${OUTPUT_FULL:-data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_dense80_full.csv}"
FORCE_REGENERATE_DENSE80="${FORCE_REGENERATE_DENSE80:-0}"

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
  if "$PYTHON_BIN" -c "import joblib, numpy, pandas, yaml, sklearn, matplotlib, six, torch" >/dev/null 2>&1; then
    return 0
  fi
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install --upgrade --prefer-binary joblib numpy pandas pyyaml scikit-learn matplotlib six
}

ensure_python_deps

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/mpl-${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"

if [ "$FORCE_REGENERATE_DENSE80" = "1" ] || [ ! -f "$PROJECT_ROOT/$OUTPUT_FULL" ] || [ ! -f "$PROJECT_ROOT/$BUMP_CSV" ]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/create_dense_bump_dataset.py" \
    --input-full "$PROJECT_ROOT/$INPUT_FULL" \
    --output-full "$PROJECT_ROOT/$OUTPUT_FULL" \
    --output-rawgap "$PROJECT_ROOT/$BUMP_CSV" \
    --target-positive-fraction "$TARGET_POSITIVE_FRACTION"
fi

echo "Starting dense80 5-seed bump train/test compare at $(date)"
echo "Base config: $BASE_CONFIG"
echo "Compare output root: $COMPARE_OUTPUT_ROOT"
echo "Bump CSV: $BUMP_CSV"
echo "Target positive fraction: $TARGET_POSITIVE_FRACTION"
echo "Seeds: $SEEDS"

export BASE_CONFIG
export RUN_SUFFIX
export SEEDS

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_multiseed.py" \
  --config "$BASE_CONFIG" \
  --device "$DEVICE" \
  --run-suffix "$RUN_SUFFIX" \
  --seeds $SEEDS

GENERATED_CONFIGS="$("$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import yaml

root = Path("/projects/p32954/dimf_liquid_sugar_repo-490")
base_config = os.environ["BASE_CONFIG"]
run_suffix = os.environ["RUN_SUFFIX"]
seed_values = [int(part) for part in os.environ["SEEDS"].split()]
cfg = yaml.safe_load((root / base_config).read_text(encoding="utf-8"))
output_dir = Path(cfg["logging"]["output_dir"])
output_root = output_dir.with_name(f"{output_dir.name}_{run_suffix}")
gen = (root / output_root / "_generated_configs").resolve()
paths = [str((gen / f"seed_{seed}.yaml").resolve()) for seed in seed_values]
print(",".join(paths))
PY
)"

COMPARE_CMD=(
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_multiseed_bump_component_compare.py"
  --configs "$GENERATED_CONFIGS"
  --bump-csv "$BUMP_CSV"
  --output-root "$COMPARE_OUTPUT_ROOT"
  --python-bin "$PYTHON_BIN"
  --device "$DEVICE"
)

if [ "$SKIP_TRAIN_IF_PRESENT" = "1" ]; then
  COMPARE_CMD+=(--skip-existing)
fi

"${COMPARE_CMD[@]}"

echo "Finished dense80 5-seed bump train/test compare at $(date)"
