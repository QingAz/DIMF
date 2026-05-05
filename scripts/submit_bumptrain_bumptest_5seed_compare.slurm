#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=bump5_cmp
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
mkdir -p "$PROJECT_ROOT/logs"
VENV_DIR="$PROJECT_ROOT/.venv-transfer-eval"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/r51_bumptrain_bumptest_5seed_compare}"
BUMP_CSV="${BUMP_CSV:-data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_rawgap.csv}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
CONFIGS="${CONFIGS:-outputs/localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_evalsafe_5seed/_generated_configs/seed_42.yaml,outputs/localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_evalsafe_5seed/_generated_configs/seed_134.yaml,outputs/localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_evalsafe_5seed/_generated_configs/seed_321.yaml,outputs/localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_evalsafe_5seed/_generated_configs/seed_456.yaml,outputs/localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_evalsafe_5seed/_generated_configs/seed_712.yaml}"
CONFIGS_DIR="${CONFIGS_DIR:-}"

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

if [ -n "$CONFIGS_DIR" ]; then
  CONFIGS="$("$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

root = Path("/projects/p32954/dimf_liquid_sugar_repo-490")
configs_dir = Path(os.environ["CONFIGS_DIR"])
if not configs_dir.is_absolute():
    configs_dir = (root / configs_dir).resolve()
paths = sorted(str(path.resolve()) for path in configs_dir.glob("seed_*.yaml"))
if not paths:
    raise SystemExit(f"No seed_*.yaml files found in {configs_dir}")
print(",".join(paths))
PY
)"
fi

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
