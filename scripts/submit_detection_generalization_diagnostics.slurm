#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=det_gen_diag
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
SCORES_PATH="${SCORES_PATH:?SCORES_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
K_LIST="${K_LIST:-4,8}"

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
    if "$PYTHON_BIN" -c "import numpy, pandas, sklearn, matplotlib" >/dev/null 2>&1; then
        return 0
    fi
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install --upgrade --prefer-binary numpy pandas scikit-learn matplotlib
}

ensure_runtime_python
ensure_python_deps

SCORES_RESOLVED="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$SCORES_PATH" "$PROJECT_ROOT")"
OUTPUT_RESOLVED="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$OUTPUT_DIR" "$PROJECT_ROOT")"

mkdir -p "$OUTPUT_RESOLVED"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Scores path: ${SCORES_RESOLVED}"
echo "Output dir: ${OUTPUT_RESOLVED}"

srun "$PYTHON_BIN" -u scripts/run_detection_generalization_diagnostics.py \
    --scores "$SCORES_RESOLVED" \
    --output-dir "$OUTPUT_RESOLVED" \
    --k-list "$K_LIST"

echo "Negative composition:"
sed -n '1,20p' "$OUTPUT_RESOLVED/negative_composition_audit.csv"

echo "Shortcut probe metrics:"
sed -n '1,20p' "$OUTPUT_RESOLVED/shortcut_probe_metrics.csv"

echo "Job finished at: $(date)"
