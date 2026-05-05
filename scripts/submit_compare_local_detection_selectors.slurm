#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=det_selcmp
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
DEVICE="${DEVICE:-cpu}"
LATE_MIN_EPOCH="${LATE_MIN_EPOCH:-30}"

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
    if "$PYTHON_BIN" -c "import numpy, pandas, matplotlib, torch, yaml" >/dev/null 2>&1; then
        return 0
    fi
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install --upgrade --prefer-binary numpy pandas matplotlib pyyaml
}

ensure_runtime_python
ensure_python_deps

echo "Job started at: $(date)"
echo "Config path: ${CONFIG_PATH}"
echo "Run dir: ${RUN_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

"$PYTHON_BIN" -u scripts/compare_local_detection_selectors.py \
    --config "$CONFIG_PATH" \
    --run-dir "$RUN_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --late-min-epoch "$LATE_MIN_EPOCH"

echo "Job finished at: $(date)"
