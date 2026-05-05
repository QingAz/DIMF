#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=det_local_diag
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
RUN_DIRS="${RUN_DIRS:?RUN_DIRS is required}"
DEVICE="${DEVICE:-cpu}"
SUCCESS_PANEL_SEEDS="${SUCCESS_PANEL_SEEDS:-2}"
FAILURE_PANEL_SEEDS="${FAILURE_PANEL_SEEDS:-2}"
PANELS_PER_SEED="${PANELS_PER_SEED:-3}"
MAX_CHECKPOINTS_PER_SEED="${MAX_CHECKPOINTS_PER_SEED:-0}"

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
echo "Output dir: ${OUTPUT_DIR}"
echo "Run dirs: ${RUN_DIRS}"

"$PYTHON_BIN" -u scripts/run_local_detection_followup_diagnostics.py \
    --run-dirs ${RUN_DIRS} \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --success-panel-seeds "$SUCCESS_PANEL_SEEDS" \
    --failure-panel-seeds "$FAILURE_PANEL_SEEDS" \
    --panels-per-seed "$PANELS_PER_SEED" \
    --max-checkpoints-per-seed "$MAX_CHECKPOINTS_PER_SEED"

echo "Job finished at: $(date)"
