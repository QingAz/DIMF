#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=unified_zero
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/r41_unified_zero_loss_sweep}"
LAMBDAS="${LAMBDAS:-1.0,2.0,5.0}"
RUNS="${RUNS:-old,seed134_e2}"
EPOCHS="${EPOCHS:-120}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-42}"

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

OUTPUT_RESOLVED="$(resolve_path "$OUTPUT_ROOT")"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Runtime Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_RESOLVED}"
echo "Lambdas: ${LAMBDAS}"
echo "Runs: ${RUNS}"
echo "Epochs: ${EPOCHS}"
echo "Device: ${DEVICE}"
"$PYTHON_BIN" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

srun "$PYTHON_BIN" -u scripts/run_unified_zero_loss_sweep.py \
    --lambdas "$LAMBDAS" \
    --runs "$RUNS" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --output-root "$OUTPUT_RESOLVED"

echo "Zero-loss sweep summary:"
sed -n '1,20p' "$OUTPUT_RESOLVED/lambda_zero_sweep_summary.csv"

echo "Job finished at: $(date)"
