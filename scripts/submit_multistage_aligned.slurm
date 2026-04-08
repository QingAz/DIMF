#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=dimf_ms_align
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
CONFIG_PATH="${CONFIG_PATH:-configs/multistage_aligned.yaml}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

module purge
module load py-torch/2.5.1-openmpi-gcc-12.4.0

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SYSTEM_PYTHON="$(which python)"

ensure_runtime_python() {
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        echo "Creating project virtualenv at $VENV_DIR"
        "$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"
    fi

    PYTHON_BIN="$VENV_DIR/bin/python"
}

ensure_python_deps() {
    ensure_runtime_python

    if "$PYTHON_BIN" -c "import numpy, pandas, sklearn, yaml, joblib, tqdm" >/dev/null 2>&1; then
        return 0
    fi

    echo "Installing missing Python packages into $VENV_DIR"
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install --upgrade --prefer-binary \
        numpy \
        pandas \
        scikit-learn \
        pyyaml \
        tqdm \
        joblib
}

ensure_runtime_python

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "System Python path: ${SYSTEM_PYTHON}"
echo "Runtime Python path: ${PYTHON_BIN}"
echo "Runtime venv: ${VENV_DIR}"
echo "Config path: ${CONFIG_PATH}"
echo "Device: ${DEVICE}"

"$PYTHON_BIN" -c "import sys; print('Python:', sys.version)"
"$PYTHON_BIN" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
"$PYTHON_BIN" -m pip --version
ensure_python_deps
"$PYTHON_BIN" -c "import numpy, pandas, sklearn, yaml, joblib, tqdm; print('Dependencies look good.')"

srun "$PYTHON_BIN" -u train.py \
    --config "${CONFIG_PATH}" \
    --device "${DEVICE}"

echo "Job finished at: $(date)"
