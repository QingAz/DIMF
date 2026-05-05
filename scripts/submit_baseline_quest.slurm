#!/bin/bash
#SBATCH --account=p32954
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --job-name=baseline
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest-baselines"
BASELINE_METHOD="${BASELINE_METHOD:?BASELINE_METHOD is required}"
CONFIG_PATH="${CONFIG_PATH:-configs/default.yaml}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-}"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"
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
        echo "Creating project virtualenv at $VENV_DIR"
        "$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"
    fi
    PYTHON_BIN="$VENV_DIR/bin/python"
}

ensure_python_deps() {
    ensure_runtime_python

    local base_modules=(numpy pandas sklearn yaml joblib)
    if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib
modules = ["numpy", "pandas", "sklearn", "yaml", "joblib"]
for name in modules:
    importlib.import_module(name)
PY
    then
        echo "Installing baseline core Python packages into $VENV_DIR"
        "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
        "$PYTHON_BIN" -m pip install --upgrade --prefer-binary \
            numpy \
            pandas \
            scikit-learn \
            pyyaml \
            joblib
    fi

    case "$BASELINE_METHOD" in
        arima)
            if ! "$PYTHON_BIN" -c "import statsmodels" >/dev/null 2>&1; then
                "$PYTHON_BIN" -m pip install --upgrade --prefer-binary statsmodels
            fi
            ;;
        lightgbm)
            if ! "$PYTHON_BIN" -c "import lightgbm" >/dev/null 2>&1; then
                "$PYTHON_BIN" -m pip install --upgrade --prefer-binary lightgbm
            fi
            ;;
        xgboost)
            if ! "$PYTHON_BIN" -c "import xgboost" >/dev/null 2>&1; then
                "$PYTHON_BIN" -m pip install --upgrade --prefer-binary xgboost
            fi
            ;;
    esac
}

ensure_runtime_python
ensure_python_deps

SCRIPT_PATH="$PROJECT_ROOT/baseline/${BASELINE_METHOD}.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Unknown baseline method: $BASELINE_METHOD" >&2
    exit 1
fi

CONFIG_RESOLVED="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$CONFIG_PATH" "$PROJECT_ROOT")"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Runtime Python path: ${PYTHON_BIN}"
echo "Runtime venv: ${VENV_DIR}"
echo "Baseline method: ${BASELINE_METHOD}"
echo "Config path: ${CONFIG_RESOLVED}"
echo "Device: ${DEVICE}"
if [ -n "$SEEDS" ]; then
    echo "Seeds override: ${SEEDS}"
fi

"$PYTHON_BIN" -c "import sys; print('Python:', sys.version)"
"$PYTHON_BIN" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

SEED_ARGS=()
if [ -n "$SEEDS" ]; then
    IFS=',' read -r -a SEED_VALUES <<< "$SEEDS"
    SEED_ARGS=(--seeds "${SEED_VALUES[@]}")
fi

case "$BASELINE_METHOD" in
    gru|lstm|tcn|transformer)
        CMD=("$PYTHON_BIN" -u "$SCRIPT_PATH" --config "$CONFIG_RESOLVED" --device "$DEVICE")
        ;;
    *)
        CMD=("$PYTHON_BIN" -u "$SCRIPT_PATH" --config "$CONFIG_RESOLVED")
        ;;
esac

if [ "${#SEED_ARGS[@]}" -gt 0 ]; then
    CMD+=("${SEED_ARGS[@]}")
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
printf '\n'

srun "${CMD[@]}"

echo "Finished at: $(date)"
