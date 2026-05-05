#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=det_segment_audit
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH is required}"
DEVICE="${DEVICE:-cuda}"
EDGE="${EDGE:-stage1_to_stage2}"
TARGET_KEY="${TARGET_KEY:-stage1_to_stage2_in_block_gt}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-segment_detection_audit}"
CONTEXT_SEGMENTS="${CONTEXT_SEGMENTS:-1}"
LOW_AUROC_K="${LOW_AUROC_K:-3}"
HIGH_GAP_K="${HIGH_GAP_K:-2}"

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
        echo "Creating project virtualenv at $VENV_DIR"
        "$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"
    fi

    PYTHON_BIN="$VENV_DIR/bin/python"
}

ensure_python_deps() {
    ensure_runtime_python

    if "$PYTHON_BIN" -c "import numpy, pandas, yaml, matplotlib" >/dev/null 2>&1; then
        return 0
    fi

    echo "Installing missing Python packages into $VENV_DIR"
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install --upgrade --prefer-binary \
        numpy \
        pandas \
        pyyaml \
        matplotlib
}

ensure_runtime_python
ensure_python_deps

CONFIG_RESOLVED="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$CONFIG_PATH" "$PROJECT_ROOT")"
RUN_OUTPUT_DIR="$("$PYTHON_BIN" -c 'import sys, yaml; from pathlib import Path; cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")); print(cfg["logging"]["output_dir"])' "$CONFIG_RESOLVED")"
RUN_OUTPUT_ABS="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$RUN_OUTPUT_DIR" "$PROJECT_ROOT")"
AUDIT_DIR="$RUN_OUTPUT_ABS/$OUTPUT_SUBDIR"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Runtime Python path: ${PYTHON_BIN}"
echo "Config path: ${CONFIG_RESOLVED}"
echo "Run output dir: ${RUN_OUTPUT_ABS}"
echo "Audit output dir: ${AUDIT_DIR}"
echo "Device: ${DEVICE}"
echo "Edge: ${EDGE}"
echo "Target key: ${TARGET_KEY}"

"$PYTHON_BIN" -c "import sys; print('Python:', sys.version)"
"$PYTHON_BIN" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

mkdir -p "$AUDIT_DIR"

echo "Running detection segment audit..."
srun "$PYTHON_BIN" -u scripts/run_detection_segment_audit.py \
    --config "$CONFIG_RESOLVED" \
    --output-dir "$AUDIT_DIR" \
    --edge "$EDGE" \
    --target-key "$TARGET_KEY" \
    --device "$DEVICE" \
    --context-segments "$CONTEXT_SEGMENTS" \
    --low-auroc-k "$LOW_AUROC_K" \
    --high-gap-k "$HIGH_GAP_K"

echo "Split detection summary:"
sed -n '1,20p' "$AUDIT_DIR/split_detection_summary.csv"

echo "Worst test segments:"
sed -n '1,20p' "$AUDIT_DIR/worst_test_segments.csv"

echo "Job finished at: $(date)"
