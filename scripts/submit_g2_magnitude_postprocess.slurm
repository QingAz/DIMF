#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=g2_mag_post
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
CONFIG_PATH="${CONFIG_PATH:-configs/r4_g2_mag_only_pos_seed42.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-artifacts/r4_g2_mag_only_pos_seed42/best.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r11_g2_magnitude_postprocess}"
DEVICE="${DEVICE:-cuda}"
EDGE="${EDGE:-stage1_to_stage2}"

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
    if "$PYTHON_BIN" -c "import numpy, pandas, yaml, joblib, tqdm, matplotlib" >/dev/null 2>&1; then
        return 0
    fi
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install --upgrade --prefer-binary \
        numpy \
        pandas \
        pyyaml \
        tqdm \
        joblib \
        matplotlib
}

resolve_path() {
    "$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$1" "$PROJECT_ROOT"
}

ensure_runtime_python
ensure_python_deps

CONFIG_RESOLVED="$(resolve_path "$CONFIG_PATH")"
CHECKPOINT_RESOLVED="$(resolve_path "$CHECKPOINT_PATH")"
OUTPUT_RESOLVED="$(resolve_path "$OUTPUT_DIR")"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Config path: ${CONFIG_RESOLVED}"
echo "Checkpoint path: ${CHECKPOINT_RESOLVED}"
echo "Output dir: ${OUTPUT_RESOLVED}"
echo "Edge: ${EDGE}"
echo "Device: ${DEVICE}"

"$PYTHON_BIN" -c "import sys; print('Python:', sys.version)"
"$PYTHON_BIN" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

srun "$PYTHON_BIN" -u scripts/run_g2_magnitude_postprocess.py \
    --config "$CONFIG_RESOLVED" \
    --checkpoint "$CHECKPOINT_RESOLVED" \
    --output-dir "$OUTPUT_RESOLVED" \
    --device "$DEVICE" \
    --edge "$EDGE"

echo "Top val grid rows:"
sed -n '1,12p' "$OUTPUT_RESOLVED/g2_postproc_val_grid.csv"

echo "Selected test rows:"
sed -n '1,12p' "$OUTPUT_RESOLVED/g2_postproc_test_selected.csv"

echo "Final comparison:"
sed -n '1,20p' "$OUTPUT_RESOLVED/g2_postproc_final_comparison.csv"

echo "Job finished at: $(date)"
