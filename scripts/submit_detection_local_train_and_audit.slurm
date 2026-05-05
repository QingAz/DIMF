#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=det_local_train
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
AUDIT_SUBDIR="${AUDIT_SUBDIR:-detection_selected_audit}"
SELECTOR_RULE="${SELECTOR_RULE:-current}"
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
    if "$PYTHON_BIN" -c "import numpy, pandas, sklearn, yaml, joblib, tqdm, matplotlib" >/dev/null 2>&1; then
        return 0
    fi
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install --upgrade --prefer-binary \
        numpy \
        pandas \
        scikit-learn \
        pyyaml \
        tqdm \
        joblib \
        matplotlib
}

ensure_runtime_python
ensure_python_deps

CONFIG_RESOLVED="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$CONFIG_PATH" "$PROJECT_ROOT")"
RUN_OUTPUT_DIR="$("$PYTHON_BIN" -c 'import sys, yaml; from pathlib import Path; cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")); print(cfg["logging"]["output_dir"])' "$CONFIG_RESOLVED")"
CHECKPOINT_DIR="$("$PYTHON_BIN" -c 'import sys, yaml; from pathlib import Path; cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")); print(cfg["logging"]["checkpoint_dir"])' "$CONFIG_RESOLVED")"
RUN_OUTPUT_ABS="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$RUN_OUTPUT_DIR" "$PROJECT_ROOT")"
CHECKPOINT_DIR_ABS="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$CHECKPOINT_DIR" "$PROJECT_ROOT")"
AUDIT_DIR="$RUN_OUTPUT_ABS/$AUDIT_SUBDIR"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Config path: ${CONFIG_RESOLVED}"
echo "Output dir: ${RUN_OUTPUT_ABS}"
echo "Checkpoint dir: ${CHECKPOINT_DIR_ABS}"
echo "Audit dir: ${AUDIT_DIR}"
echo "Selector rule: ${SELECTOR_RULE}"

echo "Training detection-local model..."
srun "$PYTHON_BIN" -u train.py \
    --config "$CONFIG_RESOLVED" \
    --device "$DEVICE"

mkdir -p "$AUDIT_DIR"

echo "Selecting checkpoint and running detection audit..."
"$PYTHON_BIN" -u scripts/select_and_audit_detection_checkpoints.py \
    --config "$CONFIG_RESOLVED" \
    --checkpoint-dir "$CHECKPOINT_DIR_ABS" \
    --output-dir "$AUDIT_DIR" \
    --edge "$EDGE" \
    --target-key "$TARGET_KEY" \
    --device "$DEVICE" \
    --selector-rule "$SELECTOR_RULE" \
    --late-min-epoch "$LATE_MIN_EPOCH"

echo "Selected checkpoint summary:"
sed -n '1,20p' "$AUDIT_DIR/checkpoint_detection_selection.csv"

echo "Detection split summary:"
sed -n '1,20p' "$AUDIT_DIR/split_detection_summary.csv"

echo "Job finished at: $(date)"
