#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=dimf_framework
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
FRAMEWORK_ROOT="/projects/p32954/liquidsugar_lag_framework"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
DEVICE="${DEVICE:-cuda}"

FRAMEWORK_RUNNER="${FRAMEWORK_RUNNER:?FRAMEWORK_RUNNER is required}"
FRAMEWORK_CONFIG="${FRAMEWORK_CONFIG:?FRAMEWORK_CONFIG is required}"
FRAMEWORK_OUTPUT_DIR="${FRAMEWORK_OUTPUT_DIR:?FRAMEWORK_OUTPUT_DIR is required}"
FULL_LAG_CSV="${FULL_LAG_CSV:?FULL_LAG_CSV is required}"
RAWGAP_CSV="${RAWGAP_CSV:?RAWGAP_CSV is required}"
ALIGNED_CONFIG="${ALIGNED_CONFIG:?ALIGNED_CONFIG is required}"
NOALIGN_CONFIG="${NOALIGN_CONFIG:?NOALIGN_CONFIG is required}"
ALIGNED_OUTPUT_DIR="${ALIGNED_OUTPUT_DIR:?ALIGNED_OUTPUT_DIR is required}"
NOALIGN_OUTPUT_DIR="${NOALIGN_OUTPUT_DIR:?NOALIGN_OUTPUT_DIR is required}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:?COMPARE_OUTPUT_DIR is required}"
VIZ_OUTPUT_DIR="${VIZ_OUTPUT_DIR:?VIZ_OUTPUT_DIR is required}"

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
    if "$PYTHON_BIN" -c "import numpy, pandas, sklearn, yaml, joblib, tqdm, matplotlib" >/dev/null 2>&1; then
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
        joblib \
        matplotlib
}

ensure_runtime_python

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Runtime Python path: ${PYTHON_BIN}"
echo "Runtime venv: ${VENV_DIR}"
echo "Device: ${DEVICE}"
echo "Framework runner: ${FRAMEWORK_RUNNER}"
echo "Framework config: ${FRAMEWORK_CONFIG}"

"$PYTHON_BIN" -c "import sys; print('Python:', sys.version)"
"$PYTHON_BIN" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
ensure_python_deps

echo "Building framework dataset..."
"$PYTHON_BIN" -u "$FRAMEWORK_ROOT/$FRAMEWORK_RUNNER" \
    --config "$FRAMEWORK_ROOT/$FRAMEWORK_CONFIG"

echo "Combining predefined train/val/test CSVs..."
"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/combine_predefined_split_csv.py" \
    --input-dir "$FRAMEWORK_OUTPUT_DIR/raw" \
    --output "$FULL_LAG_CSV"

echo "Dropping interpolated rows to keep raw-gap timeline..."
"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/prepare_raw_gap_lag_dataset.py" \
    --input "$FULL_LAG_CSV" \
    --output "$RAWGAP_CSV"

echo "Training aligned DIMF..."
srun "$PYTHON_BIN" -u "$PROJECT_ROOT/train.py" \
    --config "$ALIGNED_CONFIG" \
    --device "$DEVICE"

echo "Training no-alignment DIMF..."
srun "$PYTHON_BIN" -u "$PROJECT_ROOT/train.py" \
    --config "$NOALIGN_CONFIG" \
    --device "$DEVICE"

echo "Plotting prediction outputs..."
"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/plot_test_predictions.py" \
    --predictions "$ALIGNED_OUTPUT_DIR/test_pred_vs_true.csv" \
    --output-dir "$ALIGNED_OUTPUT_DIR/plots" \
    --title-prefix "aligned"

"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/plot_test_predictions.py" \
    --predictions "$NOALIGN_OUTPUT_DIR/test_pred_vs_true.csv" \
    --output-dir "$NOALIGN_OUTPUT_DIR/plots" \
    --title-prefix "noalign"

echo "Comparing alignment effect..."
"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/compare_alignment_lag_effect.py" \
    --aligned-estimates "$ALIGNED_OUTPUT_DIR/test_delay_estimates.csv" \
    --noalign-estimates "$NOALIGN_OUTPUT_DIR/test_delay_estimates.csv" \
    --aligned-metrics "$ALIGNED_OUTPUT_DIR/test_metrics.json" \
    --noalign-metrics "$NOALIGN_OUTPUT_DIR/test_metrics.json" \
    --raw-dataset "$RAWGAP_CSV" \
    --output-dir "$COMPARE_OUTPUT_DIR"

echo "Rendering comparison visuals..."
"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/visualize_alignment_comparison.py" \
    --summary "$COMPARE_OUTPUT_DIR/alignment_comparison_summary.json" \
    --joined "$COMPARE_OUTPUT_DIR/alignment_test_joined.csv" \
    --per-lag "$COMPARE_OUTPUT_DIR/alignment_per_lag_comparison.csv" \
    --aligned-estimates "$ALIGNED_OUTPUT_DIR/test_delay_estimates.csv" \
    --noalign-estimates "$NOALIGN_OUTPUT_DIR/test_delay_estimates.csv" \
    --output-dir "$VIZ_OUTPUT_DIR"

echo "Comparison report preview:"
sed -n '1,220p' "$COMPARE_OUTPUT_DIR/alignment_comparison_report.md"

echo "Job finished at: $(date)"
