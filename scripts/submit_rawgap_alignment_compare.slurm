#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=dimf_rawgap_align
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
LAG_ROOT="/projects/p32954/liquidsugar_lag_framework"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
DEVICE="${DEVICE:-cuda}"

LAG_OUTPUT_DIR="$LAG_ROOT/outputs_stdlib/liquidsugar_scheme_b_stage2_lag_compare"
FULL_LAG_CSV="$LAG_OUTPUT_DIR/raw/liquidsugar_stage2_lagged.csv"
RAWGAP_CSV="$PROJECT_ROOT/data/processed/LiquidSugar_stage2_lag_rawgap.csv"
COMPARE_OUTPUT_DIR="$PROJECT_ROOT/outputs/rawgap_stage2lag_alignment_compare"

ALIGNED_CONFIG="$PROJECT_ROOT/configs/multistage_rawgap_stage2lag_aligned.yaml"
NOALIGN_CONFIG="$PROJECT_ROOT/configs/multistage_rawgap_stage2lag_noalign.yaml"

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
echo "Runtime Python path: ${PYTHON_BIN}"
echo "Runtime venv: ${VENV_DIR}"
echo "Device: ${DEVICE}"

"$PYTHON_BIN" -c "import sys; print('Python:', sys.version)"
"$PYTHON_BIN" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
ensure_python_deps

echo "Building lagged LiquidSugar dataset..."
"$PYTHON_BIN" -u "$LAG_ROOT/run_stage2_lag_stdlib.py" \
    --config "$LAG_ROOT/configs/liquidsugar_lag.yaml" \
    --output-dir "$LAG_OUTPUT_DIR"

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

echo "Comparing alignment effect..."
"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/compare_alignment_lag_effect.py" \
    --aligned-estimates "$PROJECT_ROOT/outputs/rawgap_stage2lag_aligned/test_delay_estimates.csv" \
    --noalign-estimates "$PROJECT_ROOT/outputs/rawgap_stage2lag_noalign/test_delay_estimates.csv" \
    --aligned-metrics "$PROJECT_ROOT/outputs/rawgap_stage2lag_aligned/test_metrics.json" \
    --noalign-metrics "$PROJECT_ROOT/outputs/rawgap_stage2lag_noalign/test_metrics.json" \
    --raw-dataset "$RAWGAP_CSV" \
    --output-dir "$COMPARE_OUTPUT_DIR"

echo "Comparison report preview:"
sed -n '1,220p' "$COMPARE_OUTPUT_DIR/alignment_comparison_report.md"

echo "Job finished at: $(date)"
