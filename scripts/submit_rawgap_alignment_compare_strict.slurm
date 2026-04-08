#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=dimf_rawgap_align_strict
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

STRICT_ALIGNED_CONFIG="$PROJECT_ROOT/configs/multistage_rawgap_stage2lag_aligned_strict.yaml"
STRICT_NOALIGN_CONFIG="$PROJECT_ROOT/configs/multistage_rawgap_stage2lag_noalign_strict.yaml"

STRICT_ALIGNED_OUTPUT_DIR="$PROJECT_ROOT/outputs/rawgap_stage2lag_aligned_strict"
STRICT_NOALIGN_OUTPUT_DIR="$PROJECT_ROOT/outputs/rawgap_stage2lag_noalign_strict"
STRICT_COMPARE_OUTPUT_DIR="$PROJECT_ROOT/outputs/rawgap_stage2lag_alignment_compare_strict"
STRICT_VIZ_DIR="$STRICT_COMPARE_OUTPUT_DIR/visuals"
STRICT_VS_LEGACY_DIR="$PROJECT_ROOT/outputs/rawgap_stage2lag_alignment_compare_strict_vs_legacy"

LEGACY_COMPARE_DIR="$PROJECT_ROOT/outputs/rawgap_stage2lag_alignment_compare"

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

echo "Training strict aligned DIMF..."
srun "$PYTHON_BIN" -u "$PROJECT_ROOT/train.py" \
    --config "$STRICT_ALIGNED_CONFIG" \
    --device "$DEVICE"

echo "Training strict no-alignment DIMF..."
srun "$PYTHON_BIN" -u "$PROJECT_ROOT/train.py" \
    --config "$STRICT_NOALIGN_CONFIG" \
    --device "$DEVICE"

echo "Plotting strict prediction outputs..."
"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/plot_test_predictions.py" \
    --predictions "$STRICT_ALIGNED_OUTPUT_DIR/test_pred_vs_true.csv" \
    --output-dir "$STRICT_ALIGNED_OUTPUT_DIR/plots" \
    --title-prefix "strict aligned"

"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/plot_test_predictions.py" \
    --predictions "$STRICT_NOALIGN_OUTPUT_DIR/test_pred_vs_true.csv" \
    --output-dir "$STRICT_NOALIGN_OUTPUT_DIR/plots" \
    --title-prefix "strict noalign"

echo "Comparing strict alignment effect..."
"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/compare_alignment_lag_effect.py" \
    --aligned-estimates "$STRICT_ALIGNED_OUTPUT_DIR/test_delay_estimates.csv" \
    --noalign-estimates "$STRICT_NOALIGN_OUTPUT_DIR/test_delay_estimates.csv" \
    --aligned-metrics "$STRICT_ALIGNED_OUTPUT_DIR/test_metrics.json" \
    --noalign-metrics "$STRICT_NOALIGN_OUTPUT_DIR/test_metrics.json" \
    --raw-dataset "$RAWGAP_CSV" \
    --output-dir "$STRICT_COMPARE_OUTPUT_DIR"

echo "Rendering strict comparison visuals..."
"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/visualize_alignment_comparison.py" \
    --summary "$STRICT_COMPARE_OUTPUT_DIR/alignment_comparison_summary.json" \
    --joined "$STRICT_COMPARE_OUTPUT_DIR/alignment_test_joined.csv" \
    --per-lag "$STRICT_COMPARE_OUTPUT_DIR/alignment_per_lag_comparison.csv" \
    --output-dir "$STRICT_VIZ_DIR"

echo "Comparing strict results against legacy outputs..."
"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/compare_alignment_versions.py" \
    --current-summary "$STRICT_COMPARE_OUTPUT_DIR/alignment_comparison_summary.json" \
    --legacy-summary "$LEGACY_COMPARE_DIR/alignment_comparison_summary.json" \
    --current-per-lag "$STRICT_COMPARE_OUTPUT_DIR/alignment_per_lag_comparison.csv" \
    --legacy-per-lag "$LEGACY_COMPARE_DIR/alignment_per_lag_comparison.csv" \
    --output-dir "$STRICT_VS_LEGACY_DIR" \
    --current-label "strict" \
    --legacy-label "legacy"

echo "Strict comparison report preview:"
sed -n '1,220p' "$STRICT_COMPARE_OUTPUT_DIR/alignment_comparison_report.md"

echo "Strict vs legacy summary preview:"
sed -n '1,220p' "$STRICT_VS_LEGACY_DIR/strict_vs_legacy_summary.md"

echo "Job finished at: $(date)"
