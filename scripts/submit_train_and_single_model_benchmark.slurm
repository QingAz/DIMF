#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=gengpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=dimf_train_bench
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH is required}"
DEVICE="${DEVICE:-cuda}"
BENCHMARK_EDGE="${BENCHMARK_EDGE:-stage1_to_stage2}"
BENCHMARK_CONTEXT="${BENCHMARK_CONTEXT:-16}"

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
ensure_python_deps

CONFIG_RESOLVED="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$CONFIG_PATH" "$PROJECT_ROOT")"
RUN_OUTPUT_DIR="$("$PYTHON_BIN" -c 'import sys, yaml; from pathlib import Path; cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")); print(cfg["logging"]["output_dir"])' "$CONFIG_RESOLVED")"
RAW_DATASET_PATH="$("$PYTHON_BIN" -c 'import sys, yaml; from pathlib import Path; cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")); print(cfg["data"]["csv_path"])' "$CONFIG_RESOLVED")"
RUN_OUTPUT_ABS="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$RUN_OUTPUT_DIR" "$PROJECT_ROOT")"
RAW_DATASET_ABS="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$RAW_DATASET_PATH" "$PROJECT_ROOT")"
BENCHMARK_DIR="$RUN_OUTPUT_ABS/benchmark_single_model"
PANELS_DIR="$BENCHMARK_DIR/v5_lag_panels"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Runtime Python path: ${PYTHON_BIN}"
echo "Runtime venv: ${VENV_DIR}"
echo "Config path: ${CONFIG_RESOLVED}"
echo "Output dir: ${RUN_OUTPUT_ABS}"
echo "Raw dataset: ${RAW_DATASET_ABS}"
echo "Benchmark edge: ${BENCHMARK_EDGE}"
echo "Device: ${DEVICE}"

"$PYTHON_BIN" -c "import sys; print('Python:', sys.version)"
"$PYTHON_BIN" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "Training DIMF..."
srun "$PYTHON_BIN" -u train.py \
    --config "$CONFIG_RESOLVED" \
    --device "$DEVICE"

mkdir -p "$BENCHMARK_DIR" "$PANELS_DIR"

echo "Summarizing single-model lag benchmark..."
"$PYTHON_BIN" -u scripts/summarize_single_model_lag_benchmark.py \
    --estimates "$RUN_OUTPUT_ABS/test_delay_estimates.csv" \
    --raw-dataset "$RAW_DATASET_ABS" \
    --metrics "$RUN_OUTPUT_ABS/test_metrics.json" \
    --output-dir "$BENCHMARK_DIR" \
    --edge "$BENCHMARK_EDGE"

echo "Exporting lag component diagnostics..."
"$PYTHON_BIN" -u scripts/export_lag_component_diagnostics.py \
    --estimates "$RUN_OUTPUT_ABS/test_delay_estimates.csv" \
    --raw-dataset "$RAW_DATASET_ABS" \
    --output "$BENCHMARK_DIR/lag_component_diagnostics.csv" \
    --edge "$BENCHMARK_EDGE"

echo "Rendering benchmark tables..."
"$PYTHON_BIN" -u scripts/plot_single_model_benchmark_tables.py \
    --summary "$BENCHMARK_DIR/benchmark_summary.json" \
    --output-dir "$BENCHMARK_DIR"

echo "Rendering lag block panels..."
"$PYTHON_BIN" -u scripts/plot_single_model_lag_blocks.py \
    --joined "$BENCHMARK_DIR/test_joined_single_model.csv" \
    --output-dir "$PANELS_DIR" \
    --context "$BENCHMARK_CONTEXT" \
    --title-prefix "zero-loss"

echo "Benchmark overall table:"
sed -n '1,20p' "$BENCHMARK_DIR/benchmark_overall.csv"

echo "Benchmark by-dmax table:"
sed -n '1,20p' "$BENCHMARK_DIR/benchmark_by_dmax.csv"

echo "Job finished at: $(date)"
