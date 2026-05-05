#!/bin/bash
#SBATCH --account=p32954
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=g2_hybrid
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
VENV_DIR="$PROJECT_ROOT/.venv-quest"
SERIES_DIR="${SERIES_DIR:-outputs/r12b_g2_magnitude_residual_sanity}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r13_g2_hybrid_mask_amplitude}"
FAR_MAX_PRIMARY="${FAR_MAX_PRIMARY:-0.05}"
FAR_MAX_FALLBACK="${FAR_MAX_FALLBACK:-0.10}"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

module purge
module load py-torch/2.5.1-openmpi-gcc-12.4.0

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
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
    if "$PYTHON_BIN" -c "import numpy, pandas, matplotlib" >/dev/null 2>&1; then
        return 0
    fi
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install --upgrade --prefer-binary numpy pandas matplotlib
}

resolve_path() {
    "$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); root=Path(sys.argv[2]); print((p if p.is_absolute() else root / p).resolve())' "$1" "$PROJECT_ROOT"
}

ensure_runtime_python
ensure_python_deps

SERIES_RESOLVED="$(resolve_path "$SERIES_DIR")"
OUTPUT_RESOLVED="$(resolve_path "$OUTPUT_DIR")"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Series dir: ${SERIES_RESOLVED}"
echo "Output dir: ${OUTPUT_RESOLVED}"
echo "FAR primary: ${FAR_MAX_PRIMARY}"
echo "FAR fallback: ${FAR_MAX_FALLBACK}"

"$PYTHON_BIN" -u scripts/run_g2_hybrid_mask_amplitude.py \
    --series-dir "$SERIES_RESOLVED" \
    --output-dir "$OUTPUT_RESOLVED" \
    --far-max-primary "$FAR_MAX_PRIMARY" \
    --far-max-fallback "$FAR_MAX_FALLBACK"

echo "Hybrid sanity test table:"
sed -n '1,20p' "$OUTPUT_RESOLVED/g2_hybrid_sanity_test_table.csv"

echo "Hybrid final comparison:"
sed -n '1,30p' "$OUTPUT_RESOLVED/g2_hybrid_final_comparison.csv"

echo "Job finished at: $(date)"
