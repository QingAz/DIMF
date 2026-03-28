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
PYUSER_BASE="$PROJECT_ROOT/.pyuser"
PYUSER_SITE="$PYUSER_BASE/lib/python3.10/site-packages"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

module purge
module load py-torch/2.5.1-openmpi-gcc-12.4.0

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUSERBASE="$PYUSER_BASE"
export PATH="$PYUSER_BASE/bin:$PATH"
if [ -d "$PYUSER_SITE" ]; then
    export PYTHONPATH="$PYUSER_SITE:${PYTHONPATH:-}"
fi

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "PYTHONUSERBASE: ${PYTHONUSERBASE}"

python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import numpy, pandas, sklearn, yaml, joblib, tqdm; print('Dependencies look good.')"

srun python -u train.py \
    --config configs/multistage_aligned.yaml \
    --device cuda

echo "Job finished at: $(date)"
