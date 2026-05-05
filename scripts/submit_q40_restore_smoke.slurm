#!/bin/bash
#SBATCH --account=p32954
#SBATCH --job-name=q40_restore_smoke
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/p32954/dimf_liquid_sugar_repo-490
#SBATCH --output=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.out
#SBATCH --error=/projects/p32954/dimf_liquid_sugar_repo-490/logs/slurm-%x-%j.err

set -euo pipefail

REPO=/gpfs/projects/p32954/dimf_liquid_sugar_repo-490
cd "$REPO"

module purge
module load py-torch/2.5.1-openmpi-gcc-12.4.0

PY="$REPO/.venv-transfer-eval/bin/python"
RAW_CSV="$REPO/data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_rawgap.csv"
CFG="$REPO/outputs/localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_evalsafe_5seed/_generated_configs/seed_42.yaml"
CKPT="$REPO/artifacts/localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_evalsafe_5seed/seed_42/best.ckpt"
SCALER="$REPO/artifacts/localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_evalsafe_5seed/seed_42/scaler.pkl"

BASE_OUT="$REPO/outputs/q40_restore_smoke/seed_42"
TRANSFER_OUT="$BASE_OUT/transfer_eval"
FEATURE_OUT="$BASE_OUT/features"
Q40_OUT="$BASE_OUT/q40_snapshots"
R45C_OUT="$BASE_OUT/r45c"
R46B_OUT="$BASE_OUT/r46b"

"$PY" scripts/run_transfer_checkpoint_eval.py \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --scaler-path "$SCALER" \
  --csv-path "$RAW_CSV" \
  --output-dir "$TRANSFER_OUT" \
  --splits train,val,test \
  --device cuda

"$PY" scripts/export_lag_feature_tables.py \
  --raw-csv "$RAW_CSV" \
  --eval-root "$TRANSFER_OUT" \
  --output-dir "$FEATURE_OUT" \
  --splits train,val,test

"$PY" scripts/export_fixed_q40_snapshots.py \
  --fit-series "$FEATURE_OUT/train_feature_timeseries.csv" \
  --val-series "$FEATURE_OUT/val_feature_timeseries.csv" \
  --eval-series "$FEATURE_OUT/test_feature_timeseries.csv" \
  --output-dir "$Q40_OUT" \
  --run-name restore_smoke

"$PY" scripts/run_q40_segment_proposal_verifier.py \
  --proposal-root "$Q40_OUT" \
  --runs restore_smoke \
  --output-dir "$R45C_OUT" \
  --q40-segment-features

"$PY" scripts/run_q40_segment_strongkeep_veto.py \
  --source-root "$R45C_OUT" \
  --runs restore_smoke \
  --output-dir "$R46B_OUT"

echo "Wrote $BASE_OUT"
