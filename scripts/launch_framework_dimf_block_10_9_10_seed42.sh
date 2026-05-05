#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
FRAMEWORK_ROOT="/projects/p32954/liquidsugar_lag_framework"
SLURM_SCRIPT="$PROJECT_ROOT/scripts/submit_framework_dimf_compare.slurm"

sbatch \
    --job-name="dimf_block_10_9_10_s42" \
    --export=ALL,FRAMEWORK_RUNNER="run_build_local_bump_dataset.py",FRAMEWORK_CONFIG="configs/liquidsugar_local_block_mixed_evalsafe_10_9_10.yaml",FRAMEWORK_OUTPUT_DIR="$FRAMEWORK_ROOT/outputs/liquidsugar_local_block_stage12_mixed_evalsafe_10_9_10",FULL_LAG_CSV="$PROJECT_ROOT/data/processed/LiquidSugar_local_block_mixed_evalsafe_10_9_10_full.csv",RAWGAP_CSV="$PROJECT_ROOT/data/processed/LiquidSugar_local_block_mixed_evalsafe_10_9_10_rawgap.csv",ALIGNED_CONFIG="$PROJECT_ROOT/configs/multistage_localblock_mixed_evalsafe_10_9_10_aligned_seed42.yaml",NOALIGN_CONFIG="$PROJECT_ROOT/configs/multistage_localblock_mixed_evalsafe_10_9_10_noalign_seed42.yaml",ALIGNED_OUTPUT_DIR="$PROJECT_ROOT/outputs/localblock_mixed_evalsafe_10_9_10_aligned_seed42",NOALIGN_OUTPUT_DIR="$PROJECT_ROOT/outputs/localblock_mixed_evalsafe_10_9_10_noalign_seed42",COMPARE_OUTPUT_DIR="$PROJECT_ROOT/outputs/localblock_mixed_evalsafe_10_9_10_alignment_compare_seed42",VIZ_OUTPUT_DIR="$PROJECT_ROOT/outputs/localblock_mixed_evalsafe_10_9_10_alignment_compare_seed42/visuals" \
    "$SLURM_SCRIPT"
