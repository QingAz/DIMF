#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/projects/p32954/dimf_liquid_sugar_repo-490"
FRAMEWORK_ROOT="/projects/p32954/liquidsugar_lag_framework"
SLURM_SCRIPT="$PROJECT_ROOT/scripts/submit_framework_dimf_compare.slurm"

submit_job() {
    local job_name="$1"
    local framework_runner="$2"
    local framework_config="$3"
    local framework_output_dir="$4"
    local full_lag_csv="$5"
    local rawgap_csv="$6"
    local aligned_config="$7"
    local noalign_config="$8"
    local aligned_output_dir="$9"
    local noalign_output_dir="${10}"
    local compare_output_dir="${11}"
    local viz_output_dir="${12}"

    sbatch \
        --job-name="$job_name" \
        --export=ALL,FRAMEWORK_RUNNER="$framework_runner",FRAMEWORK_CONFIG="$framework_config",FRAMEWORK_OUTPUT_DIR="$framework_output_dir",FULL_LAG_CSV="$full_lag_csv",RAWGAP_CSV="$rawgap_csv",ALIGNED_CONFIG="$aligned_config",NOALIGN_CONFIG="$noalign_config",ALIGNED_OUTPUT_DIR="$aligned_output_dir",NOALIGN_OUTPUT_DIR="$noalign_output_dir",COMPARE_OUTPUT_DIR="$compare_output_dir",VIZ_OUTPUT_DIR="$viz_output_dir" \
        "$SLURM_SCRIPT"
}

submit_job \
    "dimf_v3_bump_d2" \
    "run_build_local_bump_dataset.py" \
    "configs/liquidsugar_local_bump_d2.yaml" \
    "$FRAMEWORK_ROOT/outputs/liquidsugar_local_bump_stage12_smooth_d2" \
    "$PROJECT_ROOT/data/processed/LiquidSugar_local_bump_d2_segmentsplit_v3_full.csv" \
    "$PROJECT_ROOT/data/processed/LiquidSugar_local_bump_d2_segmentsplit_v3_rawgap.csv" \
    "$PROJECT_ROOT/configs/multistage_localbump_d2_aligned_segmentsplit_v3.yaml" \
    "$PROJECT_ROOT/configs/multistage_localbump_d2_noalign_segmentsplit_v3.yaml" \
    "$PROJECT_ROOT/outputs/localbump_d2_aligned_segmentsplit_v3" \
    "$PROJECT_ROOT/outputs/localbump_d2_noalign_segmentsplit_v3" \
    "$PROJECT_ROOT/outputs/localbump_d2_alignment_compare_segmentsplit_v3" \
    "$PROJECT_ROOT/outputs/localbump_d2_alignment_compare_segmentsplit_v3/visuals"

submit_job \
    "dimf_v3_bump_d4" \
    "run_build_local_bump_dataset.py" \
    "configs/liquidsugar_local_bump_d4.yaml" \
    "$FRAMEWORK_ROOT/outputs/liquidsugar_local_bump_stage12_smooth_d4" \
    "$PROJECT_ROOT/data/processed/LiquidSugar_local_bump_d4_segmentsplit_v3_full.csv" \
    "$PROJECT_ROOT/data/processed/LiquidSugar_local_bump_d4_segmentsplit_v3_rawgap.csv" \
    "$PROJECT_ROOT/configs/multistage_localbump_d4_aligned_segmentsplit_v3.yaml" \
    "$PROJECT_ROOT/configs/multistage_localbump_d4_noalign_segmentsplit_v3.yaml" \
    "$PROJECT_ROOT/outputs/localbump_d4_aligned_segmentsplit_v3" \
    "$PROJECT_ROOT/outputs/localbump_d4_noalign_segmentsplit_v3" \
    "$PROJECT_ROOT/outputs/localbump_d4_alignment_compare_segmentsplit_v3" \
    "$PROJECT_ROOT/outputs/localbump_d4_alignment_compare_segmentsplit_v3/visuals"

submit_job \
    "dimf_v3_bump_d6" \
    "run_build_local_bump_dataset.py" \
    "configs/liquidsugar_local_bump_d6.yaml" \
    "$FRAMEWORK_ROOT/outputs/liquidsugar_local_bump_stage12_smooth_d6" \
    "$PROJECT_ROOT/data/processed/LiquidSugar_local_bump_d6_segmentsplit_v3_full.csv" \
    "$PROJECT_ROOT/data/processed/LiquidSugar_local_bump_d6_segmentsplit_v3_rawgap.csv" \
    "$PROJECT_ROOT/configs/multistage_localbump_d6_aligned_segmentsplit_v3.yaml" \
    "$PROJECT_ROOT/configs/multistage_localbump_d6_noalign_segmentsplit_v3.yaml" \
    "$PROJECT_ROOT/outputs/localbump_d6_aligned_segmentsplit_v3" \
    "$PROJECT_ROOT/outputs/localbump_d6_noalign_segmentsplit_v3" \
    "$PROJECT_ROOT/outputs/localbump_d6_alignment_compare_segmentsplit_v3" \
    "$PROJECT_ROOT/outputs/localbump_d6_alignment_compare_segmentsplit_v3/visuals"
