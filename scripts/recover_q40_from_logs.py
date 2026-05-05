#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "q40_log_recovery"


class TableSpec(object):
    def __init__(self, name, header, output_name):
        self.name = name
        self.header = header
        self.output_name = output_name


TABLE_SPECS: Sequence[TableSpec] = (
    TableSpec(
        name="q40_snapshot_metrics",
        header=(
            "split,n_rows,n_positive,n_selected,tp,fp,fn,tn,overall_recall,FAR,"
            "precision,F1,pos_MAE,zero_E_d_hat,d2_recall,d2_selected,d4_recall,"
            "d4_selected,d6_recall,d6_selected,peak_error,peak_hit_at_pm1,n_positive_blocks"
        ),
        output_name="recovered_q40_snapshot_metrics.csv",
    ),
    TableSpec(
        name="r45c_segment_verifier",
        header=(
            "run,verifier_eval_overall_recall,verifier_eval_FAR,verifier_eval_zero_E_d_hat,"
            "verifier_eval_peak_hit_at_pm1,verifier_eval_pos_MAE,q40_eval_overall_recall,"
            "q40_eval_FAR,q40_eval_zero_E_d_hat,q40_eval_peak_hit_at_pm1,q40_eval_pos_MAE,"
            "decision_threshold,n_fit_segments,n_val_segments,n_eval_segments"
        ),
        output_name="recovered_q40_r45c_segment_verifier.csv",
    ),
    TableSpec(
        name="r46b_strongkeep_veto",
        header=(
            "run,selection_status,selection_stage,theta_drop,q40_eval_recall,"
            "strongkeep_veto_eval_recall,q40_eval_FAR,strongkeep_veto_eval_FAR,"
            "q40_eval_zero_E_d_hat,strongkeep_veto_eval_zero_E_d_hat,q40_eval_pos_MAE,"
            "strongkeep_veto_eval_pos_MAE,n_eval_segments,n_eval_strong_segments"
        ),
        output_name="recovered_q40_r46b_strongkeep_veto.csv",
    ),
    TableSpec(
        name="bump_eval_by_method",
        header=(
            "n_rows,n_bump_rows,n_outside_rows,bump_in_mae,outside_far,outside_mean_d_hat,"
            "peak_time_error,peak_value_error,shape_corr,n_blocks,method"
        ),
        output_name="recovered_q40_bump_eval_by_method.csv",
    ),
    TableSpec(
        name="bump_eval_by_seed",
        header=(
            "seed_run,raw_bump_in_mae,raw_outside_far,raw_outside_mean_d_hat,raw_peak_time_error,"
            "raw_peak_value_error,raw_shape_corr,q40_bump_in_mae,q40_outside_far,"
            "q40_outside_mean_d_hat,q40_peak_time_error,q40_peak_value_error,q40_shape_corr,"
            "r46b_bump_in_mae,r46b_outside_far,r46b_outside_mean_d_hat,r46b_peak_time_error,"
            "r46b_peak_value_error,r46b_shape_corr,r46b_selection_status,r46b_selection_stage,"
            "r46b_theta_drop,q40_eval_recall,r46b_eval_recall,q40_eval_far,r46b_eval_far,"
            "dropped_positive_strong,dropped_positive_weak,dropped_false_positive_strong,"
            "dropped_false_positive_weak,r46b_minus_q40_outside_far,r46b_minus_q40_bump_in_mae,"
            "r46b_minus_q40_shape_corr"
        ),
        output_name="recovered_q40_bump_eval_by_seed.csv",
    ),
    TableSpec(
        name="bump_eval_metric_summary",
        header="metric,mean,std,median,min,max,n",
        output_name="recovered_q40_bump_eval_metric_summary.csv",
    ),
    TableSpec(
        name="unified_vs_q40",
        header=(
            "run,unified_eval_overall_recall,unified_eval_FAR,unified_eval_zero_E_d_hat,"
            "unified_eval_AUPRC,unified_eval_peak_hit_at_pm1,unified_eval_pos_MAE,"
            "threshold_selection_status,selected_threshold,selected_d_floor,"
            "selected_rank_threshold,q40_eval_overall_recall,q40_eval_FAR,"
            "q40_eval_zero_E_d_hat,q40_eval_peak_hit_at_pm1,q40_eval_pos_MAE"
        ),
        output_name="recovered_unified_vs_q40.csv",
    ),
)


def _normalize_csv_line(text: str) -> str:
    return text.strip().rstrip("\n")


def _parse_rows(header: str, rows: Sequence[str]) -> List[Dict[str, str]]:
    reader = csv.DictReader([header, *rows])
    return [dict(row) for row in reader]


def _collect_table(
    lines,  # type: Sequence[str]
    start_idx,  # type: int
    header,  # type: str
):
    # type: (...) -> Tuple[List[Dict[str, str]], int]
    header_cols = len(next(csv.reader([header])))
    row_lines: List[str] = []
    idx = start_idx + 1
    while idx < len(lines):
        line = _normalize_csv_line(lines[idx])
        if not line:
            break
        if line.startswith("+ ") or line.startswith("Wrote "):
            break
        try:
            col_count = len(next(csv.reader([line])))
        except csv.Error:
            break
        if col_count != header_cols:
            break
        row_lines.append(line)
        idx += 1
    return _parse_rows(header, row_lines), idx


def _find_next_wrote_path(lines, start_idx):
    # type: (Sequence[str], int) -> str
    for idx in range(start_idx, min(len(lines), start_idx + 80)):
        line = _normalize_csv_line(lines[idx])
        if line.startswith("Wrote "):
            return line[len("Wrote ") :]
    return ""


def recover_tables(log_dir):
    # type: (Path) -> Dict[str, List[Dict[str, str]]]
    out: Dict[str, List[Dict[str, str]]] = {spec.name: [] for spec in TABLE_SPECS}
    spec_by_header = {spec.header: spec for spec in TABLE_SPECS}
    log_paths = sorted(log_dir.glob("*.out"))
    for log_path in log_paths:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        idx = 0
        while idx < len(lines):
            line = _normalize_csv_line(lines[idx])
            spec = spec_by_header.get(line)
            if spec is None:
                idx += 1
                continue
            rows, next_idx = _collect_table(lines, idx, line)
            wrote_path = _find_next_wrote_path(lines, next_idx)
            for row in rows:
                row["source_log"] = log_path.name
                row["recovered_output_path"] = wrote_path
                out[spec.name].append(row)
            idx = max(next_idx, idx + 1)
    return out


def write_outputs(output_dir, tables):
    # type: (Path, Dict[str, List[Dict[str, str]]]) -> Dict[str, int]
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for spec in TABLE_SPECS:
        rows = tables[spec.name]
        counts[spec.name] = len(rows)
        out_path = output_dir / spec.output_name
        if not rows:
            out_path.write_text("", encoding="utf-8")
            continue
        fieldnames = list(rows[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "log_dir": LOG_DIR.as_posix(),
        "output_dir": output_dir.as_posix(),
        "counts": counts,
        "source_logs": sorted(
            {
                row["source_log"]
                for rows in tables.values()
                for row in rows
                if row.get("source_log")
            }
        ),
    }
    (output_dir / "recovery_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    tables = recover_tables(LOG_DIR)
    counts = write_outputs(DEFAULT_OUTPUT_DIR, tables)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"Wrote {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
