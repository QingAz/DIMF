#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize local-detection results across multiple seed runs."
    )
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True, help="Run output directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary tables")
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _load_one(run_dir: Path) -> Dict[str, object]:
    selection_path = run_dir / "detection_selected_audit" / "checkpoint_detection_selection.json"
    split_summary_path = run_dir / "detection_selected_audit" / "split_detection_summary.csv"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    split_summary = pd.read_csv(split_summary_path)
    test_row = split_summary.loc[split_summary["split"] == "test"].iloc[0]

    config_path = Path(selection["config"])
    seed = int(config_path.read_text(encoding="utf-8").splitlines()[0].split(":")[1].strip())
    best_epoch = int(selection["selected"]["epoch"])

    row_p_in = float(test_row["row_p_in_block"])
    row_p_out = float(test_row["row_p_out_block"])
    diff = row_p_in - row_p_out
    segment_auroc = float(test_row["segment_block_auroc"])
    success = bool(segment_auroc > 0.5 and diff > 0.0)

    return {
        "seed": seed,
        "run_dir": run_dir.as_posix(),
        "best_epoch": best_epoch,
        "row_auroc": float(test_row["row_block_auroc"]),
        "segment_auroc": segment_auroc,
        "block_auprc": float(test_row["row_block_auprc"]),
        "p_in_block": row_p_in,
        "p_out_block": row_p_out,
        "diff": diff,
        "success": success,
    }


def _summary_table(seed_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["best_epoch", "row_auroc", "segment_auroc", "block_auprc", "p_in_block", "p_out_block", "diff"]
    rows: List[Dict[str, object]] = []
    success_count = int(seed_df["success"].sum())
    for metric in metrics:
        series = seed_df[metric].astype(float)
        rows.append(
            {
                "metric": metric,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "median": float(series.median()),
                "#seeds_with_success": success_count,
            }
        )
    rows.append(
        {
            "metric": "success_rate",
            "mean": float(seed_df["success"].mean()),
            "std": 0.0,
            "median": float(seed_df["success"].median()),
            "#seeds_with_success": success_count,
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [_load_one(_absolute_path(run_dir)) for run_dir in args.run_dirs]
    seed_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary_df = _summary_table(seed_df)

    seed_df.to_csv(output_dir / "local_detection_seed_table.csv", index=False)
    summary_df.to_csv(output_dir / "local_detection_summary.csv", index=False)

    report = {
        "run_dirs": [str(_absolute_path(run_dir)) for run_dir in args.run_dirs],
        "n_runs": int(seed_df.shape[0]),
        "success_count": int(seed_df["success"].sum()),
        "success_rate": float(seed_df["success"].mean()),
    }
    (output_dir / "local_detection_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(seed_df.to_csv(index=False))
    print(summary_df.to_csv(index=False))


if __name__ == "__main__":
    main()
