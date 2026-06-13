#!/usr/bin/env python3
"""Pool per-slice CM-CPAL-SAM 5-fold CV results into one summary CSV."""

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["Dice", "IoU", "HD95", "NSD", "F1", "Precision", "Recall"]


def metric_summary(df: pd.DataFrame) -> dict:
    summary = {}
    for metric in METRICS:
        vals = df[metric].dropna()
        summary[metric] = f"{vals.mean():.4f}±{vals.std():.4f}" if len(vals) else "N/A"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pooled held-out 5-fold CM-CPAL-SAM results.")
    parser.add_argument("--result-dir", default="workdir/cv_results")
    parser.add_argument("--base-run-name", default="cpal_sam_dense_prompt_v2_cv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-per-slice-csv", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    per_fold = []
    missing = []
    for fold in range(args.n_folds):
        csv_path = result_dir / f"{args.base_run_name}_fold{fold}_results_cpal_sam_per_slice.csv"
        if not csv_path.is_file():
            missing.append(csv_path.as_posix())
            continue
        df_fold = pd.read_csv(csv_path)
        df_fold.insert(0, "fold", fold)
        per_fold.append(df_fold)

    if missing:
        preview = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing per-slice result CSVs:\n{preview}")
    if not per_fold:
        raise RuntimeError("No per-slice CSVs found.")

    df = pd.concat(per_fold, ignore_index=True)

    output_per_slice = Path(
        args.output_per_slice_csv
        or result_dir / f"{args.base_run_name}_5fold_pooled_per_slice.csv"
    )
    output_summary = Path(
        args.output_csv
        or result_dir / f"{args.base_run_name}_5fold_pooled_summary.csv"
    )
    output_per_slice.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for cat in ["core", "penumbra"]:
        df_cat = df[df["category"] == cat]
        if df_cat.empty:
            continue

        if cat == "core":
            eval_df = df_cat[df_cat["gt_nonempty"]]
            eval_label = "GT-nonempty slices"
            subset = "gt_nonempty"
        else:
            eval_df = df_cat
            eval_label = "All slices"
            subset = "all"

        nonempty_df = df_cat[df_cat["nonempty"]]
        for label, subset_name, subset_df in [
            (eval_label, subset, eval_df),
            ("Non-empty slices", "nonempty", nonempty_df),
        ]:
            stats = metric_summary(subset_df)
            print(f"\n=== CM-CPAL-SAM 5-fold pooled [{cat}] {label} (n={len(subset_df)}) ===")
            for metric in METRICS:
                print(f"  {metric}: {stats[metric]}")
            summary_rows.append(
                {
                    "Model": "CM-CPAL-SAM",
                    "Target": cat,
                    "Subset": subset_name,
                    "N": len(subset_df),
                    **stats,
                }
            )

    df.to_csv(output_per_slice, index=False)
    pd.DataFrame(summary_rows).to_csv(output_summary, index=False)
    print(f"\nPooled per-slice CSV: {output_per_slice}")
    print(f"Pooled summary CSV: {output_summary}")


if __name__ == "__main__":
    main()
