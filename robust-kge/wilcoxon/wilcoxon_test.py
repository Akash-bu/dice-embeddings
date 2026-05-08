"""
Wilcoxon Signed-Rank Test for comparing loss functions against BCE baseline.

Following the methodology from:
  "Link Prediction Under Non-targeted Attacks: Do Soft Labels Always Help?"
  (Memariani et al., ISWC 2025, Section 5.2)

Approach:
  - Data source: seed_runs/BO_runs/ where all loss functions share the same
    seeds, enabling proper paired comparison.
  - For each group (dataset, noise_ratio), pair each alternative loss against BCE
    by matching on (Model, Seed) — a true matched pair controlling for random
    initialization.
  - Apply one-sided Wilcoxon signed-rank test (alternative='greater') to check
    if the alternative loss significantly outperforms BCE (p < 0.05).

Reference:
  scipy.stats.wilcoxon — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

BASELINE_LOSS = "BCELoss"
SIGNIFICANCE_LEVEL = 0.05


def load_bo_runs(bo_runs_dir):
    """Load all per_run_results CSVs from BO_runs into a single DataFrame."""
    pattern = os.path.join(bo_runs_dir, "**", "per_run_results_*.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"No per_run_results CSVs found in {bo_runs_dir}")

    all_dfs = []
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
        print(f"  Loaded {csv_file} ({len(df)} rows)")

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def parse_dataset_and_noise(dataset_str):
    """Split 'FB15k-237/0.08' into dataset name and noise ratio."""
    parts = dataset_str.rsplit("/", 1)
    return parts[0], float(parts[1])


def perform_wilcoxon_tests(combined_df):
    """
    For each (dataset_name, noise_ratio, alternative_loss) group:
      - Pair against BCE by (Model, Seed)
      - Run one-sided Wilcoxon signed-rank test (alternative='greater')

    Returns a DataFrame with columns:
      dataset_name, noise_ratio, loss, n_pairs, stat, p_value, significant
    """
    # Parse dataset and noise
    parsed = combined_df["Dataset"].apply(parse_dataset_and_noise)
    combined_df["dataset_name"] = parsed.apply(lambda x: x[0])
    combined_df["noise_ratio"] = parsed.apply(lambda x: x[1])

    # Separate baseline and alternatives
    baseline_df = combined_df[combined_df["Loss"] == BASELINE_LOSS].copy()
    alternative_df = combined_df[combined_df["Loss"] != BASELINE_LOSS].copy()

    alternative_losses = sorted(alternative_df["Loss"].unique())
    dataset_names = sorted(combined_df["dataset_name"].unique())
    noise_ratios = sorted(combined_df["noise_ratio"].unique())

    results = []

    for loss in alternative_losses:
        loss_data = alternative_df[alternative_df["Loss"] == loss]

        for ds in dataset_names:
            for nr in noise_ratios:
                # Filter to this (dataset, noise_ratio) group
                ds_nr_str_candidates = combined_df[
                    (combined_df["dataset_name"] == ds) &
                    (combined_df["noise_ratio"] == nr)
                ]["Dataset"].unique()

                if len(ds_nr_str_candidates) == 0:
                    continue
                ds_nr_str = ds_nr_str_candidates[0]

                bce_group = baseline_df[baseline_df["Dataset"] == ds_nr_str]
                loss_group = loss_data[loss_data["Dataset"] == ds_nr_str]

                if bce_group.empty or loss_group.empty:
                    continue

                # Merge on (Model, Seed) to form true matched pairs
                merged = pd.merge(
                    bce_group[["Model", "Seed", "Test_MRR"]],
                    loss_group[["Model", "Seed", "Test_MRR"]],
                    on=["Model", "Seed"],
                    suffixes=("_bce", "_alt"),
                )

                if len(merged) < 6:
                    results.append({
                        "dataset_name": ds,
                        "noise_ratio": nr,
                        "loss": loss,
         