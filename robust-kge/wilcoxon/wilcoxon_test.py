"""
Wilcoxon Signed-Rank Test for comparing loss functions against BCE baseline.

Methodology (Memariani et al., ISWC 2025, Section 5.2 / Table 2):
  - Groups: (Model, noise_ratio).
  - For each group, pair each alternative loss against BCE by matching on
    (Dataset, Seed) — i.e., across all datasets and seeds within that
    (model, noise_ratio).
  - One-sided Wilcoxon signed-rank test (alternative='greater') with p < 0.05.

Reference:
  scipy.stats.wilcoxon — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
"""

import argparse
import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

BASELINE_LOSS = "BCELoss"
SIGNIFICANCE_LEVEL = 0.05


def load_csv_files(csv_files):
    if not csv_files:
        raise FileNotFoundError("No CSV files provided.")

    all_dfs = []
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
        print(f"  Loaded {csv_file} ({len(df)} rows)")

    return pd.concat(all_dfs, ignore_index=True)


def load_bo_runs(bo_runs_dir):
    pattern = os.path.join(bo_runs_dir, "**", "per_run_results_*.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"No per_run_results CSVs found in {bo_runs_dir}")

    return load_csv_files(csv_files)


def parse_dataset_and_noise(dataset_str):
    parts = dataset_str.rsplit("/", 1)
    return parts[0], float(parts[1])


def perform_wilcoxon_tests(combined_df):
    """
    For each (Model, noise_ratio, alternative_loss) group:
      - Pair against BCE on (Dataset, Seed) — across datasets and seeds
      - One-sided Wilcoxon (alternative='greater') vs BCE

    Returns a DataFrame with columns:
      model, noise_ratio, loss, n_pairs, stat, p_value, significant, note
    """
    df = combined_df.copy()
    parsed = df["Dataset"].apply(parse_dataset_and_noise)
    df["dataset_name"] = parsed.apply(lambda x: x[0])
    df["noise_ratio"] = parsed.apply(lambda x: x[1])

    baseline = df[df["Loss"] == BASELINE_LOSS]
    alts = df[df["Loss"] != BASELINE_LOSS]

    models = sorted(df["Model"].unique())
    noise_ratios = sorted(df["noise_ratio"].unique())
    losses = sorted(alts["Loss"].unique())

    results = []
    for model in models:
        for nr in noise_ratios:
            bce_g = baseline[(baseline["Model"] == model) & (baseline["noise_ratio"] == nr)]
            if bce_g.empty:
                continue

            for loss in losses:
                alt_g = alts[
                    (alts["Model"] == model) &
                    (alts["noise_ratio"] == nr) &
                    (alts["Loss"] == loss)
                ]
                if alt_g.empty:
                    continue

                merged = pd.merge(
                    bce_g[["dataset_name", "Seed", "Test_MRR"]],
                    alt_g[["dataset_name", "Seed", "Test_MRR"]],
                    on=["dataset_name", "Seed"],
                    suffixes=("_bce", "_alt"),
                )

                if len(merged) < 6:
                    results.append({
                        "model": model, "noise_ratio": nr, "loss": loss,
                        "n_pairs": len(merged), "stat": np.nan, "p_value": np.nan,
                        "significant": False,
                        "note": f"insufficient pairs ({len(merged)} < 6)",
                    })
                    continue

                diffs = merged["Test_MRR_alt"].values - merged["Test_MRR_bce"].values
                if np.all(diffs == 0):
                    results.append({
                        "model": model, "noise_ratio": nr, "loss": loss,
                        "n_pairs": len(merged), "stat": np.nan, "p_value": np.nan,
                        "significant": False, "note": "all differences are zero",
                    })
                    continue

                stat, p = wilcoxon(
                    merged["Test_MRR_alt"].values,
                    merged["Test_MRR_bce"].values,
                    alternative="greater",
                )
                results.append({
                    "model": model, "noise_ratio": nr, "loss": loss,
                    "n_pairs": len(merged), "stat": stat, "p_value": p,
                    "significant": p < SIGNIFICANCE_LEVEL, "note": "",
                })

    return pd.DataFrame(results)


def print_results(results_df):
    if results_df.empty:
        print("No results to display.")
        return

    for model in sorted(results_df["model"].unique()):
        for nr in sorted(results_df["noise_ratio"].unique()):
            subset = results_df[
                (results_df["model"] == model) & (results_df["noise_ratio"] == nr)
            ]
            if subset.empty:
                continue

            sig_losses = subset[subset["significant"]]["loss"].tolist()
            print(f"\n{model} | noise={nr} | n_pairs={int(subset['n_pairs'].iloc[0])} | "
                  f"significant losses: {sig_losses if sig_losses else '—'}")
            for _, row in subset.sort_values("p_value").iterrows():
                p_str = f"{row['p_value']:.4f}" if not pd.isna(row['p_value']) else "  N/A"
                marker = " *" if row["significant"] else "  "
                print(f"    {row['loss']:<45} p={p_str}{marker}  {row.get('note','')}")


def generate_summary_image(results_df, image_path):
    """
    Table 2 style:
      rows = noise_ratio
      cols = Model
      cell = comma-separated list of losses where p < 0.05 (vs BCE)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(results_df["model"].unique())
    noise_ratios = sorted(results_df["noise_ratio"].unique())

    cell_text = []
    cell_colors = []
    SIG_COLOR = "#a8d5a2"
    INSIG_COLOR = "#ffffff"

    for nr in noise_ratios:
        row_text = []
        row_colors = []
        for m in models:
            sig = results_df[
                (results_df["model"] == m) &
                (results_df["noise_ratio"] == nr) &
                (results_df["significant"] == True)
            ]["loss"].tolist()
            if sig:
                row_text.append("\n".join(sig))
                row_colors.append(SIG_COLOR)
            else:
                row_text.append("—")
                row_colors.append(INSIG_COLOR)
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    n_rows = len(noise_ratios)
    n_cols = len(models)
    max_lines_per_row = [
        max(1, max(len(t.split("\n")) for t in row)) for row in cell_text
    ]
    line_height = 0.32
    row_heights = [m * line_height for m in max_lines_per_row]
    header_height = 0.5
    fig_width = max(10, n_cols * 2.6)
    fig_height = sum(row_heights) + header_height + 1.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=[str(nr) for nr in noise_ratios],
        colLabels=models,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_height(header_height / fig_height)
            cell.set_facecolor("#4a7fc1")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_height(row_heights[r - 1] / fig_height)
            if c == -1:
                cell.set_facecolor("#d9e6f2")
                cell.set_text_props(fontweight="bold")

    ax.set_title(
        f"Wilcoxon signed-rank, p < {SIGNIFICANCE_LEVEL}, on MRR scores. "
        f"Soft label-based loss functions that outperformed BCE and are statistically significant.",
        fontsize=11, fontweight="bold", pad=16, wrap=True,
    )

    plt.tight_layout()
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary image saved to {image_path}")


def main():
    global SIGNIFICANCE_LEVEL  # noqa: PLW0603
    parser = argparse.ArgumentParser(
        description="Wilcoxon signed-rank test (paper-faithful: groups = (Model, noise_ratio); pair on Dataset, Seed)."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--files", nargs="+", metavar="CSV",
                             help="One or more per_run_results CSV files to load directly.")
    input_group.add_argument("--dirs", nargs="+", metavar="DIR",
                             help="Directories scanned recursively for per_run_results_*.csv files.")
    parser.add_argument("--output", metavar="CSV", default=None,
                        help="Path to save the results CSV.")
    parser.add_argument("--significance", type=float, default=SIGNIFICANCE_LEVEL,
                        metavar="ALPHA", help=f"Significance level (default: {SIGNIFICANCE_LEVEL}).")
    args = parser.parse_args()
    SIGNIFICANCE_LEVEL = args.significance

    print("Loading data...")
    if args.files:
        combined_df = load_csv_files(args.files)
    else:
        all_dfs = []
        for d in args.dirs:
            print(f"  Scanning {d} ...")
            all_dfs.append(load_bo_runs(d))
        combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Models: {sorted(combined_df['Model'].unique())}")
    print(f"Losses: {sorted(combined_df['Loss'].unique())}")
    print(f"Datasets: {sorted(combined_df['Dataset'].unique())}\n")

    results_df = perform_wilcoxon_tests(combined_df)
    print_results(results_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(script_dir, f"wilcoxon_results_{timestamp}.csv")

    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    image_path = output_path.replace(".csv", ".png")
    generate_summary_image(results_df, image_path)


if __name__ == "__main__":
    main()
