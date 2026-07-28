"""Run KGE experiments for UMLS + KINSHIP using BO-tuned hyperparameters.

This is a report-driven replica of ``run_sem_exp.py``. Instead of taking fixed
BATCH_SIZE / LEARNING_RATE / EMB_DIM / loss-specific constants from ``config.py``,
it reads the per-(Dataset, Model, Loss) best hyperparameters from a Bayesian
optimization report file (default:
``bo_trial_results/wo_reciprocals_umls_kinship.txt``) and passes them straight
through to ``run_dicee_eval``. All of the table/heatmap machinery is identical to
``run_sem_exp.py``.

Example:
    python run_seed_bo.py --report_dataset UMLS KINSHIP
    python run_seed_bo.py --report_dataset KINSHIP --model Keci --loss_fn WaveLoss
    python run_seed_bo.py --report_dataset FB15k-237 --noise_ratio 0.08 --num_epochs 50 --devices 0
"""
from pathlib import Path
import hashlib
import sys
import ast
import gc
import logging
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import torch
from dicee.executer import run_dicee_eval

logger = logging.getLogger(__name__)


def setup_logging(script_name):
    """Configure logging to both the console and a timestamped file under ./logs.

    Returns the path of the log file so the caller can report it.
    """
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Only add a console handler if nothing already streams to stderr (e.g. dicee).
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        root.addHandler(stream_handler)

    logger.info("Logging to %s", log_file)
    return log_file

# Add the robust-kge directory to the path for config import
robust_kge_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(robust_kge_dir))

# Get project root (parent of robust-kge directory)
project_root = robust_kge_dir.parent

# Add project root to Python path so dicee can be imported
sys.path.insert(0, str(project_root))

MODELS = ["Pykeen_RotatE", "Pykeen_MuRE", "Keci"]
LOSSES = ["AGCELoss", "AELoss", "RoBoSS", "WaveLoss", "BCELoss"]

NUM_EPOCHS = "100"
SCORING_TECH = "KvsAll"
OPTIM = "Adam"
EVAL_MODEL = "test"

# Random seeds to run each experiment under; metrics are averaged across them.
SEEDS = ["42", "67", "83"]

_BO_DIR = project_root / "bo_trial_results"
# Which report file holds the BO params for each dataset.
DATASET_REPORTS = {
    "KINSHIP": _BO_DIR / "wo_reciprocals_umls_kinship.txt",
    "UMLS": _BO_DIR / "wo_reciprocals_umls_kinship.txt",
    "NELL-995-h100": _BO_DIR / "wo_reciprocals_nell.txt",
    "FB15k-237": _BO_DIR / "wo_reciprocals_fb15k.txt",
}

# Supply data
DEFAULT_DATASETS_ROOT = project_root / "Datasets_Perturbed"

# ---------------------------------------------------------------------------
# Report parsing (best params per (Dataset, Model, Loss))
# ---------------------------------------------------------------------------
def abs_path(path_str, base_dir):
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def normalize_dataset_key(dataset_str):
    parts = [p for p in dataset_str.strip().replace("\\", "/").split("/") if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-2]}/{parts[-1]}"


def parse_report_file(report_path):
    """Return {(Dataset, Model, Loss): params} for the best (highest Value) row per key."""
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    best_by_key = {}
    with open(report_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "Params:" not in line:
                continue

            params_idx = line.find("Params:")
            dataset_idx = line.find("Dataset:")
            model_idx = line.find("Model:")
            loss_idx = line.find("Loss:")
            if min(params_idx, dataset_idx, model_idx, loss_idx) == -1:
                continue

            value = None
            if line.startswith("Value:"):
                value_str = line[len("Value:"):params_idx].strip().rstrip(",")
                try:
                    value = float(value_str)
                except ValueError:
                    value = None

            params_str = line[params_idx + len("Params:"):dataset_idx].strip().rstrip(",")
            dataset_str = line[dataset_idx + len("Dataset:"):model_idx].strip().rstrip(",")
            model_str = line[model_idx + len("Model:"):loss_idx].strip().rstrip(",")
            loss_str = line[loss_idx + len("Loss:"):].strip().rstrip(",")

            params = ast.literal_eval(params_str)
            if not isinstance(params, dict):
                continue

            key = (normalize_dataset_key(dataset_str), model_str.strip(), loss_str.strip())
            prev = best_by_key.get(key)
            if prev is None:
                best_by_key[key] = {"value": value, "params": params}
                continue
            prev_value = prev["value"]
            if prev_value is None and value is not None:
                best_by_key[key] = {"value": value, "params": params}
            elif value is not None and prev_value is not None and value > prev_value:
                best_by_key[key] = {"value": value, "params": params}

    return {key: entry["params"] for key, entry in best_by_key.items()}


# ---------------------------------------------------------------------------
# Results tables and plotting (identical to run_sem_exp.py)
# ---------------------------------------------------------------------------
def _average_metrics(metrics_list):
    """Aggregate per-seed metric dicts into mean and std per metric.

    For every metric key the result holds ``<k>`` (mean across seeds) and
    ``<k>_std`` (sample std, ddof=1; 0.0 when only one seed produced a value).
    Ignores failed runs (None/empty) and non-numeric values. Returns None if no
    seed produced usable metrics.
    """
    valid = [m for m in metrics_list if m]
    if not valid:
        return None
    keys = set().union(*(m.keys() for m in valid))
    averaged = {}
    for k in keys:
        vals = [m[k] for m in valid if isinstance(m.get(k), (int, float))]
        if vals:
            averaged[k] = statistics.mean(vals)
            averaged[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return averaged


def create_results_table(results_dict):
    # Structure: results_dict[dataset][model][loss][scoring][neg_ratio] = mrr_value
    rows = []
    for dataset, models_dict in results_dict.items():
        for model, losses_dict in models_dict.items():
            for loss, scoring_dict in losses_dict.items():
                for scoring, neg_dict in scoring_dict.items():
                    for neg_ratio, metrics in neg_dict.items():
                        metrics = metrics or {}
                        rows.append({
                            'Dataset': dataset,
                            'Model': model,
                            'Loss': loss,
                            'Scoring': scoring,
                            'Neg_Ratio': neg_ratio,
                            'Test_MRR': metrics.get('MRR'),
                            'Test_MRR_std': metrics.get('MRR_std'),
                            'Hits_H@1': metrics.get('H@1'),
                            'Hits_H@1_std': metrics.get('H@1_std'),
                            'Hits_H@3': metrics.get('H@3'),
                            'Hits_H@3_std': metrics.get('H@3_std'),
                            'Hits_H@10': metrics.get('H@10'),
                            'Hits_H@10_std': metrics.get('H@10_std'),
                        })

    df = pd.DataFrame(rows)

    # Pivot: models as rows, (Dataset, Loss, Scoring, Neg_Ratio) as MultiIndex columns
    pivot_df = df.pivot_table(
        index='Model',
        columns=['Dataset', 'Loss', 'Scoring', 'Neg_Ratio'],
        values='Test_MRR',
        aggfunc='first'
    )
    pivot_df = pivot_df.sort_index(axis=1, level=[0, 1, 2, 3])

    return df, pivot_df


def create_metrics_table(df):
    """Pivot MRR and Hits@k together as 'mean ± std' strings: models as rows.

    Columns are a MultiIndex (Metric, Dataset, Loss, Scoring, Neg_Ratio) so every
    seed-averaged metric (MRR, H@1, H@3, H@10) appears side by side, each cell
    formatted as ``<mean> ± <std>`` across seeds, as final results.
    """
    metric_cols = ['Test_MRR', 'Hits_H@1', 'Hits_H@3', 'Hits_H@10']

    fmt_df = df[['Dataset', 'Model', 'Loss', 'Scoring', 'Neg_Ratio']].copy()
    for col in metric_cols:
        mean = df[col]
        std = df.get(f"{col}_std")
        fmt_df[col] = [
            "" if pd.isna(m) else (
                f"{m:.3f} ± {s:.3f}" if (std is not None and pd.notna(s)) else f"{m:.3f}"
            )
            for m, s in zip(mean, (std if std is not None else [None] * len(mean)))
        ]

    metrics_pivot = fmt_df.pivot_table(
        index='Model',
        columns=['Dataset', 'Loss', 'Scoring', 'Neg_Ratio'],
        values=metric_cols,
        aggfunc='first',
    )
    # Name the metric level and order columns by metric, then config.
    metrics_pivot.columns = metrics_pivot.columns.set_names('Metric', level=0)
    metrics_pivot = metrics_pivot.sort_index(axis=1, level=[0, 1, 2, 3, 4])
    return metrics_pivot


def _loss_suffix(configs):
    """Build a filename-safe suffix from one or more (loss, scoring) pairs or loss names."""
    if not configs:
        return ""
    if isinstance(configs, str):
        return f"_{configs}"
    parts = []
    for item in configs:
        if isinstance(item, (list, tuple)):
            parts.append(f"{item[0]}-{item[1]}")
        else:
            parts.append(str(item))
    if len(parts) == 1:
        return f"_{parts[0]}"
    return "_" + "_vs_".join(parts)


def _models_suffix(models):
    """Build a filename-safe suffix from one or more model names."""
    if not models:
        return ""
    if isinstance(models, str):
        return f"_{models}"
    if len(models) == 1:
        return f"_{models[0]}"
    return "_" + "-".join(models)


def _neg_ratio_suffix(neg_ratios):
    """Build a filename-safe suffix from one or more neg_ratio values."""
    if not neg_ratios:
        return ""
    if isinstance(neg_ratios, str):
        return f"_neg{neg_ratios}"
    if len(neg_ratios) == 1:
        return f"_neg{neg_ratios[0]}"
    return "_neg" + "-".join(str(n) for n in neg_ratios)


def _safe_suffix(suffix, max_len=120):
    """Bound a filename suffix so the full path stays under the OS limit.

    If the assembled suffix exceeds max_len characters, truncate it and append
    a short hash of the full suffix so distinct configurations stay unique.
    """
    if len(suffix) <= max_len:
        return suffix
    digest = hashlib.sha1(suffix.encode()).hexdigest()[:8]
    return f"{suffix[:max_len]}_{digest}"


def _format_inner_label(col):
    """Compact column label for a single facet: '<loss>/<scoring>' with optional 'n=k'."""
    parts = [str(c) for c in col] if isinstance(col, tuple) else [str(col)]
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}\n$n={parts[2]}$"
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return "/".join(parts)


# def create_visualization(pivot_df, output_dir, configs=None, models=None, neg_ratios=None, kind=None):
#     """Render a publication-quality faceted heatmap of Test MRR.

#     One panel per dataset, shared model axis, single colorbar. Outputs both a
#     high-DPI PNG and a vector PDF (with editable text) for camera-ready use.
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Build the dataset -> sub-DataFrame mapping. Each sub-frame has (Loss, Scoring, Neg_Ratio) cols.
#     if isinstance(pivot_df.columns, pd.MultiIndex) and "Dataset" in (pivot_df.columns.names or []):
#         datasets = list(dict.fromkeys(pivot_df.columns.get_level_values("Dataset")))
#         sub_dfs = {ds: pivot_df.xs(ds, axis=1, level="Dataset") for ds in datasets}
#     else:
#         datasets = [""]
#         sub_dfs = {"": pivot_df}

#     # Compact config / neg-ratio strings for the suptitle.
#     if not configs:
#         config_name = "default"
#     elif isinstance(configs, str):
#         config_name = configs
#     else:
#         parts = []
#         for item in configs:
#             if isinstance(item, (list, tuple)):
#                 parts.append(f"{item[0]}/{item[1]}")
#             else:
#                 parts.append(str(item))
#         config_name = ", ".join(parts)
#     if neg_ratios:
#         neg_str = neg_ratios if isinstance(neg_ratios, str) else ", ".join(str(n) for n in neg_ratios)
#     else:
#         neg_str = None

#     pub_rc = {
#         "font.family": "serif",
#         "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
#         "mathtext.fontset": "cm",
#         "font.size": 10,
#         "axes.titlesize": 11,
#         "axes.labelsize": 10,
#         "xtick.labelsize": 9,
#         "ytick.labelsize": 10,
#         "axes.linewidth": 0.8,
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#     }

#     sub_widths = [max(1, sub_dfs[ds].shape[1]) for ds in datasets]
#     n_ds = len(datasets)
#     n_rows = max(1, pivot_df.shape[0])

#     # Square-ish cells; pad for y-labels, colorbar, titles and 2-line x-labels.
#     cell = 1.1
#     fig_w = sum(sub_widths) * cell + 1.8 + 1.2 + 0.35 * max(0, n_ds - 1)
#     fig_h = n_rows * cell + 4.0

#     with plt.rc_context(pub_rc):
#         sns.set_style("white")
#         fig, ax_row = plt.subplots(
#             1,
#             n_ds + 1,
#             figsize=(fig_w, fig_h),
#             gridspec_kw={"width_ratios": sub_widths + [0.35]},
#             constrained_layout=True,
#         )
#         axes = list(ax_row[:-1])
#         cbar_ax = ax_row[-1]

#         for i, (ax, ds) in enumerate(zip(axes, datasets)):
#             sub = sub_dfs[ds].copy()
#             if isinstance(sub.columns, pd.MultiIndex):
#                 sub.columns = [_format_inner_label(c) for c in sub.columns]

#             show_y = i == 0
#             show_cbar = i == n_ds - 1

#             sns.heatmap(
#                 sub,
#                 annot=True,
#                 fmt=".3f",
#                 cmap="viridis",
#                 vmin=0.0,
#                 vmax=1.0,
#                 linewidths=0.5,
#                 linecolor="white",
#                 ax=ax,
#                 cbar=show_cbar,
#                 cbar_ax=cbar_ax if show_cbar else None,
#                 cbar_kws={"label": "Test MRR"} if show_cbar else None,
#                 annot_kws={"fontsize": 9},
#                 yticklabels=show_y,
#             )

#             if ds:
#                 ax.set_title(ds, pad=8)
#             ax.set_xlabel("")
#             ax.set_ylabel("Model" if show_y else "")
#             ax.tick_params(axis="both", which="both", length=0)
#             plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va="top")
#             if show_y:
#                 plt.setp(ax.get_yticklabels(), rotation=0)

#             for spine in ax.spines.values():
#                 spine.set_visible(True)
#                 spine.set_linewidth(0.6)
#                 spine.set_edgecolor("#333333")

#         cbar = axes[-1].collections[0].colorbar
#         cbar.outline.set_linewidth(0.5)
#         cbar.outline.set_edgecolor("#333333")
#         cbar.ax.tick_params(width=0.5)

#         suptitle = f"Test MRR — configurations: {config_name}"
#         if neg_str is not None:
#             suptitle += f"  (negative ratios: {neg_str})"
#         fig.suptitle(suptitle, fontsize=11)

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         suffix = _safe_suffix(
#             f"{_models_suffix(models)}{_loss_suffix(configs)}{_neg_ratio_suffix(neg_ratios)}"
#         )
#         if kind:
#             suffix = f"{suffix}_{kind}"
#         base = f"mrr_comparison_heatmap{suffix}_{timestamp}"
#         image_path = output_dir / f"{base}.png"
#         pdf_path = output_dir / f"{base}.pdf"
#         fig.savefig(image_path, dpi=600)
#         fig.savefig(pdf_path)
#         logger.info(f"Heatmap saved to: {image_path}")
#         logger.info(f"Heatmap (vector PDF) saved to: {pdf_path}")

#         plt.close(fig)

#     return image_path


def save_results_table(df, pivot_df, output_dir, configs=None, models=None, neg_ratios=None, kind=None, ds_name = None):
    """Save results to CSV files and create visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = _safe_suffix(
        f"_{ds_name}{_models_suffix(models)}{_loss_suffix(configs)}{_neg_ratio_suffix(neg_ratios)}"
    )
    if kind:
        suffix = f"{suffix}_{kind}"

    # Save detailed results
    detailed_path = output_dir / f"detailed_results{suffix}_{timestamp}.csv"
    df.to_csv(detailed_path, index=False)
    logger.info(f"Detailed results saved to: {detailed_path}")

    # Save pivot table
    pivot_path = output_dir / f"comparison_table{suffix}_{timestamp}.csv"
    pivot_df.to_csv(pivot_path)
    logger.info(f"Comparison table saved to: {pivot_path}")

    # Save full metrics table: averaged MRR + Hits@1/3/10 side by side.
    metrics_pivot = create_metrics_table(df)
    metrics_path = output_dir / f"metrics_table{suffix}_{timestamp}.csv"
    metrics_pivot.to_csv(metrics_path)
    logger.info(f"Averaged metrics table (MRR + Hits@k) saved to: {metrics_path}")

    # Log formatted tables
    logger.info("MRR Comparison Table (Test Set)\n%s\n%s\n%s",
                "=" * 80, pivot_df.to_string(), "=" * 80)
    logger.info("Averaged Metrics Table — MRR + Hits@k (Test Set)\n%s\n%s\n%s",
                "=" * 80, metrics_pivot.to_string(), "=" * 80)

    # Create heatmap visualization
    # logger.info("Generating heatmap visualization...")
    # create_visualization(pivot_df, output_dir, configs, models, neg_ratios, kind)

    return detailed_path, pivot_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run KGE experiments with BO-tuned hyperparameters from a report file')
    parser.add_argument('--report_dataset', type=str, nargs='+', required=True,
                        choices=sorted(DATASET_REPORTS.keys()),
                        help='One or more datasets to run (e.g., --report_dataset UMLS KINSHIP). '
                             'Selects the matching BO report file(s) and runs only these datasets.')
    parser.add_argument('--report_file', type=str, nargs='+', default=None,
                        help='Override the BO report file(s) for the chosen --report_dataset '
                             '(default: the files mapped in DATASET_REPORTS). Configs are merged.')
    parser.add_argument('--datasets_root', type=str, default=str(DEFAULT_DATASETS_ROOT),
                        help='Root directory containing <DB>/<noise>/ dataset folders. '
                             f'Default: {DEFAULT_DATASETS_ROOT}')
    parser.add_argument('--loss_fn', type=str, nargs='+', default=None,
                        help='One or more loss functions to use (e.g., --loss_fn BCELoss WaveLoss). '
                             'If not provided, uses LOSSES from the top of the script.')
    parser.add_argument('--model', type=str, nargs='+', default=None,
                        help='One or more models to run (e.g., --model Keci Pykeen_MuRE). '
                             'If not provided, uses MODELS from the top of the script.')
    parser.add_argument('--num_epochs', type=str, default=None,
                        help='Number of epochs (e.g., 100). If not provided, uses NUM_EPOCHS from the script.')
    parser.add_argument('--lr', type=str, default=None,
                        help='Override learning rate from the report for all configs (e.g., 0.05).')
    parser.add_argument('--batch_size', type=str, default=None,
                        help='Override batch size from the report for all configs (e.g., 256).')
    parser.add_argument('--emb_dim', type=str, default=None,
                        help='Override embedding dimension from the report for all configs (e.g., 32).')
    parser.add_argument('--scoring_technique', type=str, nargs='+', default=None,
                        help='One or more scoring techniques (e.g., KvsAll NegSample). '
                             'If multiple are given, must match --loss_fn in length and is paired 1:1. '
                             'A single value is broadcast to every loss. If omitted, uses SCORING_TECH.')
    parser.add_argument('--optim', type=str, default=None,
                        help='Optimizer to use (e.g., Adam). If not provided, uses OPTIM from the script.')
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Evaluation model to use (e.g., test). If not provided, uses EVAL_MODEL.')
    parser.add_argument("--trainer", type=str, default="PL")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--random_seed", type=str, nargs='+', default=None,
                        help="One or more random seeds. Each runs as a separate experiment per "
                             "(loss, scoring, neg_ratio) and the metrics are averaged across seeds. "
                             "If omitted, uses SEEDS from the top of the script.")
    parser.add_argument("--neg_ratio", type=str, nargs='+', default=None,
                        help="One or more negative sample ratios for NegSample scoring "
                             "(e.g., --neg_ratio 2 5 10). Default: ['2'].")
    parser.add_argument("--num_of_output_channels", type=str, default=None)
    parser.add_argument("--block_size", type=str, default=None)
    parser.add_argument("--noise_ratio", type=str, default=None,
                        choices=["0.0", "0.08", "0.16", "0.32"],
                        help="Single noise ratio subdir to run (e.g., 0.08). "
                             "If omitted, all of 0.0/0.08/0.16/0.32 are run.")
    args = parser.parse_args()

    setup_logging(Path(__file__).stem)

    # The chosen datasets select their report file(s) (overridable via --report_file)
    # and are the only DBs we run.
    dbs_to_run = list(dict.fromkeys(args.report_dataset))
    if args.report_file:
        report_sources = args.report_file
    else:
        report_sources = [DATASET_REPORTS[db] for db in dbs_to_run]
    # De-duplicate while preserving order (UMLS + KINSHIP share one file).
    report_files = list(dict.fromkeys(abs_path(rf, project_root) for rf in report_sources))

    datasets_root = abs_path(args.datasets_root, project_root)

    # Tag output filenames by perturbation type, inferred from the datasets root.
    # Defaults to "Random" for the generic Datasets_Perturbed root; still detects
    # semantic/adversarial roots when --datasets_root is pointed at one of those.
    root_str = str(datasets_root).lower()
    if "semantic" in root_str or "_sem" in root_str:
        run_kind = "Sem"
    elif "adversarial" in root_str or "_adv" in root_str:
        run_kind = "Adv"
    else:
        run_kind = "Random"

    params_by_key = {}
    for rf in report_files:
        file_params = parse_report_file(rf)
        params_by_key.update(file_params)
        logger.info(f"Loaded {len(file_params)} (Dataset, Model, Loss) configurations from {rf}")
    logger.info(f"Total merged configurations: {len(params_by_key)}")

    # Loss functions: CLI overrides the LOSSES list at the top of the script.
    loss_functions = args.loss_fn if args.loss_fn else list(LOSSES)

    # Models: CLI overrides the MODELS list at the top of the script.
    models_to_run = args.model if args.model else list(MODELS)

    num_epochs = args.num_epochs if args.num_epochs else NUM_EPOCHS

    # Scoring techniques: build a list, then pair 1:1 with loss_functions.
    if args.scoring_technique:
        scoring_techniques = args.scoring_technique
    elif isinstance(SCORING_TECH, (list, tuple)):
        scoring_techniques = list(SCORING_TECH)
    else:
        scoring_techniques = [SCORING_TECH]

    if len(scoring_techniques) == 1:
        scoring_techniques = scoring_techniques * len(loss_functions)
    elif len(scoring_techniques) != len(loss_functions):
        raise ValueError(
            f"--scoring_technique has {len(scoring_techniques)} values but --loss_fn has "
            f"{len(loss_functions)}. Pass either a single scoring technique (broadcast) "
            f"or the same number as losses (paired)."
        )
    loss_scoring_pairs = list(zip(loss_functions, scoring_techniques))

    optim = args.optim if args.optim else OPTIM
    eval_model = args.eval_model if args.eval_model else EVAL_MODEL
    trainer = args.trainer if args.trainer else None
    accelerator = args.accelerator if args.accelerator else None
    devices = args.devices if args.devices else None
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)
    precision = args.precision if args.precision else None
    # random_seeds: list of one or more seeds; each experiment runs once per seed
    # and the per-seed metrics are averaged into the results table.
    random_seeds = args.random_seed if args.random_seed else SEEDS
    neg_ratios = args.neg_ratio if args.neg_ratio else ["2"]
    num_of_output_channels = args.num_of_output_channels if args.num_of_output_channels else None
    block_size = args.block_size if args.block_size else None

    # Optional per-run hyperparameter overrides (applied on top of the report params).
    param_overrides = {}
    if args.emb_dim:
        param_overrides["embedding_dim"] = args.emb_dim
    if args.batch_size:
        param_overrides["batch_size"] = args.batch_size
    if args.lr:
        param_overrides["learning_rate"] = args.lr

    allowed_subdirs = {args.noise_ratio} if args.noise_ratio else {"0.0", "0.08", "0.16", "0.32"}

    # Dictionary to store all results: {dataset: {model: {loss: {scoring: {neg_ratio: metrics}}}}}
    all_results = {}

    # Saved single-run models go here (kept separate from run_sem_exp's saved_models_sem).
    results_dir = robust_kge_dir / "saved_models_bo"

    for DB in dbs_to_run:
        db_path = datasets_root / DB
        if not db_path.exists():
            logger.warning(f"{db_path} does not exist, skipping {DB}")
            continue
        subdirs = sorted(
            d.name for d in db_path.iterdir()
            if d.is_dir() and d.name in allowed_subdirs
        )
        if not subdirs:
            logger.warning(f"No matching noise subdirs found under {db_path}, skipping {DB}")
            continue

        for subdir in subdirs:
            dataset_name = f"{DB}/{subdir}"
            all_results.setdefault(dataset_name, {})

            ds_name = f"{DB}"

            for MODEL in models_to_run:
                all_results[dataset_name].setdefault(MODEL, {})

                for loss_function, scoring_technique in loss_scoring_pairs:
                    # Hyperparameters come from the report for the *clean* (base) config,
                    # keyed by (DB, model, loss); reused across all noise levels.
                    params = params_by_key.get((DB, MODEL, loss_function))
                    if params is None:
                        logger.warning(
                            f"No report params for ({DB}, {MODEL}, {loss_function}), "
                            f"skipping on {dataset_name}"
                        )
                        all_results[dataset_name][MODEL].setdefault(loss_function, {}).setdefault(
                            scoring_technique, {})[neg_ratios[0]] = None
                        continue
                    params = {**params, **param_overrides}

                    dataset_folder = datasets_root / DB / subdir
                    # neg_ratio only changes the experiment for NegSample; other
                    # scoring techniques are invariant to it, so run once.
                    effective_neg_ratios = (
                        neg_ratios if scoring_technique == "NegSample" else neg_ratios[:1]
                    )
                    if not dataset_folder.exists():
                        logger.warning(
                            f"{dataset_folder} does not exist, skipping "
                            f"{MODEL} on {dataset_name} with loss={loss_function}"
                        )
                        for neg_ratio in effective_neg_ratios:
                            all_results[dataset_name][MODEL].setdefault(loss_function, {}).setdefault(
                                scoring_technique, {})[neg_ratio] = None
                        continue
                    for neg_ratio in effective_neg_ratios:
                        seed_metrics = []
                        for seed in random_seeds:
                            tag = (f"loss={loss_function} scoring={scoring_technique} "
                                   f"neg_ratio={neg_ratio} seed={seed}")
                            logger.info(f"Running experiment: {MODEL} on {dataset_name} with {tag} | params={params}")
                            try:
                                result = run_dicee_eval(
                                    dataset_folder=str(dataset_folder),
                                    model=MODEL,
                                    num_epochs=num_epochs,
                                    loss_function=loss_function,
                                    path_to_store_single_run=str(
                                        results_dir / DB / subdir / MODEL / f"{loss_function}_{scoring_technique}_neg{neg_ratio}" / f"seed{seed}" / ""
                                    ),
                                    scoring_technique=scoring_technique,
                                    optim=optim,
                                    num_core=16,
                                    eval_model=eval_model,
                                    trainer=trainer,
                                    accelerator=accelerator,
                                    devices=devices,
                                    precision=precision,
                                    random_seed=seed,
                                    neg_ratio=neg_ratio,
                                    num_of_output_channels=num_of_output_channels,
                                    block_size=block_size,
                                    **params,
                                )

                                test_metrics = result.get('Test', {})
                                seed_metrics.append(test_metrics)
                                logger.info(f"Completed: {MODEL} on {dataset_name} {tag} - Test MRR: {test_metrics.get('MRR')}")
                            except Exception:
                                logger.exception(f"Error running {MODEL} on {dataset_name} with {tag}")
                            finally:
                                # Drop references to the trained model/results before
                                # clearing the cache, otherwise empty_cache() has nothing
                                # to release and GPU memory accumulates across runs.
                                try:
                                    del result
                                except NameError:
                                    pass
                                try:
                                    del test_metrics
                                except NameError:
                                    pass
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                        averaged_metrics = _average_metrics(seed_metrics)
                        all_results[dataset_name][MODEL].setdefault(loss_function, {}).setdefault(
                            scoring_technique, {})[neg_ratio] = averaged_metrics
                        if averaged_metrics:
                            logger.info(
                                f"Averaged over {len(random_seeds)} seeds: {MODEL} on {dataset_name} "
                                f"loss={loss_function} scoring={scoring_technique} neg_ratio={neg_ratio} "
                                f"- Mean Test MRR: {averaged_metrics.get('MRR')}"
                            )

    # Create and save results table (single combined table for all (loss, scoring) configs)
    if all_results:
        df, pivot_df = create_results_table(all_results)
        save_results_table(
            df,
            pivot_df,
            project_root / "robust-kge" / "bo_results" / "new_random_perturbed_seed",
            loss_scoring_pairs,
            models_to_run,
            neg_ratios,
            kind=run_kind,
            ds_name = ds_name
        )
    else:
        logger.info("No results to save.")
