from pathlib import Path
import hashlib
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from dicee.executer import run_dicee_eval

# Add the robust-kge directory to the path for config import
robust_kge_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(robust_kge_dir))

# Get project root (parent of robust-kge directory)
project_root = robust_kge_dir.parent

# Add project root to Python path so dicee can be imported
sys.path.insert(0, str(project_root))

from config import (MODELS,
                    BATCH_SIZE,
                    LEARNING_RATE,
                    EMB_DIM,
                    LOSS_FN
                    )
NUM_EPOCHS = 100
SCORING_TECH = "KvsAll"
OPTIM = "Adam"
EVAL_MODEL = "test"

# Base dataset providing the clean train.txt + valid.txt + test.txt for all
# noise levels. The CKRL N1/N2/N3 noise files were generated against FB15K
# (483,142 train triples) — pairing them with FB15k-237 is incorrect.
BASE_DB = "FB15k"
BASE_DATASET_DIR = project_root / "KGs" / BASE_DB

# Working dataset root: per-noise-level subdirs are created here with the
# concatenated train.txt and valid/test symlinks. Source FB15K-N* dirs are
# left untouched.
CRKL_ROOT = project_root / "crkl"

# Mapping: noise-level token -> (source dir with neg_train.txt, working subdir name).
NOISE_LEVEL_DIRS = {
    "10": (project_root / "FB15K-N1-10%", "N1-10%"),
    "20": (project_root / "FB15K-N2-20%", "N2-20%"),
    "40": (project_root / "FB15K-N3-40%", "N3-40%"),
}
ALLOWED_NOISE_LEVELS = list(NOISE_LEVEL_DIRS.keys())


def _ensure_symlink(link_path: Path, target_path: Path):
    """Create link_path -> target_path if it doesn't already point there."""
    if link_path.is_symlink():
        if Path(os.readlink(link_path)).resolve() == target_path.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target_path)


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _last_byte(path: Path) -> bytes:
    with path.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1)


def _build_concat_train(clean_train: Path, neg_train: Path, out_path: Path) -> None:
    """Write out_path = clean_train ++ neg_train (CKRL noise-injection).

    Idempotent: if out_path already exists as a regular file with the expected
    line count, skip the rewrite.
    """
    expected = _count_lines(clean_train) + _count_lines(neg_train)
    if out_path.is_file() and not out_path.is_symlink() and _count_lines(out_path) == expected:
        return
    if out_path.exists() or out_path.is_symlink():
        out_path.unlink()
    needs_sep = _last_byte(clean_train) != b"\n"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as out_f:
        with clean_train.open("rb") as in_f:
            for chunk in iter(lambda: in_f.read(1 << 20), b""):
                out_f.write(chunk)
        if needs_sep:
            out_f.write(b"\n")
        with neg_train.open("rb") as in_f:
            for chunk in iter(lambda: in_f.read(1 << 20), b""):
                out_f.write(chunk)
    tmp.replace(out_path)


def prepare_noise_level_dir(source_dir: Path, subdir_name: str) -> Path:
    """Build a working dataset dir under CRKL_ROOT for this noise level.

    Creates ``crkl/<subdir_name>/`` with:
      - train.txt : real file = KGs/FB15k/train.txt ++ source_dir/neg_train.txt
      - valid.txt -> KGs/FB15k/valid.txt
      - test.txt  -> KGs/FB15k/test.txt
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    neg_train = source_dir / "neg_train.txt"
    if not neg_train.exists():
        raise FileNotFoundError(f"Missing neg_train.txt in {source_dir}")
    if not BASE_DATASET_DIR.exists():
        raise FileNotFoundError(f"Base dataset not found: {BASE_DATASET_DIR}")

    work_dir = CRKL_ROOT / subdir_name
    work_dir.mkdir(parents=True, exist_ok=True)

    _build_concat_train(
        clean_train=BASE_DATASET_DIR / "train.txt",
        neg_train=neg_train,
        out_path=work_dir / "train.txt",
    )
    _ensure_symlink(work_dir / "valid.txt", BASE_DATASET_DIR / "valid.txt")
    _ensure_symlink(work_dir / "test.txt", BASE_DATASET_DIR / "test.txt")
    return work_dir


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
                            'Test_H@1': metrics.get('H@1'),
                            'Test_H@3': metrics.get('H@3'),
                            'Test_H@10': metrics.get('H@10'),
                        })

    df = pd.DataFrame(rows)

    pivot_df = df.pivot_table(
        index='Model',
        columns=['Dataset', 'Loss', 'Scoring', 'Neg_Ratio'],
        values='Test_MRR',
        aggfunc='first'
    )
    pivot_df = pivot_df.sort_index(axis=1, level=[0, 1, 2, 3])

    return df, pivot_df


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


def create_visualization(pivot_df, output_dir, configs=None, models=None, neg_ratios=None):
    """Render a publication-quality faceted heatmap of Test MRR.

    One panel per dataset, shared model axis, single colorbar. Outputs both a
    high-DPI PNG and a vector PDF (with editable text) for camera-ready use.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(pivot_df.columns, pd.MultiIndex) and "Dataset" in (pivot_df.columns.names or []):
        datasets = list(dict.fromkeys(pivot_df.columns.get_level_values("Dataset")))
        sub_dfs = {ds: pivot_df.xs(ds, axis=1, level="Dataset") for ds in datasets}
    else:
        datasets = [""]
        sub_dfs = {"": pivot_df}

    if not configs:
        config_name = "default"
    elif isinstance(configs, str):
        config_name = configs
    else:
        parts = []
        for item in configs:
            if isinstance(item, (list, tuple)):
                parts.append(f"{item[0]}/{item[1]}")
            else:
                parts.append(str(item))
        config_name = ", ".join(parts)
    if neg_ratios:
        neg_str = neg_ratios if isinstance(neg_ratios, str) else ", ".join(str(n) for n in neg_ratios)
    else:
        neg_str = None

    pub_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    sub_widths = [max(1, sub_dfs[ds].shape[1]) for ds in datasets]
    n_ds = len(datasets)
    n_rows = max(1, pivot_df.shape[0])

    cell = 0.75
    fig_w = sum(sub_widths) * cell + 1.8 + 1.2 + 0.35 * max(0, n_ds - 1)
    fig_h = n_rows * cell + 2.4

    with plt.rc_context(pub_rc):
        sns.set_style("white")
        fig, ax_row = plt.subplots(
            1,
            n_ds + 1,
            figsize=(fig_w, fig_h),
            gridspec_kw={"width_ratios": sub_widths + [0.35]},
            constrained_layout=True,
        )
        axes = list(ax_row[:-1])
        cbar_ax = ax_row[-1]

        for i, (ax, ds) in enumerate(zip(axes, datasets)):
            sub = sub_dfs[ds].copy()
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = [_format_inner_label(c) for c in sub.columns]

            show_y = i == 0
            show_cbar = i == n_ds - 1

            sns.heatmap(
                sub,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                linewidths=0.5,
                linecolor="white",
                ax=ax,
                cbar=show_cbar,
                cbar_ax=cbar_ax if show_cbar else None,
                cbar_kws={"label": "Test MRR"} if show_cbar else None,
                annot_kws={"fontsize": 9},
                yticklabels=show_y,
            )

            if ds:
                ax.set_title(ds, pad=8)
            ax.set_xlabel("")
            ax.set_ylabel("Model" if show_y else "")
            ax.tick_params(axis="both", which="both", length=0)
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center", va="top")
            if show_y:
                plt.setp(ax.get_yticklabels(), rotation=0)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#333333")

        cbar = axes[-1].collections[0].colorbar
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor("#333333")
        cbar.ax.tick_params(width=0.5)

        suptitle = f"Test MRR — configurations: {config_name}"
        if neg_str is not None:
            suptitle += f"  (negative ratios: {neg_str})"
        fig.suptitle(suptitle, fontsize=11)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = _safe_suffix(
            f"{_models_suffix(models)}{_loss_suffix(configs)}{_neg_ratio_suffix(neg_ratios)}"
        )
        base = f"mrr_comparison_heatmap_fb15k_n{suffix}_{timestamp}"
        image_path = output_dir / f"{base}.png"
        pdf_path = output_dir / f"{base}.pdf"
        fig.savefig(image_path, dpi=600)
        fig.savefig(pdf_path)
        print(f"Heatmap saved to: {image_path}")
        print(f"Heatmap (vector PDF) saved to: {pdf_path}")

        plt.close(fig)

    return image_path


def save_results_table(df, pivot_df, output_dir, configs=None, models=None, neg_ratios=None):
    """Save results to CSV files and create visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = _safe_suffix(
        f"{_models_suffix(models)}{_loss_suffix(configs)}{_neg_ratio_suffix(neg_ratios)}"
    )

    detailed_path = output_dir / f"detailed_results_fb15k_n{suffix}_{timestamp}.csv"
    df.to_csv(detailed_path, index=False)
    print(f"\nDetailed results saved to: {detailed_path}")

    pivot_path = output_dir / f"comparison_table_fb15k_n{suffix}_{timestamp}.csv"
    pivot_df.to_csv(pivot_path)
    print(f"Comparison table saved to: {pivot_path}")

    print("\n" + "="*80)
    print("MRR Comparison Table (Test Set) — FB15K-N1/N2/N3")
    print("="*80)
    print(pivot_df.to_string())
    print("="*80)

    print("\nGenerating heatmap visualization...")
    create_visualization(pivot_df, output_dir, configs, models, neg_ratios)

    return detailed_path, pivot_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run KGE experiments on FB15K-N1/N2/N3 (10/20/40%) datasets')
    parser.add_argument('--loss_fn', type=str, nargs='+', default=None,
                        help='One or more loss functions to use (e.g., --loss_fn BCELoss MSELoss). '
                             'If not provided, uses value from config.py')
    parser.add_argument('--model', type=str, nargs='+', default=None,
                        help='One or more models to run (e.g., --model Pykeen_TransE DistMult). '
                             'If not provided, uses MODELS from config.py')
    parser.add_argument('--lr', type=str, default=None,
    help='Learning rate to use (e.g., 0.1). If not provided, uses value from config.py')
    parser.add_argument('--batch_size', type=str, default=None,
    help='Batch size to use (e.g., 1024). If not provided, uses value from config.py')
    parser.add_argument('--num_epochs', type=str, default=None,
    help='Number of epochs to use (e.g., 100). If not provided, uses value from config.py')
    parser.add_argument('--emb_dim', type=str, default=None,
    help='Embedding dimension to use (e.g., 32). If not provided, uses value from config.py')
    parser.add_argument('--scoring_technique', type=str, nargs='+', default=None,
    help='One or more scoring techniques (e.g., KvsAll NegSample). '
         'If multiple are given, must match --loss_fn in length and is paired 1:1. '
         'A single value is broadcast to every loss. If omitted, uses SCORING_TECH from config.')
    parser.add_argument('--optim', type=str, default=None,
    help='Optimizer to use (e.g., Adam). If not provided, uses value from config.py')
    parser.add_argument('--eval_model', type=str, default=None,
    help='Evaluation model to use (e.g., train_val_test). If not provided, uses value from config.py')
    parser.add_argument("--trainer", type=str, default="PL")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--random_seed", type=str, default=None)
    parser.add_argument("--neg_ratio", type=str, nargs='+', default=None,
        help="One or more negative sample ratios for NegSample scoring technique "
             "(e.g., --neg_ratio 2 5 10). Each value runs as a separate experiment "
             "per (loss, scoring) pair. Default: ['2'], matching run.py.")
    parser.add_argument("--num_of_output_channels", type=str, default=None)
    parser.add_argument("--block_size", type=str, default=None)
    parser.add_argument("--noise_ratio", type=str, nargs='+', default=None,
        choices=ALLOWED_NOISE_LEVELS,
        help="One or more FB15K-N noise levels to run (10, 20, 40). "
             "If omitted, all of 10/20/40 are run.")
    args = parser.parse_args()

    if args.loss_fn:
        loss_functions = args.loss_fn
    elif isinstance(LOSS_FN, (list, tuple)):
        loss_functions = list(LOSS_FN)
    else:
        loss_functions = [LOSS_FN]

    if args.model:
        models_to_run = args.model
    elif isinstance(MODELS, (list, tuple)):
        models_to_run = list(MODELS)
    else:
        models_to_run = [MODELS]

    learning_rate = args.lr if args.lr else LEARNING_RATE
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    num_epochs = args.num_epochs if args.num_epochs else NUM_EPOCHS
    embedding_dim = args.emb_dim if args.emb_dim else EMB_DIM

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
    random_seed = args.random_seed if args.random_seed else None
    neg_ratios = args.neg_ratio if args.neg_ratio else ["2"]
    num_of_output_channels = args.num_of_output_channels if args.num_of_output_channels else None
    block_size = args.block_size if args.block_size else None

    noise_levels = args.noise_ratio if args.noise_ratio else list(ALLOWED_NOISE_LEVELS)

    if not BASE_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Base dataset {BASE_DATASET_DIR} not found — needed for valid/test splits."
        )

    all_results = {}
    results_dir = robust_kge_dir / "saved_models_test"

    DB = BASE_DB
    for level in noise_levels:
        source_dir, subdir = NOISE_LEVEL_DIRS[level]
        try:
            work_dir = prepare_noise_level_dir(source_dir, subdir)
        except FileNotFoundError as e:
            print(f"Warning: {e}, skipping noise level {level}%")
            continue

        dataset_name = f"{DB}/{subdir}"
        all_results.setdefault(dataset_name, {})

        for MODEL in models_to_run:
            all_results[dataset_name].setdefault(MODEL, {})

            for loss_function, scoring_technique in loss_scoring_pairs:
                for neg_ratio in neg_ratios:
                    tag = f"loss={loss_function} scoring={scoring_technique} neg_ratio={neg_ratio}"
                    print(f"Running experiment: {MODEL} on {dataset_name} with {tag}")
                    try:
                        result = run_dicee_eval(
                            dataset_folder=str(work_dir),
                            model=MODEL,
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            embedding_dim=embedding_dim,
                            loss_function=loss_function,
                            path_to_store_single_run=str(
                                results_dir / DB / subdir / MODEL / f"{loss_function}_{scoring_technique}_neg{neg_ratio}" / ""
                            ),
                            scoring_technique=scoring_technique,
                            optim=optim,
                            eval_model=EVAL_MODEL,
                            trainer=trainer,
                            accelerator=accelerator,
                            devices=devices,
                            precision=precision,
                            random_seed=random_seed,
                            neg_ratio=neg_ratio,
                            num_of_output_channels=num_of_output_channels,
                            block_size=block_size
                        )

                        test_metrics = result.get('Test', {})

                        all_results[dataset_name][MODEL].setdefault(loss_function, {}).setdefault(scoring_technique, {})[neg_ratio] = test_metrics
                        print(f"Completed: {MODEL} on {dataset_name} {tag} - Test MRR: {test_metrics.get('MRR')}")
                    except Exception as e:
                        print(f"Error running {MODEL} on {dataset_name} with {tag}: {e}")
                        all_results[dataset_name][MODEL].setdefault(loss_function, {}).setdefault(scoring_technique, {})[neg_ratio] = None

    if all_results:
        df, pivot_df = create_results_table(all_results)
        save_results_table(
            df,
            pivot_df,
            project_root / "robust-kge" / "test_results",
            loss_scoring_pairs,
            models_to_run,
            neg_ratios,
        )
    else:
        print("No results to save.")
