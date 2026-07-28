import gc
import random

import numpy as np
import torch

from pathlib import Path
import hashlib
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from dicee.executer import run_dicee_eval

# Add the robust-kge directory to the path for config import
robust_kge_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(robust_kge_dir))

# Get project root (parent of robust-kge directory)
project_root = robust_kge_dir.parent.resolve()

# Add project root to Python path so dicee can be imported
sys.path.insert(0, str(project_root))

from config import (DBS,
                    MODELS,
                    BATCH_SIZE,
                    LEARNING_RATE,
                    NUM_EPOCHS,
                    EMB_DIM,
                    LOSS_FN,
                    SCORING_TECH,
                    OPTIM,
                    EVAL_MODEL_TEST
                    )

N_SEEDS = 3


# ---------------------------------------------------------------------------
# Results table helpers
# ---------------------------------------------------------------------------

def _safe_suffix(suffix, max_len=120):
    """Bound a filename suffix so the full path stays under the OS limit.

    If the assembled suffix exceeds max_len characters, truncate it and append
    a short hash of the full suffix so distinct configurations stay unique.
    """
    if len(suffix) <= max_len:
        return suffix
    digest = hashlib.sha1(suffix.encode()).hexdigest()[:8]
    return f"{suffix[:max_len]}_{digest}"


def create_results_table(results_dict):
    """results_dict[dataset][model] = list of test_mrr values (one per seed)."""
    rows = []
    for dataset, models_dict in results_dict.items():
        for model, mrr_list in models_dict.items():
            valid = [v for v in mrr_list if v is not None]
            mean_mrr = float(np.mean(valid)) if valid else None
            std_mrr = float(np.std(valid, ddof=1)) if len(valid) > 1 else None
            rows.append({
                'Dataset': dataset,
                'Model': model,
                'Test_MRR_mean': mean_mrr,
                'Test_MRR_std': std_mrr,
                'Runs': len(valid),
            })

    df = pd.DataFrame(rows)

    mean_pivot = df.pivot_table(
        index='Model',
        columns='Dataset',
        values='Test_MRR_mean',
        aggfunc='first'
    )
    std_pivot = df.pivot_table(
        index='Model',
        columns='Dataset',
        values='Test_MRR_std',
        aggfunc='first'
    )

    return df, mean_pivot, std_pivot


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_visualization(mean_pivot, std_pivot, output_dir, loss_fn=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mean_pivot.empty:
        return None

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=(max(14, len(mean_pivot.columns) * 1.8), max(10, len(mean_pivot.index) * 1.0))
    )

    # Build annotation matrix: "mean\n±std"
    annot = mean_pivot.copy().astype(object)
    for i in range(mean_pivot.shape[0]):
        for j in range(mean_pivot.shape[1]):
            mean_val = mean_pivot.iat[i, j]
            std_val = std_pivot.iat[i, j]
            if pd.isna(mean_val):
                annot.iat[i, j] = ""
            elif pd.isna(std_val):
                annot.iat[i, j] = f"{mean_val:.4f}"
            else:
                annot.iat[i, j] = f"{mean_val:.4f}\n±{std_val:.4f}"

    sns.heatmap(
        mean_pivot,
        annot=annot,
        fmt="",
        cmap='RdYlGn',
        cbar_kws={'label': 'Test MRR (Mean)'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,
        vmax=1.0
    )

    loss_name = loss_fn if loss_fn else "Default"
    title = f'Test MRR Mean±Std: Models vs Datasets ({N_SEEDS} seeds) - Loss: {loss_name}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    loss_suffix = _safe_suffix(f"_{loss_fn}" if loss_fn else "")
    image_path = output_dir / f"test_mrr_mean_std_heatmap{loss_suffix}_{N_SEEDS}seeds_{timestamp}.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap visualization saved to: {image_path}")

    plt.close()
    return image_path


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_results_table(df, mean_pivot, std_pivot, output_dir, loss_fn=None):
    """Save results to CSV files and create visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    loss_suffix = _safe_suffix(f"_{loss_fn}" if loss_fn else "")

    # Detailed per-dataset/model summary
    detailed_path = output_dir / f"detailed_results_test{loss_suffix}_{N_SEEDS}seeds_{timestamp}.csv"
    df.to_csv(detailed_path, index=False)
    print(f"\nDetailed results saved to: {detailed_path}")

    # Mean pivot
    mean_path = output_dir / f"comparison_table_mean_test{loss_suffix}_{N_SEEDS}seeds_{timestamp}.csv"
    mean_pivot.to_csv(mean_path)
    print(f"Mean comparison table saved to: {mean_path}")

    # Std pivot
    std_path = output_dir / f"comparison_table_std_test{loss_suffix}_{N_SEEDS}seeds_{timestamp}.csv"
    std_pivot.to_csv(std_path)
    print(f"Std comparison table saved to: {std_path}")

    # Pretty-print
    print("\n" + "=" * 80)
    print(f"Test MRR Comparison Table (Mean across {N_SEEDS} seeds)")
    print("=" * 80)
    print(mean_pivot.to_string())
    print("=" * 80)

    print("\nGenerating heatmap visualization...")
    create_visualization(mean_pivot, std_pivot, output_dir, loss_fn)

    return detailed_path, mean_path, std_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run KGE experiments with 3 seeds, reporting Test MRR')
    parser.add_argument('--loss_fn', type=str, default=None,
                        help='Loss function to use (e.g., BCELoss). If not provided, uses value from config.py')
    parser.add_argument('--lr', type=str, default=None,
        help='Learning rate to use (e.g., 0.1). If not provided, uses value from config.py')
    parser.add_argument('--batch_size', type=str, default=None,
        help='Batch size to use (e.g., 1024). If not provided, uses value from config.py')
    parser.add_argument('--num_epochs', type=str, default=None,
        help='Number of epochs to use (e.g., 100). If not provided, uses value from config.py')
    parser.add_argument('--emb_dim', type=str, default=None,
        help='Embedding dimension to use (e.g., 32). If not provided, uses value from config.py')
    parser.add_argument('--scoring_technique', type=str, default=None,
        help='Scoring technique to use (e.g., KvsAll). If not provided, uses value from config.py')
    parser.add_argument('--optim', type=str, default=None,
        help='Optimizer to use (e.g., Adam). If not provided, uses value from config.py')
    parser.add_argument('--eval_model', type=str, default=None,
        help='Evaluation model to use (e.g., train_val_test). If not provided, uses value from config.py')
    parser.add_argument("--trainer", type=str, default="PL")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--neg_ratio", type=str, default="2",
        help="Negative sample ratio for NegSample scoring technique (default: 2, matching run.py)")
    parser.add_argument("--num_of_output_channels", type=str, default=None)
    parser.add_argument("--block_size", type=str, default=None)
    parser.add_argument("--seed_min", type=int, default=10001,
        help="Lower bound of the seed sampling range (inclusive).")
    parser.add_argument("--seed_max", type=int, default=999999,
        help="Upper bound of the seed sampling range (inclusive).")
    parser.add_argument("--datasets_root", type=str, default="Datasets_Perturbed",
        choices=["Datasets_Perturbed", "Datasets_Perturbed_pcra"],
        help="Dataset directory to use. 'Datasets_Perturbed_pcra' includes pre-generated PCRA path files "
             "required for LocalTripleWithPriorPathLoss and similar losses (default: Datasets_Perturbed).")
    args = parser.parse_args()

    # Resolve hyperparameters (CLI overrides config)
    loss_function = args.loss_fn if args.loss_fn else LOSS_FN
    learning_rate = args.lr if args.lr else LEARNING_RATE
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    num_epochs = args.num_epochs if args.num_epochs else NUM_EPOCHS
    embedding_dim = args.emb_dim if args.emb_dim else EMB_DIM
    scoring_technique = args.scoring_technique if args.scoring_technique else SCORING_TECH
    optim = args.optim if args.optim else OPTIM
    eval_model = args.eval_model if args.eval_model else EVAL_MODEL_TEST
    trainer = args.trainer if args.trainer else None
    accelerator = args.accelerator if args.accelerator else None
    devices = args.devices if args.devices else None
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)
    precision = args.precision if args.precision else None
    neg_ratio = args.neg_ratio if args.neg_ratio else "2"  # default matches run.py's --neg_ratio default
    num_of_output_channels = args.num_of_output_channels if args.num_of_output_channels else None
    block_size = args.block_size if args.block_size else None
    datasets_root = args.datasets_root
    print(f"Using dataset root: {datasets_root}")

    # Draw N_SEEDS random seeds
    seed_range_size = args.seed_max - args.seed_min + 1
    if seed_range_size < N_SEEDS:
        raise ValueError(f"Seed range must contain at least {N_SEEDS} values.")
    seeds = random.SystemRandom().sample(range(args.seed_min, args.seed_max + 1), N_SEEDS)
    print(f"Running {N_SEEDS} seeds: {seeds}")

    allowed_subdirs = {"0.0", "0.08", "0.16", "0.32"}

    # {dataset_name: {model: [mrr_seed1, mrr_seed2, mrr_seed3]}}
    all_results = {}

    loss_folder = loss_function if loss_function else "default"

    for seed in seeds:
        # Set global RNG state for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n{'='*60}")
        print(f"  Starting seed {seed}")
        print(f"{'='*60}")

        for DB in DBS:
            db_path = project_root / datasets_root / DB
            if db_path.exists():
                subdirs = sorted(
                    d.name for d in db_path.iterdir()
                    if d.is_dir() and d.name in allowed_subdirs
                )
            else:
                print(f"Warning: {db_path} does not exist, skipping {DB}")
                continue

            # Saved models: saved_models/<DB>/<loss>/
            db_models_dir = robust_kge_dir / "saved_models" / DB / loss_folder

            for subdir in subdirs:
                dataset_name = f"{DB}/{subdir}"
                if dataset_name not in all_results:
                    all_results[dataset_name] = {}

                for MODEL in MODELS:
                    if MODEL not in all_results[dataset_name]:
                        all_results[dataset_name][MODEL] = []

                    print(f"[seed={seed}] Running experiment: {MODEL} on {dataset_name}")
                    try:
                        result = run_dicee_eval(
                            dataset_folder=str(project_root / datasets_root / DB / subdir),
                            model=MODEL,
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            embedding_dim=embedding_dim,
                            loss_function=loss_function,
                            path_to_store_single_run=str(
                                db_models_dir / f"seed_{seed}" / subdir / MODEL / ""
                            ),
                            scoring_technique=scoring_technique,
                            optim=optim,
                            eval_model=eval_model,
                            trainer=trainer,
                            accelerator=accelerator,
                            devices=devices,
                            precision=precision,
                            random_seed=seed,
                            neg_ratio=neg_ratio,
                            num_of_output_channels=num_of_output_channels,
                            block_size=block_size,
                        )

                        test_mrr = result.get('Test', {}).get('MRR', None)
                        all_results[dataset_name][MODEL].append(test_mrr)
                        print(f"  Completed: {MODEL} on {dataset_name} - Test MRR: {test_mrr}")

                        # Free GPU memory between runs
                        del result
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"  Error running {MODEL} on {dataset_name}: {e}")
                        all_results[dataset_name][MODEL].append(None)

    # ------------------------------------------------------------------
    # Save per-seed raw rows, grouped by dataset (DB)
    # ------------------------------------------------------------------
    timestamp_final = datetime.now().strftime('%Y%m%d_%H%M%S')
    loss_suffix_final = f"_{loss_function}" if loss_function else ""

    # Group results by DB so each dataset gets its own results folder
    db_results: dict[str, dict] = {}  # {DB: {dataset_name: {model: [mrrs]}}}
    for dataset_name, models_dict in all_results.items():
        db_name = dataset_name.split("/")[0]
        if db_name not in db_results:
            db_results[db_name] = {}
        db_results[db_name][dataset_name] = models_dict

    all_raw_rows = []
    for db_name, db_data in db_results.items():
        # Per-dataset output directory: results/<DB>/<loss>/
        db_out_dir = project_root / "robust-kge" / "results" / db_name / loss_folder
        db_out_dir.mkdir(parents=True, exist_ok=True)

        # Per-run CSV for this dataset
        db_raw_rows = []
        for dataset_name, models_dict in db_data.items():
            for model, mrr_list in models_dict.items():
                for seed, mrr in zip(seeds, mrr_list):
                    row = {
                        'Seed': seed,
                        'Dataset': dataset_name,
                        'Model': model,
                        'Loss': loss_function,
                        'Test_MRR': mrr,
                    }
                    db_raw_rows.append(row)
                    all_raw_rows.append(row)

        db_raw_df = pd.DataFrame(db_raw_rows)
        db_per_run_path = (
            db_out_dir
            / f"per_run_test_mrr{loss_suffix_final}_{N_SEEDS}seeds_{timestamp_final}.csv"
        )
        db_raw_df.to_csv(db_per_run_path, index=False)
        print(f"\n[{db_name}] Per-run results saved to: {db_per_run_path}")

        # Aggregate summary + heatmap for this dataset
        if db_data:
            db_df, db_mean_pivot, db_std_pivot = create_results_table(db_data)
            save_results_table(db_df, db_mean_pivot, db_std_pivot, db_out_dir, loss_function)

    if not all_results:
        print("No results to save.") 


# Example usage:
# python robust-kge/run_experiment_test_3_seeds.py \
#   --trainer PL \
#   --accelerator gpu \
#   --devices 1 \
#   --loss_fn general_robust_loss
