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
robust_kge_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(robust_kge_dir))

# Get project root (parent of robust-kge directory)
project_root = robust_kge_dir.parent

# Add project root to Python path so dicee can be imported
sys.path.insert(0, str(project_root))

from config import (DBS,
                    MODELS,
                    BATCH_SIZE,
                    LEARNING_RATE,
                    EMB_DIM,
                    LOSS_FN
                    )
NUM_EPOCHS = 100
SCORING_TECH = "KvsAll"
OPTIM = "Adam"
EVAL_MODEL = "val"

def create_results_table(results_dict):
    # Structure: results_dict[dataset][model][loss][scoring] = mrr_value
    rows = []
    for dataset, models_dict in results_dict.items():
        for model, losses_dict in models_dict.items():
            for loss, scoring_dict in losses_dict.items():
                for scoring, mrr in scoring_dict.items():
                    rows.append({
                        'Dataset': dataset,
                        'Model': model,
                        'Loss': loss,
                        'Scoring': scoring,
                        'Val_MRR': mrr
                    })

    df = pd.DataFrame(rows)

    # Pivot: models as rows, (Dataset, Loss, Scoring) as MultiIndex columns
    pivot_df = df.pivot_table(
        index='Model',
        columns=['Dataset', 'Loss', 'Scoring'],
        values='Val_MRR',
        aggfunc='first'
    )
    pivot_df = pivot_df.sort_index(axis=1, level=[0, 1, 2])

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


def _safe_suffix(suffix, max_len=120):
    """Bound a filename suffix so the full path stays under the OS limit.

    If the assembled suffix exceeds max_len characters, truncate it and append
    a short hash of the full suffix so distinct configurations stay unique.
    """
    if len(suffix) <= max_len:
        return suffix
    digest = hashlib.sha1(suffix.encode()).hexdigest()[:8]
    return f"{suffix[:max_len]}_{digest}"


def create_visualization(pivot_df, output_dir, configs=None, models=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten MultiIndex columns to "Dataset\nLoss/Scoring" labels for the heatmap
    display_df = pivot_df.copy()
    if isinstance(display_df.columns, pd.MultiIndex):
        flat = []
        for col in display_df.columns:
            if len(col) == 3:
                ds, loss, scoring = col
                flat.append(f"{ds}\n{loss}/{scoring}")
            else:
                flat.append("/".join(str(c) for c in col))
        display_df.columns = flat

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (max(12, len(display_df.columns) * 1.5), max(8, len(display_df.index) * 0.8))

    fig, ax = plt.subplots(figsize=(max(14, len(display_df.columns) * 1.8), max(10, len(display_df.index) * 1.0)))

    sns.heatmap(
        display_df,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Val MRR'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,
        vmax=1.0
    )

    # Title: list all (loss, scoring) configs
    if not configs:
        config_name = "Default"
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
    title = f'MRR Comparison: Models vs Datasets (Val Set) - Configs: {config_name}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset / Loss / Scoring', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = _safe_suffix(f"{_models_suffix(models)}{_loss_suffix(configs)}")
    image_path = output_dir / f"mrr_comparison_heatmap{suffix}_{timestamp}.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap visualization saved to: {image_path}")

    plt.close()

    return image_path

def save_results_table(df, pivot_df, output_dir, configs=None, models=None):
    """Save results to CSV files and create visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = _safe_suffix(f"{_models_suffix(models)}{_loss_suffix(configs)}")

    # Save detailed results
    detailed_path = output_dir / f"detailed_results{suffix}_{timestamp}.csv"
    df.to_csv(detailed_path, index=False)
    print(f"\nDetailed results saved to: {detailed_path}")

    # Save pivot table
    pivot_path = output_dir / f"comparison_table{suffix}_{timestamp}.csv"
    pivot_df.to_csv(pivot_path)
    print(f"Comparison table saved to: {pivot_path}")

    # Print formatted table
    print("\n" + "="*80)
    print("MRR Comparison Table (Val Set)")
    print("="*80)
    print(pivot_df.to_string())
    print("="*80)

    # Create heatmap visualization
    print("\nGenerating heatmap visualization...")
    create_visualization(pivot_df, output_dir, configs, models)

    return detailed_path, pivot_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run KGE experiments')
    parser.add_argument('--loss_fn', type=str, nargs='+', default=None,
                        help='One or more loss functions to use (e.g., --loss_fn BCELoss MSELoss). '
                             'If not provided, uses value from config.py')
    parser.add_argument('--model', type=str, nargs='+', default=None,
                        help='One or more models to run (e.g., --model Pykeen_TransE DistMult). '
                             'If not provided, uses MODELS from config.py')
    parser.add_argument('--lr', type=str, default=None,
    help = 'Learning rate to use (e.g., 0.1). If not provided, uses value from config.py')
    parser.add_argument('--batch_size', type=str, default=None,
    help = 'Batch size to use (e.g., 1024). If not provided, uses value from config.py')
    parser.add_argument('--num_epochs', type=str, default=None,
    help = 'Number of epochs to use (e.g., 100). If not provided, uses value from config.py')
    parser.add_argument('--emb_dim', type=str, default=None,
    help = 'Embedding dimension to use (e.g., 32). If not provided, uses value from config.py')
    parser.add_argument('--scoring_technique', type=str, nargs='+', default=None,
    help = 'One or more scoring techniques (e.g., KvsAll NegSample). '
           'If multiple are given, must match --loss_fn in length and is paired 1:1. '
           'A single value is broadcast to every loss. If omitted, uses SCORING_TECH from config.')
    parser.add_argument('--optim', type=str, default=None,
    help = 'Optimizer to use (e.g., Adam). If not provided, uses value from config.py')
    parser.add_argument('--eval_model', type=str, default=None,
    help = 'Evaluation model to use (e.g., train_val_test). If not provided, uses value from config.py')
    parser.add_argument("--trainer", type=str, default="PL")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--random_seed", type=str, default=None)
    parser.add_argument("--neg_ratio", type=str, default="2",
        help="Negative sample ratio for NegSample scoring technique (default: 2, matching run.py)")
    parser.add_argument("--num_of_output_channels", type=str, default=None)
    parser.add_argument("--block_size", type=str, default=None)
    parser.add_argument("--noise_ratio", type=str, default=None,
        choices=["0.0", "0.08", "0.16", "0.32"],
        help="Single noise ratio subdir to run (e.g., 0.08). If omitted, all of 0.0/0.08/0.16/0.32 are run.")
    args = parser.parse_args()
    
    # Use command-line argument if provided, otherwise use config value.
    # loss_functions is always a list so we can iterate uniformly.
    if args.loss_fn:
        loss_functions = args.loss_fn
    elif isinstance(LOSS_FN, (list, tuple)):
        loss_functions = list(LOSS_FN)
    else:
        loss_functions = [LOSS_FN]

    # Models: CLI overrides config; config may be a list or a single string.
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
    random_seed = args.random_seed if args.random_seed else None
    neg_ratio = args.neg_ratio if args.neg_ratio else "2"  # default matches run.py's --neg_ratio default
    num_of_output_channels = args.num_of_output_channels if args.num_of_output_channels else None
    block_size = args.block_size if args.block_size else None
    
    allowed_subdirs = {args.noise_ratio} if args.noise_ratio else {"0.0", "0.08", "0.16", "0.32"}

    # Dictionary to store all results: {dataset: {model: mrr}}
    all_results = {}
    
    # Default to saved_models in robust-kge directory
    results_dir = robust_kge_dir / "saved_models_val"
    
    # Run new experiments
    for DB in DBS:
        
        db_path = project_root / "Datasets_Perturbed" / DB
        if db_path.exists():
            subdirs = sorted(
                d.name for d in db_path.iterdir()
                if d.is_dir() and d.name in allowed_subdirs
            )
        else:
            print(f"Warning: {db_path} does not exist, skipping {DB}")
            continue
        
        for subdir in subdirs:
            dataset_name = f"{DB}/{subdir}"
            all_results.setdefault(dataset_name, {})

            for MODEL in models_to_run:
                all_results[dataset_name].setdefault(MODEL, {})

                for loss_function, scoring_technique in loss_scoring_pairs:
                    tag = f"loss={loss_function} scoring={scoring_technique}"
                    print(f"Running experiment: {MODEL} on {dataset_name} with {tag}")
                    try:
                        result = run_dicee_eval(
                            dataset_folder=str(project_root / "Datasets_Perturbed" / DB / subdir),
                            model=MODEL,
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            embedding_dim=embedding_dim,
                            loss_function=loss_function,
                            path_to_store_single_run=str(
                                results_dir / DB / subdir / MODEL / f"{loss_function}_{scoring_technique}" / ""
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

                        val_mrr = result.get('Val', {}).get('MRR', None)

                        all_results[dataset_name][MODEL].setdefault(loss_function, {})[scoring_technique] = val_mrr
                        print(f"Completed: {MODEL} on {dataset_name} {tag} - Val MRR: {val_mrr}")
                    except Exception as e:
                        print(f"Error running {MODEL} on {dataset_name} with {tag}: {e}")
                        all_results[dataset_name][MODEL].setdefault(loss_function, {})[scoring_technique] = None

    # Create and save results table (single combined table for all (loss, scoring) configs)
    if all_results:
        df, pivot_df = create_results_table(all_results)
        save_results_table(
            df,
            pivot_df,
            project_root / "robust-kge" / "val_results",
            loss_scoring_pairs,
            models_to_run,
        )
    else:
        print("No results to save.")
