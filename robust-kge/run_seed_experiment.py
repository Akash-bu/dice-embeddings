import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import sys
import ast
import gc
import random
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from dicee.executer import run_dicee_eval


robust_kge_dir = Path(__file__).resolve().parent
project_root = robust_kge_dir.parent.resolve()
sys.path.insert(0, str(robust_kge_dir))
sys.path.insert(0, str(project_root))


NUM_EPOCHS = 100
SCORING_TECH = "KvsAll"
OPTIM = "Adam"
EVAL_MODEL = "test"
N_SEEDS = 3

# Losses that require pre-generated PCRA path files (Datasets_Perturbed_pcra)
PCRA_LOSSES = {
    "LocalTripleWithPriorPathLoss",
    "LocalTripleWithPriorAndAdaptivePathLoss",
    "DSKRLLoss",
}

# Losses that require NegSample scoring (incompatible with KvsAll due to x_batch shape)
NEGSAMPLE_LOSSES = {
    "LocalTripleLoss",
    "LocalTripleWithPriorPathLoss",
    "LocalTripleWithPriorAndAdaptivePathLoss",
    "DSKRLLoss",
    "PTrustELoss",
}


def abs_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def normalize_dataset_key(dataset_str: str) -> str:
    parts = [p for p in dataset_str.strip().replace("\\", "/").split("/") if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-2]}/{parts[-1]}"


def parse_report_file(report_path: Path):
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    rows = []
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

            rows.append(
                {
                    "Dataset": normalize_dataset_key(dataset_str),
                    "Model": model_str.strip(),
                    "Loss": loss_str.strip(),
                    "Value": value,
                    "Params": params,
                }
            )

    best_by_key = {}
    for row in rows:
        key = (row["Dataset"], row["Model"], row["Loss"])
        prev = best_by_key.get(key)
        if prev is None:
            best_by_key[key] = row
            continue

        prev_value = prev["Value"]
        curr_value = row["Value"]
        if prev_value is None and curr_value is not None:
            best_by_key[key] = row
        elif curr_value is not None and prev_value is not None and curr_value > prev_value:
            best_by_key[key] = row

    return list(best_by_key.values())


def build_zero_point_plan(entries):
    grouped = {}
    for entry in entries:
        dataset_key = entry["Dataset"]
        db = dataset_key.split("/", 1)[0]
        key = (db, entry["Model"], entry["Loss"])
        grouped.setdefault(key, []).append(entry)

    plan = []
    for (db, model, loss_fn), group_entries in grouped.items():
        source = None
        for entry in group_entries:
            if entry["Dataset"] == f"{db}/0.0":
                source = entry
                break
        if source is None:
            for entry in group_entries:
                if entry["Dataset"] == db:
                    source = entry
                    break

        if source is None:
            continue

        plan.append(
            {
                "DB": db,
                "Model": model,
                "Loss": loss_fn,
                "Params": source["Params"],
            }
        )

    return plan


def resolve_dataset_targets(datasets_root: Path, db: str, noise_levels: set = None):
    db_path = (datasets_root / db).resolve()
    if not db_path.exists():
        return []
    subdirs = sorted(d.name for d in db_path.iterdir() if d.is_dir())
    if noise_levels is not None:
        subdirs = [s for s in subdirs if s in noise_levels]
    return [f"{db}/{subdir}" for subdir in subdirs]


def create_visualization_mean_std(mean_pivot, std_pivot, output_dir: Path):
    if mean_pivot.empty:
        return

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=(max(14, len(mean_pivot.columns) * 1.8), max(10, len(mean_pivot.index) * 1.0))
    )

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
                annot.iat[i, j] = f"{mean_val:.4f}\\n±{std_val:.4f}"

    sns.heatmap(
        mean_pivot,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        cbar_kws={"label": "Test MRR (Mean)"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        vmin=0,
        vmax=1.0,
    )

    ax.set_title("MRR Mean±Std: (Model, Loss) vs Datasets (Test Set)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model / Loss", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"mrr_mean_std_report_params_3seeds_{timestamp}.png"
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Mean±Std heatmap visualization saved to: {image_path}")


def save_pretty_table(pivot_df: pd.DataFrame, out_path: Path, title: str):
    line = "=" * 80
    with open(out_path, "w") as f:
        f.write(f"{line}\n")
        f.write(f"{title}\n")
        f.write(f"{line}\n")
        f.write(pivot_df.to_string())
        f.write(f"\n{line}\n")
    print(f"Formatted table saved to: {out_path}")


def run_one_seed(seed: int, plan, datasets_root: Path, datasets_root_pcra: Path, saved_models_dir: Path, num_epochs: int, scoring_technique: str, optim: str, eval_model: str, trainer=None, accelerator=None, devices=None, precision=None, noise_levels: set = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rows = []
    for entry in plan:
        db = entry["DB"]
        model = entry["Model"]
        loss_fn = entry["Loss"]
        params = entry["Params"]

        # Use pcra dataset directory for losses that need PCRA path files
        if loss_fn in PCRA_LOSSES:
            effective_root = datasets_root_pcra
        else:
            effective_root = datasets_root

        # Override scoring technique for losses that require NegSample
        effective_scoring = "NegSample" if loss_fn in NEGSAMPLE_LOSSES else scoring_technique

        targets = resolve_dataset_targets(effective_root, db, noise_levels=noise_levels)
        for dataset_name in targets:
            dataset_folder = (effective_root / dataset_name).resolve()
            store_path = (saved_models_dir / f"seed_{seed}" / dataset_name / model / loss_fn).resolve()

            print("\n\n")
            print("============================================================")
            print(f"[seed={seed}] {dataset_name} | {model} | {loss_fn} | scoring={effective_scoring}")
            test_mrr = None
            try:
                result = run_dicee_eval(
                    dataset_folder=str(dataset_folder),
                    model=model,
                    num_epochs=num_epochs,
                    loss_function=loss_fn,
                    path_to_store_single_run=str(store_path),
                    scoring_technique=effective_scoring,
                    optim=optim,
                    eval_model=eval_model,
                    random_seed=seed,
                    trainer=trainer,
                    accelerator=accelerator,
                    devices=devices,
                    precision=precision,
                    **params,
                )
                test_mrr = result.get("Test", {}).get("MRR", None)
                del result
            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] {dataset_name} | {model} | {loss_fn}: {e}")
            except Exception as e:
                print(f"  [ERROR] {dataset_name} | {model} | {loss_fn}: {e}")
            finally:
                # Always free GPU memory before the next run
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            rows.append(
                {
                    "Seed": seed,
                    "Dataset": dataset_name,
                    "Model": model,
                    "Loss": loss_fn,
                    "Test_MRR": test_mrr,
                }
            )

    return rows


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run 3-seed experiments using 0.0 BO params for each DB/model/loss.")
    parser.add_argument("--report_file", type=str, default=None)
    parser.add_argument("--datasets_root", type=str, default=str(project_root / "Datasets_Perturbed"))
    parser.add_argument("--datasets_root_pcra", type=str, default=str(project_root / "Datasets_Perturbed_pcra"),
                        help="Dataset directory for PCRA-based losses (DSKRLLoss, LocalTripleWithPriorPathLoss, etc.). "
                             "Automatically used when a loss is in PCRA_LOSSES.")
    parser.add_argument("--seed_min", type=int, default=10001)
    parser.add_argument("--seed_max", type=int, default=999999)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--scoring_technique", type=str, default=SCORING_TECH)
    parser.add_argument("--optim", type=str, default=OPTIM)
    parser.add_argument("--eval_model", type=str, default=EVAL_MODEL)
    parser.add_argument("--trainer", type=str, default="PL")
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--noise_levels", type=str, nargs="+", default=None,
                        help="Noise level subdirs to include, e.g. --noise_levels 0.0 0.08 0.16 0.32. Default: all.")
    parser.add_argument("--results_dir", type=str, default=str(robust_kge_dir / "seed_runs"))
    parser.add_argument("--saved_models_dir", type=str, default=str(robust_kge_dir / "saved_models_seed_runs"))
    args = parser.parse_args()

    devices = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)

    noise_levels = set(args.noise_levels) if args.noise_levels is not None else None

    report_file = abs_path(args.report_file, project_root)
    datasets_root = abs_path(args.datasets_root, project_root)
    datasets_root_pcra = abs_path(args.datasets_root_pcra, project_root)
    results_dir = abs_path(args.results_dir, project_root)
    saved_models_dir = abs_path(args.saved_models_dir, project_root)

    entries = parse_report_file(report_file)
    plan = build_zero_point_plan(entries)
    if not plan:
        raise ValueError("No valid (DB, model, loss) plan could be built from report file.")

    seed_range_size = args.seed_max - args.seed_min + 1
    if seed_range_size < N_SEEDS:
        raise ValueError(f"Seed range must contain at least {N_SEEDS} values.")

    seeds = random.SystemRandom().sample(range(args.seed_min, args.seed_max + 1), N_SEEDS)
    print(f"Running {N_SEEDS} seeds: {seeds}")

    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = results_dir / f"seed_batch_{N_SEEDS}_{args.seed_min}-{args.seed_max}_{batch_ts}"
    batch_root.mkdir(parents=True, exist_ok=True)

    # Log which losses will use pcra or NegSample
    pcra_in_plan = [e["Loss"] for e in plan if e["Loss"] in PCRA_LOSSES]
    if pcra_in_plan:
        print(f"PCRA losses detected: {sorted(set(pcra_in_plan))} -> using {datasets_root_pcra}")
    negsample_in_plan = [e["Loss"] for e in plan if e["Loss"] in NEGSAMPLE_LOSSES]
    if negsample_in_plan:
        print(f"NegSample losses detected: {sorted(set(negsample_in_plan))} -> scoring_technique=NegSample")

    all_rows = []
    for seed in seeds:
        all_rows.extend(
            run_one_seed(
                seed=seed,
                plan=plan,
                datasets_root=datasets_root,
                datasets_root_pcra=datasets_root_pcra,
                saved_models_dir=saved_models_dir,
                num_epochs=args.num_epochs,
                scoring_technique=args.scoring_technique,
                optim=args.optim,
                eval_model=args.eval_model,
                trainer=args.trainer,
                accelerator=args.accelerator,
                devices=devices,
                precision=args.precision,
                noise_levels=noise_levels,
            )
        )

    df = pd.DataFrame(all_rows)
    df["Test_MRR"] = pd.to_numeric(df["Test_MRR"], errors="coerce")

    out_dir = batch_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_run_path = out_dir / f"per_run_results_report_params_{N_SEEDS}seeds_{ts}.csv"
    df.to_csv(per_run_path, index=False)
    print(f"Per-run results saved to: {per_run_path}")

    stats_df = df.groupby(["Dataset", "Model", "Loss"])["Test_MRR"].agg(["mean", "std", "count"]).reset_index()
    stats_df = stats_df.rename(columns={"mean": "MRR_mean", "std": "MRR_std", "count": "Runs"})
    stats_path = out_dir / f"summary_mean_std_report_params_{N_SEEDS}seeds_{ts}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Summary mean/std saved to: {stats_path}")

    mean_pivot = stats_df.pivot_table(index=["Model", "Loss"], columns="Dataset", values="MRR_mean", aggfunc="first")
    std_pivot = stats_df.pivot_table(index=["Model", "Loss"], columns="Dataset", values="MRR_std", aggfunc="first")

    mean_path = out_dir / f"comparison_table_mean_report_params_{N_SEEDS}seeds_{ts}.csv"
    std_path = out_dir / f"comparison_table_std_report_params_{N_SEEDS}seeds_{ts}.csv"
    mean_pivot.to_csv(mean_path)
    std_pivot.to_csv(std_path)
    print(f"Mean table saved to: {mean_path}")
    print(f"Std table saved to: {std_path}")

    table_txt_path = out_dir / f"comparison_table_mean_report_params_{N_SEEDS}seeds_{ts}.txt"
    save_pretty_table(mean_pivot, table_txt_path, "MRR Comparison Table (Mean Across Seeds)")

    create_visualization_mean_std(mean_pivot, std_pivot, out_dir)


