"""Run 3-seed experiments for losses, grouped by their required scoring technique.

Losses are split into groups that share a training/scoring technique so each
group can be launched independently:

  - 1vsall    (scoring_technique="1vsAll")  : NCELoss, NCEandAGCELoss, NCEandAULoss
  - negsample (scoring_technique="NegSample"): LocalTripleLoss, LocalTripleWithPriorPathLoss,
                                               LocalTripleWithPriorAndAdaptivePathLoss,
                                               DSKRLLoss, PTrustELoss
  - kvsall    (scoring_technique="KvsAll")  : GCELoss, general_robust_loss, RDALoss, CORESLoss

For each group, N_SEEDS=3 random seeds are drawn ONCE and reused across every
loss in the group, so losses are compared on the same seeds. Each loss is run
once per (seed, dataset, model).

PCRA-path losses automatically switch to the Datasets_Perturbed_pcra dataset
root because they need pre-generated PCRA path files.

Examples
--------
    python robust-kge/run_seed_groups.py --group 1vsall    --trainer PL --accelerator gpu --devices 1
    python robust-kge/run_seed_groups.py --group negsample --trainer PL --accelerator gpu --devices 1
    python robust-kge/run_seed_groups.py --group kvsall     --trainer PL --accelerator gpu --devices 1
    python robust-kge/run_seed_groups.py --group all        --trainer PL --accelerator gpu --devices 1
"""

import gc
import sys
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

robust_kge_dir = Path(__file__).resolve().parent
project_root = robust_kge_dir.parent.resolve()
sys.path.insert(0, str(robust_kge_dir))
sys.path.insert(0, str(project_root))

from dicee.executer import run_dicee_eval

DBS = ["FB15k-237"] #, "NELL-995-h100", "WN18RR"
# MODELS = ["Pykeen_RotatE", "Pykeen_MuRE" ,"Keci"]
 
MODELS = ["Pykeen_TransE"]
#'Pykeen_TransE', 'Pykeen_TransH', "DistMult", "ComplEx", "DeCaL"


BATCH_SIZE = "1024"
LEARNING_RATE = "0.1"

NUM_EPOCHS = "100"
EMB_DIM = "32"
LOSS_FN = "BCELoss"
SCORING_TECH = "KvsAll"
OPTIM = "Adam"
EVAL_MODE = "test"

N_SEEDS = 3

# Loss groups keyed by the scoring technique they require.
GROUPS = {
    "1vsall": {
        "scoring_technique": "1vsAll",
        "losses": ["NCELoss", "NCEandAGCELoss", "NCEandAULoss"],
    },
    "negsample": {
        "scoring_technique": "NegSample",
        "losses": [
            "LocalTripleLoss",
            "LocalTripleWithPriorPathLoss",
            "LocalTripleWithPriorAndAdaptivePathLoss",
            "DSKRLLoss",
            "PTrustELoss",
        ],
    },
    "kvsall": {
        "scoring_technique": "KvsAll",
        "losses": ["GCELoss", "general_robust_loss", "RDALoss", "CORESLoss"],
    },
}

# Losses needing pre-generated PCRA path files (Datasets_Perturbed_pcra).
PCRA_LOSSES = {
    "LocalTripleWithPriorPathLoss",
    "LocalTripleWithPriorAndAdaptivePathLoss",
    "DSKRLLoss",
}


def run_group(group_name, group_spec, seeds, args, datasets_root, datasets_root_pcra,
              out_dir, allowed_subdirs):
    scoring_technique = group_spec["scoring_technique"]
    losses = group_spec["losses"]

    print(f"\n{'#' * 70}")
    print(f"# GROUP '{group_name}' | scoring_technique={scoring_technique}")
    print(f"# losses: {losses}")
    print(f"# seeds : {seeds}")
    print(f"{'#' * 70}")

    rows = []
    for loss_fn in losses:
        effective_root = datasets_root_pcra if loss_fn in PCRA_LOSSES else datasets_root
        if loss_fn in PCRA_LOSSES:
            print(f"  [{loss_fn}] PCRA loss -> dataset root {effective_root}")

        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            for DB in DBS:
                db_path = effective_root / DB
                if not db_path.exists():
                    print(f"  Warning: {db_path} missing, skipping {DB}")
                    continue
                subdirs = sorted(
                    d.name for d in db_path.iterdir()
                    if d.is_dir() and (allowed_subdirs is None or d.name in allowed_subdirs)
                )

                for subdir in subdirs:
                    dataset_name = f"{DB}/{subdir}"
                    for MODEL in MODELS:
                        store_path = (
                            out_dir / "saved_models" / f"seed_{seed}"
                            / dataset_name / MODEL / loss_fn
                        )
                        print(
                            f"\n[group={group_name}][seed={seed}] "
                            f"{dataset_name} | {MODEL} | {loss_fn} | scoring={scoring_technique}"
                        )
                        test_mrr = None
                        try:
                            result = run_dicee_eval(
                                dataset_folder=str((effective_root / DB / subdir).resolve()),
                                model=MODEL,
                                num_epochs=args.num_epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.lr,
                                embedding_dim=args.emb_dim,
                                loss_function=loss_fn,
                                path_to_store_single_run=str(store_path.resolve()),
                                scoring_technique=scoring_technique,
                                optim=args.optim,
                                eval_model=args.eval_model,
                                trainer=args.trainer,
                                accelerator=args.accelerator,
                                devices=args.devices,
                                precision=args.precision,
                                random_seed=seed,
                                neg_ratio=args.neg_ratio,
                            )
                            test_mrr = result.get("Test", {}).get("MRR", None)
                            print(f"  Completed: Test MRR = {test_mrr}")
                            del result
                        except Exception as e:
                            print(f"  [ERROR] {dataset_name} | {MODEL} | {loss_fn}: {e}")
                        finally:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        rows.append({
                            "Group": group_name,
                            "Scoring": scoring_technique,
                            "Seed": seed,
                            "Dataset": dataset_name,
                            "Model": MODEL,
                            "Loss": loss_fn,
                            "Test_MRR": test_mrr,
                        })

    return rows


def save_results(rows, group_name, out_dir):
    if not rows:
        print(f"[{group_name}] No results to save.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(rows)
    df["Test_MRR"] = pd.to_numeric(df["Test_MRR"], errors="coerce")

    per_run_path = out_dir / f"per_run_{group_name}_{N_SEEDS}seeds_{ts}.csv"
    df.to_csv(per_run_path, index=False)
    print(f"[{group_name}] Per-run results -> {per_run_path}")

    stats = (
        df.groupby(["Dataset", "Model", "Loss"])["Test_MRR"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "MRR_mean", "std": "MRR_std", "count": "Runs"})
    )
    summary_path = out_dir / f"summary_{group_name}_{N_SEEDS}seeds_{ts}.csv"
    stats.to_csv(summary_path, index=False)
    print(f"[{group_name}] Summary mean/std -> {summary_path}")

    pivot = stats.pivot_table(
        index=["Model", "Loss"], columns="Dataset", values="MRR_mean", aggfunc="first"
    )
    print("\n" + "=" * 80)
    print(f"Test MRR (mean across {N_SEEDS} seeds) - group '{group_name}'")
    print("=" * 80)
    print(pivot.to_string())
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-seed experiments grouped by scoring technique.")
    parser.add_argument("--group", type=str, default="all",
                        choices=list(GROUPS.keys()) + ["all"],
                        help="Which loss group to run (default: all, sequentially).")
    parser.add_argument("--num_epochs", type=str, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=str, default=BATCH_SIZE)
    parser.add_argument("--lr", type=str, default=LEARNING_RATE)
    parser.add_argument("--emb_dim", type=str, default=EMB_DIM)
    parser.add_argument("--optim", type=str, default=OPTIM)
    parser.add_argument("--eval_model", type=str, default=EVAL_MODE)
    parser.add_argument("--trainer", type=str, default="PL")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--neg_ratio", type=str, default="2")
    parser.add_argument("--datasets_root", type=str, default="Datasets_Perturbed")
    parser.add_argument("--datasets_root_pcra", type=str, default="Datasets_Perturbed_pcra")
    parser.add_argument("--noise_levels", type=str, nargs="+",
                        default=["0.0", "0.08", "0.16", "0.32"],
                        help="Noise-level subdirs to include. Pass 'all' for every subdir.")
    parser.add_argument("--seed_min", type=int, default=10001)
    parser.add_argument("--seed_max", type=int, default=999999)
    parser.add_argument("--results_dir", type=str, default=str(robust_kge_dir / "seed_group_runs"))
    args = parser.parse_args()

    devices = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)
    args.devices = devices

    datasets_root = (project_root / args.datasets_root).resolve()
    datasets_root_pcra = (project_root / args.datasets_root_pcra).resolve()

    allowed_subdirs = None if args.noise_levels == ["all"] else set(args.noise_levels)

    if args.seed_max - args.seed_min + 1 < N_SEEDS:
        raise ValueError(f"Seed range must contain at least {N_SEEDS} values.")
    # Draw the seeds ONCE so every loss/group in this invocation shares them.
    seeds = random.SystemRandom().sample(range(args.seed_min, args.seed_max + 1), N_SEEDS)
    print(f"Running {N_SEEDS} seeds: {seeds}")
    print(f"Datasets root: {datasets_root}")
    print(f"PCRA datasets root: {datasets_root_pcra}")
    print(f"DBs: {DBS} | Models: {MODELS} | noise levels: {allowed_subdirs or 'all'}")

    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    groups_to_run = list(GROUPS.keys()) if args.group == "all" else [args.group]

    for group_name in groups_to_run:
        out_dir = Path(args.results_dir) / group_name / batch_ts
        rows = run_group(
            group_name=group_name,
            group_spec=GROUPS[group_name],
            seeds=seeds,
            args=args,
            datasets_root=datasets_root,
            datasets_root_pcra=datasets_root_pcra,
            out_dir=out_dir,
            allowed_subdirs=allowed_subdirs,
        )
        save_results(rows, group_name, out_dir / "results")
