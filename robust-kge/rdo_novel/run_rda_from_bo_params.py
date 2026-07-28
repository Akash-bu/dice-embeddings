"""
Run RDARoBossLoss / RDAWaveLoss across all perturbation levels, seeding the
structural-bound hyperparameters (a, lambda) from a Bayesian-optimization report
obtained for the base RoBoSS / WaveLoss runs.

Source of params: bo_trial_results/wo_reciprocals_umls_kinship.txt (or any BO
report in the same format). For each (Dataset, Model, source-loss) we take the
clean-split (DB/0.0) entry and reuse its tuned a/lambda + general training
hyperparameters; the RDA gate parameters (beta_start, beta_end, etc.) fall back
to the defaults defined in dicee/models/base_model.py.
"""

from pathlib import Path
import sys
import ast
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dicee.executer import run_dicee_eval

script_dir = Path(__file__).resolve().parent              # robust-kge/rdo_novel
robust_kge_dir = script_dir.parent.resolve()              # robust-kge
project_root = robust_kge_dir.parent.resolve()            # dice-embeddings
sys.path.insert(0, str(robust_kge_dir))
sys.path.insert(0, str(project_root))


NUM_EPOCHS = 100
SCORING_TECH = "KvsAll"
OPTIM = "Adam"
EVAL_MODEL = "test"

# Perturbation levels to evaluate per dataset (matches subdir names under Datasets_Perturbed/<DB>/).
PERTURBATION_LEVELS = ("0.0", "0.08", "0.16", "0.32")

# Source loss -> RDA-gated target loss + BO-name -> RDA-arg-name translation.
LOSS_MAP = {
    "RoBoSS": {
        "target_loss": "RDARoBossLoss",
        "param_map": {
            "a_roboss": "rda_roboss_a",
            "lambda_roboss": "rda_roboss_lambda",
        },
    },
    "WaveLoss": {
        "target_loss": "RDAWaveLoss",
        "param_map": {
            "wave_a": "rda_wave_a",
            "lambda_param": "rda_wave_lambda",
        },
    },
}

# Keys in the BO Params dict that are general training hyperparameters
# (passed through to run_dicee_eval as named kwargs).
PASSTHROUGH_PARAMS = {"embedding_dim", "batch_size", "learning_rate"}


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

            rows.append({
                "Dataset": normalize_dataset_key(dataset_str),
                "Model": model_str.strip(),
                "Loss": loss_str.strip(),
                "Value": value,
                "Params": params,
            })

    best = {}
    for row in rows:
        key = (row["Dataset"], row["Model"], row["Loss"])
        prev = best.get(key)
        if prev is None:
            best[key] = row
        elif row["Value"] is not None and (prev["Value"] is None or row["Value"] > prev["Value"]):
            best[key] = row
    return list(best.values())


def build_rda_plan(entries):
    """For each (DB, Model, source-loss) pick the entry from the clean (DB/0.0) split."""
    grouped = {}
    for e in entries:
        if e["Loss"] not in LOSS_MAP:
            continue
        db = e["Dataset"].split("/", 1)[0]
        key = (db, e["Model"], e["Loss"])
        grouped.setdefault(key, []).append(e)

    plan = []
    for (db, model, source_loss), group in grouped.items():
        source = next((g for g in group if g["Dataset"] == f"{db}/0.0"), None)
        if source is None:
            source = next((g for g in group if g["Dataset"] == db), None)
        if source is None:
            continue
        plan.append({
            "DB": db,
            "Model": model,
            "SourceLoss": source_loss,
            "TargetLoss": LOSS_MAP[source_loss]["target_loss"],
            "ParamMap": LOSS_MAP[source_loss]["param_map"],
            "Params": source["Params"],
            "SourceDataset": source["Dataset"],
        })
    return plan


def resolve_dataset_targets(datasets_root, db):
    """Return the configured PERTURBATION_LEVELS that actually exist on disk for db."""
    db_path = (datasets_root / db).resolve()
    if not db_path.exists():
        return []
    targets = []
    for level in PERTURBATION_LEVELS:
        if (db_path / level).is_dir():
            targets.append(f"{db}/{level}")
        else:
            print(f"  [skip] {db}/{level} not found on disk")
    return targets


def split_params(params, loss_param_map):
    """Split BO params into (passthrough kwargs, loss-specific kwargs renamed for RDA).

    Anything outside PASSTHROUGH_PARAMS or the source-loss param_map is silently
    dropped (e.g. source-loss specific keys that don't have an RDA equivalent).
    """
    passthrough = {}
    loss_kwargs = {}
    for k, v in params.items():
        if k in PASSTHROUGH_PARAMS:
            passthrough[k] = v
        elif k in loss_param_map:
            loss_kwargs[loss_param_map[k]] = v
    return passthrough, loss_kwargs


def create_results_table(records):
    df = pd.DataFrame(records)
    if df.empty:
        return df, pd.DataFrame()
    pivot_df = df.pivot_table(
        index=["Model", "Loss"],
        columns="Dataset",
        values="Test_MRR",
        aggfunc="first",
    )
    return df, pivot_df


def create_visualization(pivot_df, output_dir):
    if pivot_df.empty:
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=(max(14, len(pivot_df.columns) * 1.8), max(10, len(pivot_df.index) * 0.5))
    )
    sns.heatmap(
        pivot_df, annot=True, fmt=".4f", cmap="RdYlGn",
        cbar_kws={"label": "Test MRR"}, linewidths=0.5, linecolor="gray",
        ax=ax, vmin=0, vmax=1.0,
    )
    ax.set_title("RDA-Gated Losses with BO-seeded params (Test MRR)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model / Loss", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"mrr_rda_from_bo_{ts}.png"
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to: {image_path}")


def save_pretty_table(pivot_df, out_path, title):
    line = "=" * 80
    with open(out_path, "w") as f:
        f.write(f"{line}\n{title}\n{line}\n")
        f.write(pivot_df.to_string())
        f.write(f"\n{line}\n")
    print(f"Formatted table saved to: {out_path}")


def save_results(df, pivot_df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_path = output_dir / f"detailed_rda_from_bo_{ts}.csv"
    df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")
    pivot_path = output_dir / f"comparison_rda_from_bo_{ts}.csv"
    pivot_df.to_csv(pivot_path)
    print(f"Comparison table saved to: {pivot_path}")
    save_pretty_table(
        pivot_df,
        output_dir / f"comparison_rda_from_bo_{ts}.txt",
        "MRR Comparison Table - RDA-Gated Losses (BO-seeded params)",
    )
    create_visualization(pivot_df, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RDARoBossLoss/RDAWaveLoss using BO-tuned a/lambda from a base RoBoSS/WaveLoss BO report."
    )
    parser.add_argument("--report_file", type=str,
                        default=str(project_root / "bo_trial_results" / "wo_reciprocals_umls_kinship.txt"))
    parser.add_argument("--datasets_root", type=str, default=str(project_root / "Datasets_Perturbed"))
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--scoring_technique", type=str, default=SCORING_TECH)
    parser.add_argument("--optim", type=str, default=OPTIM)
    parser.add_argument("--eval_model", type=str, default=EVAL_MODEL)
    parser.add_argument("--trainer", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--results_dir", type=str,
                        default=str(script_dir / "results_rdo"))
    parser.add_argument("--saved_models_dir", type=str,
                        default=str(script_dir / "saved_models_rda_from_bo"))
    parser.add_argument(
        "--only_loss", type=str, default=None,
        choices=[None, "RoBoSS", "WaveLoss"],
        help="If set, restrict to a single source loss type (RoBoSS or WaveLoss).",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Datasets (DB names) to run, e.g. --datasets KINSHIP UMLS. "
             "Default: every DB present in the BO report.",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Model names to run, e.g. --models Pykeen_RotatE Pykeen_MuRE. "
             "Default: every model present in the BO report.",
    )
    args = parser.parse_args()

    devices = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)

    report_file = abs_path(args.report_file, project_root)
    datasets_root = abs_path(args.datasets_root, project_root)
    results_dir = abs_path(args.results_dir, project_root)
    saved_models_dir = abs_path(args.saved_models_dir, project_root)

    entries = parse_report_file(report_file)
    plan = build_rda_plan(entries)
    if args.only_loss is not None:
        plan = [p for p in plan if p["SourceLoss"] == args.only_loss]
    if args.datasets is not None:
        requested = set(args.datasets)
        unknown = requested - {p["DB"] for p in plan}
        if unknown:
            print(f"Warning: requested datasets not present in BO report: {sorted(unknown)}")
        plan = [p for p in plan if p["DB"] in requested]
    if args.models is not None:
        requested = set(args.models)
        unknown = requested - {p["Model"] for p in plan}
        if unknown:
            print(f"Warning: requested models not present in BO report: {sorted(unknown)} "
                  f"(they will be skipped — no a/lambda to seed from)")
        plan = [p for p in plan if p["Model"] in requested]
    if not plan:
        raise ValueError(
            "No RoBoSS/WaveLoss entries found for the requested filter — "
            "check --report_file, --only_loss, and --datasets."
        )

    records = []
    for entry in plan:
        db = entry["DB"]
        model = entry["Model"]
        target_loss = entry["TargetLoss"]
        passthrough, loss_kwargs = split_params(entry["Params"], entry["ParamMap"])

        targets = resolve_dataset_targets(datasets_root, db)
        for dataset_name in targets:
            dataset_folder = (datasets_root / dataset_name).resolve()
            store_path = (saved_models_dir / dataset_name / model / target_loss).resolve()

            print(
                f"Running: dataset={dataset_name}, model={model}, loss={target_loss}, "
                f"src={entry['SourceLoss']}@{entry['SourceDataset']}, "
                f"passthrough={passthrough}, loss_kwargs={loss_kwargs}"
            )
            result = run_dicee_eval(
                dataset_folder=str(dataset_folder),
                model=model,
                num_epochs=args.num_epochs,
                loss_function=target_loss,
                path_to_store_single_run=str(store_path),
                scoring_technique=args.scoring_technique,
                optim=args.optim,
                eval_model=args.eval_model,
                trainer=args.trainer,
                accelerator=args.accelerator,
                devices=devices,
                precision=args.precision,
                **passthrough,
                **loss_kwargs,
            )
            test_mrr = result.get("Test", {}).get("MRR", None)
            records.append({
                "Dataset": dataset_name,
                "Model": model,
                "Loss": target_loss,
                "SourceLoss": entry["SourceLoss"],
                "Test_MRR": test_mrr,
            })

    df, pivot_df = create_results_table(records)
    save_results(df, pivot_df, results_dir)
