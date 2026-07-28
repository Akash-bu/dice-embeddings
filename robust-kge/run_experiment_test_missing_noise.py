"""Run test experiments for specific (missing) noise levels using BO report params.

The sibling script ``run_experiment_test_from_bayes_report.py`` caps the noise
levels it runs at 0.08 (see its ``resolve_dataset_targets``). This script reuses
the same report-parsing / planning / saving helpers but lets you run an explicit
list of noise levels instead -- e.g. the missing 0.16 and 0.32 perturbations.

Example:
    python run_experiment_test_missing_noise.py \
        --model_losses "Pykeen_RotatE:BCELoss,Pykeen_MuRE:AELoss" \
        --noise_levels "0.16,0.32"
"""

from pathlib import Path
import sys

from dicee.executer import run_dicee_eval

robust_kge_dir = Path(__file__).resolve().parent
project_root = robust_kge_dir.parent.resolve()
sys.path.insert(0, str(robust_kge_dir))
sys.path.insert(0, str(project_root))

from run_experiment_test_from_bayes_report import (  # noqa: E402
    NUM_EPOCHS,
    SCORING_TECH,
    OPTIM,
    EVAL_MODEL,
    abs_path,
    parse_report_file,
    build_zero_point_plan,
    create_results_table,
    save_results,
)


def resolve_noise_targets(datasets_root: Path, db: str, noise_levels):
    """Resolve dataset folders for an explicit list of noise levels.

    Only returns targets whose perturbed dataset folder actually exists on disk.
    """
    db_path = (datasets_root / db).resolve()
    targets = []
    missing = []
    for level in noise_levels:
        subdir = (db_path / level)
        if subdir.is_dir():
            targets.append(f"{db}/{level}")
        else:
            missing.append(f"{db}/{level}")
    if missing:
        print(f"  [warn] no perturbed dataset folder for: {', '.join(missing)} (skipped)")
    return targets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run test experiments for explicit noise levels from BO report params."
    )
    parser.add_argument("--report_file", type=str, default="bo_trial_results/wo_reciprocals_fb15k.txt")
    parser.add_argument(
        "--noise_levels",
        type=str,
        default="0.16,0.32",
        help="Comma-separated noise levels to run (e.g. '0.16,0.32').",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default=None,
        help="Comma-separated loss names to run (e.g. 'AELoss,BCELoss'). Default: all losses in report.",
    )
    parser.add_argument(
        "--model_losses",
        type=str,
        default="Pykeen_RotatE:BCELoss,Pykeen_MuRE:AELoss",
        help=(
            "Comma-separated Model:Loss pairs to run (e.g. 'Pykeen_RotatE:BCELoss,Pykeen_MuRE:AELoss'). "
            "Only these exact (model, loss) combinations are run. Default: all pairs in report."
        ),
    )
    parser.add_argument("--datasets_root", type=str, default=str(project_root / "Datasets_Perturbed"))
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--scoring_technique", type=str, default=SCORING_TECH)
    parser.add_argument("--optim", type=str, default=OPTIM)
    parser.add_argument("--eval_model", type=str, default=EVAL_MODEL)
    parser.add_argument("--trainer", type=str, default="PL")
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=str(project_root / "robust-kge" / "results"))
    parser.add_argument("--saved_models_dir", type=str, default=str(robust_kge_dir / "saved_models_from_report"))
    args = parser.parse_args()

    devices = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)

    noise_levels = [n.strip() for n in args.noise_levels.split(",") if n.strip()]
    if not noise_levels:
        raise ValueError("No noise levels provided via --noise_levels.")

    report_file = abs_path(args.report_file, project_root)
    datasets_root = abs_path(args.datasets_root, project_root)
    results_dir = abs_path(args.results_dir, project_root)
    saved_models_dir = abs_path(args.saved_models_dir, project_root)

    entries = parse_report_file(report_file)

    if args.losses:
        wanted = {l.strip() for l in args.losses.split(",") if l.strip()}
        entries = [e for e in entries if e["Loss"] in wanted]
        if not entries:
            raise ValueError(f"No entries matched --losses={args.losses}. Available losses are in the report file.")

    if args.model_losses:
        wanted_pairs = set()
        for token in args.model_losses.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"Invalid --model_losses entry '{token}'. Expected 'Model:Loss'.")
            model_name, loss_name = token.split(":", 1)
            wanted_pairs.add((model_name.strip(), loss_name.strip()))
        entries = [e for e in entries if (e["Model"], e["Loss"]) in wanted_pairs]
        if not entries:
            raise ValueError(
                f"No entries matched --model_losses={args.model_losses}. "
                f"Available (model, loss) pairs are in the report file."
            )

    plan = build_zero_point_plan(entries)
    if not plan:
        raise ValueError("No valid (DB, model, loss) plan could be built from report file.")

    records = []
    for entry in plan:
        db = entry["DB"]
        model = entry["Model"]
        loss_fn = entry["Loss"]
        params = entry["Params"]
        source_dataset = entry["SourceDataset"]

        targets = resolve_noise_targets(datasets_root, db, noise_levels)
        for dataset_name in targets:
            dataset_folder = (datasets_root / dataset_name).resolve()
            store_path = (saved_models_dir / dataset_name / model / loss_fn).resolve()

            print(
                f"Running: dataset={dataset_name}, model={model}, loss={loss_fn}, "
                f"source_params={source_dataset}, params={params}"
            )
            result = run_dicee_eval(
                dataset_folder=str(dataset_folder),
                model=model,
                num_epochs=args.num_epochs,
                loss_function=loss_fn,
                path_to_store_single_run=str(store_path),
                scoring_technique=args.scoring_technique,
                optim=args.optim,
                eval_model=args.eval_model,
                trainer=args.trainer,
                accelerator=args.accelerator,
                devices=devices,
                precision=args.precision,
                **params,
            )
            test_metrics = result.get("Test", {})

            records.append(
                {
                    "Dataset": dataset_name,
                    "Model": model,
                    "Loss": loss_fn,
                    "Test_MRR": test_metrics.get("MRR", None),
                    "Test_H@1": test_metrics.get("H@1", None),
                    "Test_H@3": test_metrics.get("H@3", None),
                    "Test_H@10": test_metrics.get("H@10", None),
                }
            )

    df, pivot_df = create_results_table(records)
    save_results(df, pivot_df, results_dir)
