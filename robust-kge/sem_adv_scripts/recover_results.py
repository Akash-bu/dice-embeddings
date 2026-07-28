"""Rebuild the aggregate result tables from per-run checkpoints on disk.

run_sem_exp.py runs every (model, loss, scoring, neg_ratio) experiment and saves
each one to disk under saved_models_sem/<DB>/<subdir>/<MODEL>/<loss>_<scoring>_neg<n>/
(via path_to_store_single_run). Only the *final* aggregation step builds the big
combined CSV/heatmap in memory -- and that step crashed with "File name too long".

Because every individual run wrote its Test metrics to eval_report.json, no compute
is lost: this script walks those checkpoints, reconstructs the same all_results dict
run_sem_exp would have held in memory, and re-runs the (now length-safe) save logic.

Usage:
    # Recover the failed UMLS sweep (tmux main2):
    python recover_results.py --datasets UMLS

    # Recover the FB15k-237 sweep once it finishes / after it crashes (tmux main):
    python recover_results.py --datasets FB15k-237

Defaults mirror the exact command both runs used, so only --datasets need change.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Import the live run_sem_exp module so we reuse its (now patched) table/figure
# builders verbatim -- guarantees the output format matches what the run intended.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sem_exp as R  # noqa: E402

logger = logging.getLogger(__name__)


def read_test_metrics(leaf_dir):
    """Return the Test metrics dict for one run checkpoint, or None if absent."""
    report = leaf_dir / "eval_report.json"
    if not report.exists():
        return None
    try:
        data = json.loads(report.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read/parse %s, treating as missing", report)
        return None
    test = data.get("Test")
    return test or None


def main():
    parser = argparse.ArgumentParser(description="Rebuild result tables from on-disk run checkpoints")
    parser.add_argument("--loss_fn", nargs="+",
                        default=["LocalTripleLoss", "LocalTripleWithPriorPathLoss",
                                 "LocalTripleWithPriorAndAdaptivePathLoss", "DSKRLLoss",
                                 "PTrustELoss", "BCELoss"])
    parser.add_argument("--scoring_technique", nargs="+",
                        default=["NegSample", "NegSample", "NegSample", "NegSample",
                                 "NegSample", "KvsAll"])
    parser.add_argument("--neg_ratio", nargs="+",
                        default=["10", "25", "50", "100", "200", "500"])
    parser.add_argument("--datasets", nargs="+", default=["FB15k-237"])
    parser.add_argument("--model", nargs="+", default=list(R.MODELS))
    parser.add_argument("--results_dir", default=str(R.robust_kge_dir / "saved_models_sem"),
                        help="Root holding the per-run checkpoints.")
    parser.add_argument("--output_dir", default=str(R.project_root / "robust-kge" / "sem_results"),
                        help="Where to write the recovered CSVs/heatmap.")
    args = parser.parse_args()

    R.setup_logging(Path(__file__).stem)

    # Pair losses with scoring techniques exactly as run_sem_exp does.
    losses = args.loss_fn
    scoring = args.scoring_technique
    if len(scoring) == 1:
        scoring = scoring * len(losses)
    elif len(scoring) != len(losses):
        parser.error(f"--scoring_technique has {len(scoring)} values but --loss_fn has {len(losses)}.")
    loss_scoring_pairs = list(zip(losses, scoring))
    neg_ratios = args.neg_ratio
    models = args.model
    results_dir = Path(args.results_dir)

    all_results = {}
    found, missing = 0, 0
    for db in args.datasets:
        db_root = results_dir / db
        if not db_root.exists():
            logger.warning(f"{db_root} does not exist, skipping {db}")
            continue
        subdirs = sorted(d.name for d in db_root.iterdir() if d.is_dir())
        for subdir in subdirs:
            dataset_name = f"{db}/{subdir}"
            all_results.setdefault(dataset_name, {})
            for model in models:
                all_results[dataset_name].setdefault(model, {})
                for loss, score in loss_scoring_pairs:
                    # Non-NegSample scoring is invariant to neg_ratio -> only the first was run.
                    effective = neg_ratios if score == "NegSample" else neg_ratios[:1]
                    for neg in effective:
                        leaf = results_dir / db / subdir / model / f"{loss}_{score}_neg{neg}"
                        metrics = read_test_metrics(leaf)
                        if metrics is None:
                            missing += 1
                        else:
                            found += 1
                        (all_results[dataset_name][model]
                         .setdefault(loss, {})
                         .setdefault(score, {})[neg]) = metrics

    if not any(all_results.values()):
        logger.info("No checkpoints found -- nothing to recover.")
        return

    logger.info(f"Recovered {found} runs from disk ({missing} missing/not-yet-run).")
    df, pivot_df = R.create_results_table(all_results)
    R.save_results_table(df, pivot_df, args.output_dir, loss_scoring_pairs, models, neg_ratios, kind="Sem")


if __name__ == "__main__":
    main()
