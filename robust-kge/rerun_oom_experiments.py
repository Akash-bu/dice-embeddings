"""Restart OOM-affected seed configurations in isolated subprocesses.

Each manifest row is intentionally executed in a fresh Python process.  The
underlying seed drivers already reuse completed ``eval_report.json`` files, so
rerunning a row only fills missing seeds while process isolation releases all
CUDA state before the next configuration starts.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


ROBUST_KGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROBUST_KGE_DIR.parent
DEFAULT_MANIFEST = ROBUST_KGE_DIR / "oom_reruns.tsv"
DEFAULT_METRICS_DIR = ROBUST_KGE_DIR / "oom_rerun_results"

DRIVERS = {
    "Adv": ROBUST_KGE_DIR / "sem_adv_scripts" / "seed_runs_adv.py",
    "Rand": ROBUST_KGE_DIR / "Random_perturbed_seed" / "random_perturb_seed.py",
    "Sem": ROBUST_KGE_DIR / "sem_adv_scripts" / "seed_runs_sem.py",
}

DATASET_ROOTS = {
    "Adv": PROJECT_ROOT / "Datasets_Perturbed_adversarial",
    "Rand": PROJECT_ROOT / "Datasets_Perturbed",
    "Sem": PROJECT_ROOT / "Datasets_Perturbed_semantic",
}

RESULT_ROOTS = {
    "Adv": ROBUST_KGE_DIR / "saved_models_adv",
    "Rand": ROBUST_KGE_DIR / "saved_models_bo_rand",
    "Sem": ROBUST_KGE_DIR / "saved_models_sem",
}

MODELS = {"Pykeen_TransE", "Pykeen_RotatE", "Pykeen_MuRE", "Keci"}
SEEDS = ("42", "67", "83")

SCORING_TECHNIQUES = {"KvsAll", "1vsAll", "NegSample"}
METRIC_KEYS = ("MRR", "H@1", "H@3", "H@10")
METRIC_ID_FIELDS = ("Label", "Family", "Dataset", "Noise", "Model", "Loss", "Scoring", "Seed")


def load_manifest(path: Path) -> list[tuple[str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str]] = []
    seen = set()
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not row or all(not item.strip() for item in row):
                continue
            if len(row) != 6:
                raise ValueError(
                    f"{path}:{line_number}: expected 6 tab-separated fields "
                    "(label, dataset, noise, model, loss, scoring)"
                )
            item = tuple(field.strip() for field in row)
            if item in seen:
                raise ValueError(f"{path}:{line_number}: duplicate row: {item}")
            seen.add(item)
            rows.append(item)
    return rows


def family_from_label(label: str) -> str:
    family = label.split("_", 1)[0]
    if family not in DRIVERS:
        raise ValueError(f"Unsupported run family in label {label!r}")
    return family


def command_for(row: tuple[str, str, str, str, str, str], args: argparse.Namespace) -> list[str]:
    label, dataset, noise, model, loss, scoring = row
    family = family_from_label(label)
    command = [
        sys.executable,
        str(DRIVERS[family]),
        "--datasets",
        dataset,
        "--noise_ratio",
        noise,
        "--model",
        model,
        "--loss_fn",
        loss,
        "--scoring_technique",
        scoring,
        "--devices",
        args.devices,
    ]
    if args.batch_size is not None:
        command.extend(("--batch_size", args.batch_size))
    return command


def validate(rows: list[tuple[str, str, str, str, str, str]]) -> None:
    for label, dataset, noise, model, _loss, scoring in rows:
        family = family_from_label(label)
        if model not in MODELS:
            raise ValueError(f"Unsupported model in manifest: {model}")
        if scoring not in SCORING_TECHNIQUES:
            raise ValueError(f"Unsupported scoring technique in manifest: {scoring}")
        dataset_dir = DATASET_ROOTS[family] / dataset / noise
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset directory for {label}: {dataset_dir}")


def pid_is_running(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        # Field 3 is the process state. Zombies no longer hold GPU resources.
        return stat_path.read_text().split()[2] != "Z"
    except (FileNotFoundError, IndexError, PermissionError):
        return False


def wait_for_processes(pids: list[int], poll_seconds: int) -> None:
    remaining = [pid for pid in pids if pid_is_running(pid)]
    while remaining:
        logging.info("Waiting for active GPU process(es) to finish: %s", remaining)
        time.sleep(poll_seconds)
        remaining = [pid for pid in remaining if pid_is_running(pid)]


def collect_metrics(
    row: tuple[str, str, str, str, str, str],
) -> tuple[list[dict], list[Path]]:
    label, dataset, noise, model, loss, scoring = row
    family = family_from_label(label)
    metrics = []
    missing = []
    for seed in SEEDS:
        report = (
            RESULT_ROOTS[family]
            / dataset
            / noise
            / model
            / f"{loss}_{scoring}_neg2"
            / f"seed{seed}"
            / "eval_report.json"
        )
        try:
            with report.open() as handle:
                result = json.load(handle)
            test_metrics = result.get("Test", {})
            if not all(key in test_metrics for key in METRIC_KEYS):
                missing.append(report)
                continue
            metrics.append(
                {
                    "Label": label,
                    "Family": family,
                    "Dataset": dataset,
                    "Noise": noise,
                    "Model": model,
                    "Loss": loss,
                    "Scoring": scoring,
                    "Seed": seed,
                    **{key: test_metrics[key] for key in METRIC_KEYS},
                }
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            missing.append(report)
    return metrics, missing


def append_metrics(metrics: list[dict], metrics_path: Path) -> None:
    with metrics_path.open("a", newline="") as handle:
        fieldnames = (*METRIC_ID_FIELDS, *METRIC_KEYS)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help=f"Directory for the combined rerun metrics CSV (default: {DEFAULT_METRICS_DIR}).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only run manifest rows for this dataset (for example: FB15k-237).",
    )
    parser.add_argument("--devices", default="1", help="Lightning device count/index argument (default: 1)")
    parser.add_argument(
        "--batch_size",
        default=None,
        help="Optional override. Omit to preserve the original batch size (1024).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-at", type=int, default=1, help="One-based manifest row to start at")
    # parser.add_argument(
    #     "--wait-for-pid",
    #     type=int,
    #     nargs="*",
    #     default=[],
    #     help="Wait until these existing process IDs exit before starting the queue.",
    # )
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    rows = load_manifest(args.manifest.resolve())
    if args.dataset is not None:
        available_datasets = sorted({row[1] for row in rows})
        rows = [row for row in rows if row[1] == args.dataset]
        if not rows:
            parser.error(
                f"dataset {args.dataset!r} is not present in the manifest; "
                f"available datasets: {', '.join(available_datasets)}"
            )
    validate(rows)
    if args.start_at < 1 or args.start_at > len(rows) + 1:
        parser.error(f"--start-at must be between 1 and {len(rows) + 1}")

    selected = rows[args.start_at - 1 :]
    if args.dry_run:
        for index, row in enumerate(selected, args.start_at):
            print(f"[{index}/{len(rows)}]", subprocess.list2cmdline(command_for(row, args)))
        print(f"Validated {len(rows)} unique manifest rows; selected {len(selected)}.")
        return 0

    log_dir = ROBUST_KGE_DIR / "oom_rerun_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.metrics_dir.resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = log_dir / f"queue_{timestamp}.log"
    dataset_tag = args.dataset or "all"
    metrics_path = metrics_dir / f"rerun_metrics_per_seed_{dataset_tag}_{timestamp}.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=(logging.FileHandler(summary_path), logging.StreamHandler()),
    )

    lock_path = log_dir / "queue.lock"
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logging.error("Another OOM rerun queue already holds %s", lock_path)
        return 2

    env = os.environ.copy()
    # Do not enable ``expandable_segments`` here. The H100-40C/vGPU driver used
    # by these experiments rejects the required CUDA operation when the model
    # is first moved to the GPU. Process isolation already releases CUDA state
    # between manifest rows.
    failed = []
    logging.info("Starting %d OOM reruns from manifest row %d", len(selected), args.start_at)
    logging.info("Queue log: %s", summary_path)
    logging.info("Rerun metrics: %s", metrics_path)
    append_metrics([], metrics_path)
    # wait_for_processes(args.wait_for_pid, args.poll_seconds)

    for index, row in enumerate(selected, args.start_at):
        command = command_for(row, args)
        label, dataset, noise, model, loss, scoring = row
        logging.info(
            "[%d/%d] START %s | %s | %s | %s | %s | %s",
            index,
            len(rows),
            label,
            dataset,
            noise,
            model,
            loss,
            scoring,
        )
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        metrics, missing = collect_metrics(row)
        append_metrics(metrics, metrics_path)
        if completed.returncode or missing:
            failed.append((index, row, completed.returncode, len(missing)))
            logging.error(
                "[%d/%d] FAILED exit=%d missing_outputs=%d",
                index,
                len(rows),
                completed.returncode,
                len(missing),
            )
            for path in missing:
                logging.error("Missing or incomplete: %s", path)
        else:
            logging.info("[%d/%d] DONE; verified %d outputs", index, len(rows), len(SEEDS))

    if failed:
        logging.error("Queue completed with %d failed row(s):", len(failed))
        for index, row, returncode, missing_count in failed:
            logging.error(
                "row=%d exit=%d missing_outputs=%d config=%s",
                index,
                returncode,
                missing_count,
                row,
            )
        return 1

    logging.info("Queue completed successfully: %d/%d rows", len(selected), len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
