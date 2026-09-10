"""Grid-search joint BERT + KGE BCE link prediction.

The model, data pipeline, negative sampling, optimizer groups, and filtered
link-prediction evaluation are inherited from ``bert_bce_link_prediction.py``.
Every trial is selected on validation MRR and uses early stopping; test metrics
are deliberately not evaluated during hyperparameter selection.
"""

import argparse
import csv
import gc
import itertools
import json
import os
import random
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from dicee.scripts.bert_bce_link_prediction import (
    BertTripleClassifier,
    GroupedNegativeBatchSampler,
    JointBCEModel,
    TripleBCEDataset,
    add_nbert_tokens,
    collate_text,
    create_kge_model,
    create_unique_output_dir,
    dataset_run_name,
    evaluate_link_prediction,
    negative_filter_triples,
    negative_sampling_bce_loss,
    post_kge_parameter_update,
    read_support,
    read_triples,
    resolve_path,
    training_entity_ids,
)


DEFAULT_INITIAL_LAMBDAS = [0.5, 0.7, 0.8, 0.9, 0.95]
DEFAULT_LAMBDA_LRS = [1e-4, 1e-3, 1e-2]
DEFAULT_KGE_LRS = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2]
DEFAULT_KGE_EMBEDDING_DIMS = [32, 64, 128, 256]
METRIC_NAMES = ("MRR", "H@1", "H@3", "H@10")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search joint BERT + KGE BCE link prediction using "
            "validation-MRR early stopping."
        )
    )

    # These defaults match bert_bce_link_prediction.py.
    parser.add_argument("--dataset_path", type=str, default="KGs/UMLS")
    parser.add_argument("--support_path", type=str, default=None)
    parser.add_argument(
        "--bert_model_path",
        type=str,
        default="checkpoints/umls/bert-pretrained",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="bert-base-cased",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=(
            "Target examples per optimizer step. Grouped batching uses the "
            "largest multiple of 1 + --negative_ratio not exceeding this."
        ),
    )
    parser.add_argument("--candidate_batch_size", type=int, default=256)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Maximum epochs per trial; early stopping may finish sooner.",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--negative_ratio", type=int, default=1)
    parser.add_argument(
        "--negative_loss_weighting",
        choices=["balanced", "sampled"],
        default="balanced",
    )
    parser.add_argument(
        "--negative_filter_scope",
        choices=["train", "train_valid", "all"],
        default="train",
    )
    parser.add_argument(
        "--calibration_lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kge_model",
        type=str,
        default="TransE",
        choices=["TransE", "RotatE", "Keci", "MuRE"],
    )

    parser.add_argument(
        "--initial_lambda_grid",
        type=float,
        nargs="+",
        default=DEFAULT_INITIAL_LAMBDAS,
    )
    parser.add_argument(
        "--lambda_lr_grid",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDA_LRS,
    )
    parser.add_argument(
        "--kge_lr_grid",
        type=float,
        nargs="+",
        default=DEFAULT_KGE_LRS,
    )
    parser.add_argument(
        "--kge_embedding_dim_grid",
        type=int,
        nargs="+",
        default=DEFAULT_KGE_EMBEDDING_DIMS,
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help=(
            "Stop after this many validation evaluations without an MRR "
            "improvement of at least --early_stopping_min_delta."
        ),
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--early_stopping_warmup_epochs",
        type=int,
        default=2,
        help="Do not count non-improvements before this epoch.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help=(
            "Evaluate validation MRR every N epochs and always at the "
            "last epoch."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Sweep directory. A new directory is created when omitted. Pass "
            "an existing sweep directory to resume compatible completed trials."
        ),
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=None,
        help=(
            "Maximum number of new trials to run in this invocation. This is "
            "useful for staged execution; resume with the same --output_dir."
        ),
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Retry trials previously recorded as failed because of GPU OOM.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_epochs < 1:
        raise ValueError("--num_epochs must be at least 1.")
    if args.negative_ratio < 1:
        raise ValueError("--negative_ratio must be at least 1.")
    if args.batch_size < 1 + args.negative_ratio:
        raise ValueError(
            "--batch_size must be at least 1 + --negative_ratio for grouped "
            "negative batches."
        )
    if args.calibration_lr <= 0.0:
        raise ValueError("--calibration_lr must be positive.")
    if args.early_stopping_patience < 1:
        raise ValueError("--early_stopping_patience must be at least 1.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early_stopping_min_delta cannot be negative.")
    if args.early_stopping_warmup_epochs < 0:
        raise ValueError("--early_stopping_warmup_epochs cannot be negative.")
    if args.eval_every < 1:
        raise ValueError("--eval_every must be at least 1.")
    if args.max_trials is not None and args.max_trials < 1:
        raise ValueError("--max_trials must be at least 1 when provided.")

    if not args.initial_lambda_grid:
        raise ValueError("--initial_lambda_grid cannot be empty.")
    if not all(0.0 < value < 1.0 for value in args.initial_lambda_grid):
        raise ValueError("Every initial lambda must be between 0 and 1.")
    if not args.lambda_lr_grid or not all(
        value > 0.0 for value in args.lambda_lr_grid
    ):
        raise ValueError("Every lambda learning rate must be positive.")
    if not args.kge_lr_grid or not all(
        value > 0.0 for value in args.kge_lr_grid
    ):
        raise ValueError("Every KGE learning rate must be positive.")
    if not args.kge_embedding_dim_grid or not all(
        value > 0 for value in args.kge_embedding_dim_grid
    ):
        raise ValueError("Every KGE embedding dimension must be positive.")
    if args.kge_model == "Keci" and any(
        value % 2 != 0 for value in args.kge_embedding_dim_grid
    ):
        raise ValueError("Keci requires even KGE embedding dimensions.")

    for argument_name in (
        "initial_lambda_grid",
        "lambda_lr_grid",
        "kge_lr_grid",
        "kge_embedding_dim_grid",
    ):
        values = getattr(args, argument_name)
        if len(values) != len(set(values)):
            raise ValueError(f"--{argument_name} contains duplicate values.")


def build_grid(args):
    grid = []
    combinations = itertools.product(
        args.initial_lambda_grid,
        args.lambda_lr_grid,
        args.kge_lr_grid,
        args.kge_embedding_dim_grid,
    )
    for trial_id, values in enumerate(combinations, start=1):
        initial_lambda, lambda_lr, kge_lr, embedding_dim = values
        grid.append(
            {
                "trial_id": trial_id,
                "hyperparameters": {
                    "initial_lambda": initial_lambda,
                    "lambda_lr": lambda_lr,
                    "kge_lr": kge_lr,
                    "kge_embedding_dim": embedding_dim,
                },
            }
        )
    return grid


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path, value):
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary_path, path)


def atomic_torch_save(path, value):
    temporary_path = f"{path}.tmp"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def is_complete_metrics(metrics):
    return (
        isinstance(metrics, dict)
        and all(
            isinstance(metrics.get(metric_name), (int, float))
            for metric_name in METRIC_NAMES
        )
    )


def read_trial_result(path, expected_trial):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    if result.get("trial_id") != expected_trial["trial_id"]:
        return None
    if result.get("hyperparameters") != expected_trial["hyperparameters"]:
        return None

    status = result.get("status")
    if status == "completed" and is_complete_metrics(
        result.get("best_validation_metrics")
    ):
        return result
    if status == "failed_oom":
        return result
    return None


def result_sort_key(result):
    metrics = result["best_validation_metrics"]
    return (
        float(metrics["MRR"]),
        float(metrics["H@1"]),
        float(metrics["H@3"]),
        float(metrics["H@10"]),
        -int(result["best_epoch"]),
        -int(result["trial_id"]),
    )


def write_aggregate_results(output_dir, trial_results):
    ordered_results = [
        trial_results[trial_id]
        for trial_id in sorted(trial_results)
    ]
    atomic_write_json(
        os.path.join(output_dir, "grid_results.json"),
        ordered_results,
    )

    fieldnames = [
        "trial_id",
        "status",
        "initial_lambda",
        "lambda_lr",
        "kge_lr",
        "kge_embedding_dim",
        "best_epoch",
        "epochs_ran",
        "early_stopped",
        "best_final_lambda",
        "best_bert_scale",
        "best_bert_bias",
        "best_kge_scale",
        "best_kge_bias",
        "best_effective_bert_weight",
        "best_effective_kge_weight",
        "best_joint_bias",
        "valid_MRR",
        "valid_H@1",
        "valid_H@3",
        "valid_H@10",
        "runtime_min",
        "error",
    ]
    csv_path = os.path.join(output_dir, "grid_results.csv")
    temporary_path = f"{csv_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in ordered_results:
            hyperparameters = result["hyperparameters"]
            metrics = result.get("best_validation_metrics") or {}
            calibration = result.get("best_calibration") or {}
            writer.writerow(
                {
                    "trial_id": result["trial_id"],
                    "status": result["status"],
                    "initial_lambda": hyperparameters["initial_lambda"],
                    "lambda_lr": hyperparameters["lambda_lr"],
                    "kge_lr": hyperparameters["kge_lr"],
                    "kge_embedding_dim": hyperparameters[
                        "kge_embedding_dim"
                    ],
                    "best_epoch": result.get("best_epoch"),
                    "epochs_ran": result.get("epochs_ran"),
                    "early_stopped": result.get("early_stopped"),
                    "best_final_lambda": result.get("best_final_lambda"),
                    "best_bert_scale": calibration.get("bert_scale"),
                    "best_bert_bias": calibration.get("bert_bias"),
                    "best_kge_scale": calibration.get("kge_scale"),
                    "best_kge_bias": calibration.get("kge_bias"),
                    "best_effective_bert_weight": calibration.get(
                        "effective_bert_weight"
                    ),
                    "best_effective_kge_weight": calibration.get(
                        "effective_kge_weight"
                    ),
                    "best_joint_bias": calibration.get("joint_bias"),
                    "valid_MRR": metrics.get("MRR"),
                    "valid_H@1": metrics.get("H@1"),
                    "valid_H@3": metrics.get("H@3"),
                    "valid_H@10": metrics.get("H@10"),
                    "runtime_min": result.get("runtime_min"),
                    "error": result.get("error"),
                }
            )
    os.replace(temporary_path, csv_path)

    completed_results = [
        result
        for result in ordered_results
        if result.get("status") == "completed"
    ]
    if completed_results:
        best_result = max(completed_results, key=result_sort_key)
        atomic_write_json(
            os.path.join(output_dir, "best_result.json"),
            best_result,
        )
        return best_result
    return None


def build_search_definition(
    args,
    dataset_path,
    support_path,
    bert_model_path,
    grid,
):
    return {
        "dataset_path": dataset_path,
        "support_path": support_path,
        "bert_model_path": bert_model_path,
        "tokenizer_path": args.tokenizer_path,
        "device": args.device,
        "batch_size": args.batch_size,
        "candidate_batch_size": args.candidate_batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "negative_ratio": args.negative_ratio,
        "negative_loss_weighting": args.negative_loss_weighting,
        "negative_filter_scope": args.negative_filter_scope,
        "negative_entity_pool": "train",
        "negative_batching": "positive_groups",
        "calibration_lr": args.calibration_lr,
        "fusion_calibration": "affine_v1",
        "max_seq_length": args.max_seq_length,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "kge_model": args.kge_model,
        "initial_lambda_grid": args.initial_lambda_grid,
        "lambda_lr_grid": args.lambda_lr_grid,
        "kge_lr_grid": args.kge_lr_grid,
        "kge_embedding_dim_grid": args.kge_embedding_dim_grid,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "early_stopping_warmup_epochs": (
            args.early_stopping_warmup_epochs
        ),
        "eval_every": args.eval_every,
        "selection_split": "valid",
        "selection_metric": "MRR",
        "total_trials": len(grid),
    }


def prepare_output_dir(repo_root, args, search_definition):
    if args.output_dir:
        output_dir = resolve_path(repo_root, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        base_output_dir = os.path.join(
            repo_root,
            "bert_bce_grid_runs",
        )
        dataset_name = dataset_run_name(args.dataset_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"{dataset_name}_joint_{args.kge_model}_grid_valid_{timestamp}"
        )
        output_dir = create_unique_output_dir(
            base_output_dir,
            run_name,
        )

    config_path = os.path.join(output_dir, "grid_config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            existing_definition = json.load(handle)
        if existing_definition != search_definition:
            raise ValueError(
                "The existing --output_dir was created for a different "
                "grid or training configuration."
            )
    else:
        unexpected_entries = [
            entry
            for entry in os.listdir(output_dir)
            if not entry.endswith(".tmp")
        ]
        if unexpected_entries:
            raise ValueError(
                "The requested --output_dir is non-empty but has no "
                "grid_config.json, so it cannot be resumed safely."
            )
        atomic_write_json(config_path, search_definition)

    trials_dir = os.path.join(output_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)
    return output_dir, trials_dir


def train_trial(
    args,
    trial,
    shared_data,
    tokenizer,
    device,
    trial_checkpoint_path,
):
    trial_start_time = time.perf_counter()
    hyperparameters = trial["hyperparameters"]
    seed_everything(args.seed)

    bert_model = BertTripleClassifier(
        shared_data["bert_model_path"],
        tokenizer,
    )
    kge_model = create_kge_model(
        model_name=args.kge_model,
        num_entities=len(shared_data["entity_to_idx"]),
        num_relations=len(shared_data["relation_to_idx"]),
        embedding_dim=hyperparameters["kge_embedding_dim"],
        random_seed=args.seed,
        learning_rate=hyperparameters["kge_lr"],
        negative_ratio=args.negative_ratio,
    )
    model = JointBCEModel(
        bert_model=bert_model,
        kge_model=kge_model,
        initial_lambda=hyperparameters["initial_lambda"],
    )
    model.to(device)

    # Model construction can consume RNG state differently across embedding
    # dimensions. Reset the sampling RNGs and use a dedicated shuffle generator
    # so every trial sees the same data order and corruptions.
    random.seed(args.seed)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_dataset = TripleBCEDataset(
        triples=shared_data["train_triples"],
        entities=shared_data["entities"],
        relations=shared_data["relations"],
        entity_to_idx=shared_data["entity_to_idx"],
        relation_to_idx=shared_data["relation_to_idx"],
        max_seq_length=args.max_seq_length,
        negative_ratio=args.negative_ratio,
        negative_entity_ids=shared_data["negative_entity_ids"],
        known_true_triples=shared_data["training_negative_filter"],
    )
    train_batch_sampler = GroupedNegativeBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        generator=train_generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_text(
            batch,
            tokenizer,
            args.max_seq_length,
        ),
    )
    print(
        f"trial={trial['trial_id']} "
        f"negative_group_size={train_batch_sampler.group_size} "
        f"positive_groups_per_batch={train_batch_sampler.groups_per_batch} "
        f"effective_batch_size="
        f"{train_batch_sampler.group_size * train_batch_sampler.groups_per_batch}"
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.bert_model.parameters(),
                "lr": args.lr,
            },
            {
                "params": model.kge_model.parameters(),
                "lr": hyperparameters["kge_lr"],
            },
            {
                "params": model.calibration_parameters(),
                "lr": args.calibration_lr,
            },
            {
                "params": [model.lambda_logit],
                "lr": hyperparameters["lambda_lr"],
            },
        ]
    )

    best_mrr = -float("inf")
    patience_reference_mrr = -float("inf")
    best_metrics = None
    best_epoch = None
    best_final_lambda = None
    best_calibration = None
    evaluations_without_improvement = 0
    history = []
    epochs_ran = 0

    for epoch in range(1, args.num_epochs + 1):
        epochs_ran = epoch
        model.train()
        losses = []

        for batch in tqdm(
            train_loader,
            desc=(
                f"Trial {trial['trial_id']} epoch "
                f"{epoch}/{args.num_epochs}"
            ),
        ):
            labels = batch.pop("labels").to(device)
            indexed_triples = batch.pop("indexed_triples").to(device)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            logits = model(
                indexed_triples=indexed_triples,
                **batch,
            )
            loss = negative_sampling_bce_loss(
                logits,
                labels,
                args.negative_ratio,
                weighting=args.negative_loss_weighting,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            post_kge_parameter_update(model.kge_model)
            losses.append(float(loss.detach().cpu()))

        mean_loss = sum(losses) / max(len(losses), 1)
        lambda_value = float(model.mixing_weight.detach().cpu())
        calibration_state = model.calibration_state()
        should_evaluate = (
            epoch % args.eval_every == 0
            or epoch == args.num_epochs
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "lambda": lambda_value,
            **calibration_state,
            "validation_metrics": None,
        }

        if should_evaluate:
            validation_metrics = evaluate_link_prediction(
                model=model,
                tokenizer=tokenizer,
                eval_triples=shared_data["valid_triples"],
                all_true_triples=shared_data["all_true_triples"],
                entities=shared_data["entities"],
                relations=shared_data["relations"],
                entity_to_idx=shared_data["entity_to_idx"],
                relation_to_idx=shared_data["relation_to_idx"],
                device=device,
                max_seq_length=args.max_seq_length,
                candidate_batch_size=args.candidate_batch_size,
            )
            epoch_record["validation_metrics"] = validation_metrics
            validation_mrr = float(validation_metrics["MRR"])

            if validation_mrr > best_mrr:
                best_mrr = validation_mrr
                best_metrics = validation_metrics
                best_epoch = epoch
                best_final_lambda = lambda_value
                best_calibration = dict(calibration_state)
                checkpoint_args = dict(vars(args))
                checkpoint_args.update(hyperparameters)
                checkpoint_args["eval_split"] = "valid"
                atomic_torch_save(
                    trial_checkpoint_path,
                    {
                        "model_state_dict": {
                            name: parameter.detach().cpu().clone()
                            for name, parameter in model.state_dict().items()
                        },
                        "metrics": best_metrics,
                        "validation_metrics": best_metrics,
                        "best_validation_metrics": best_metrics,
                        "hyperparameters": hyperparameters,
                        "best_epoch": best_epoch,
                        "epochs_ran": epoch,
                        "early_stopped": False,
                        "lambda_mode": "learned",
                        "lambda_requires_grad": True,
                        "configured_lambda": hyperparameters[
                            "initial_lambda"
                        ],
                        "final_lambda": best_final_lambda,
                        "best_final_lambda": best_final_lambda,
                        "selection_split": "valid",
                        "selection_metric": "MRR",
                        "validation_filter_scope": "train_valid_test",
                        "negative_batching": "positive_groups",
                        "fusion_calibration": "affine_v1",
                        "calibration": best_calibration,
                        "args": checkpoint_args,
                    },
                )

            significant_improvement = (
                validation_mrr
                > patience_reference_mrr
                + args.early_stopping_min_delta
            )
            if significant_improvement:
                patience_reference_mrr = validation_mrr
                evaluations_without_improvement = 0
            elif epoch >= args.early_stopping_warmup_epochs:
                evaluations_without_improvement += 1

            print(
                f"trial={trial['trial_id']} epoch={epoch} "
                f"loss={mean_loss:.6f} "
                f"valid_mrr={validation_mrr:.6f} "
                f"lambda={lambda_value:.6f} "
                f"kge_scale={calibration_state['kge_scale']:.6f} "
                f"kge_bias={calibration_state['kge_bias']:.6f} "
                f"patience={evaluations_without_improvement}/"
                f"{args.early_stopping_patience}"
            )
        else:
            print(
                f"trial={trial['trial_id']} epoch={epoch} "
                f"loss={mean_loss:.6f} "
                f"lambda={lambda_value:.6f} "
                f"kge_scale={calibration_state['kge_scale']:.6f} "
                f"kge_bias={calibration_state['kge_bias']:.6f}"
            )

        history.append(epoch_record)
        if (
            should_evaluate
            and epoch >= args.early_stopping_warmup_epochs
            and evaluations_without_improvement
            >= args.early_stopping_patience
        ):
            print(
                f"Early stopping trial {trial['trial_id']} at epoch "
                f"{epoch}; best epoch was {best_epoch}."
            )
            break

    if best_metrics is None:
        raise RuntimeError(
            "The trial completed without a validation evaluation."
        )

    return {
        "trial_id": trial["trial_id"],
        "status": "completed",
        "hyperparameters": hyperparameters,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "early_stopped": epochs_ran < args.num_epochs,
        "best_validation_metrics": best_metrics,
        "best_final_lambda": best_final_lambda,
        "best_calibration": best_calibration,
        "runtime_min": (
            time.perf_counter() - trial_start_time
        ) / 60.0,
        "history": history,
    }


def clean_up_trial(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def add_run_metadata(result, args, shared_data, output_dir):
    """Retain the single-run result fields for downstream compatibility."""
    trial_args = dict(vars(args))
    trial_args.update(result["hyperparameters"])
    trial_args["eval_split"] = "valid"
    result.update(
        {
            "metrics": result.get("best_validation_metrics"),
            "args": trial_args,
            "dataset_path": shared_data["dataset_path"],
            "bert_model_path": shared_data["bert_model_path"],
            "num_train_triples": len(shared_data["train_triples"]),
            "num_valid_triples": len(shared_data["valid_triples"]),
            "num_test_triples": len(shared_data["test_triples"]),
            "num_entities": len(shared_data["entities"]),
            "num_relations": len(shared_data["relations"]),
            "num_negative_entities": len(
                shared_data["negative_entity_ids"]
            ),
            "negative_filter_size": len(
                shared_data["training_negative_filter"]
            ),
            "negative_batching": "positive_groups",
            "fusion_calibration": "affine_v1",
            "final_lambda": result.get("best_final_lambda"),
            "output_dir": output_dir,
        }
    )
    return result


def main():
    args = parse_args()
    validate_args(args)
    run_start_time = time.perf_counter()

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    dataset_path = resolve_path(repo_root, args.dataset_path)
    support_path = (
        resolve_path(repo_root, args.support_path)
        if args.support_path
        else None
    )
    bert_model_path = resolve_path(repo_root, args.bert_model_path)
    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    grid = build_grid(args)
    search_definition = build_search_definition(
        args,
        dataset_path,
        support_path,
        bert_model_path,
        grid,
    )
    output_dir, trials_dir = prepare_output_dir(
        repo_root,
        args,
        search_definition,
    )
    print(f"grid_output_dir={output_dir}")
    print(f"total_grid_trials={len(grid)}")
    print("selection_split=valid selection_metric=MRR")

    train_triples = read_triples(
        os.path.join(dataset_path, "train.txt")
    )
    valid_triples = read_triples(
        os.path.join(dataset_path, "valid.txt")
    )
    test_triples = read_triples(
        os.path.join(dataset_path, "test.txt")
    )
    entities, relations = read_support(dataset_path, support_path)
    negative_entity_ids = training_entity_ids(train_triples, entities)
    training_negative_filter = negative_filter_triples(
        args.negative_filter_scope,
        train_triples,
        valid_triples,
        test_triples,
    )
    print(
        f"negative_entity_pool=train ({len(negative_entity_ids)}/"
        f"{len(entities)} entities) "
        f"negative_filter_scope={args.negative_filter_scope} "
        f"negative_loss_weighting={args.negative_loss_weighting}"
    )
    entity_to_idx = {
        entity_id: index
        for index, entity_id in enumerate(entities)
    }
    relation_to_idx = {
        relation_id: index
        for index, relation_id in enumerate(relations)
    }
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_path,
        do_basic_tokenize=False,
    )
    add_nbert_tokens(tokenizer, entities, relations)
    tokenizer.save_pretrained(output_dir)

    shared_data = {
        "dataset_path": dataset_path,
        "bert_model_path": bert_model_path,
        "train_triples": train_triples,
        "valid_triples": valid_triples,
        "test_triples": test_triples,
        "all_true_triples": (
            set(train_triples)
            | set(valid_triples)
            | set(test_triples)
        ),
        "negative_entity_ids": negative_entity_ids,
        "training_negative_filter": training_negative_filter,
        "entities": entities,
        "relations": relations,
        "entity_to_idx": entity_to_idx,
        "relation_to_idx": relation_to_idx,
    }

    trial_results = {}
    for trial in grid:
        trial_path = os.path.join(
            trials_dir,
            f"trial_{trial['trial_id']:04d}.json",
        )
        existing_result = read_trial_result(trial_path, trial)
        if existing_result is None:
            continue
        if (
            existing_result["status"] == "failed_oom"
            and args.retry_failed
        ):
            continue
        trial_results[trial["trial_id"]] = existing_result

    best_result = write_aggregate_results(
        output_dir,
        trial_results,
    )
    best_checkpoint_path = os.path.join(
        output_dir,
        "best_model.pt",
    )
    force_rerun_trial_id = None
    if best_result and not os.path.isfile(best_checkpoint_path):
        force_rerun_trial_id = best_result["trial_id"]
        print(
            "Best checkpoint is missing; rerunning trial "
            f"{force_rerun_trial_id} to recover it."
        )

    new_trials_run = 0
    for trial in grid:
        trial_id = trial["trial_id"]
        existing_result = trial_results.get(trial_id)
        is_terminal = (
            existing_result is not None
            and (
                existing_result.get("status") == "completed"
                or (
                    existing_result.get("status") == "failed_oom"
                    and not args.retry_failed
                )
            )
        )
        if is_terminal and trial_id != force_rerun_trial_id:
            continue
        if (
            args.max_trials is not None
            and new_trials_run >= args.max_trials
        ):
            break

        hyperparameters = trial["hyperparameters"]
        print(
            "\n"
            f"Starting trial {trial_id}/{len(grid)}: "
            f"{json.dumps(hyperparameters, sort_keys=True)}"
        )
        trial_checkpoint_path = os.path.join(
            output_dir,
            f".trial_{trial_id:04d}_best.pt",
        )
        trial_start_time = time.perf_counter()

        try:
            result = train_trial(
                args=args,
                trial=trial,
                shared_data=shared_data,
                tokenizer=tokenizer,
                device=device,
                trial_checkpoint_path=trial_checkpoint_path,
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            result = {
                "trial_id": trial_id,
                "status": "failed_oom",
                "hyperparameters": hyperparameters,
                "best_epoch": None,
                "epochs_ran": None,
                "early_stopped": None,
                "best_validation_metrics": None,
                "best_final_lambda": None,
                "best_calibration": None,
                "runtime_min": (
                    time.perf_counter() - trial_start_time
                ) / 60.0,
                "history": [],
                "error": str(error),
            }
            print(f"Trial {trial_id} failed with GPU OOM: {error}")
            if os.path.isfile(trial_checkpoint_path):
                os.unlink(trial_checkpoint_path)
        finally:
            clean_up_trial(device)

        result = add_run_metadata(
            result,
            args,
            shared_data,
            output_dir,
        )
        completed_results = [
            previous_result
            for previous_result in trial_results.values()
            if previous_result.get("status") == "completed"
            and previous_result.get("trial_id") != trial_id
        ]
        previous_best = (
            max(completed_results, key=result_sort_key)
            if completed_results
            else None
        )
        is_new_global_best = (
            result["status"] == "completed"
            and (
                previous_best is None
                or result_sort_key(result)
                > result_sort_key(previous_best)
            )
        )
        if is_new_global_best:
            os.replace(
                trial_checkpoint_path,
                best_checkpoint_path,
            )
            print(
                "New global best: "
                f"trial={trial_id} "
                f"valid_mrr="
                f"{result['best_validation_metrics']['MRR']:.6f}"
            )
        elif os.path.isfile(trial_checkpoint_path):
            os.unlink(trial_checkpoint_path)

        trial_path = os.path.join(
            trials_dir,
            f"trial_{trial_id:04d}.json",
        )
        atomic_write_json(trial_path, result)
        trial_results[trial_id] = result
        best_result = write_aggregate_results(
            output_dir,
            trial_results,
        )
        new_trials_run += 1

    best_result = write_aggregate_results(
        output_dir,
        trial_results,
    )
    completed_count = sum(
        result.get("status") == "completed"
        for result in trial_results.values()
    )
    failed_count = sum(
        result.get("status") == "failed_oom"
        for result in trial_results.values()
    )
    print(
        f"completed_trials={completed_count}/{len(grid)} "
        f"failed_oom_trials={failed_count}"
    )
    if best_result:
        print(
            "best_trial="
            f"{best_result['trial_id']} "
            "best_valid_mrr="
            f"{best_result['best_validation_metrics']['MRR']:.6f} "
            "best_hyperparameters="
            f"{json.dumps(best_result['hyperparameters'], sort_keys=True)}"
        )
    print(
        "grid_runtime_minutes="
        f"{(time.perf_counter() - run_start_time) / 60.0:.2f}"
    )


if __name__ == "__main__":
    main()
