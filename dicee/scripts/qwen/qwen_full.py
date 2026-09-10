"""End-to-end Qwen link prediction with trainable Qwen, W, and E.

Unlike ``train_qwen.py``, this script does not cache Qwen features. The Qwen
hidden states must be recomputed because every Qwen parameter is optimized.
"""

import argparse
import json
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from dicee.scripts.qwen.qwen import (
        KvsAllCollator,
        QwenKvsAllDataset,
        build_prediction_queries,
        build_true_entity_indexes,
        dataset_name_from_path,
        loss_func,
        query_prompt,
        rank_of_target,
        ranks_to_metrics,
        read_dataset_splits,
        read_support,
        set_seed,
        should_early_stop,
    )
except ModuleNotFoundError:
    # Also support: python dicee/scripts/decoder_based/train_qwen_full.py
    from train_qwen import (
        KvsAllCollator,
        QwenKvsAllDataset,
        build_prediction_queries,
        build_true_entity_indexes,
        dataset_name_from_path,
        loss_func,
        query_prompt,
        rank_of_target,
        ranks_to_metrics,
        read_dataset_splits,
        read_support,
        set_seed,
        should_early_stop,
    )


PROMPT_VERSION = "qwen_full_v1_head_tail"


def default_output_paths(dataset_path, timestamp=None):
    """Return timestamped paths in one run-specific directory."""
    dataset_name = dataset_name_from_path(dataset_path)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{dataset_name}_qwen_full_{timestamp}"
    stem = os.path.join("decoder_runs", run_name, run_name)
    return f"{stem}.pt", f"{stem}.json"


def atomic_torch_save(value, path):
    """Write a checkpoint without leaving a partially written final file."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def load_torch_checkpoint(path):
    """Load a trusted local training checkpoint on CPU."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_json(value, path):
    """Atomically save JSON output."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary_path, path)


def resolve_dtype(dtype_name, device):
    """Resolve the model dtype, preferring BF16 only when CUDA supports it."""
    if dtype_name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    if dtype_name == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError(
                "This CUDA device does not support bfloat16; use --dtype float32."
            )
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name!r}")


class FullQwenLinkPredictor(nn.Module):
    """Rank entities while fine-tuning every Qwen, W, and E parameter."""

    def __init__(
        self,
        model_name,
        num_entities,
        embedding_dim=256,
        dtype=torch.bfloat16,
        gradient_checkpointing=True,
    ):
        super().__init__()
        self.qwen = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            local_files_only=False,
        )
        self.qwen.requires_grad_(True)
        if hasattr(self.qwen.config, "use_cache"):
            self.qwen.config.use_cache = False

        if gradient_checkpointing:
            enable_checkpointing = getattr(
                self.qwen,
                "gradient_checkpointing_enable",
                None,
            )
            if not callable(enable_checkpointing):
                raise ValueError(
                    f"{model_name} does not support gradient checkpointing. "
                    "Run with --disable_gradient_checkpointing."
                )
            enable_checkpointing()

        hidden_size = self.qwen.config.hidden_size
        self.proj = nn.Linear(hidden_size, embedding_dim, bias=False)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        self.register_buffer(
            "logit_scale",
            torch.tensor(embedding_dim**-0.5),
        )

    def encode_queries(self, input_ids, attention_mask):
        """Return the final non-padding token state with gradients enabled."""
        output = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = output.last_hidden_state
        last_token_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device,
        )
        return hidden_states[batch_indices, last_token_positions]

    def forward(self, input_ids, attention_mask):
        features = self.encode_queries(input_ids, attention_mask)
        query = self.proj(features.to(self.proj.weight.dtype))
        return (
            query @ self.entity_embeddings.weight.T
        ) * self.logit_scale


def print_trainable_parameters(model):
    qwen_trainable = sum(
        parameter.numel()
        for parameter in model.qwen.parameters()
        if parameter.requires_grad
    )
    head_trainable = sum(
        parameter.numel()
        for module in (model.proj, model.entity_embeddings)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    total_trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"qwen_trainable_parameters={qwen_trainable:,} "
        f"head_trainable_parameters={head_trainable:,} "
        f"total_trainable_parameters={total_trainable:,} "
        f"total_parameters={total_parameters:,}"
    )
    if total_trainable != total_parameters:
        raise RuntimeError(
            "Full fine-tuning requested, but some model parameters are frozen."
        )


def create_optimizer(
    model,
    qwen_learning_rate,
    head_learning_rate,
    qwen_weight_decay,
    head_weight_decay,
):
    """Use a small Qwen LR and a larger LR for newly initialized W and E."""
    qwen_decay = []
    qwen_no_decay = []
    no_decay_suffixes = (
        "bias",
        "norm.weight",
        "layernorm.weight",
        "layer_norm.weight",
    )
    for name, parameter in model.qwen.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.lower().endswith(no_decay_suffixes):
            qwen_no_decay.append(parameter)
        else:
            qwen_decay.append(parameter)

    parameter_groups = []
    if qwen_decay:
        parameter_groups.append(
            {
                "params": qwen_decay,
                "lr": qwen_learning_rate,
                "weight_decay": qwen_weight_decay,
            }
        )
    if qwen_no_decay:
        parameter_groups.append(
            {
                "params": qwen_no_decay,
                "lr": qwen_learning_rate,
                "weight_decay": 0.0,
            }
        )
    parameter_groups.append(
        {
            "params": list(model.proj.parameters())
            + list(model.entity_embeddings.parameters()),
            "lr": head_learning_rate,
            "weight_decay": head_weight_decay,
        }
    )
    return torch.optim.AdamW(parameter_groups)


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    gradient_accumulation_steps,
    max_grad_norm,
):
    """Fine-tune Qwen and the prediction head for one epoch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    optimizer_steps = 0
    num_batches = len(loader)
    progress = tqdm(loader, desc="Training full Qwen")

    for batch_index, batch in enumerate(progress):
        batch = move_batch_to_device(batch, device)
        targets = batch.pop("targets")
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = loss_func(logits, targets)

        accumulation_window_start = (
            batch_index // gradient_accumulation_steps
        ) * gradient_accumulation_steps
        accumulation_window_size = min(
            gradient_accumulation_steps,
            num_batches - accumulation_window_start,
        )
        (loss / accumulation_window_size).backward()

        should_step = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == num_batches
        )
        if should_step:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            optimizer_steps=optimizer_steps,
        )

    if total_examples == 0:
        raise RuntimeError("The training loader produced no examples.")
    return total_loss / total_examples, optimizer_steps


def evaluate_live_ranking(
    model,
    tokenizer,
    queries,
    entities,
    relations,
    all_true_entities,
    entity_to_idx,
    device,
    max_length,
    batch_size,
    description,
):
    """Evaluate current Qwen weights with filtered head and tail ranking."""
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be at least 1.")

    ranks_by_direction = {"head": [], "tail": []}
    model.eval()
    progress = tqdm(range(0, len(queries), batch_size), desc=description)
    with torch.no_grad():
        for start in progress:
            query_batch = queries[start:start + batch_size]
            prompts = [
                query_prompt(query, entities, relations)
                for query in query_batch
            ]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            for row, query in enumerate(query_batch):
                direction, source_entity, relation, targets = query
                scores = logits[row]
                known_target_indices = [
                    entity_to_idx[entity]
                    for entity in all_true_entities[direction][
                        (source_entity, relation)
                    ]
                ]
                for target in targets:
                    target_index = entity_to_idx[target]
                    target_score = scores[target_index].clone()
                    filtered_scores = scores.clone()
                    filtered_scores[known_target_indices] = -float("inf")
                    filtered_scores[target_index] = target_score
                    ranks_by_direction[direction].append(
                        rank_of_target(filtered_scores, target_index)
                    )

            progress.set_postfix(
                head_ranks=len(ranks_by_direction["head"]),
                tail_ranks=len(ranks_by_direction["tail"]),
            )

    combined_ranks = ranks_by_direction["head"] + ranks_by_direction["tail"]
    metrics = ranks_to_metrics(combined_ranks)
    metrics["head_metrics"] = ranks_to_metrics(ranks_by_direction["head"])
    metrics["tail_metrics"] = ranks_to_metrics(ranks_by_direction["tail"])
    return metrics


def model_state_to_cpu(model):
    """Copy the complete Qwen + W + E state for checkpointing."""
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


def build_checkpoint(
    model,
    args,
    entity_ids,
    entity_to_idx,
    best_epoch,
    best_validation_metrics,
    training_history,
):
    return {
        "version": 1,
        "model_state_dict": model_state_to_cpu(model),
        "entity_ids": entity_ids,
        "entity_to_idx": entity_to_idx,
        "model_name": args.model_name,
        "embedding_dim": args.embedding_dim,
        "max_length": args.max_length,
        "model_dtype": args.resolved_dtype,
        "prompt_version": PROMPT_VERSION,
        "training_mode": "full_qwen_w_e",
        "prediction_direction": "head_and_tail_no_reciprocals",
        "selection_split": "valid",
        "selection_metric": "MRR",
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "training_history": training_history,
        "args": vars(args),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune all Qwen weights plus W and E for head/tail link "
            "prediction."
        )
    )
    parser.add_argument(
        "--dataset_path",
        default="bert_datasets/UMLS/0.0",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-0.6B-Base",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help=(
            "Checkpoint path; defaults to "
            "decoder_runs/{dataset}_qwen_full_{datetime}/"
            "{dataset}_qwen_full_{datetime}.pt."
        ),
    )
    parser.add_argument(
        "--results_path",
        default=None,
        help=(
            "Results path; defaults to the same timestamped run folder as "
            "the checkpoint."
        ),
    )
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of grouped queries per forward/backward pass.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Accumulate this many batches before each optimizer update.",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Evaluate validation every N epochs and always at the final epoch.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Number of live Qwen queries per evaluation batch.",
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help="Tune on train/valid only without reading test.txt.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help=(
            "Stop after this many validation evaluations without an MRR "
            "improvement; 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation-MRR increase counted as improvement.",
    )
    parser.add_argument(
        "--qwen_learning_rate",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--head_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for projection W and entity embeddings E.",
    )
    parser.add_argument("--qwen_weight_decay", type=float, default=0.01)
    parser.add_argument("--head_weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of optimizer updates used for linear LR warmup.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float32"],
        default="auto",
        help="Qwen parameter dtype; auto uses BF16 on supported CUDA devices.",
    )
    parser.add_argument(
        "--disable_gradient_checkpointing",
        action="store_true",
        help="Use more activation memory in exchange for faster training.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args, parser):
    positive_integer_fields = (
        "batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "eval_every",
        "eval_batch_size",
        "embedding_dim",
        "max_length",
    )
    for field in positive_integer_fields:
        if getattr(args, field) < 1:
            parser.error(f"--{field} must be at least 1")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if args.early_stopping_patience < 0:
        parser.error("--early_stopping_patience cannot be negative")
    if args.early_stopping_min_delta < 0:
        parser.error("--early_stopping_min_delta cannot be negative")
    if args.qwen_learning_rate <= 0:
        parser.error("--qwen_learning_rate must be positive")
    if args.head_learning_rate <= 0:
        parser.error("--head_learning_rate must be positive")
    if args.qwen_weight_decay < 0 or args.head_weight_decay < 0:
        parser.error("weight decay cannot be negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")
    if args.max_grad_norm <= 0:
        parser.error("--max_grad_norm must be positive")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    default_checkpoint_path, default_results_path = default_output_paths(
        args.dataset_path
    )
    if args.output_path is None:
        args.output_path = default_checkpoint_path
    if args.results_path is None:
        args.results_path = default_results_path

    set_seed(args.seed)
    device = torch.device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    args.resolved_dtype = str(model_dtype).removeprefix("torch.")
    print(
        f"training_mode=full_qwen_w_e device={device} "
        f"model_dtype={model_dtype} "
        f"gradient_checkpointing={not args.disable_gradient_checkpointing}"
    )

    entities, relations = read_support(args.dataset_path)
    train_triples, valid_triples, test_triples = read_dataset_splits(
        args.dataset_path,
        validation_only=args.validation_only,
    )
    validation_true_entities = build_true_entity_indexes(
        train_triples,
        valid_triples,
    )
    test_true_entities = None
    if test_triples is not None:
        test_true_entities = build_true_entity_indexes(
            train_triples,
            valid_triples,
            test_triples,
        )

    entity_ids = list(entities.keys())
    entity_to_idx = {
        entity_id: index
        for index, entity_id in enumerate(entity_ids)
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        local_files_only=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer has neither a pad nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FullQwenLinkPredictor(
        model_name=args.model_name,
        num_entities=len(entity_ids),
        embedding_dim=args.embedding_dim,
        dtype=model_dtype,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
    ).to(device)
    print_trainable_parameters(model)

    train_dataset = QwenKvsAllDataset(
        triples=train_triples,
        entities=entities,
        relations=relations,
        entity_to_idx=entity_to_idx,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=KvsAllCollator(
            tokenizer=tokenizer,
            num_entities=len(entity_ids),
            max_length=args.max_length,
        ),
    )
    valid_queries = build_prediction_queries(valid_triples)
    test_queries = (
        []
        if test_triples is None
        else build_prediction_queries(test_triples)
    )
    print(
        f"train_queries={len(train_dataset)} "
        f"valid_queries={len(valid_queries)} "
        f"test_queries={len(test_queries)} "
        f"batches_per_epoch={len(train_loader)} "
        f"optimizer_updates_per_epoch="
        f"{(len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps}"
    )

    optimizer = create_optimizer(
        model=model,
        qwen_learning_rate=args.qwen_learning_rate,
        head_learning_rate=args.head_learning_rate,
        qwen_weight_decay=args.qwen_weight_decay,
        head_weight_decay=args.head_weight_decay,
    )
    optimizer_updates_per_epoch = (
        len(train_loader) + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps
    total_scheduled_steps = optimizer_updates_per_epoch * args.num_epochs
    warmup_steps = int(total_scheduled_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_scheduled_steps,
    )
    print(
        f"total_scheduled_steps={total_scheduled_steps} "
        f"warmup_steps={warmup_steps}"
    )

    best_validation_mrr = -float("inf")
    best_validation_metrics = None
    best_epoch = None
    training_history = []
    evaluations_without_improvement = 0
    early_stopped = False
    total_optimizer_steps = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss, optimizer_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        total_optimizer_steps += optimizer_steps
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "optimizer_steps": optimizer_steps,
            "total_optimizer_steps": total_optimizer_steps,
        }
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"optimizer_steps={optimizer_steps}"
        )

        should_evaluate = (
            epoch % args.eval_every == 0
            or epoch == args.num_epochs
        )
        if not should_evaluate:
            training_history.append(epoch_record)
            continue

        validation_metrics = evaluate_live_ranking(
            model=model,
            tokenizer=tokenizer,
            queries=valid_queries,
            entities=entities,
            relations=relations,
            all_true_entities=validation_true_entities,
            entity_to_idx=entity_to_idx,
            device=device,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            description=f"Validating epoch {epoch}",
        )
        epoch_record["validation_metrics"] = validation_metrics
        validation_mrr = validation_metrics["MRR"]
        is_best = (
            validation_mrr
            > best_validation_mrr + args.early_stopping_min_delta
        )
        epoch_record["is_best"] = is_best
        if is_best:
            evaluations_without_improvement = 0
            best_validation_mrr = validation_mrr
            best_validation_metrics = dict(validation_metrics)
            best_epoch = epoch
        else:
            evaluations_without_improvement += 1
        epoch_record["evaluations_without_improvement"] = (
            evaluations_without_improvement
        )
        training_history.append(epoch_record)

        print(
            f"validation_epoch={epoch} "
            f"mrr={validation_metrics['MRR']:.6f} "
            f"h1={validation_metrics['H@1']:.6f} "
            f"h3={validation_metrics['H@3']:.6f} "
            f"h10={validation_metrics['H@10']:.6f} "
            f"head_mrr={validation_metrics['head_metrics']['MRR']:.6f} "
            f"tail_mrr={validation_metrics['tail_metrics']['MRR']:.6f}"
        )

        if is_best:
            checkpoint = build_checkpoint(
                model=model,
                args=args,
                entity_ids=entity_ids,
                entity_to_idx=entity_to_idx,
                best_epoch=best_epoch,
                best_validation_metrics=best_validation_metrics,
                training_history=training_history,
            )
            atomic_torch_save(checkpoint, args.output_path)
            del checkpoint
            print(
                f"saved_best_checkpoint={args.output_path} "
                f"best_epoch={best_epoch} "
                f"best_valid_mrr={best_validation_mrr:.6f}"
            )

        if should_early_stop(
            evaluations_without_improvement=evaluations_without_improvement,
            patience=args.early_stopping_patience,
            epoch=epoch,
            num_epochs=args.num_epochs,
        ):
            early_stopped = True
            print(
                f"early_stopping_epoch={epoch} "
                f"best_epoch={best_epoch} "
                f"best_valid_mrr={best_validation_mrr:.6f}"
            )
            break

    if best_epoch is None:
        raise RuntimeError("Training completed without validation evaluation.")

    best_checkpoint = load_torch_checkpoint(args.output_path)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    test_metrics = None
    if args.validation_only:
        print("validation_only=true test_set_not_loaded=true")
    else:
        test_metrics = evaluate_live_ranking(
            model=model,
            tokenizer=tokenizer,
            queries=test_queries,
            entities=entities,
            relations=relations,
            all_true_entities=test_true_entities,
            entity_to_idx=entity_to_idx,
            device=device,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            description="Testing best validation checkpoint",
        )
        print(
            f"test_best_epoch={best_epoch} "
            f"mrr={test_metrics['MRR']:.6f} "
            f"h1={test_metrics['H@1']:.6f} "
            f"h3={test_metrics['H@3']:.6f} "
            f"h10={test_metrics['H@10']:.6f} "
            f"head_mrr={test_metrics['head_metrics']['MRR']:.6f} "
            f"tail_mrr={test_metrics['tail_metrics']['MRR']:.6f}"
        )

    best_checkpoint.update(
        {
            "test_metrics": test_metrics,
            "training_history": training_history,
            "early_stopped": early_stopped,
            "total_optimizer_steps": total_optimizer_steps,
        }
    )
    atomic_torch_save(best_checkpoint, args.output_path)
    del best_checkpoint

    results = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "model_name": args.model_name,
        "prompt_version": PROMPT_VERSION,
        "training_mode": "full_qwen_w_e",
        "prediction_direction": "head_and_tail_no_reciprocals",
        "validation_only": args.validation_only,
        "selection_split": "valid",
        "selection_metric": "MRR",
        "metrics": (
            best_validation_metrics
            if args.validation_only
            else test_metrics
        ),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "num_train_triples": len(train_triples),
        "num_valid_triples": len(valid_triples),
        "num_test_triples": (
            None if test_triples is None else len(test_triples)
        ),
        "num_train_queries": len(train_dataset),
        "num_valid_queries": len(valid_queries),
        "num_test_queries": len(test_queries),
        "num_entities": len(entity_ids),
        "training_history": training_history,
        "early_stopped": early_stopped,
        "total_optimizer_steps": total_optimizer_steps,
        "args": vars(args),
        "checkpoint_path": os.path.abspath(args.output_path),
    }
    save_json(results, args.results_path)
    print(f"saved_checkpoint={args.output_path}")
    print(f"saved_results={args.results_path}")


if __name__ == "__main__":
    main()
