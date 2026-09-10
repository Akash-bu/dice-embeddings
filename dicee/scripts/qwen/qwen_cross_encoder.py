"""Candidate-aware Qwen cross-encoder for knowledge-graph link prediction.

Every Qwen weight is trainable. Unlike the retrieval-style Qwen scripts, this
model reads the candidate entity's name and description, uses learned masked
attention pooling over all prompt tokens, and produces one plausibility logit.
Training follows core Dice ``NegSample`` semantics: one positive item per
triple, K uniformly sampled corruptions, and one batch-wide head/tail coin
flip. It does not use grouped K-vs-all queries.
"""

import argparse
import json
import math
import os
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from dicee.scripts.qwen.qwen import (
        build_true_entity_indexes,
        dataset_name_from_path,
        entity_text,
        rank_of_target,
        ranks_to_metrics,
        read_dataset_splits,
        read_support,
        relation_text,
        set_seed,
        should_early_stop,
    )
    from dicee.scripts.qwen.qwen_full import resolve_dtype
except ModuleNotFoundError:
    # Also support: python dicee/scripts/decoder_based/train_qwen_cross_encoder.py
    from qwen import (
        build_true_entity_indexes,
        dataset_name_from_path,
        entity_text,
        rank_of_target,
        ranks_to_metrics,
        read_dataset_splits,
        read_support,
        relation_text,
        set_seed,
        should_early_stop,
    )
    from qwen_full import resolve_dtype


PROMPT_VERSION = "qwen_cross_encoder_v1_candidate_attention_pool"


def default_output_paths(dataset_path, timestamp=None):
    """Return timestamped checkpoint and result paths in one folder."""
    dataset_name = dataset_name_from_path(dataset_path)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{dataset_name}_qwen_cross_encoder_{timestamp}"
    stem = os.path.join("decoder_runs", run_name, run_name)
    return f"{stem}.pt", f"{stem}.json"


def prepare_output_directories(output_path, results_path):
    """Create output folders before the first epoch starts."""
    output_path = os.path.abspath(output_path)
    results_path = os.path.abspath(results_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    return output_path, results_path


def atomic_torch_save(value, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_json(value, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary_path, path)


def build_candidate_prompt(
    source_entity_id,
    relation_id,
    candidate_entity_id,
    entities,
    relations,
    direction,
):
    """Build a direction-neutral prompt containing the complete candidate fact."""
    source = entity_text(source_entity_id, entities)
    candidate = entity_text(candidate_entity_id, entities)
    relation = relation_text(relation_id, relations)

    if direction == "tail":
        head, tail = source, candidate
    elif direction == "head":
        head, tail = candidate, source
    else:
        raise ValueError(f"Unsupported prediction direction: {direction!r}")
    return (
        "Knowledge graph link prediction.\n"
        f"Head entity: {head}\n"
        f"Relation: {relation}\n"
        f"Tail entity: {tail}\n"
        "Candidate fact plausibility:"
    )


class DiceNegativeSamplingDataset(Dataset):
    """Expose one item per positive triple, as core Dice NegSample does."""

    def __init__(
        self,
        triples,
    ):
        # Core Dice lexicographically sorts indexed triples before training.
        # Sorting string IDs here preserves the same order-independent input
        # convention before DataLoader shuffling.
        self.triples = sorted(triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        return self.triples[index]


class DiceNegativeSamplingCollator:
    """Reproduce core Dice TriplePredictionDataset.collate_fn for text."""

    def __init__(
        self,
        tokenizer,
        entities,
        relations,
        max_length,
        negative_ratio,
        label_smoothing_rate=0.0,
    ):
        if negative_ratio < 1:
            raise ValueError("negative_ratio must be at least 1.")
        self.tokenizer = tokenizer
        self.entities = entities
        self.relations = relations
        self.entity_ids = list(entities)
        self.max_length = max_length
        self.negative_ratio = negative_ratio
        self.label_smoothing_rate = label_smoothing_rate

    def __call__(self, positive_triples):
        batch_size = len(positive_triples)
        if batch_size == 0:
            raise ValueError("Cannot collate an empty batch.")

        # Core Dice flips one coin for the complete batch, then corrupts that
        # side for every sampled negative in the batch.
        corrupt_head = bool(torch.rand(1) >= 0.5)
        direction = "head" if corrupt_head else "tail"
        sampled_indices = torch.randint(
            low=0,
            high=len(self.entity_ids),
            size=(batch_size * self.negative_ratio,),
        ).tolist()
        negative_triples = []
        for negative_round in range(self.negative_ratio):
            for row, (head, relation, tail) in enumerate(positive_triples):
                replacement = self.entity_ids[
                    sampled_indices[negative_round * batch_size + row]
                ]
                if corrupt_head:
                    negative_triples.append((replacement, relation, tail))
                else:
                    negative_triples.append((head, relation, replacement))

        all_triples = list(positive_triples) + negative_triples
        prompts = []
        for head, relation, tail in all_triples:
            source = tail if direction == "head" else head
            candidate = head if direction == "head" else tail
            prompts.append(build_candidate_prompt(
                source_entity_id=source,
                relation_id=relation,
                candidate_entity_id=candidate,
                entities=self.entities,
                relations=self.relations,
                direction=direction,
            ))
        labels = (
            [1.0 - self.label_smoothing_rate] * batch_size
            + [self.label_smoothing_rate] * len(negative_triples)
        )
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.float32)
        encoded["corruption_direction"] = direction
        return encoded


class QwenCrossEncoder(nn.Module):
    """Full Qwen plus learned all-token attention pooling and BCE head."""

    def __init__(
        self,
        model_name,
        dtype,
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
                    "Use --disable_gradient_checkpointing."
                )
            enable_checkpointing()

        hidden_size = self.qwen.config.hidden_size
        self.pool_score = nn.Linear(hidden_size, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        nn.init.normal_(self.pool_score.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

    def pool_hidden_states(self, hidden_states, attention_mask):
        """Learn a normalized contribution from every non-padding token."""
        hidden_states = hidden_states.to(self.pool_score.weight.dtype)
        attention_logits = self.pool_score(hidden_states).squeeze(-1)
        attention_logits = attention_logits.masked_fill(
            attention_mask == 0,
            torch.finfo(attention_logits.dtype).min,
        )
        attention_weights = torch.softmax(attention_logits, dim=-1)
        return torch.bmm(
            attention_weights.unsqueeze(1),
            hidden_states,
        ).squeeze(1)

    def forward(self, input_ids, attention_mask):
        output = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        pooled = self.pool_hidden_states(
            output.last_hidden_state,
            attention_mask,
        )
        return self.classifier(pooled).squeeze(-1)


def dice_negsample_bce(logits, labels):
    """Use core Dice DefaultBCELoss semantics without class reweighting."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def create_optimizer(
    model,
    qwen_learning_rate,
    head_learning_rate,
    qwen_weight_decay,
    head_weight_decay,
):
    qwen_decay = []
    qwen_no_decay = []
    no_decay_suffixes = (
        "bias",
        "norm.weight",
        "layernorm.weight",
        "layer_norm.weight",
    )
    for name, parameter in model.qwen.named_parameters():
        destination = (
            qwen_no_decay
            if name.lower().endswith(no_decay_suffixes)
            else qwen_decay
        )
        destination.append(parameter)
    groups = []
    if qwen_decay:
        groups.append({
            "params": qwen_decay,
            "lr": qwen_learning_rate,
            "weight_decay": qwen_weight_decay,
        })
    if qwen_no_decay:
        groups.append({
            "params": qwen_no_decay,
            "lr": qwen_learning_rate,
            "weight_decay": 0.0,
        })
    groups.append({
        "params": list(model.pool_score.parameters())
        + list(model.classifier.parameters()),
        "lr": head_learning_rate,
        "weight_decay": head_weight_decay,
    })
    return torch.optim.AdamW(groups)


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
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    optimizer_steps = 0
    num_batches = len(loader)
    progress = tqdm(loader, desc="Training Qwen cross-encoder")

    for batch_index, batch in enumerate(progress):
        corruption_direction = batch.pop("corruption_direction")
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("labels")
        logits = model(**batch)
        # This is the same unweighted BCEWithLogits objective used by core
        # Dice's DefaultBCELoss for NegSample training.
        loss = dice_negsample_bce(logits, labels)
        window_start = (
            batch_index // gradient_accumulation_steps
        ) * gradient_accumulation_steps
        window_size = min(
            gradient_accumulation_steps,
            num_batches - window_start,
        )
        (loss / window_size).backward()

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

        examples = labels.numel()
        total_loss += float(loss.detach().cpu()) * examples
        total_examples += examples
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            optimizer_steps=optimizer_steps,
            corrupt=corruption_direction,
        )

    if total_examples == 0:
        raise RuntimeError("The training loader produced no examples.")
    return total_loss / total_examples, optimizer_steps


def score_candidate_prompts(
    model,
    tokenizer,
    prompts,
    device,
    max_length,
    batch_size,
):
    scores = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                prompts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = move_batch_to_device(encoded, device)
            scores.append(model(**encoded).detach().cpu())
    return torch.cat(scores)


def evaluate_cross_encoder(
    model,
    tokenizer,
    eval_triples,
    all_true_entities,
    entities,
    relations,
    device,
    max_length,
    candidate_batch_size,
    description,
):
    """Enumerate candidate entities and compute filtered head/tail ranks."""
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be at least 1.")
    entity_ids = list(entities)
    entity_to_index = {
        entity_id: index
        for index, entity_id in enumerate(entity_ids)
    }
    ranks_by_direction = {"head": [], "tail": []}
    model.eval()

    for head, relation, tail in tqdm(eval_triples, desc=description):
        for direction in ("tail", "head"):
            source_entity = head if direction == "tail" else tail
            target_entity = tail if direction == "tail" else head
            prompts = [
                build_candidate_prompt(
                    source_entity_id=source_entity,
                    relation_id=relation,
                    candidate_entity_id=candidate,
                    entities=entities,
                    relations=relations,
                    direction=direction,
                )
                for candidate in entity_ids
            ]
            scores = score_candidate_prompts(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                max_length=max_length,
                batch_size=candidate_batch_size,
            )
            target_index = entity_to_index[target_entity]
            target_score = scores[target_index].clone()
            known_entities = all_true_entities[direction][
                (source_entity, relation)
            ]
            known_indices = [
                entity_to_index[entity]
                for entity in known_entities
            ]
            scores[known_indices] = -float("inf")
            scores[target_index] = target_score
            ranks_by_direction[direction].append(
                rank_of_target(scores, target_index)
            )

    combined = ranks_by_direction["head"] + ranks_by_direction["tail"]
    metrics = ranks_to_metrics(combined)
    metrics["head_metrics"] = ranks_to_metrics(ranks_by_direction["head"])
    metrics["tail_metrics"] = ranks_to_metrics(ranks_by_direction["tail"])
    return metrics


def model_state_to_cpu(model):
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


def build_checkpoint(
    model,
    args,
    entity_ids,
    best_epoch,
    best_validation_metrics,
    training_history,
):
    return {
        "version": 1,
        "model_state_dict": model_state_to_cpu(model),
        "entity_ids": entity_ids,
        "model_name": args.model_name,
        "model_dtype": args.resolved_dtype,
        "max_length": args.max_length,
        "prompt_version": PROMPT_VERSION,
        "training_mode": "full_qwen_candidate_cross_encoder_dice_negsample",
        "negative_sampling": "core_dice_negsample",
        "pooling": "learned_masked_attention_all_tokens",
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
        description="Fully fine-tune a candidate-aware Qwen cross-encoder."
    )
    parser.add_argument(
        "--dataset_path",
        default="bert_datasets/UMLS/0.0",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-0.6B-Base",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--results_path", default=None)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help=(
            "Positive triples per forward pass; actual prompts equal "
            "batch_size * (1 + negative_ratio)."
        ),
    )
    parser.add_argument(
        "--negative_ratio",
        type=int,
        default=2,
        help=(
            "Uniform corruptions per positive, matching core Dice's "
            "--neg_ratio (default: 2)."
        ),
    )
    parser.add_argument(
        "--label_smoothing_rate",
        type=float,
        default=0.0,
        help="Core Dice-style positive/negative label smoothing.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument(
        "--candidate_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0001,
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
    )
    parser.add_argument("--qwen_weight_decay", type=float, default=0.01)
    parser.add_argument("--head_weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--disable_gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args, parser):
    for name in (
        "batch_size",
        "negative_ratio",
        "gradient_accumulation_steps",
        "num_epochs",
        "eval_every",
        "candidate_batch_size",
        "max_length",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.early_stopping_patience < 0:
        parser.error("--early_stopping_patience cannot be negative")
    if args.early_stopping_min_delta < 0:
        parser.error("--early_stopping_min_delta cannot be negative")
    if not 0.0 <= args.label_smoothing_rate < 0.5:
        parser.error("--label_smoothing_rate must be in [0, 0.5)")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if args.qwen_learning_rate <= 0 or args.head_learning_rate <= 0:
        parser.error("learning rates must be positive")
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

    default_checkpoint, default_results = default_output_paths(
        args.dataset_path
    )
    if args.output_path is None:
        args.output_path = default_checkpoint
    if args.results_path is None:
        args.results_path = default_results
    args.output_path, args.results_path = prepare_output_directories(
        args.output_path,
        args.results_path,
    )
    print(f"run_output_dir={os.path.dirname(args.output_path)}")
    print(f"checkpoint_path={args.output_path}")
    print(f"results_path={args.results_path}")

    set_seed(args.seed)
    device = torch.device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    args.resolved_dtype = str(model_dtype).removeprefix("torch.")

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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        local_files_only=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = QwenCrossEncoder(
        model_name=args.model_name,
        dtype=model_dtype,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
    ).to(device)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable != total:
        raise RuntimeError("Some Qwen cross-encoder parameters are frozen.")
    print(
        f"training_mode=full_qwen_candidate_cross_encoder "
        f"trainable_parameters={trainable:,} model_dtype={model_dtype}"
    )

    train_dataset = DiceNegativeSamplingDataset(
        triples=train_triples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=DiceNegativeSamplingCollator(
            tokenizer=tokenizer,
            entities=entities,
            relations=relations,
            max_length=args.max_length,
            negative_ratio=args.negative_ratio,
            label_smoothing_rate=args.label_smoothing_rate,
        ),
    )
    optimizer = create_optimizer(
        model=model,
        qwen_learning_rate=args.qwen_learning_rate,
        head_learning_rate=args.head_learning_rate,
        qwen_weight_decay=args.qwen_weight_decay,
        head_weight_decay=args.head_weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_scheduled_steps = updates_per_epoch * args.num_epochs
    warmup_steps = int(total_scheduled_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_scheduled_steps,
    )
    print(
        f"train_triples={len(train_triples)} "
        f"train_positive_items={len(train_dataset)} "
        f"prompts_per_positive={1 + args.negative_ratio} "
        f"negative_sampling=core_dice_negsample "
        f"batches_per_epoch={len(train_loader)} "
        f"updates_per_epoch={updates_per_epoch}"
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
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "optimizer_steps": optimizer_steps,
            "total_optimizer_steps": total_optimizer_steps,
        }
        print(f"epoch={epoch} train_loss={train_loss:.6f}")

        should_evaluate = (
            epoch % args.eval_every == 0
            or epoch == args.num_epochs
        )
        if not should_evaluate:
            training_history.append(record)
            continue

        validation_metrics = evaluate_cross_encoder(
            model=model,
            tokenizer=tokenizer,
            eval_triples=valid_triples,
            all_true_entities=validation_true_entities,
            entities=entities,
            relations=relations,
            device=device,
            max_length=args.max_length,
            candidate_batch_size=args.candidate_batch_size,
            description=f"Validating epoch {epoch}",
        )
        record["validation_metrics"] = validation_metrics
        validation_mrr = validation_metrics["MRR"]
        is_best = (
            validation_mrr
            > best_validation_mrr + args.early_stopping_min_delta
        )
        record["is_best"] = is_best
        if is_best:
            evaluations_without_improvement = 0
            best_validation_mrr = validation_mrr
            best_validation_metrics = dict(validation_metrics)
            best_epoch = epoch
        else:
            evaluations_without_improvement += 1
        record["evaluations_without_improvement"] = (
            evaluations_without_improvement
        )
        training_history.append(record)
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
                entity_ids=list(entities),
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
                f"early_stopping_epoch={epoch} best_epoch={best_epoch} "
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
        test_metrics = evaluate_cross_encoder(
            model=model,
            tokenizer=tokenizer,
            eval_triples=test_triples,
            all_true_entities=test_true_entities,
            entities=entities,
            relations=relations,
            device=device,
            max_length=args.max_length,
            candidate_batch_size=args.candidate_batch_size,
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

    best_checkpoint.update({
        "test_metrics": test_metrics,
        "training_history": training_history,
        "early_stopped": early_stopped,
        "total_optimizer_steps": total_optimizer_steps,
    })
    atomic_torch_save(best_checkpoint, args.output_path)

    results = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "model_name": args.model_name,
        "prompt_version": PROMPT_VERSION,
        "training_mode": "full_qwen_candidate_cross_encoder_dice_negsample",
        "negative_sampling": "core_dice_negsample",
        "pooling": "learned_masked_attention_all_tokens",
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
        "num_train_positive_items": len(train_dataset),
        "num_entities": len(entities),
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
