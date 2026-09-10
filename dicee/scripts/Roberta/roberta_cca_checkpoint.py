"""Create RoBERTa checkpoints with the CCA N-BERT pretraining recipe.

This is a RoBERTa adaptation of the checkpoint-only path in
https://github.com/nju-websoft/CCA. It deliberately preserves the important
CCA behavior:

* add deterministic entity and relation tokens;
* create 33 description prompts per entity (11 prompts x 3 replicas);
* predict the masked entity over entity-token logits only;
* apply 0.8 cross-entropy label smoothing;
* train only the tied word-embedding table;
* use AdamW and cosine decay with 10 percent warmup; and
* save the epoch with the best rounded training MRR.

Triple files are used only to recognize a dataset directory. Their contents
are never read for this checkpoint-pretraining objective.
"""

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_cosine_schedule_with_warmup,
)

from dicee.scripts.Roberta.roberta_check_point import (
    atomic_json_dump,
    build_token_mapping,
    checkpoint_folder_name,
    dataset_name,
    resolve_dataset_paths,
    set_seed,
    utc_now,
)
from dicee.scripts.bert_bce_link_prediction import read_support


CHECKPOINT_FORMAT = "roberta_cca_entity_description_v1"
CCA_NOMINAL_SPLITS = ("train", "dev", "test")
NEIGHBOR_TOKEN = "[Neighbor]"
NO_RELATION_TOKEN = "[R_None]"


def description_prompt(mask_token, description):
    """Render the exact natural-language template used by CCA."""
    return f"The description of {mask_token} is that {description}"


def build_cca_pretraining_examples(
    entities,
    mask_token,
    max_length,
    random_spans_per_replica=10,
    nominal_split_replicas=3,
    seed=2022,
    shuffle=True,
):
    """Create CCA's full-description and random-span entity examples."""
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if random_spans_per_replica < 0:
        raise ValueError("random_spans_per_replica cannot be negative")
    if nominal_split_replicas < 1:
        raise ValueError("nominal_split_replicas must be at least 1")

    rng = random.Random(seed)
    examples = []
    for replica in range(nominal_split_replicas):
        split_name = (
            CCA_NOMINAL_SPLITS[replica]
            if replica < len(CCA_NOMINAL_SPLITS)
            else f"replica_{replica + 1}"
        )
        for entity_id, entity in entities.items():
            description = str(entity.get("desc") or "")
            description_tokens = description.split()
            label = int(entity["token_id"])
            examples.append(
                {
                    "entity_id": entity_id,
                    "label": label,
                    "prompt": description_prompt(mask_token, description),
                    "replica": split_name,
                    "variant": "full_description",
                }
            )
            for span_index in range(random_spans_per_replica):
                begin = rng.randint(0, len(description_tokens))
                end = min(begin + max_length, len(description_tokens))
                span = " ".join(description_tokens[begin:end])
                examples.append(
                    {
                        "entity_id": entity_id,
                        "label": label,
                        "prompt": description_prompt(mask_token, span),
                        "replica": split_name,
                        "variant": f"random_span_{span_index + 1}",
                    }
                )

    if shuffle:
        rng.shuffle(examples)
    return examples


class CCAEntityDescriptionDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CCAPretrainingCollator:
    """Tokenize prompts and return the single mask position in each row."""

    def __init__(self, tokenizer, max_length):
        if tokenizer.mask_token_id is None:
            raise ValueError("The tokenizer must define a mask token")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        encoded = self.tokenizer(
            [example["prompt"] for example in examples],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        mask_locations = input_ids.eq(self.tokenizer.mask_token_id)
        mask_counts = mask_locations.sum(dim=1)
        if not torch.all(mask_counts.eq(1)):
            invalid = mask_counts.ne(1).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "Every CCA pretraining prompt must contain exactly one mask; "
                f"invalid batch rows={invalid[:10]}"
            )
        encoded["mask_positions"] = mask_locations.to(torch.long).argmax(dim=1)
        encoded["entity_labels"] = torch.tensor(
            [example["label"] for example in examples],
            dtype=torch.long,
        ) #target labels
        encoded.pop("token_type_ids", None)
        return encoded


def cca_entity_prediction_loss_and_ranks(
    logits,
    mask_positions,
    entity_labels,
    entity_token_ids,
    label_smoothing=0.8,
):
    """Compute CCA's entity-only smoothed CE and one-indexed ranks."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    batch_size = logits.shape[0]
    if mask_positions.shape != (batch_size,):
        raise ValueError("mask_positions must have one value per batch row")
    if entity_labels.shape != (batch_size,):
        raise ValueError("entity_labels must have one value per batch row")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")

    entity_token_ids = torch.as_tensor(
        entity_token_ids,
        dtype=torch.long,
        device=logits.device,
    )
    if entity_token_ids.ndim != 1 or entity_token_ids.numel() < 1:
        raise ValueError("entity_token_ids must be a non-empty vector")
    if torch.unique(entity_token_ids).numel() != entity_token_ids.numel():
        raise ValueError("entity_token_ids must be unique")
    if entity_labels.min() < 0 or entity_labels.max() >= entity_token_ids.numel():
        raise ValueError("entity_labels contain an invalid entity index")

    batch_indices = torch.arange(batch_size, device=logits.device)
    masked_logits = logits[batch_indices, mask_positions]
    entity_logits = masked_logits.index_select(-1, entity_token_ids).float()
    loss = F.cross_entropy(
        entity_logits,
        entity_labels,
        label_smoothing=label_smoothing,
    )
    sorted_indices = torch.argsort(entity_logits, dim=-1, descending=True)
    matches = sorted_indices.eq(entity_labels[:, None])
    ranks = matches.to(torch.long).argmax(dim=1) + 1
    return loss, ranks


def add_cca_tokens(tokenizer, entities, relations):
    """Extend RoBERTa in CCA order and retain all additions as special tokens."""
    entity_tokens = [entity["name"] for entity in entities.values()]
    relation_tokens = []
    for separator_index in range(1, 6):
        separator_name = f"sep{separator_index}"
        relation_tokens.extend(
            relation[separator_name] for relation in relations.values()
        )
    extra_tokens = [NEIGHBOR_TOKEN, NO_RELATION_TOKEN]
    all_tokens = entity_tokens + relation_tokens + extra_tokens
    if len(all_tokens) != len(set(all_tokens)):
        raise ValueError("CCA extended vocabulary contains duplicate tokens")
    tokenizer.add_special_tokens({"additional_special_tokens": all_tokens})
    token_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    if len(set(token_ids)) != len(all_tokens):
        raise RuntimeError("CCA tokens did not receive distinct token IDs")
    if tokenizer.unk_token_id in token_ids:
        raise RuntimeError("At least one CCA token maps to the unknown token")
    return {
        "entity_tokens": entity_tokens,
        "entity_token_ids": tokenizer.convert_tokens_to_ids(entity_tokens),
        "relation_tokens": relation_tokens,
        "extra_tokens": extra_tokens,
        "all_token_ids": token_ids,
    }


def freeze_except_word_embeddings(model):
    """Match CCA by training the complete input embedding table and nothing else."""
    model.requires_grad_(False)
    embedding_weight = model.get_input_embeddings().weight
    embedding_weight.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable) != 1 or trainable[0] is not embedding_weight:
        raise RuntimeError(
            "Expected the tied word-embedding weight to be the only trainable "
            "parameter"
        )
    return embedding_weight


def resize_token_embeddings_with_cca_initialization(model, vocabulary_size):
    """Use the independent random initialization used by CCA-era Transformers."""
    try:
        return model.resize_token_embeddings(
            vocabulary_size,
            mean_resizing=False,
        )
    except TypeError:
        # Older Transformers releases predate mean-resizing and already use
        # the initialization behavior required here.
        return model.resize_token_embeddings(vocabulary_size)


def build_cca_optimizer_and_scheduler(
    model,
    total_steps,
    learning_rate=5e-5,
    epsilon=1e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
):
    """Build upstream CCA's AdamW and ten-percent-warmup cosine schedule."""
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("The model has no trainable parameters")
    optimizer = AdamW(
        [
            {
                "params": trainable,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ],
        eps=epsilon,
    )
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler, warmup_steps


def ranks_to_training_metrics(ranks, mean_loss):
    ranks = torch.as_tensor(ranks, dtype=torch.float64)
    if ranks.numel() == 0:
        raise ValueError("Cannot calculate metrics without ranks")
    return {
        "loss": round(float(mean_loss), 2),
        "H@1": round(float(ranks.le(1).double().mean() * 100), 2),
        "H@3": round(float(ranks.le(3).double().mean() * 100), 2),
        "H@10": round(float(ranks.le(10).double().mean() * 100), 2),
        "MRR": round(float(ranks.reciprocal().mean()), 4),
    }


def training_mrr_improved(candidate_mrr, best_mrr):
    """CCA overwrites the checkpoint only after a strict rounded-MRR gain."""
    return best_mrr is None or candidate_mrr > best_mrr


def save_best_checkpoint(model, tokenizer, token_mapping, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    atomic_json_dump(token_mapping, output_dir / "kg_token_mapping.json")


def train_one_dataset(args, source_path, device):
    name = dataset_name(source_path)
    output_dir = (
        Path(args.output_root)
        / checkpoint_folder_name(name)
        / "roberta-pretrained"
    ).resolve()
    manifest_path = output_dir / "pretraining_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Checkpoint already exists at {output_dir}; use --overwrite"
        )

    entities, relations = read_support(str(source_path))
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    base_vocab_size = len(tokenizer)
    token_info = add_cca_tokens(tokenizer, entities, relations)
    token_mapping = build_token_mapping(
        tokenizer,
        entities,
        relations,
        base_vocab_size,
    )
    examples = build_cca_pretraining_examples(
        entities=entities,
        mask_token=tokenizer.mask_token,
        max_length=args.max_length,
        random_spans_per_replica=args.random_spans_per_replica,
        nominal_split_replicas=args.nominal_split_replicas,
        seed=args.seed,
        shuffle=True,
    )
    corpus = CCAEntityDescriptionDataset(examples)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        corpus,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        collate_fn=CCAPretrainingCollator(tokenizer, args.max_length),
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.num_epochs

    model = RobertaForMaskedLM.from_pretrained(
        args.model_name,
        dtype=torch.float32,
    )
    resize_token_embeddings_with_cca_initialization(model, len(tokenizer))
    model.tie_weights()
    model.to(device)
    embedding_weight = freeze_except_word_embeddings(model)
    optimizer, scheduler, warmup_steps = build_cca_optimizer_and_scheduler(
        model=model,
        total_steps=total_steps,
        learning_rate=args.learning_rate,
        epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )
    entity_token_ids = torch.tensor(
        token_info["entity_token_ids"],
        dtype=torch.long,
        device=device,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    manifest = {
        "status": "training",
        "checkpoint_format": CHECKPOINT_FORMAT,
        "upstream_recipe": "nju-websoft/CCA Code/models/nbert.py",
        "base_model": args.model_name,
        "dataset": name,
        "dataset_path": str(source_path),
        "checkpoint_path": str(output_dir),
        "edge_splits_used": [],
        "support_metadata_used": ["entity.json", "relation.json"],
        "num_entities": len(entities),
        "num_relations": len(relations),
        "base_vocab_size": base_vocab_size,
        "extended_vocab_size": len(tokenizer),
        "num_entity_tokens": len(entities),
        "num_relation_tokens": 5 * len(relations),
        "extra_tokens": token_info["extra_tokens"],
        "token_mapping_sha256": token_mapping["sha256"],
        "prompt_template": "The description of <mask> is that {description}",
        "random_spans_per_replica": args.random_spans_per_replica,
        "nominal_split_replicas": args.nominal_split_replicas,
        "examples_per_entity": (
            (1 + args.random_spans_per_replica)
            * args.nominal_split_replicas
        ),
        "num_training_examples": len(corpus),
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "max_length": args.max_length,
        "loss": {
            "name": "cross_entropy",
            "candidate_space": "entity_tokens_only",
            "num_candidates": len(entities),
            "label_smoothing": args.label_smoothing,
        },
        "trainable_scope": "complete_tied_word_embedding_table_only",
        "trainable_parameters": trainable_parameters,
        "embedding_shape": list(embedding_weight.shape),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": list(optimizer.defaults["betas"]),
            "epsilon": args.adam_epsilon,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "name": "cosine_with_warmup",
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        },
        "seed": args.seed,
        "parameter_dtype": "float32",
        "new_embedding_initialization": "independent_model_normal",
        "checkpoint_selection": "best_rounded_training_mrr",
        "started_at": utc_now(),
        "history": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(manifest, manifest_path)
    atomic_json_dump(token_mapping, output_dir / "kg_token_mapping.json")

    print(
        f"dataset={name} entities={len(entities):,} "
        f"relations={len(relations):,} examples={len(corpus):,} "
        f"examples_per_entity={manifest['examples_per_entity']} "
        f"steps_per_epoch={steps_per_epoch:,} total_steps={total_steps:,} "
        f"trainable_parameters={trainable_parameters:,}"
    )

    best_mrr = None
    best_epoch = None
    history = []
    global_step = 0
    training_started = time.monotonic()
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        batch_losses = []
        epoch_ranks = []
        progress = tqdm(loader, desc=f"CCA pretraining {name} epoch {epoch}")
        for batch in progress:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            mask_positions = batch.pop("mask_positions")
            entity_labels = batch.pop("entity_labels")
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss, ranks = cca_entity_prediction_loss_and_ranks(
                logits=logits,
                mask_positions=mask_positions,
                entity_labels=entity_labels,
                entity_token_ids=entity_token_ids,
                label_smoothing=args.label_smoothing,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss for {name} at epoch {epoch}, "
                    f"step {global_step + 1}"
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            loss_value = float(loss.detach().cpu())
            batch_losses.append(loss_value)
            epoch_ranks.extend(ranks.detach().cpu().tolist())
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                step=global_step,
            )

        mean_loss = sum(batch_losses) / len(batch_losses)
        metrics = ranks_to_training_metrics(epoch_ranks, mean_loss)
        learning_rate = scheduler.get_last_lr()[0]
        record = {
            "epoch": epoch,
            "mean_training_loss": mean_loss,
            "rounded_training_metrics": metrics,
            "global_step": global_step,
            "learning_rate": learning_rate,
        }
        history.append(record)
        improved = training_mrr_improved(metrics["MRR"], best_mrr)
        if improved:
            best_mrr = metrics["MRR"]
            best_epoch = epoch
            save_best_checkpoint(
                model=model,
                tokenizer=tokenizer,
                token_mapping=token_mapping,
                output_dir=output_dir,
            )
        manifest.update(
            {
                "history": history,
                "global_step": global_step,
                "best_training_mrr": best_mrr,
                "best_epoch": best_epoch,
                "last_saved_at": utc_now(),
            }
        )
        atomic_json_dump(manifest, manifest_path)
        print(
            f"dataset={name} epoch={epoch} loss={metrics['loss']:.2f} "
            f"MRR={metrics['MRR']:.4f} H@1={metrics['H@1']:.2f} "
            f"best_epoch={best_epoch}"
        )

    if global_step != total_steps:
        raise RuntimeError(
            f"Training stopped at step {global_step}, expected {total_steps}"
        )
    manifest.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "elapsed_seconds": time.monotonic() - training_started,
            "history": history,
            "global_step": global_step,
            "best_training_mrr": best_mrr,
            "best_epoch": best_epoch,
        }
    )
    atomic_json_dump(manifest, manifest_path)
    print(
        f"dataset={name} status=complete best_epoch={best_epoch} "
        f"best_training_mrr={best_mrr:.4f} checkpoint={output_dir}"
    )
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_dir


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create a RoBERTa entity-description checkpoint using the exact "
            "checkpoint-pretraining recipe from CCA N-BERT."
        )
    )
    parser.add_argument("--dataset_root", default="bert_datasets")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", default=None)
    selection.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--subset", default="0.0")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_root", default="checkpoints_roberta_cca")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.8)
    parser.add_argument("--random_spans_per_replica", type=int, default=10)
    parser.add_argument("--nominal_split_replicas", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args, parser):
    for name in ("num_epochs", "batch_size", "max_length"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label_smoothing must be in [0, 1)")
    if args.random_spans_per_replica < 0:
        parser.error("--random_spans_per_replica cannot be negative")
    if args.nominal_split_replicas < 1:
        parser.error("--nominal_split_replicas must be at least 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")
    if args.adam_epsilon <= 0:
        parser.error("--adam_epsilon must be positive")
    if args.weight_decay < 0:
        parser.error("--weight_decay cannot be negative")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    paths = resolve_dataset_paths(
        dataset_root=args.dataset_root,
        datasets=[args.dataset] if args.dataset is not None else args.datasets,
        subset=args.subset,
    )
    print(
        f"datasets={len(paths)} device={device} base_model={args.model_name} "
        f"recipe=CCA_N-BERT"
    )
    checkpoints = []
    for path in paths:
        set_seed(args.seed)
        checkpoints.append(train_one_dataset(args, path, device))
    print("Created checkpoints:")
    for checkpoint in checkpoints:
        print(checkpoint)


if __name__ == "__main__":
    main()