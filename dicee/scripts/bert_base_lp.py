"""Out-of-the-box BERT-base-cased baseline for KG link prediction.

The model is a KG-BERT-style cross encoder.  It learns to distinguish true
``(head, relation, tail)`` triples from randomly corrupted triples with binary
cross entropy.  At evaluation time every entity is tried as the missing head
and tail, and standard filtered MRR/Hits metrics are reported.
Only the split selected by ``--eval_split`` is evaluated. All three splits
still contribute known facts for filtered ranking.

Expected dataset layout::

    dataset/
      train.txt            # tab/space separated: head relation tail
      valid.txt
      test.txt
      support/             # optional textual metadata
        entity.json
        relation.json

The support JSON files may map IDs either to strings or to dictionaries with
``name`` and optional ``desc`` fields.  If they are absent, IDs are converted
to readable text by replacing underscores and slashes with spaces.
Entity names and relation names are always retained. Descriptions share the
remaining token budget; increase ``--max_length`` if the names alone do not fit.

Example::

    python -m dicee.scripts.bert_base_lp \
        --dataset_path bert_datasets/UMLS/0.0 \
        --output_dir runs/umls_bert_base_cased \
        --epochs 3 --batch_size 16

To continue an interrupted run, pass either its run directory or its
``training_state.pt`` file to ``--resume_from_checkpoint``.  ``--epochs`` is
the total epoch budget, not the number of additional epochs.

Install the optional runtime dependency with ``pip install transformers``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm

from dicee.scripts.bert_bce_link_prediction import (
    atomic_torch_save,
    dataset_run_name,
    load_torch_checkpoint,
)

Triple = tuple[str, str, str]
LabeledTriple = tuple[Triple, float]
TRAINING_RECIPE = "bert_base_fixed_epoch_budgeted_bce_v2"


@dataclass(frozen=True)
class TextFields:
    name: str
    description: str = ""

    def __str__(self) -> str:
        return f"{self.name}. {self.description}" if self.description else self.name


def read_triples(path: Path) -> list[Triple]:
    """Read triples while accepting either tabs or arbitrary whitespace."""
    triples: list[Triple] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) != 3:
                fields = stripped.split()
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected three fields, got {len(fields)}"
                )
            triples.append((fields[0], fields[1], fields[2]))
    if not triples:
        raise ValueError(f"No triples found in {path}")
    return triples


def _fallback_text(identifier: str) -> str:
    return identifier.replace("_", " ").replace("/", " ").strip()


def _metadata_text(identifier: str, value: object) -> TextFields:
    if isinstance(value, str):
        return TextFields(value.strip() or _fallback_text(identifier))
    if isinstance(value, Mapping):
        name = str(value.get("name") or _fallback_text(identifier)).strip()
        description = str(value.get("desc") or "").strip()
        if description and description.casefold() != name.casefold():
            return TextFields(name, description)
        return TextFields(name)
    return TextFields(_fallback_text(identifier))


def load_text_map(
    metadata_path: Path,
    identifiers: Iterable[str],
) -> dict[str, TextFields]:
    """Load display text, falling back gracefully for missing metadata."""
    metadata: Mapping[str, object] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a JSON object in {metadata_path}")
        metadata = loaded

    return {
        identifier: _metadata_text(identifier, metadata.get(identifier))
        for identifier in identifiers
    }


def triple_to_text(
    triple: Triple,
    entity_text: Mapping[str, str | TextFields],
    relation_text: Mapping[str, str | TextFields],
) -> str:
    """Render a triple as one candidate-aware cross-encoder prompt."""
    head, relation, tail = triple
    return (
        f"Head: {entity_text[head]} "
        f"Relation: {relation_text[relation]} "
        f"Tail: {entity_text[tail]}"
    )


def corrupt_triple(
    positive: Triple,
    entities: Sequence[str],
    known_true: set[Triple],
    rng: random.Random,
) -> Triple:
    """Replace a head or tail without accidentally producing a known fact."""
    if len(entities) < 2:
        raise ValueError("Negative sampling requires at least two entities.")

    head, relation, tail = positive
    corrupt_head_first = rng.random() < 0.5
    for corrupt_head in (corrupt_head_first, not corrupt_head_first):
        start = rng.randrange(len(entities))
        for offset in range(len(entities)):
            replacement = entities[(start + offset) % len(entities)]
            candidate = (
                (replacement, relation, tail)
                if corrupt_head
                else (head, relation, replacement)
            )
            if candidate not in known_true:
                return candidate

    raise RuntimeError(
        f"No valid head or tail corruption exists for triple {positive!r}."
    )


def make_labeled_examples(
    positives: Sequence[Triple],
    entities: Sequence[str],
    known_true: set[Triple],
    negative_ratio: int,
    seed: int,
) -> list[LabeledTriple]:
    """Create shuffled positive groups while retaining each group's negatives."""
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1")
    rng = random.Random(seed)
    groups: list[list[LabeledTriple]] = []
    for positive in positives:
        group: list[LabeledTriple] = [(positive, 1.0)]
        for _ in range(negative_ratio):
            group.append(
                (corrupt_triple(positive, entities, known_true, rng), 0.0)
            )
        groups.append(group)
    rng.shuffle(groups)
    return [example for group in groups for example in group]


class BudgetedTripleEncoder:
    """Preserve all names and share spare tokens between descriptions."""

    def __init__(self, tokenizer, entity_text, relation_text, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
            raise ValueError("BERT tokenizer must define CLS and SEP tokens.")
        self.special_tokens = 2
        self.markers = [self.tokenize(label) for label in ("Head:", "Relation:", "Tail:")]
        self.entities = {key: self.fields(value) for key, value in entity_text.items()}
        self.relations = {key: self.fields(value) for key, value in relation_text.items()}

    def tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=False)

    def fields(self, value):
        fields = value if isinstance(value, TextFields) else TextFields(value)
        name = self.tokenize(fields.name)
        # No description can use more than the entire sequence budget.
        description = self.tokenizer.encode(
            f". {fields.description}" if fields.description else "",
            add_special_tokens=False, truncation=True, max_length=self.max_length,
        )
        return name, description

    def encode(self, triple: Triple):
        head, relation, tail = triple
        fields = (self.entities[head], self.relations[relation], self.entities[tail])
        required = self.special_tokens + sum(
            len(marker) + len(name)
            for marker, (name, _) in zip(self.markers, fields)
        )
        if required > self.max_length:
            raise ValueError(
                f"Triple {triple!r} needs {required} tokens for its names and "
                f"markers; increase --max_length from {self.max_length}."
            )
        remaining = self.max_length - required
        counts = [0, 0, 0]
        # Give unused space from short descriptions to longer ones, without
        # allowing any description to displace a name.
        while remaining:
            active = [i for i, (_, desc) in enumerate(fields) if counts[i] < len(desc)]
            if not active:
                break
            share = max(1, remaining // len(active))
            for i in active:
                if not remaining:
                    break
                take = min(share, remaining, len(fields[i][1]) - counts[i])
                counts[i] += take
                remaining -= take
        ids = []
        for marker, (name, description), count in zip(self.markers, fields, counts):
            ids.extend(marker + name + description[:count])
        ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "token_type_ids": [0] * len(ids),
        }

    def batch(self, triples):
        return self.tokenizer.pad(
            [self.encode(triple) for triple in triples], padding=True, return_tensors="pt"
        )


class TripleDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[LabeledTriple],
    ) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LabeledTriple:
        return self.examples[index]


@dataclass
class BatchCollator:
    encoder: BudgetedTripleEncoder

    def __call__(self, batch: Sequence[LabeledTriple]) -> dict[str, torch.Tensor]:
        triples, labels = zip(*batch)
        encoded = self.encoder.batch(triples)
        encoded["labels"] = torch.tensor(labels, dtype=torch.float32)
        return encoded


def load_model_and_tokenizer(source: str):
    """Load vanilla BERT with its standard sequence-classification head."""
    tokenizer = BertTokenizer.from_pretrained(source)
    model = BertForSequenceClassification.from_pretrained(
        source,
        num_labels=1,
    )
    return model, tokenizer


def resolve_resume_checkpoint(path: Path) -> Path:
    """Resolve a training-state file from either a file or run directory."""
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "training_state.pt"
    if not resolved.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resolved}")
    return resolved


def training_signature(args: argparse.Namespace) -> dict[str, object]:
    """Return settings which must remain fixed for an exact continuation."""
    return {
        "training_recipe": TRAINING_RECIPE,
        "dataset_path": str(Path(args.dataset_path).expanduser().resolve()),
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "negative_ratio": args.negative_ratio,
        "max_length": args.max_length,
        "seed": args.seed,
    }


def validate_resume_state(
    state: Mapping[str, object],
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    """Reject incompatible or malformed training checkpoints early."""
    if state.get("version") != 1:
        raise ValueError(
            f"Unsupported training checkpoint version in {checkpoint_path}: "
            f"{state.get('version')!r}"
        )

    saved_signature = state.get("training_signature")
    if not isinstance(saved_signature, Mapping):
        raise ValueError(
            f"Training checkpoint has no training signature: {checkpoint_path}"
        )
    current_signature = training_signature(args)
    mismatches = [
        name
        for name, value in current_signature.items()
        if saved_signature.get(name) != value
    ]
    if mismatches:
        details = ", ".join(
            f"{name}: saved={saved_signature.get(name)!r}, "
            f"current={current_signature[name]!r}"
            for name in mismatches
        )
        raise ValueError(
            f"Resume arguments do not match the checkpoint ({details})."
        )

    required_keys = {
        "epoch_completed",
        "model_state_dict",
        "optimizer_state_dict",
        "training_history",
        "python_random_state",
        "torch_random_state",
    }
    missing = sorted(required_keys.difference(state))
    if missing:
        raise ValueError(
            f"Training checkpoint is missing required fields: {', '.join(missing)}"
        )
    epoch_completed = int(state["epoch_completed"])
    history = state["training_history"]
    if epoch_completed < 0 or epoch_completed > args.epochs:
        raise ValueError(
            "Checkpoint epoch_completed must be between zero and the requested "
            f"epoch budget; got {epoch_completed} for --epochs {args.epochs}."
        )
    if not isinstance(history, Sequence) or len(history) != epoch_completed:
        history_length = len(history) if isinstance(history, Sequence) else "invalid"
        raise ValueError(
            "Checkpoint training history does not match epoch_completed: "
            f"got {history_length} records for {epoch_completed} epochs."
        )


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train(
    model: nn.Module,
    tokenizer: object,
    train_triples: Sequence[Triple],
    negative_entities: Sequence[str],
    negative_filter: set[Triple],
    entity_text: Mapping[str, str | TextFields],
    relation_text: Mapping[str, str | TextFields],
    args: argparse.Namespace,
    device: torch.device,
    resume_state: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Fine-tune BERT, checkpointing each epoch for exact continuation."""
    output_dir = Path(args.output_dir)
    training_state_path = output_dir / "training_state.pt"
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    collator = BatchCollator(
        BudgetedTripleEncoder(tokenizer, entity_text, relation_text, args.max_length)
    )
    history: list[dict[str, object]] = []
    epoch_completed = 0
    group_size = 1 + args.negative_ratio
    groups_per_batch = args.batch_size // group_size
    effective_batch_size = groups_per_batch * group_size

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        epoch_completed = int(resume_state["epoch_completed"])
        history = list(resume_state["training_history"])
        if epoch_completed < 0 or epoch_completed > args.epochs:
            raise ValueError(
                "Checkpoint epoch_completed must be between zero and the "
                f"requested epoch budget; got {epoch_completed} for "
                f"--epochs {args.epochs}."
            )
        if len(history) != epoch_completed:
            raise ValueError(
                "Checkpoint training history does not match epoch_completed: "
                f"got {len(history)} records for {epoch_completed} epochs."
            )
        random.setstate(resume_state["python_random_state"])
        torch.set_rng_state(resume_state["torch_random_state"])
        if (
            torch.cuda.is_available()
            and resume_state.get("cuda_random_state") is not None
        ):
            torch.cuda.set_rng_state_all(resume_state["cuda_random_state"])

    def save_training_state(phase: str) -> None:
        atomic_torch_save(
            training_state_path,
            {
                "version": 1,
                "phase": phase,
                "epoch_completed": epoch_completed,
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "training_history": history,
                "training_signature": training_signature(args),
                "args": vars(args),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        )
        print(
            f"saved_training_checkpoint={training_state_path} "
            f"phase={phase} epoch={epoch_completed}"
        )

    if resume_state is None:
        # This epoch-zero state also makes interruptions during epoch one resumable.
        save_training_state("training")

    for epoch in range(epoch_completed + 1, args.epochs + 1):
        train_examples = make_labeled_examples(
            train_triples,
            negative_entities,
            negative_filter,
            args.negative_ratio,
            args.seed + epoch,
        )
        train_loader = DataLoader(
            TripleDataset(train_examples),
            batch_size=effective_batch_size,
            # Groups were shuffled above; do not split positives from negatives.
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )

        model.train()
        loss_sum = 0.0
        example_count = 0
        progress = tqdm(train_loader, desc=f"train epoch {epoch}")
        for batch in progress:
            labels = batch.pop("labels").to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**move_batch(batch, device)).logits.squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * labels.numel()
            example_count += labels.numel()
            progress.set_postfix(loss=f"{loss_sum / example_count:.4f}")

        train_loss = loss_sum / max(example_count, 1)
        history.append(
            {
                "epoch": float(epoch),
                "loss": train_loss,
                "sampled_examples": example_count,
                "sampled_batches": len(train_loader),
                "validation_metrics": None,
            }
        )
        epoch_completed = epoch
        print(f"epoch={epoch} train_bce={train_loss:.6f}")
        save_training_state("training")

    save_training_state("training_complete")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return history


def score_triples(
    model: nn.Module,
    tokenizer: object,
    triples: Sequence[Triple],
    entity_text: Mapping[str, str | TextFields],
    relation_text: Mapping[str, str | TextFields],
    device: torch.device,
    max_length: int,
    batch_size: int,
    encoder: BudgetedTripleEncoder | None = None,
) -> torch.Tensor:
    """Return one plausibility logit for each supplied triple."""
    model.eval()
    if encoder is None:
        encoder = BudgetedTripleEncoder(tokenizer, entity_text, relation_text, max_length)
    scores: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(triples), batch_size):
            encoded = encoder.batch(triples[start : start + batch_size])
            logits = model(**move_batch(encoded, device)).logits.squeeze(-1)
            scores.append(logits.detach().cpu())
    return torch.cat(scores) if scores else torch.empty(0)


def filtered_rank(
    scores: torch.Tensor,
    candidates: Sequence[Triple],
    target_index: int,
    known_true: set[Triple],
) -> int:
    """Compute DICE's one-based position after filtered descending sorting."""
    filtered = scores.clone()
    for index, candidate in enumerate(candidates):
        if index != target_index and candidate in known_true:
            filtered[index] = -torch.inf

    _, sorted_indices = torch.sort(filtered, descending=True)
    target_position = (sorted_indices == target_index).nonzero(as_tuple=True)[0]
    return int(target_position.item()) + 1


def metrics_from_ranks(ranks: Sequence[float]) -> dict[str, float]:
    values = torch.tensor(ranks, dtype=torch.float32)
    if values.numel() == 0:
        raise ValueError("Cannot compute metrics without ranks.")
    return {
        "MRR": float((1.0 / values).mean().item()),
        "Hits@1": float((values <= 1).float().mean().item()),
        "Hits@3": float((values <= 3).float().mean().item()),
        "Hits@10": float((values <= 10).float().mean().item()),
        "mean_rank": float(values.mean().item()),
    }


def evaluate_link_prediction(
    model: nn.Module,
    tokenizer: object,
    eval_triples: Sequence[Triple],
    entities: Sequence[str],
    known_true: set[Triple],
    entity_text: Mapping[str, str | TextFields],
    relation_text: Mapping[str, str | TextFields],
    device: torch.device,
    max_length: int,
    candidate_batch_size: int,
    progress_description: str = "filtered evaluation",
) -> dict[str, object]:
    """Run filtered head and tail prediction against every entity."""
    entity_position = {entity: index for index, entity in enumerate(entities)}
    encoder = BudgetedTripleEncoder(tokenizer, entity_text, relation_text, max_length)
    head_ranks: list[int] = []
    tail_ranks: list[int] = []

    for head, relation, tail in tqdm(eval_triples, desc=progress_description):
        tail_candidates = [(head, relation, candidate) for candidate in entities]
        tail_scores = score_triples(
            model,
            tokenizer,
            tail_candidates,
            entity_text,
            relation_text,
            device,
            max_length,
            candidate_batch_size,
            encoder,
        )
        tail_ranks.append(
            filtered_rank(
                tail_scores, tail_candidates, entity_position[tail], known_true
            )
        )

        head_candidates = [(candidate, relation, tail) for candidate in entities]
        head_scores = score_triples(
            model,
            tokenizer,
            head_candidates,
            entity_text,
            relation_text,
            device,
            max_length,
            candidate_batch_size,
            encoder,
        )
        head_ranks.append(
            filtered_rank(
                head_scores, head_candidates, entity_position[head], known_true
            )
        )

    return {
        "both": metrics_from_ranks(head_ranks + tail_ranks),
        "head": metrics_from_ranks(head_ranks),
        "tail": metrics_from_ranks(tail_ranks),
        "num_evaluated_triples": len(eval_triples),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune bert-base-cased for filtered KG link prediction."
    )
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help=(
            "Base directory for a new run. When resuming, this may be omitted "
            "or name the original base/run directory."
        ),
    )
    parser.add_argument("--model_name", default="bert-base-cased")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate this saved Hugging Face checkpoint without training.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        help=(
            "Resume training from a training_state.pt file or the run "
            "directory containing it. --epochs is the total epoch budget."
        ),
    )
    parser.add_argument("--eval_split", choices=("valid", "test"), default="test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--candidate_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--negative_ratio", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_eval_triples",
        type=int,
        help="Evaluate only the first N triples (useful for a quick smoke run).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.checkpoint is not None and args.resume_from_checkpoint is not None:
        raise ValueError("checkpoint and resume_from_checkpoint are mutually exclusive")
    if args.output_dir is None and args.resume_from_checkpoint is None:
        raise ValueError("output_dir is required unless resuming from a checkpoint")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.batch_size < 1 or args.candidate_batch_size < 1:
        raise ValueError("batch sizes must be at least 1")
    if args.negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1")
    if args.batch_size < 1 + args.negative_ratio:
        raise ValueError("batch_size must fit a positive and all its negatives")
    if args.max_length < 8:
        raise ValueError("max_length must be at least 8")
    if args.max_eval_triples is not None and args.max_eval_triples < 1:
        raise ValueError("max_eval_triples must be at least 1")


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    validate_args(args)

    resume_checkpoint_path: Path | None = None
    resume_state: Mapping[str, object] | None = None
    requested_output_dir = args.output_dir
    if args.resume_from_checkpoint is not None:
        resume_checkpoint_path = resolve_resume_checkpoint(
            args.resume_from_checkpoint
        )
        loaded_state = load_torch_checkpoint(resume_checkpoint_path)
        if not isinstance(loaded_state, Mapping):
            raise ValueError(
                f"Training checkpoint must contain a mapping: "
                f"{resume_checkpoint_path}"
            )
        resume_state = loaded_state
        run_output_dir = resume_checkpoint_path.parent
        if requested_output_dir is not None:
            requested = requested_output_dir.expanduser().resolve()
            if requested not in {run_output_dir, run_output_dir.parent}:
                raise ValueError(
                    "output_dir must match the resumed run directory or its "
                    "original base directory."
                )
        args.output_dir = run_output_dir
        validate_resume_state(resume_state, resume_checkpoint_path, args)

    dataset_name = dataset_run_name(args.dataset_path)
    run_datetime = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{dataset_name}_bert_base_"
        f"{args.eval_split}_{run_datetime}"
    )
    if resume_checkpoint_path is None:
        args.output_dir = requested_output_dir / run_name
        suffix = 1
        while True:
            try:
                args.output_dir.mkdir(parents=True)
            except FileExistsError:
                suffix += 1
                args.output_dir = requested_output_dir / f"{run_name}_{suffix}"
            else:
                break
    print(f"run_output_dir={args.output_dir}")
    if resume_checkpoint_path is not None:
        print(f"resumed_training_checkpoint={resume_checkpoint_path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_triples = read_triples(args.dataset_path / "train.txt")
    valid_triples = read_triples(args.dataset_path / "valid.txt")
    test_triples = read_triples(args.dataset_path / "test.txt")
    all_triples = train_triples + valid_triples + test_triples
    known_true = set(all_triples)
    entities = sorted(
        {head for head, _, _ in all_triples}
        | {tail for _, _, tail in all_triples}
    )
    relations = sorted({relation for _, relation, _ in all_triples})
    support_dir = args.dataset_path / "support"
    entity_text = load_text_map(support_dir / "entity.json", entities)
    relation_text = load_text_map(support_dir / "relation.json", relations)

    source = str(args.checkpoint) if args.checkpoint else args.model_name
    model, tokenizer = load_model_and_tokenizer(source)
    model.to(device)

    history: list[dict[str, object]] = []
    if args.checkpoint is None:
        tokenizer.save_pretrained(args.output_dir)
        observed_training_entities = {
            entity
            for head, _, tail in train_triples
            for entity in (head, tail)
        }
        negative_entities = [
            entity for entity in entities if entity in observed_training_entities
        ]
        history = train(
            model,
            tokenizer,
            train_triples,
            negative_entities,
            set(train_triples),
            entity_text,
            relation_text,
            args,
            device,
            resume_state,
        )

    eval_triples = valid_triples if args.eval_split == "valid" else test_triples
    if args.max_eval_triples is not None:
        eval_triples = eval_triples[: args.max_eval_triples]
    metrics = evaluate_link_prediction(
        model,
        tokenizer,
        eval_triples,
        entities,
        known_true,
        entity_text,
        relation_text,
        device,
        args.max_length,
        args.candidate_batch_size,
        progress_description=f"filtered {args.eval_split} evaluation",
    )
    validation_metrics = metrics if args.eval_split == "valid" else None

    results = {
        "training_recipe": TRAINING_RECIPE,
        "model": str(args.checkpoint) if args.checkpoint else str(args.output_dir),
        "base_model": args.model_name,
        "eval_split": args.eval_split,
        "metrics": metrics,
        "best_validation_metrics": validation_metrics,
        "best_epoch": args.epochs if args.checkpoint is None else None,
        "epochs_ran": args.epochs if args.checkpoint is None else 0,
        "negative_filter_scope": "train",
        "negative_entity_scope": "train",
        "negative_loss_weighting": "sampled_mean",
        "checkpoint_policy": "final_epoch",
        "validation_policy": (
            "once_after_training" if args.eval_split == "valid" else "not_run"
        ),
        "rank_policy": "dice_torch_sort",
        "training_history": history,
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
