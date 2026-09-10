"""Continue pretraining RoBERTa with an N-BERT-style KG vocabulary.

Each dataset starts from the same base RoBERTa model and adds deterministic
entity tokens (``[E_i]``) and relation boundary tokens
(``[R_i_SEP1]`` ... ``[R_i_SEP5]``). The default objective mirrors the BERT
checkpoint stage: every entity token is explicitly predicted from its
description, with cross entropy restricted to valid entity-token candidates,
and no graph edges are read. An optional combined objective also uses training
edges for explicit head/tail prediction and relation metadata for auxiliary
dynamic full-vocabulary MLM. Validation and test edges are never read.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from dicee.scripts.bert_bce_link_prediction import (
    read_support,
    truncate_desc,
)

DEFAULT_CHECKPOINT_FOLDERS = {
    "UMLS": "umls",
    "codex-s": "codex-s",
    "fb15k-237-sem": "fb15k-237",
    "nell-995-sem": "nell-995-h100",
    "wn18rr-sem": "wn18rr-cp",
}

CHECKPOINT_FORMAT = "roberta_nbert_entity_prediction_v3"
TRAINING_STATE_VERSION = 4
PARAMETER_DTYPE = torch.float32
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ENTITY_DESCRIPTION_OBJECTIVE = "entity_description"
COMBINED_OBJECTIVE = "entity_description_and_links"
OBJECTIVE_NAMES = {
    ENTITY_DESCRIPTION_OBJECTIVE: "forced_entity_prediction_from_description",
    COMBINED_OBJECTIVE: "forced_entity_prediction_with_relation_mlm",
}

DATASET_ALIASES = {
    "umls": "UMLS",
    "codex-s": "codex-s",
    "fb15k-237": "fb15k-237-sem",
    "fb15k-237-sem": "fb15k-237-sem",
    "nell-995": "nell-995-sem",
    "nell-995-h100": "nell-995-sem",
    "nell-995-sem": "nell-995-sem",
    "wn18rr": "wn18rr-sem",
    "wn18rr-cp": "wn18rr-sem",
    "wn18rr-sem": "wn18rr-sem",
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary_path, path)


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_triples(path):
    triples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 3:
                parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid triple at {path}:{line_number}: {stripped!r}"
                )
            triples.append(tuple(parts))
    return triples


def is_dataset_path(path):
    path = Path(path)
    return all(
        candidate.is_file()
        for candidate in (
            path / "train.txt",
            path / "support" / "entity.json",
            path / "support" / "relation.json",
        )
    )


def resolve_dataset_paths(dataset_root, datasets=None, subset="0.0"):
    """Resolve selected datasets or discover every dataset under a root."""
    requested_root = Path(dataset_root).expanduser()
    root_candidates = [requested_root]
    if not requested_root.is_absolute():
        root_candidates.append(PROJECT_ROOT / requested_root)
    dataset_root = next(
        (candidate for candidate in root_candidates if candidate.is_dir()),
        requested_root,
    )

    if datasets:
        candidates = []
        for value in datasets:
            supplied = Path(value).expanduser()
            supplied_candidates = [supplied]
            if not supplied.is_absolute():
                supplied_candidates.append(PROJECT_ROOT / supplied)
            existing_supplied = next(
                (
                    candidate
                    for candidate in supplied_candidates
                    if candidate.exists()
                ),
                None,
            )

            if existing_supplied is not None:
                candidate = existing_supplied
            else:
                candidate = dataset_root / value
                if not candidate.exists():
                    canonical_name = DATASET_ALIASES.get(value.casefold())
                    if canonical_name is not None:
                        candidate = dataset_root / canonical_name
            if not is_dataset_path(candidate):
                candidate = candidate / subset
            if not is_dataset_path(candidate):
                raise FileNotFoundError(
                    f"Could not find train/support files for dataset {value!r} "
                    f"(looked under {candidate})."
                )
            candidates.append(candidate)
    else:
        if not dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset root does not exist: {dataset_root}"
            )
        candidates = [
            child / subset
            for child in sorted(dataset_root.iterdir())
            if child.is_dir() and is_dataset_path(child / subset)
        ]

    if not candidates:
        raise FileNotFoundError(
            f"No datasets with subset {subset!r} found under {dataset_root}."
        )

    resolved = []
    seen = set()
    for path in candidates:
        normalized = path.resolve()
        if normalized not in seen:
            resolved.append(normalized)
            seen.add(normalized)
    return resolved


def dataset_name(dataset_path):
    path = Path(dataset_path)
    try:
        float(path.name)
    except ValueError:
        return path.name
    return path.parent.name


def checkpoint_folder_name(name):
    return DEFAULT_CHECKPOINT_FOLDERS.get(
        name,
        name.removesuffix("-sem").lower(),
    )


def collapse_whitespace(value):
    return " ".join(str(value).split())


def validate_triples(triples, entities, relations, dataset_path):
    missing_entities = set()
    missing_relations = set()
    for head, relation, tail in triples:
        if head not in entities:
            missing_entities.add(head)
        if tail not in entities:
            missing_entities.add(tail)
        if relation not in relations:
            missing_relations.add(relation)

    if missing_entities or missing_relations:
        details = []
        if missing_entities:
            details.append(f"missing entities={sorted(missing_entities)[:10]}")
        if missing_relations:
            details.append(
                f"missing relations={sorted(missing_relations)[:10]}"
            )
        raise ValueError(
            f"Support metadata is incomplete for {dataset_path}: "
            + "; ".join(details)
        )


def load_pretraining_triples(
    source_path,
    entities,
    relations,
    pretraining_objective,
):
    """Read train edges only when the selected objective consumes them."""
    if pretraining_objective == ENTITY_DESCRIPTION_OBJECTIVE:
        return []
    if pretraining_objective != COMBINED_OBJECTIVE:
        raise ValueError(
            f"Unsupported pretraining objective: {pretraining_objective!r}"
        )
    triples = read_triples(Path(source_path) / "train.txt")
    validate_triples(triples, entities, relations, source_path)
    return triples


def get_kg_tokens(entities, relations):
    """Return tokens in the same deterministic order used downstream."""
    tokens = [entity["name"] for entity in entities.values()]
    for relation in relations.values():
        tokens.extend(
            [
                relation["sep1"],
                relation["sep2"],
                relation["sep3"],
                relation["sep4"],
                relation["sep5"],
            ]
        )
    if len(tokens) != len(set(tokens)):
        raise ValueError("Generated KG vocabulary contains duplicate tokens.")
    return tokens


def build_token_mapping(tokenizer, entities, relations, base_vocab_size):
    entity_mapping = {
        entity_id: {
            "token": entity["name"],
            "token_id": tokenizer.convert_tokens_to_ids(entity["name"]),
            "raw_name": entity["raw_name"],
        }
        for entity_id, entity in entities.items()
    }
    relation_mapping = {}
    for relation_id, relation in relations.items():
        tokens = [
            relation["sep1"],
            relation["sep2"],
            relation["sep3"],
            relation["sep4"],
            relation["sep5"],
        ]
        relation_mapping[relation_id] = {
            "tokens": tokens,
            "token_ids": tokenizer.convert_tokens_to_ids(tokens),
            "name": relation["name"],
        }

    mapping = {
        "version": 1,
        "strategy": "nbert_extended_vocabulary",
        "base_vocab_size": base_vocab_size,
        "extended_vocab_size": len(tokenizer),
        "num_added_tokens": len(tokenizer) - base_vocab_size,
        "entities": entity_mapping,
        "relations": relation_mapping,
    }
    canonical_mapping = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
    )
    mapping["sha256"] = hashlib.sha256(
        canonical_mapping.encode("utf-8")
    ).hexdigest()
    return mapping


def entity_metadata_text(entity):
    raw_name = collapse_whitespace(
        entity.get("raw_name") or entity["name"]
    )
    description = collapse_whitespace(entity.get("desc") or "")
    values = [entity["name"], "Knowledge graph entity:", raw_name]
    if description:
        values.extend(["Entity description:", description])
    return " ".join(values)


def entity_description_prediction_text(entity):
    """Render the entity-description prompt used by BERT pretraining."""
    description = collapse_whitespace(entity.get("desc") or "")
    if not description:
        description = collapse_whitespace(
            entity.get("raw_name") or entity["name"]
        )
    return " ".join(
        ["The description of", entity["name"], "is", description]
    )


def relation_metadata_text(relation):
    return " ".join(
        [
            relation["sep1"],
            "Knowledge graph relation:",
            relation["name"],
            relation["sep2"],
            "head-relation boundary",
            relation["sep3"],
            "relation-tail boundary",
            relation["sep4"],
            "head-description boundary",
            relation["sep5"],
            "tail-description boundary",
        ]
    )


def tail_prediction_text(
    triple,
    entities,
    relations,
    max_seq_length,
):
    """Render a tail query without leaking the target description."""
    head_id, relation_id, tail_id = triple
    head = entities[head_id]
    relation = relations[relation_id]
    tail = entities[tail_id]
    return " ".join(
        [
            "Tail prediction.",
            relation["sep1"],
            head["name"],
            relation["sep2"],
            relation["name"],
            relation["sep3"],
            tail["name"],
            relation["sep4"],
            truncate_desc(head["desc"], max_seq_length),
        ]
    )


def head_prediction_text(
    triple,
    entities,
    relations,
    max_seq_length,
):
    """Render a head query without leaking the target description."""
    head_id, relation_id, tail_id = triple
    head = entities[head_id]
    relation = relations[relation_id]
    tail = entities[tail_id]
    return " ".join(
        [
            "Head prediction.",
            relation["sep1"],
            tail["name"],
            relation["sep2"],
            relation["name"],
            relation["sep3"],
            head["name"],
            relation["sep4"],
            truncate_desc(tail["desc"], max_seq_length),
        ]
    )


class KnowledgeGraphTokenMLMDataset(Dataset):
    """Lazily render selected KG pretraining examples."""

    def __init__(
        self,
        triples,
        entities,
        relations,
        max_length,
        directions=("tail", "head"),
        include_metadata=True,
        objective=ENTITY_DESCRIPTION_OBJECTIVE,
    ):
        if objective not in OBJECTIVE_NAMES:
            raise ValueError(f"Unsupported pretraining objective: {objective!r}")
        invalid = set(directions).difference({"head", "tail"})
        if invalid:
            raise ValueError(f"Unsupported directions: {sorted(invalid)}")
        if objective == COMBINED_OBJECTIVE and not directions:
            raise ValueError("At least one prediction direction is required.")
        if objective == ENTITY_DESCRIPTION_OBJECTIVE and not include_metadata:
            raise ValueError(
                "Entity-description pretraining requires entity metadata."
            )

        self.triples = triples
        self.entities = entities
        self.relations = relations
        self.max_length = max_length
        self.objective = objective
        self.directions = (
            tuple(directions) if objective == COMBINED_OBJECTIVE else ()
        )
        self.include_metadata = include_metadata
        self.entity_ids = list(entities)
        self.relation_ids = list(relations)
        self.entity_metadata_count = (
            len(self.entity_ids) if include_metadata else 0
        )
        self.relation_metadata_count = (
            len(self.relation_ids)
            if include_metadata and objective == COMBINED_OBJECTIVE
            else 0
        )
        self.metadata_count = (
            self.entity_metadata_count + self.relation_metadata_count
        )

    def __len__(self):
        return self.metadata_count + len(self.triples) * len(self.directions)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        if index < self.entity_metadata_count:
            entity_id = self.entity_ids[index]
            entity = self.entities[entity_id]
            return {
                "example_id": index,
                "kind": "entity_metadata",
                "text": (
                    entity_description_prediction_text(entity)
                    if self.objective == ENTITY_DESCRIPTION_OBJECTIVE
                    else entity_metadata_text(entity)
                ),
                "target_token": entity["name"],
                "target_occurrence": 0,
            }

        if index < self.metadata_count:
            relation_id = self.relation_ids[
                index - self.entity_metadata_count
            ]
            return {
                "example_id": index,
                "kind": "relation_metadata",
                "text": relation_metadata_text(self.relations[relation_id]),
            }

        triple_example_index = index - self.metadata_count
        triple_index, direction_index = divmod(
            triple_example_index,
            len(self.directions),
        )
        triple = self.triples[triple_index]
        direction = self.directions[direction_index]
        if direction == "tail":
            text = tail_prediction_text(
                triple,
                self.entities,
                self.relations,
                self.max_length,
            )
            target_token = self.entities[triple[2]]["name"]
        else:
            text = head_prediction_text(
                triple,
                self.entities,
                self.relations,
                self.max_length,
            )
            target_token = self.entities[triple[0]]["name"]
        return {
            "example_id": index,
            "kind": f"{direction}_prediction",
            "text": text,
            "target_token": target_token,
            # The query target is the second entity token in either prompt.
            # Selecting the last occurrence also handles self-loop triples.
            "target_occurrence": -1,
        }


class ExtendedVocabularyMLMCollator:
    """Create forced entity targets and reproducible auxiliary MLM targets."""

    def __init__(
        self,
        tokenizer,
        kg_token_ids,
        max_length,
        mlm_probability,
        seed,
        epoch,
        mask_kg_tokens=True,
    ):
        if tokenizer.mask_token_id is None:
            raise ValueError("The tokenizer must define a mask token.")
        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must define a padding token.")
        self.tokenizer = tokenizer
        self.kg_token_ids = {int(token_id) for token_id in kg_token_ids}
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.seed = seed
        self.epoch = epoch
        self.mask_kg_tokens = mask_kg_tokens

    def kg_token_positions(self, input_ids):
        positions = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.kg_token_ids:
            positions |= input_ids.eq(token_id)
        return positions

    def mask_target_row(
        self,
        input_ids,
        target_token,
        target_occurrence,
    ):
        """Replace one requested KG token and supervise only that position."""
        target_id = int(
            self.tokenizer.convert_tokens_to_ids(target_token)
        )
        if target_id not in self.kg_token_ids:
            raise ValueError(
                f"Forced target is not a registered KG token: {target_token!r}"
            )
        positions = input_ids.eq(target_id).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError(
                "Forced target token is absent after tokenization/truncation: "
                f"{target_token!r}"
            )
        try:
            target_position = positions[int(target_occurrence)]
        except IndexError as error:
            raise ValueError(
                f"Target occurrence {target_occurrence} is unavailable for "
                f"{target_token!r}; found {positions.numel()} occurrence(s)."
            ) from error

        labels = torch.full_like(input_ids, -100)
        labels[target_position] = target_id
        input_ids[target_position] = self.tokenizer.mask_token_id
        return input_ids, labels

    def mask_row(
        self,
        input_ids,
        special_tokens_mask,
        attention_mask,
        example_id,
    ):
        labels = input_ids.clone()
        kg_positions = self.kg_token_positions(input_ids)
        excluded = special_tokens_mask.bool() | ~attention_mask.bool()
        if self.mask_kg_tokens:
            excluded &= ~kg_positions
        else:
            excluded |= kg_positions
        eligible = ~excluded

        generator = torch.Generator()
        generator.manual_seed(
            self.seed + self.epoch * 1_000_003 + int(example_id)
        )
        probabilities = torch.full(
            input_ids.shape,
            self.mlm_probability,
            dtype=torch.float,
        )
        probabilities.masked_fill_(~eligible, 0.0)
        masked = torch.bernoulli(
            probabilities,
            generator=generator,
        ).bool()

        eligible_positions = eligible.nonzero(as_tuple=False).flatten()
        if not masked.any() and eligible_positions.numel() > 0:
            selected = torch.randint(
                eligible_positions.numel(),
                (1,),
                generator=generator,
            ).item()
            masked[eligible_positions[selected]] = True

        labels[~masked] = -100
        replace_with_mask = (
            torch.rand(input_ids.shape, generator=generator) < 0.8
        ) & masked
        input_ids[replace_with_mask] = self.tokenizer.mask_token_id
        replace_with_random = (
            torch.rand(input_ids.shape, generator=generator) < 0.5
        ) & masked & ~replace_with_mask
        random_tokens = torch.randint(
            len(self.tokenizer),
            input_ids.shape,
            generator=generator,
        )
        input_ids[replace_with_random] = random_tokens[replace_with_random]
        return input_ids, labels

    def __call__(self, examples):
        encoded = self.tokenizer(
            [example["text"] for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        special_tokens_mask = encoded.pop("special_tokens_mask")
        labels = torch.full_like(input_ids, -100)
        forced_entity_rows = torch.zeros(len(examples), dtype=torch.bool)
        for row, example in enumerate(examples):
            if "target_token" in example:
                forced_entity_rows[row] = True
                input_ids[row], labels[row] = self.mask_target_row(
                    input_ids=input_ids[row].clone(),
                    target_token=example["target_token"],
                    target_occurrence=example["target_occurrence"],
                )
            else:
                input_ids[row], labels[row] = self.mask_row(
                    input_ids=input_ids[row].clone(),
                    special_tokens_mask=special_tokens_mask[row],
                    attention_mask=attention_mask[row],
                    example_id=example["example_id"],
                )
        encoded["input_ids"] = input_ids
        encoded["labels"] = labels
        encoded["forced_entity_rows"] = forced_entity_rows
        return encoded


def entity_restricted_pretraining_loss(
    logits,
    labels,
    forced_entity_rows,
    entity_token_ids,
):
    """Use entity-only classes for forced targets and full MLM elsewhere."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits and labels have incompatible shapes")
    if forced_entity_rows.shape != (logits.shape[0],):
        raise ValueError("forced_entity_rows must have one value per batch row")

    entity_token_ids = torch.as_tensor(
        entity_token_ids,
        dtype=torch.long,
        device=logits.device,
    )
    if entity_token_ids.ndim != 1 or entity_token_ids.numel() < 1:
        raise ValueError("entity_token_ids must be a non-empty vector")
    if torch.unique(entity_token_ids).numel() != entity_token_ids.numel():
        raise ValueError("entity_token_ids must be unique")
    if entity_token_ids.min() < 0 or entity_token_ids.max() >= logits.shape[-1]:
        raise ValueError("entity_token_ids contain an out-of-vocabulary ID")

    forced_entity_rows = forced_entity_rows.to(
        device=logits.device,
        dtype=torch.bool,
    )
    total_loss = logits.new_zeros((), dtype=torch.float32)
    supervised_positions = 0

    if forced_entity_rows.any():
        forced_labels = labels[forced_entity_rows]
        forced_positions = forced_labels.ne(-100)
        counts = forced_positions.sum(dim=1)
        if not torch.all(counts.eq(1)):
            raise ValueError(
                "Every forced entity row must contain exactly one target label"
            )

        target_token_ids = forced_labels[forced_positions]
        target_matches = target_token_ids[:, None].eq(entity_token_ids[None, :])
        if not torch.all(target_matches.sum(dim=1).eq(1)):
            invalid = target_token_ids[
                target_matches.sum(dim=1).ne(1)
            ].detach().cpu().tolist()
            raise ValueError(
                "Forced target labels are not registered entity tokens: "
                f"{invalid[:10]}"
            )

        forced_logits = logits[forced_entity_rows][forced_positions]
        candidate_logits = forced_logits.index_select(-1, entity_token_ids)
        target_classes = target_matches.to(torch.long).argmax(dim=1)
        total_loss = total_loss + F.cross_entropy(
            candidate_logits.float(),
            target_classes,
            reduction="sum",
        )
        supervised_positions += int(target_classes.numel())

    auxiliary_rows = ~forced_entity_rows
    if auxiliary_rows.any():
        auxiliary_labels = labels[auxiliary_rows]
        auxiliary_positions = auxiliary_labels.ne(-100)
        auxiliary_count = int(auxiliary_positions.sum().item())
        if auxiliary_count:
            total_loss = total_loss + F.cross_entropy(
                logits[auxiliary_rows].float().reshape(-1, logits.shape[-1]),
                auxiliary_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            supervised_positions += auxiliary_count

    if supervised_positions == 0:
        raise ValueError("The batch contains no supervised MLM positions")
    return total_loss / supervised_positions


def resolve_compute_dtype(dtype_name, device):
    """Choose an autocast dtype while retaining FP32 model parameters."""
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


def build_optimizer(
    model,
    learning_rate,
    beta1,
    beta2,
    epsilon,
    weight_decay,
):
    """Build the Adam optimizer from the RoBERTa pretraining recipe."""
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return Adam(
        parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
    )


def resolve_training_plan(args, updates_per_epoch):
    """Resolve the linear-decay horizon and the epochs needed to reach it."""
    if updates_per_epoch < 1:
        raise ValueError("updates_per_epoch must be at least 1")
    total_updates = (
        args.max_steps
        if args.max_steps is not None
        else updates_per_epoch * args.num_epochs
    )
    if total_updates <= args.warmup_steps:
        raise ValueError(
            "The training schedule must contain a decay phase: total optimizer "
            f"steps ({total_updates}) must exceed --warmup_steps "
            f"({args.warmup_steps}). Pass --max_steps with the complete "
            "RoBERTa training horizon."
        )
    planned_epochs = math.ceil(total_updates / updates_per_epoch)
    return total_updates, planned_epochs


def checkpoint_signature(
    args,
    source_path,
    corpus,
    token_mapping,
    total_updates,
    compute_dtype,
):
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "base_model": args.model_name,
        "dataset_path": str(source_path),
        "objective": OBJECTIVE_NAMES[args.pretraining_objective],
        "directions": list(corpus.directions),
        "include_metadata": corpus.include_metadata,
        "mask_kg_tokens": (
            args.mask_kg_tokens
            if args.pretraining_objective == COMBINED_OBJECTIVE
            else None
        ),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "total_optimizer_steps": total_updates,
        "max_length": args.max_length,
        "mlm_probability": (
            args.mlm_probability
            if args.pretraining_objective == COMBINED_OBJECTIVE
            else None
        ),
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "scheduler": "linear_warmup_then_decay",
        "parameter_dtype": "float32",
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "seed": args.seed,
        "num_examples": len(corpus),
        "base_vocab_size": token_mapping["base_vocab_size"],
        "extended_vocab_size": token_mapping["extended_vocab_size"],
        "num_added_tokens": token_mapping["num_added_tokens"],
        "forced_entity_loss_space": "entity_tokens_only",
        "num_entity_candidates": len(token_mapping["entities"]),
        "token_mapping_sha256": token_mapping["sha256"],
    }


def assert_resume_compatible(saved_signature, expected_signature):
    differences = {
        key: (saved_signature.get(key), expected_value)
        for key, expected_value in expected_signature.items()
        if saved_signature.get(key) != expected_value
    }
    if differences:
        formatted = ", ".join(
            f"{key}: saved={saved!r}, requested={requested!r}"
            for key, (saved, requested) in differences.items()
        )
        raise ValueError(
            "The existing RoBERTa pretraining state is incompatible with "
            f"this command ({formatted}). Use --overwrite to regenerate it."
        )


def save_training_checkpoint(
    model,
    tokenizer,
    output_dir,
    optimizer,
    scheduler,
    signature,
    token_mapping,
    epoch,
    next_batch,
    global_step,
    history,
    manifest,
    partial_epoch_loss_sum,
    partial_epoch_batches,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    atomic_json_dump(token_mapping, output_dir / "kg_token_mapping.json")
    atomic_torch_save(
        {
            "version": TRAINING_STATE_VERSION,
            "signature": signature,
            "epoch": epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "partial_epoch_loss_sum": partial_epoch_loss_sum,
            "partial_epoch_batches": partial_epoch_batches,
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        output_dir / "training_state.pt",
    )
    updated_manifest = dict(manifest)
    updated_manifest.update(
        {
            "status": "training",
            "last_saved_at": utc_now(),
            "global_step": global_step,
            "next_epoch": epoch + 1,
            "next_batch": next_batch,
            "history": history,
        }
    )
    atomic_json_dump(
        updated_manifest,
        output_dir / "pretraining_manifest.json",
    )


def make_dataloader(corpus, tokenizer, kg_token_ids, args, device, epoch):
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch)
    return DataLoader(
        corpus,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=ExtendedVocabularyMLMCollator(
            tokenizer=tokenizer,
            kg_token_ids=kg_token_ids,
            max_length=args.max_length,
            mlm_probability=args.mlm_probability,
            seed=args.seed,
            epoch=epoch,
            mask_kg_tokens=args.mask_kg_tokens,
        ),
    )


def restore_random_state(resume_state):
    if "python_random_state" in resume_state:
        random.setstate(resume_state["python_random_state"])
    if "torch_random_state" in resume_state:
        torch.set_rng_state(resume_state["torch_random_state"])
    if (
        torch.cuda.is_available()
        and resume_state.get("cuda_random_state") is not None
    ):
        torch.cuda.set_rng_state_all(resume_state["cuda_random_state"])


def train_one_dataset(args, source_path, device, compute_dtype):
    name = dataset_name(source_path)
    output_dir = (
        Path(args.output_root)
        / checkpoint_folder_name(name)
        / "roberta-pretrained"
    ).resolve()
    manifest_path = output_dir / "pretraining_manifest.json"
    state_path = output_dir / "training_state.pt"
    existing_manifest = (
        read_json(manifest_path) if manifest_path.is_file() else None
    )

    # read_support creates the exact [E_i] and [R_i_SEPj] mapping used by LP.
    entities, relations = read_support(str(source_path))
    uses_training_edges = args.pretraining_objective == COMBINED_OBJECTIVE
    triples = load_pretraining_triples(
        source_path,
        entities,
        relations,
        args.pretraining_objective,
    )
    corpus = KnowledgeGraphTokenMLMDataset(
        triples=triples,
        entities=entities,
        relations=relations,
        max_length=args.max_length,
        directions=args.directions,
        include_metadata=args.include_metadata,
        objective=args.pretraining_objective,
    )
    batches_per_epoch = math.ceil(len(corpus) / args.batch_size)
    updates_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    total_updates, planned_epochs = resolve_training_plan(
        args,
        updates_per_epoch,
    )

    if state_path.is_file() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Incomplete checkpoint exists at {output_dir}. Use --resume "
            "or --overwrite."
        )

    resume_state = None
    if args.resume and not args.overwrite and state_path.is_file():
        resume_state = load_torch_checkpoint(state_path)
        model_source = str(output_dir)
        print(
            f"dataset={name} status=resuming global_step="
            f"{resume_state['global_step']} checkpoint={output_dir}"
        )
    else:
        model_source = args.model_name
        print(
            f"dataset={name} status=starting base_model={args.model_name} "
            f"checkpoint={output_dir}"
        )
        if args.overwrite and state_path.is_file():
            state_path.unlink()

    tokenizer = RobertaTokenizer.from_pretrained(model_source)
    base_vocab_size = (
        int(existing_manifest["base_vocab_size"])
        if resume_state is not None
        and existing_manifest
        and "base_vocab_size" in existing_manifest
        else len(tokenizer)
    )
    kg_tokens = get_kg_tokens(entities, relations)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": kg_tokens}
    )
    kg_token_ids = tokenizer.convert_tokens_to_ids(kg_tokens)
    entity_token_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(
            [entity["name"] for entity in entities.values()]
        ),
        dtype=torch.long,
        device=device,
    )
    if len(set(kg_token_ids)) != len(kg_tokens):
        raise RuntimeError("KG tokens did not receive distinct token IDs.")
    if tokenizer.unk_token_id in kg_token_ids:
        raise RuntimeError("At least one KG token maps to the unknown token.")

    token_mapping = build_token_mapping(
        tokenizer,
        entities,
        relations,
        base_vocab_size,
    )
    signature = checkpoint_signature(
        args,
        source_path,
        corpus,
        token_mapping,
        total_updates,
        compute_dtype,
    )

    if resume_state is not None:
        assert_resume_compatible(resume_state["signature"], signature)

    if (
        existing_manifest
        and existing_manifest.get("status") == "complete"
        and not args.overwrite
    ):
        assert_resume_compatible(
            existing_manifest.get("signature", {}),
            signature,
        )
        print(
            f"dataset={name} status=already_complete checkpoint={output_dir}"
        )
        return output_dir

    model = RobertaForMaskedLM.from_pretrained(
        model_source,
        dtype=PARAMETER_DTYPE,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()
    model.to(device)
    model.requires_grad_(True)
    input_vocab_size = model.get_input_embeddings().weight.shape[0]
    output_vocab_size = model.get_output_embeddings().weight.shape[0]
    if input_vocab_size != len(tokenizer) or output_vocab_size != len(tokenizer):
        raise RuntimeError(
            "Tokenizer, input embeddings, and MLM decoder vocabulary sizes "
            "do not match."
        )

    if not args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters != total_parameters:
        raise RuntimeError("Some RoBERTa parameters are unexpectedly frozen.")

    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_updates,
    )

    start_epoch = 0
    start_batch = 0
    global_step = 0
    history = []
    partial_epoch_loss_sum = 0.0
    partial_epoch_batches = 0
    if resume_state is not None:
        if resume_state.get("version") != TRAINING_STATE_VERSION:
            raise ValueError(
                "Only entity-prediction training-state version "
                f"{TRAINING_STATE_VERSION} can resume; "
                f"got version {resume_state.get('version')!r}. Use --overwrite."
            )
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = int(resume_state["epoch"])
        start_batch = int(resume_state["next_batch"])
        global_step = int(resume_state["global_step"])
        history = resume_state.get("history", [])
        partial_epoch_loss_sum = float(
            resume_state.get("partial_epoch_loss_sum", 0.0)
        )
        partial_epoch_batches = int(
            resume_state.get("partial_epoch_batches", 0)
        )
        restore_random_state(resume_state)

    manifest = {
        "status": "training",
        "objective": OBJECTIVE_NAMES[args.pretraining_objective],
        "checkpoint_format": CHECKPOINT_FORMAT,
        "base_model": args.model_name,
        "dataset": name,
        "dataset_path": str(source_path),
        "checkpoint_path": str(output_dir),
        "edge_splits_used": ["train"] if uses_training_edges else [],
        "edge_splits_not_read": (
            ["valid", "test"]
            if uses_training_edges
            else ["train", "valid", "test"]
        ),
        "support_metadata_used": ["entity.json", "relation.json"],
        "base_vocab_size": base_vocab_size,
        "extended_vocab_size": len(tokenizer),
        "num_added_tokens": len(tokenizer) - base_vocab_size,
        "num_entity_tokens": len(entities),
        "num_relation_tokens": 5 * len(relations),
        "token_mapping_sha256": token_mapping["sha256"],
        "mask_kg_tokens": (
            args.mask_kg_tokens if uses_training_edges else None
        ),
        "num_train_triples": len(triples),
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_training_examples": len(corpus),
        "forced_entity_examples_per_epoch": (
            len(entities) if args.include_metadata else 0
        ),
        "forced_link_prediction_examples_per_epoch": (
            len(triples) * len(corpus.directions)
        ),
        "auxiliary_relation_mlm_examples_per_epoch": (
            len(relations)
            if args.include_metadata and uses_training_edges
            else 0
        ),
        "loss": {
            "forced_entity_candidate_space": "entity_tokens_only",
            "num_entity_candidates": len(entity_token_ids),
            "uniform_forced_entity_loss": math.log(len(entity_token_ids)),
            "auxiliary_mlm_candidate_space": (
                "full_extended_vocabulary" if uses_training_edges else None
            ),
            "reduction": "mean_over_supervised_positions",
        },
        "directions": list(corpus.directions),
        "include_metadata": corpus.include_metadata,
        "mlm_probability": (
            args.mlm_probability if uses_training_edges else None
        ),
        "trainable_parameters": trainable_parameters,
        "total_optimizer_steps": total_updates,
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.learning_rate,
            "betas": [args.adam_beta1, args.adam_beta2],
            "epsilon": args.adam_epsilon,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "name": "linear_warmup_then_decay",
            "warmup_steps": args.warmup_steps,
            "total_steps": total_updates,
        },
        "parameter_dtype": "float32",
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "gradient_checkpointing": not args.disable_gradient_checkpointing,
        "signature": signature,
        "started_at": (
            existing_manifest.get("started_at")
            if resume_state is not None and existing_manifest
            else utc_now()
        ),
    }
    atomic_json_dump(manifest, manifest_path)
    atomic_json_dump(token_mapping, output_dir / "kg_token_mapping.json")

    print(
        f"dataset={name} objective={args.pretraining_objective} "
        f"triples_used={len(triples):,} entities={len(entities):,} "
        f"relations={len(relations):,} examples={len(corpus):,} "
        f"base_vocab={base_vocab_size:,} extended_vocab={len(tokenizer):,} "
        f"added_tokens={len(tokenizer) - base_vocab_size:,} "
        f"optimizer_steps={total_updates:,} warmup_steps={args.warmup_steps:,} "
        f"parameter_dtype=float32 compute_dtype={compute_dtype}"
    )

    optimizer.zero_grad(set_to_none=True)
    training_started = time.monotonic()
    stop_training = global_step >= total_updates
    autocast_enabled = compute_dtype != torch.float32
    for epoch in range(start_epoch, planned_epochs):
        if stop_training:
            break
        loader = make_dataloader(
            corpus,
            tokenizer,
            kg_token_ids,
            args,
            device,
            epoch,
        )
        first_batch = start_batch if epoch == start_epoch else 0
        epoch_loss_sum = (
            partial_epoch_loss_sum if epoch == start_epoch else 0.0
        )
        epoch_batches = (
            partial_epoch_batches if epoch == start_epoch else 0
        )
        model.train()
        accumulation_index = 0
        accumulation_group_size = 0
        progress = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Pretraining {name} epoch {epoch + 1}/{planned_epochs}",
        )
        next_batch = first_batch
        epoch_finished = False

        for batch_index, batch in progress:
            if batch_index < first_batch:
                continue
            if accumulation_index == 0:
                accumulation_group_size = min(
                    args.gradient_accumulation_steps,
                    len(loader) - batch_index,
                )

            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            forced_entity_rows = batch.pop("forced_entity_rows")
            with torch.autocast(
                device_type=device.type,
                dtype=compute_dtype,
                enabled=autocast_enabled,
            ):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = entity_restricted_pretraining_loss(
                    logits=logits,
                    labels=batch["labels"],
                    forced_entity_rows=forced_entity_rows,
                    entity_token_ids=entity_token_ids,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss for {name} at epoch {epoch + 1}, "
                    f"batch {batch_index + 1}: {loss.item()}"
                )
            (loss / accumulation_group_size).backward()
            loss_value = loss.detach().float().item()
            epoch_loss_sum += loss_value
            epoch_batches += 1
            accumulation_index += 1

            if accumulation_index == accumulation_group_size:
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accumulation_index = 0
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    optimizer_steps=global_step,
                )

                next_batch = batch_index + 1
                if global_step >= total_updates:
                    stop_training = True
                    epoch_finished = next_batch == len(loader)
                    break
                should_save_step = (
                    args.save_steps
                    and global_step % args.save_steps == 0
                    and next_batch < len(loader)
                )
                if should_save_step:
                    save_training_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=output_dir,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        signature=signature,
                        token_mapping=token_mapping,
                        epoch=epoch,
                        next_batch=next_batch,
                        global_step=global_step,
                        history=history,
                        manifest=manifest,
                        partial_epoch_loss_sum=epoch_loss_sum,
                        partial_epoch_batches=epoch_batches,
                    )

        if not stop_training:
            epoch_finished = True

        mean_loss = epoch_loss_sum / max(epoch_batches, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": mean_loss,
                "global_step": global_step,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_complete": epoch_finished,
            }
        )
        print(
            f"dataset={name} epoch={epoch + 1} mean_loss={mean_loss:.6f} "
            f"optimizer_steps={global_step}"
        )
        save_training_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            signature=signature,
            token_mapping=token_mapping,
            epoch=epoch + 1 if epoch_finished else epoch,
            next_batch=0 if epoch_finished else next_batch,
            global_step=global_step,
            history=history,
            manifest=manifest,
            partial_epoch_loss_sum=0.0,
            partial_epoch_batches=0,
        )
        start_batch = 0
        partial_epoch_loss_sum = 0.0
        partial_epoch_batches = 0

    if global_step != total_updates:
        raise RuntimeError(
            f"Training for {name} stopped at optimizer step {global_step}, "
            f"expected {total_updates}."
        )

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    atomic_json_dump(token_mapping, output_dir / "kg_token_mapping.json")
    completed_manifest = dict(manifest)
    completed_manifest.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "elapsed_seconds_this_process": time.monotonic() - training_started,
            "global_step": global_step,
            "history": history,
        }
    )
    atomic_json_dump(completed_manifest, manifest_path)
    if state_path.is_file() and not args.keep_optimizer_state:
        state_path.unlink()

    print(f"dataset={name} status=complete checkpoint={output_dir}")
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_dir


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Continue pretraining RoBERTa separately on each KG dataset "
            "with an N-BERT-style extended entity/relation vocabulary."
        )
    )
    parser.add_argument("--dataset_root", default="bert_datasets")
    dataset_selection = parser.add_mutually_exclusive_group()
    dataset_selection.add_argument(
        "--dataset",
        default=None,
        help=(
            "Train one dataset, specified as a name, alias, or path "
            "(for example: UMLS or bert_datasets/UMLS/0.0)."
        ),
    )
    dataset_selection.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names or paths; omit to discover every dataset.",
    )
    parser.add_argument("--subset", default="0.0")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_root", default="checkpoints")
    parser.add_argument(
        "--pretraining_objective",
        choices=[ENTITY_DESCRIPTION_OBJECTIVE, COMBINED_OBJECTIVE],
        default=ENTITY_DESCRIPTION_OBJECTIVE,
        help=(
            "Use entity_description to match BERT checkpoint creation from "
            "support/entity.json only. entity_description_and_links also "
            "uses train.txt head/tail targets and relation-metadata MLM."
        ),
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=["head", "tail"],
        default=["tail", "head"],
        help="Prediction directions for entity_description_and_links only.",
    )
    parser.add_argument(
        "--include_metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mask_kg_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow KG-specific tokens in auxiliary relation-metadata MLM. "
            "Forced entity targets are always masked."
        ),
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Training epochs used only when --max_steps is omitted.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=(
            "Exact optimizer-step horizon for linear decay. Required when "
            "the epoch-derived horizon does not exceed --warmup_steps."
        ),
    )
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=24_000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Refresh resumable state every N optimizer steps; "
            "0 saves only at epoch boundaries."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float32"],
        default="auto",
        help=(
            "Forward-pass autocast dtype. Model parameters and optimizer "
            "state always remain float32."
        ),
    )
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate an existing ordinary or extended-vocabulary checkpoint.",
    )
    parser.add_argument("--keep_optimizer_state", action="store_true")
    return parser


def validate_args(args, parser):
    for name in (
        "max_length",
        "batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if args.save_steps < 0:
        parser.error("--save_steps cannot be negative")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max_steps must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    if not 0.0 <= args.adam_beta1 < 1.0:
        parser.error("--adam_beta1 must be in [0, 1)")
    if not 0.0 <= args.adam_beta2 < 1.0:
        parser.error("--adam_beta2 must be in [0, 1)")
    if args.adam_epsilon <= 0:
        parser.error("--adam_epsilon must be positive")
    if args.weight_decay < 0:
        parser.error("--weight_decay cannot be negative")
    if args.warmup_steps < 0:
        parser.error("--warmup_steps cannot be negative")
    if not 0.0 < args.mlm_probability < 1.0:
        parser.error("--mlm_probability must be between 0 and 1")
    if args.max_grad_norm <= 0:
        parser.error("--max_grad_norm must be positive")
    if len(set(args.directions)) != len(args.directions):
        parser.error("--directions cannot contain duplicates")
    if (
        args.pretraining_objective == ENTITY_DESCRIPTION_OBJECTIVE
        and not args.include_metadata
    ):
        parser.error(
            "--no-include_metadata cannot be used with "
            "--pretraining_objective entity_description"
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    compute_dtype = resolve_compute_dtype(args.dtype, device)
    paths = resolve_dataset_paths(
        dataset_root=args.dataset_root,
        datasets=[args.dataset] if args.dataset is not None else args.datasets,
        subset=args.subset,
    )
    print(
        f"datasets={len(paths)} device={device} "
        f"parameter_dtype={PARAMETER_DTYPE} compute_dtype={compute_dtype} "
        f"base_model={args.model_name}"
    )

    checkpoints = []
    for path in paths:
        set_seed(args.seed)
        checkpoints.append(
            train_one_dataset(
                args=args,
                source_path=path,
                device=device,
                compute_dtype=compute_dtype,
            )
        )

    print("Created/reused checkpoints:")
    for checkpoint in checkpoints:
        print(checkpoint)


if __name__ == "__main__":
    main()
