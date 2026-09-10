import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from dicee.static_funcs import intialize_model


def resolve_path(base_path, path):
    # Normalize relative CLI paths against the Dice repository root.
    if os.path.isabs(path):
        return path
    return os.path.join(base_path, path)


def dataset_run_name(dataset_path):
    """Include the dataset name when the path ends in a perturbation ratio."""
    normalized_path = os.path.normpath(dataset_path)
    path_leaf = os.path.basename(normalized_path)

    try:
        perturbation_ratio = float(path_leaf)
    except ValueError:
        return path_leaf

    if not 0.0 <= perturbation_ratio <= 1.0:
        return path_leaf

    dataset_name = os.path.basename(os.path.dirname(normalized_path))
    return f"{dataset_name}_{path_leaf}"


def create_unique_output_dir(base_output_dir, run_name):
    """Create a new run directory without ever reusing an existing one."""
    os.makedirs(base_output_dir, exist_ok=True)
    suffix = 1

    while True:
        candidate_name = run_name if suffix == 1 else f"{run_name}_{suffix}"
        output_dir = os.path.join(base_output_dir, candidate_name)
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            suffix += 1
        else:
            return output_dir


def atomic_torch_save(path, value):
    """Replace a checkpoint only after its temporary file is fully written."""
    temporary_path = f"{path}.tmp"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def load_torch_checkpoint(path):
    """Load a trusted local checkpoint across PyTorch default changes."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_json_save(path, value):
    """Atomically replace a JSON file."""
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary_path, path)


def evaluation_fingerprint(eval_triples, entity_ids):
    """Identify the ordered triples and candidates behind rank progress."""
    digest = hashlib.sha256()
    for values in (entity_ids, eval_triples):
        for value in values:
            if isinstance(value, tuple):
                value = "\0".join(value)
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\n")
        digest.update(b"\0")
    return digest.hexdigest()


def ranks_to_metrics(ranks):
    """Compute filtered link-prediction metrics from saved integer ranks."""
    ranks = torch.tensor(ranks, dtype=torch.float)
    if ranks.numel() == 0:
        raise ValueError("Cannot compute link-prediction metrics without ranks.")
    return {
        "MRR": float((1.0 / ranks).mean().item()),
        "H@1": float((ranks <= 1).float().mean().item()),
        "H@3": float((ranks <= 3).float().mean().item()),
        "H@10": float((ranks <= 10).float().mean().item()),
    }


def load_evaluation_progress(checkpoint_path, fingerprint, num_triples):
    """Restore ranks only when they belong to this exact evaluation."""
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        return [], 0

    progress = load_torch_checkpoint(checkpoint_path)
    if progress.get("fingerprint") != fingerprint:
        raise ValueError(
            "Evaluation checkpoint does not match the requested triples "
            f"and entities: {checkpoint_path}"
        )
    if progress.get("version") != 1:
        raise ValueError(
            f"Unsupported evaluation checkpoint version in "
            f"{checkpoint_path}: {progress.get('version')!r}"
        )
    if int(progress.get("num_triples", -1)) != num_triples:
        raise ValueError(
            f"Evaluation checkpoint triple count does not match: "
            f"{checkpoint_path}"
        )

    next_triple_index = int(progress["next_triple_index"])
    ranks = [int(rank) for rank in progress["ranks"]]
    if not 0 <= next_triple_index <= num_triples:
        raise ValueError(
            f"Invalid next_triple_index in {checkpoint_path}: "
            f"{next_triple_index}"
        )
    if len(ranks) != 2 * next_triple_index:
        raise ValueError(
            f"Invalid rank count in {checkpoint_path}: got {len(ranks)}, "
            f"expected {2 * next_triple_index}."
        )
    return ranks, next_triple_index


def save_evaluation_progress(
    checkpoint_path,
    fingerprint,
    next_triple_index,
    num_triples,
    ranks,
):
    """Persist completed head/tail ranks without including model weights."""
    if checkpoint_path is None:
        return
    atomic_torch_save(
        checkpoint_path,
        {
            "version": 1,
            "fingerprint": fingerprint,
            "next_triple_index": next_triple_index,
            "num_triples": num_triples,
            "ranks": ranks,
            "complete": next_triple_index == num_triples,
        },
    )


def read_triples(path):
    # Load tab-separated or whitespace-separated triples from a split file.
    triples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                parts = line.strip().split()
            if len(parts) != 3:
                raise ValueError(f"Could not parse triple line in {path}: {line!r}")
            triples.append(tuple(parts))
    return triples


def read_support(dataset_path, support_path=None):
    # Load entity/relation metadata and convert it to N-BERT special-token form.
    support_path = support_path or os.path.join(dataset_path, "support")
    entity_path = os.path.join(support_path, "entity.json")
    relation_path = os.path.join(support_path, "relation.json")
    with open(entity_path, "r", encoding="utf-8") as handle:
        entities = json.load(handle)
    with open(relation_path, "r", encoding="utf-8") as handle:
        relations = json.load(handle)

    for idx, entity_id in enumerate(entities):
        raw_name = entities[entity_id]["name"]
        entities[entity_id] = {
            "token_id": idx,
            "name": f"[E_{idx}]",
            "desc": entities[entity_id]["desc"],
            "raw_name": raw_name,
        }

    for idx, relation_id in enumerate(relations):
        relations[relation_id] = {
            "sep1": f"[R_{idx}_SEP1]",
            "sep2": f"[R_{idx}_SEP2]",
            "sep3": f"[R_{idx}_SEP3]",
            "sep4": f"[R_{idx}_SEP4]",
            "sep5": f"[R_{idx}_SEP5]",
            "name": relations[relation_id]["name"],
        }
    return entities, relations


def add_nbert_tokens(tokenizer, entities, relations):
    # Extend the tokenizer with dataset-specific entity and relation prompt tokens.
    entity_tokens = [entity["name"] for entity in entities.values()]
    tokenizer.add_special_tokens({"additional_special_tokens": entity_tokens})
    relation_tokens = []
    for relation in relations.values():
        relation_tokens.extend(
            [relation["sep1"], relation["sep2"], relation["sep3"], relation["sep4"], relation["sep5"]]
        )
    tokenizer.add_special_tokens({"additional_special_tokens": relation_tokens})


def truncate_desc(desc, max_seq_length):
    # Keep descriptions short enough that the full triple prompt fits BERT input length.
    tokens = str(desc).split()
    return " ".join(tokens[: min(max_seq_length - 8, len(tokens))])


def triple_prompt(triple, entities, relations, max_seq_length):
    # Render one triple as the textual prompt consumed by the BERT classifier.
    head_id, relation_id, tail_id = triple
    head = entities[head_id]
    relation = relations[relation_id]
    tail = entities[tail_id]
    return " ".join(
        [
            relation["sep1"],
            head["name"],
            relation["sep2"],
            relation["name"],
            relation["sep3"],
            tail["name"],
            relation["sep4"],
            truncate_desc(head["desc"], max_seq_length),
            relation["sep5"],
            truncate_desc(tail["desc"], max_seq_length),
        ]
    )


class BertTripleClassifier(nn.Module):
    def __init__(self, model_path, tokenizer):
        # Load BERT and attach a scalar binary-classification head.
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Score each prompt as a single triple plausibility logit.
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = output.pooler_output
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        return self.classifier(pooled).squeeze(-1)

def create_kge_model(model_name, 
    num_entities,  
    num_relations, 
    embedding_dim = 32, 
    random_seed=42, 
    learning_rate=0.1,
    negative_ratio=1,
    optimizer_name="Adam",
    eval_model="test"):

    # Create a KGE model based on the specified architecture.
    model_name = {
        "TransE": "Pykeen_TransE",
        "RotatE": "Pykeen_RotatE",
        "Keci": "Keci",
        "MuRE": "Pykeen_MuRE",
    }[model_name] 

    if model_name == "Keci" and embedding_dim % 2 != 0:
        raise ValueError(
            "Keci requires an even --kge_embedding_dim when p=0 and q=1."
        )

    kge_args = {
        "model": model_name,
        "embedding_dim": embedding_dim,
        "random_seed": random_seed,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "learning_rate": learning_rate,
        "scoring_technique": "NegSample",
        "neg_ratio": negative_ratio,
        "optim": optimizer_name,
        "eval_model": eval_model,
        "loss_fn": "BCELoss",
        "input_dropout_rate": 0.0,
        "hidden_dropout_rate": 0.0,
        "feature_map_dropout_rate": 0.0,
        "weight_decay": 0.0,
        "normalization": None,
        "init_param": None,
        "byte_pair_encoding": False,
        "pykeen_model_kwargs": {},
        "p": 0,
        "q": 1,
    }

    kge_model, _ = intialize_model(kge_args)
    return kge_model


CALIBRATION_STATE_KEYS = {
    "bert_log_scale",
    "bert_bias",
    "kge_log_scale",
    "kge_bias",
}


def negative_sampling_bce_loss(
    logits,
    labels,
    negative_ratio,
    weighting="balanced",
):
    """Compute sampled BCE with an explicit, scale-stable weighting policy."""
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1.")
    if weighting == "balanced":
        example_losses = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )
        # A group contains one positive and ``negative_ratio`` negatives.
        # Giving its positive weight K balances the two classes; dividing by
        # the sum of weights keeps the loss scale stable as K changes.
        example_weights = 1.0 + labels * float(negative_ratio - 1)
        return (example_losses * example_weights).sum() / example_weights.sum()
    elif weighting == "sampled":
        return F.binary_cross_entropy_with_logits(logits, labels)
    else:
        raise ValueError(
            "negative loss weighting must be 'balanced' or 'sampled'."
        )


def post_kge_parameter_update(kge_model):
    """Run a wrapped PyKEEN model's post-update constraints when present."""
    wrapped_model = getattr(kge_model, "model", None)
    callback = getattr(wrapped_model, "post_parameter_update", None)
    if callback is None:
        callback = getattr(kge_model, "post_parameter_update", None)
    if callable(callback):
        with torch.no_grad():
            callback()


def training_entity_ids(triples, entities):
    """Return support-ordered entities that occur in positive training data."""
    observed = {
        entity_id
        for head, _, tail in triples
        for entity_id in (head, tail)
    }
    missing = observed.difference(entities)
    if missing:
        missing_preview = ", ".join(sorted(missing)[:5])
        raise ValueError(
            "Training triples contain entities absent from support metadata: "
            f"{missing_preview}"
        )
    entity_ids = [entity_id for entity_id in entities if entity_id in observed]
    if not entity_ids:
        raise ValueError("No training entities are available for corruption.")
    return entity_ids


def negative_filter_triples(scope, train_triples, valid_triples, test_triples):
    """Build the truth set excluded by the requested training protocol."""
    filtered = set(train_triples)
    if scope in {"train_valid", "all"}:
        filtered.update(valid_triples)
    if scope == "all":
        filtered.update(test_triples)
    if scope not in {"train", "train_valid", "all"}:
        raise ValueError(f"Unknown negative filter scope: {scope!r}")
    return filtered


def load_joint_model_state(model, state_dict, allow_legacy_calibration):
    """Load joint weights, optionally accepting identity-calibrated legacy runs."""
    expected_keys = set(model.state_dict())
    provided_keys = set(state_dict)
    missing = expected_keys.difference(provided_keys)
    unexpected = provided_keys.difference(expected_keys)
    legacy_calibration = missing == CALIBRATION_STATE_KEYS
    if unexpected or (missing and not legacy_calibration):
        raise RuntimeError(
            "Joint checkpoint state is incompatible: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if legacy_calibration and not allow_legacy_calibration:
        raise ValueError(
            "This checkpoint predates affine branch calibration and cannot be "
            "resumed safely. It can still be loaded for evaluation with "
            "identity calibration."
        )
    if legacy_calibration:
        with torch.no_grad():
            for parameter in model.calibration_parameters():
                parameter.zero_()
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    return legacy_calibration


class JointBCEModel(nn.Module):
    def __init__(self,
    bert_model, 
    kge_model,
    initial_lambda = 0.5
    ):

        super().__init__() 

        if not 0.0 < initial_lambda < 1.0:
            raise ValueError("initial_lambda must be between 0 and 1.") 
        
        self.bert_model = bert_model
        self.kge_model = kge_model

        initial_lambda = torch.tensor(initial_lambda, dtype=torch.float32) 

        self.lambda_logit = nn.Parameter(torch.logit(initial_lambda))
        # Positive scales preserve each branch's ranking direction. The KGE
        # bias supplies the missing margin for non-positive distance scores
        # such as RotatE, while retaining compatibility with biased models
        # such as MuRE.
        self.bert_log_scale = nn.Parameter(torch.zeros(()))
        self.bert_bias = nn.Parameter(torch.zeros(()))
        self.kge_log_scale = nn.Parameter(torch.zeros(()))
        self.kge_bias = nn.Parameter(torch.zeros(()))
    
    @property
    def mixing_weight(self): 
        return torch.sigmoid(self.lambda_logit)

    @property
    def bert_scale(self):
        return torch.exp(self.bert_log_scale)

    @property
    def kge_scale(self):
        return torch.exp(self.kge_log_scale)

    def calibration_parameters(self):
        return (
            self.bert_log_scale,
            self.bert_bias,
            self.kge_log_scale,
            self.kge_bias,
        )

    def calibration_state(self):
        mixing_weight = float(self.mixing_weight.detach().cpu())
        bert_scale = float(self.bert_scale.detach().cpu())
        bert_bias = float(self.bert_bias.detach().cpu())
        kge_scale = float(self.kge_scale.detach().cpu())
        kge_bias = float(self.kge_bias.detach().cpu())
        return {
            "bert_scale": bert_scale,
            "bert_bias": bert_bias,
            "kge_scale": kge_scale,
            "kge_bias": kge_bias,
            "effective_bert_weight": mixing_weight * bert_scale,
            "effective_kge_weight": (1.0 - mixing_weight) * kge_scale,
            "joint_bias": (
                mixing_weight * bert_bias
                + (1.0 - mixing_weight) * kge_bias
            ),
        }
    
    def forward(
        self, 
        indexed_triples,
        input_ids,
        attention_mask,
        token_type_ids=None,
        return_components=False,
    ):

        raw_bert_logits = self.bert_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )

        raw_kge_logits = self.kge_model.forward_triples(indexed_triples)
        raw_kge_logits = raw_kge_logits.reshape_as(raw_bert_logits)
        
        bert_logits = self.bert_scale * raw_bert_logits + self.bert_bias
        kge_logits = self.kge_scale * raw_kge_logits + self.kge_bias
        
        lambda_ = self.mixing_weight 

        joint_logits = (
            lambda_ * bert_logits + (1 - lambda_) * kge_logits
        )
        if return_components:
            return joint_logits, bert_logits, kge_logits
        return joint_logits

def index_triple(
    triple, entity_to_index, relation_to_index
):
    head, rel, tail = triple

    head = entity_to_index[head]
    rel = relation_to_index[rel]
    tail = entity_to_index[tail]
    return head, rel, tail


class TripleBCEDataset(Dataset):
    def __init__(self, triples, entities, relations, entity_to_idx,
        relation_to_idx, max_seq_length, negative_ratio=1,
        negative_entity_ids=None, known_true_triples=None):
        # Store triples and metadata needed to create positive and corrupted examples.
        if negative_ratio < 1:
            raise ValueError("negative_ratio must be at least 1.")
        self.triples = triples
        self.entities = entities
        self.relations = relations
        if negative_entity_ids is None:
            negative_entity_ids = training_entity_ids(triples, entities)
        self.entity_ids = list(dict.fromkeys(negative_entity_ids))
        if not self.entity_ids:
            raise ValueError("negative_entity_ids cannot be empty.")
        unknown_entities = set(self.entity_ids).difference(entities)
        if unknown_entities:
            unknown_preview = ", ".join(sorted(unknown_entities)[:5])
            raise ValueError(
                "Negative entity pool contains entities absent from support: "
                f"{unknown_preview}"
            )
        self.true_triples = (
            set(triples)
            if known_true_triples is None
            else set(known_true_triples)
        )
        missing_positives = set(triples).difference(self.true_triples)
        if missing_positives:
            raise ValueError(
                "known_true_triples must contain every positive training triple."
            )
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.max_seq_length = max_seq_length
        self.negative_ratio = negative_ratio

    def __len__(self):
        # Count each true triple plus its requested number of negatives.
        return len(self.triples) * (1 + self.negative_ratio)

    def __getitem__(self, index):
        # Return a positive prompt or a randomly corrupted negative prompt.
        pos_index = index // (1 + self.negative_ratio)
        offset = index % (1 + self.negative_ratio)
        triple = self.triples[pos_index]
        label = 1.0
        if offset != 0:
            triple = self.corrupt_triple(triple)
            label = 0.0

        prompt = triple_prompt(triple, self.entities, self.relations, self.max_seq_length)

        indexed_triple = index_triple(triple, self.entity_to_idx, self.relation_to_idx)
        return prompt, label, indexed_triple

    def corrupt_triple(self, triple):
        # Corrupt the head or tail without producing a known training-positive triple.
        head, relation, tail = triple
        corrupt_head_first = random.random() < 0.5

        # Rejection sampling is fast for the normally sparse set of true triples.
        for _ in range(100):
            replacement = random.choice(self.entity_ids)
            if corrupt_head_first:
                candidate = (replacement, relation, tail)
            else:
                candidate = (head, relation, replacement)
            if candidate not in self.true_triples:
                return candidate

        # Try both corruption directions exhaustively in case the first one is dense.
        for corrupt_head in (corrupt_head_first, not corrupt_head_first):
            start = random.randrange(len(self.entity_ids))
            for offset in range(len(self.entity_ids)):
                replacement = self.entity_ids[(start + offset) % len(self.entity_ids)]
                if corrupt_head:
                    candidate = (replacement, relation, tail)
                else:
                    candidate = (head, relation, replacement)
                if candidate not in self.true_triples:
                    return candidate

        raise RuntimeError(
            f"Could not generate a negative corruption for triple {triple!r}; "
            "all head and tail corruptions are known training positives."
        )


class GroupedNegativeBatchSampler:
    """Shuffle positive groups while keeping each positive with its K negatives."""

    def __init__(self, dataset, batch_size, generator=None):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        self.num_positive_triples = len(dataset.triples)
        self.group_size = 1 + dataset.negative_ratio
        if batch_size < self.group_size:
            raise ValueError(
                "batch_size must be at least 1 + negative_ratio when using "
                "grouped negative batches."
            )
        self.groups_per_batch = batch_size // self.group_size
        self.generator = generator

    def __iter__(self):
        group_order = torch.randperm(
            self.num_positive_triples,
            generator=self.generator,
        ).tolist()
        for start in range(0, len(group_order), self.groups_per_batch):
            batch = []
            for positive_index in group_order[
                start:start + self.groups_per_batch
            ]:
                group_start = positive_index * self.group_size
                batch.extend(range(group_start, group_start + self.group_size))
            yield batch

    def __len__(self):
        return math.ceil(
            self.num_positive_triples / self.groups_per_batch
        )


def collate_text(batch, tokenizer, max_seq_length):
    # Tokenize a list of text prompts and attach BCE labels.
    prompts, labels, indexed_triples = zip(*batch)
    encoded = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    encoded["indexed_triples"] = torch.tensor(indexed_triples, dtype=torch.long)
    encoded["labels"] = torch.tensor(labels, dtype=torch.float)
    return encoded


def score_prompts(model, tokenizer, prompts, device, max_seq_length, batch_size, indexed_triples):
    # Score candidate prompts in mini-batches to avoid evaluation-time memory spikes.
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            batch_prompts = prompts[start : end]
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            batch_triples = torch.tensor(
                indexed_triples[start: end],
                dtype = torch.long,
                device = device
            )

            joint_logits = model(
                indexed_triples = batch_triples,
                **encoded
            )
            scores.append(joint_logits.detach().cpu())
    return torch.cat(scores, dim=0)


def rank_of_target(scores, target_index):
    # Convert candidate scores into the one-based rank of the correct entity.
    target_score = scores[target_index]
    return int((scores > target_score).sum().item()) + 1


def evaluate_link_prediction(
    model,
    tokenizer,
    eval_triples,
    all_true_triples,
    entities,
    relations,
    entity_to_idx,
    relation_to_idx,
    device,
    max_seq_length,
    candidate_batch_size,
    progress_description="Evaluating joint BERT BCE MRR",
    checkpoint_path=None,
    checkpoint_every=1,
):
    # Evaluate filtered head and tail prediction ranks for every held-out triple.
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1.")
    entity_ids = list(entities.keys())
    entity_to_pos = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
    fingerprint = evaluation_fingerprint(eval_triples, entity_ids)
    ranks, next_triple_index = load_evaluation_progress(
        checkpoint_path,
        fingerprint,
        len(eval_triples),
    )
    if next_triple_index:
        print(
            f"resuming_evaluation={checkpoint_path} "
            f"completed_triples={next_triple_index}/{len(eval_triples)}"
        )

    for triple_index in tqdm(
        range(next_triple_index, len(eval_triples)),
        desc=progress_description,
        initial=next_triple_index,
        total=len(eval_triples),
    ):
        head, relation, tail = eval_triples[triple_index]

        # Tail prediction (h, r, ?): keep the head and relation fixed and
        # score every entity as a possible tail, including the correct tail.
        tail_candidates = [(head, relation, candidate_tail) for candidate_tail in entity_ids]
        tail_prompts = [
            triple_prompt(triple, entities, relations, max_seq_length) for triple in tail_candidates
        ]
        indexed_tail_candidates = [
            index_triple(
                triple, entity_to_idx, relation_to_idx
            )
            for triple in tail_candidates
        ]
        tail_scores = score_prompts(
            model=model, tokenizer=tokenizer, prompts=tail_prompts, device=device, max_seq_length=max_seq_length, batch_size=candidate_batch_size, indexed_triples=indexed_tail_candidates
        )
        # Filter other known true tails so they do not worsen the target's
        # rank; retain the correct tail being evaluated.
        for idx, candidate_tail in enumerate(entity_ids):
            candidate = (head, relation, candidate_tail)
            if candidate_tail != tail and candidate in all_true_triples:
                tail_scores[idx] = -float("inf")
        ranks.append(rank_of_target(tail_scores, entity_to_pos[tail]))

        # Head prediction (?, r, t): keep the relation and tail fixed and
        # score every entity as a possible head, including the correct head.
        head_candidates = [(candidate_head, relation, tail) for candidate_head in entity_ids]
        head_prompts = [
            triple_prompt(triple, entities, relations, max_seq_length) for triple in head_candidates
        ]
        indexed_head_candidates = [
            index_triple(
                triple, entity_to_idx, relation_to_idx
            )
            for triple in head_candidates
        ]
        head_scores = score_prompts(
            model=model, tokenizer=tokenizer, prompts=head_prompts, device=device, max_seq_length=max_seq_length, batch_size=candidate_batch_size, indexed_triples=indexed_head_candidates
        )
        # Apply the same filtering to other known true heads, retaining
        # the correct head being evaluated.
        for idx, candidate_head in enumerate(entity_ids):
            candidate = (candidate_head, relation, tail)
            if candidate_head != head and candidate in all_true_triples:
                head_scores[idx] = -float("inf")
        ranks.append(rank_of_target(head_scores, entity_to_pos[head]))

        next_triple_index = triple_index + 1
        if (
            next_triple_index % checkpoint_every == 0
            or next_triple_index == len(eval_triples)
        ):
            save_evaluation_progress(
                checkpoint_path,
                fingerprint,
                next_triple_index,
                len(eval_triples),
                ranks,
            )

    # Each triple contributes two ranks (tail and head), so MRR and Hits@K
    # aggregate both prediction directions with equal weight.
    return ranks_to_metrics(ranks)


def parse_args(use_fixed_lambda=False):
    # Define CLI options for training and evaluating the BERT BCE baseline.
    parser = argparse.ArgumentParser(description="Train/evaluate a BERT BCE link-prediction baseline.")
    parser.add_argument("--dataset_path", type=str, default="KGs/UMLS")
    parser.add_argument("--support_path", type=str, default=None)
    parser.add_argument("--bert_model_path", type=str, default="checkpoints/umls/bert-pretrained")
    parser.add_argument("--tokenizer_path", type=str, default="bert-base-cased")
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
        help=(
            "Maximum number of training epochs. Training-loss early stopping "
            "may finish sooner; validation then runs once."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--negative_ratio", type=int, default=1)
    parser.add_argument(
        "--negative_loss_weighting",
        choices=["balanced", "sampled"],
        default="balanced",
        help=(
            "Use 'balanced' to keep total positive and negative class weight "
            "comparable as --negative_ratio changes; 'sampled' reproduces "
            "the unweighted example mean."
        ),
    )
    parser.add_argument(
        "--negative_filter_scope",
        choices=["train", "train_valid", "all"],
        default="train",
        help=(
            "Facts excluded from corruption. 'train' is split-strict; "
            "'train_valid' uses validation membership; 'all' also uses test "
            "membership and is therefore transductive."
        ),
    )
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Run directory. A unique directory is created when omitted.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Resume from a training_state.pt file or the run directory "
            "containing it."
        ),
    )
    parser.add_argument(
        "--resume_legacy_v1",
        action="store_true",
        help=(
            "Explicitly continue a version-1 training checkpoint with its "
            "original uncalibrated, shuffled-example, sampled-BCE protocol."
        ),
    )
    parser.add_argument(
        "--evaluate_checkpoint",
        type=str,
        default=None,
        help=(
            "Skip training and evaluate an existing best-model checkpoint. "
            "This also upgrades legacy runs to resumable evaluation."
        ),
    )
    parser.add_argument(
        "--eval_checkpoint_every",
        type=int,
        default=1,
        help="Save evaluation ranks after this many completed triples.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help=(
            "Stop after this many training epochs without a loss decrease of "
            "at least --early_stopping_min_delta."
        ),
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Minimum training-loss decrease counted as an improvement.",
    )
    parser.add_argument(
        "--early_stopping_warmup_epochs",
        type=int,
        default=2,
        help="Do not count non-improving training epochs before this epoch.",
    )
    parser.add_argument(
    "--kge_model",
    type=str,
    default="TransE",
    choices=["TransE", "RotatE", "Keci", "MuRE"],
    )
    parser.add_argument(
        "--kge_embedding_dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--kge_lr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--calibration_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the affine BERT/KGE score calibrators.",
    )
    if use_fixed_lambda:
        parser.add_argument(
            "--lambda_val",
            type=float,
            required=True,
            help="Fixed BERT mixing weight; must be strictly between 0 and 1.",
        )
    else:
        parser.add_argument(
            "--lambda_lr",
            type=float,
            default=1e-3,
        )
        parser.add_argument(
            "--initial_lambda",
            type=float,
            default=0.5,
        )
    return parser.parse_args()


def main(use_fixed_lambda=False):
    # Orchestrate data loading, BCE training, filtered MRR evaluation, and checkpoint saving.
    args = parse_args(use_fixed_lambda=use_fixed_lambda)
    fixed_lambda = args.lambda_val if use_fixed_lambda else None
    if fixed_lambda is not None and not 0.0 < fixed_lambda < 1.0:
        raise ValueError("--lambda_val must be between 0 and 1.")
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
    if args.eval_checkpoint_every < 1:
        raise ValueError("--eval_checkpoint_every must be at least 1.")
    if args.resume_from_checkpoint and args.evaluate_checkpoint:
        raise ValueError(
            "--resume_from_checkpoint and --evaluate_checkpoint are "
            "mutually exclusive."
        )

    lambda_mode = "learned" if fixed_lambda is None else "fixed"
    configured_lambda = (
        args.initial_lambda if fixed_lambda is None else fixed_lambda
    )
    run_start_time = time.perf_counter()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = resolve_path(repo_root, args.dataset_path)
    support_path = resolve_path(repo_root, args.support_path) if args.support_path else None
    bert_model_path = resolve_path(repo_root, args.bert_model_path)
    base_output_dir = os.path.join(repo_root, "bert_bce_runs")
    dataset_name = dataset_run_name(args.dataset_path)
    run_datetime = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_variant = (
        "joint"
        if fixed_lambda is None
        else f"joint_fixed_lambda_{fixed_lambda:g}"
    )
    run_name = (
        f"{dataset_name}_{run_variant}_{args.kge_model}_"
        f"{args.eval_split}_{run_datetime}"
    )
    resume_checkpoint_path = None
    resume_state = None
    evaluation_checkpoint_path = None
    evaluation_checkpoint_state = None
    if args.evaluate_checkpoint:
        evaluation_checkpoint_path = resolve_path(
            repo_root,
            args.evaluate_checkpoint,
        )
        if os.path.isdir(evaluation_checkpoint_path):
            evaluation_checkpoint_path = os.path.join(
                evaluation_checkpoint_path,
                f"joint_bert_{args.kge_model}_bce_link_prediction.pt",
            )
        if not os.path.isfile(evaluation_checkpoint_path):
            raise FileNotFoundError(
                f"Evaluation checkpoint not found: "
                f"{evaluation_checkpoint_path}"
            )
        evaluation_checkpoint_state = load_torch_checkpoint(
            evaluation_checkpoint_path
        )
        output_dir = os.path.dirname(evaluation_checkpoint_path)
        if args.output_dir:
            requested_output_dir = resolve_path(repo_root, args.output_dir)
            if os.path.abspath(requested_output_dir) != os.path.abspath(output_dir):
                raise ValueError(
                    "--output_dir must match the directory containing "
                    "--evaluate_checkpoint."
                )
    elif args.resume_from_checkpoint:
        resume_checkpoint_path = resolve_path(
            repo_root,
            args.resume_from_checkpoint,
        )
        if os.path.isdir(resume_checkpoint_path):
            resume_checkpoint_path = os.path.join(
                resume_checkpoint_path,
                "training_state.pt",
            )
        if not os.path.isfile(resume_checkpoint_path):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_checkpoint_path}"
            )
        resume_state = load_torch_checkpoint(resume_checkpoint_path)
        output_dir = os.path.dirname(resume_checkpoint_path)
        if args.output_dir:
            requested_output_dir = resolve_path(repo_root, args.output_dir)
            if os.path.abspath(requested_output_dir) != os.path.abspath(output_dir):
                raise ValueError(
                    "--output_dir must match the directory containing "
                    "--resume_from_checkpoint."
                )
    elif args.output_dir:
        output_dir = resolve_path(repo_root, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(os.path.join(output_dir, "training_state.pt")):
            raise FileExistsError(
                f"{output_dir} already contains training_state.pt; pass "
                "--resume_from_checkpoint to resume it."
            )
    else:
        output_dir = create_unique_output_dir(base_output_dir, run_name)

    legacy_v1_resume = (
        resume_state is not None
        and resume_state.get("version") == 1
    )
    resume_legacy_v1_requested = getattr(args, "resume_legacy_v1", False)
    if legacy_v1_resume and not resume_legacy_v1_requested:
        raise ValueError(
            "Version-1 training checkpoints require the explicit "
            "--resume_legacy_v1 flag so they are continued with the legacy "
            "training protocol."
        )
    if resume_legacy_v1_requested and not legacy_v1_resume:
        checkpoint_version = (
            None if resume_state is None else resume_state.get("version")
        )
        raise ValueError(
            "--resume_legacy_v1 requires a version-1 training checkpoint; "
            f"got version {checkpoint_version!r}."
        )

    fusion_calibration_label = (
        "identity_legacy_v1" if legacy_v1_resume else "affine_v1"
    )
    negative_batching_label = (
        "shuffled_examples_legacy_v1"
        if legacy_v1_resume
        else "positive_groups"
    )
    effective_negative_loss_weighting = (
        "sampled" if legacy_v1_resume else args.negative_loss_weighting
    )

    training_state_path = os.path.join(output_dir, "training_state.pt")
    checkpoint_path = (
        evaluation_checkpoint_path
        if evaluation_checkpoint_path is not None
        else os.path.join(
            output_dir,
            f"joint_bert_{args.kge_model}_bce_link_prediction.pt",
        )
    )
    validation_progress_path = os.path.join(
        output_dir,
        "validation_evaluation_progress.pt",
    )
    test_progress_path = os.path.join(
        output_dir,
        "test_evaluation_progress.pt",
    )
    print(f"run_output_dir={output_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_triples = read_triples(os.path.join(dataset_path, "train.txt"))
    valid_triples = read_triples(os.path.join(dataset_path, "valid.txt"))
    test_triples = read_triples(os.path.join(dataset_path, "test.txt"))
    if not valid_triples:
        raise ValueError(
            "Validation triples are required for checkpoint selection "
            "and early stopping."
        )
    all_true_triples = (
        set(train_triples)
        | set(valid_triples)
        | set(test_triples)
    )
    print("validation_filter_scope=train_valid_test")
    entities, relations = read_support(dataset_path, support_path)
    if legacy_v1_resume:
        negative_entity_ids = list(entities)
        training_negative_filter = set(train_triples)
        negative_entity_pool_label = "all_legacy_v1"
        negative_filter_scope_label = "train_legacy_v1"
    else:
        negative_entity_ids = training_entity_ids(train_triples, entities)
        training_negative_filter = negative_filter_triples(
            args.negative_filter_scope,
            train_triples,
            valid_triples,
            test_triples,
        )
        negative_entity_pool_label = "train"
        negative_filter_scope_label = args.negative_filter_scope
    print(
        f"negative_entity_pool={negative_entity_pool_label} "
        f"({len(negative_entity_ids)}/"
        f"{len(entities)} entities) "
        f"negative_filter_scope={negative_filter_scope_label} "
        f"negative_loss_weighting={effective_negative_loss_weighting}"
    )
    entity_to_idx = {
        entity_id: idx
        for idx, entity_id in enumerate(entities)
    }
    relation_to_idx = {
        relation_id: idx
        for idx, relation_id in enumerate(relations)
    }

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_basic_tokenize=False)
    add_nbert_tokens(tokenizer, entities, relations)
    bert_model = BertTripleClassifier(bert_model_path, tokenizer)

    kge_model = create_kge_model(
        model_name=args.kge_model,
        num_entities=len(entity_to_idx),
        num_relations=len(relation_to_idx),
        embedding_dim=args.kge_embedding_dim,
        random_seed=args.seed,
        learning_rate=args.kge_lr,
        negative_ratio=args.negative_ratio,
    )

    model = JointBCEModel(
        bert_model=bert_model,
        kge_model=kge_model,
        initial_lambda=configured_lambda,
    )
    if fixed_lambda is not None:
        model.lambda_logit.requires_grad_(False)
    if legacy_v1_resume:
        for parameter in model.calibration_parameters():
            parameter.requires_grad_(False)
            with torch.no_grad():
                parameter.zero_()
    model.to(device)

    train_dataset = TripleBCEDataset(
        triples=train_triples,
        entities=entities,
        relations=relations,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        max_seq_length=args.max_seq_length,
        negative_ratio=args.negative_ratio,
        negative_entity_ids=negative_entity_ids,
        known_true_triples=training_negative_filter,
    )

    if legacy_v1_resume:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_text(
                batch,
                tokenizer,
                args.max_seq_length,
            ),
        )
        print(
            "negative_batching=shuffled_examples_legacy_v1 "
            f"effective_batch_size={args.batch_size}"
        )
    else:
        train_batch_sampler = GroupedNegativeBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
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
            f"negative_group_size={train_batch_sampler.group_size} "
            f"positive_groups_per_batch={train_batch_sampler.groups_per_batch} "
            f"effective_batch_size="
            f"{train_batch_sampler.group_size * train_batch_sampler.groups_per_batch}"
        )

    optimizer_parameter_groups = [
        {
            "params": model.bert_model.parameters(),
            "lr": args.lr,
        },
        {
            "params": model.kge_model.parameters(),
            "lr": args.kge_lr,
        },
    ]
    if not legacy_v1_resume:
        optimizer_parameter_groups.append(
            {
                "params": model.calibration_parameters(),
                "lr": args.calibration_lr,
            }
        )
    if fixed_lambda is None:
        optimizer_parameter_groups.append(
            {
                "params": [model.lambda_logit],
                "lr": args.lambda_lr,
            }
        )
    optimizer = torch.optim.Adam(optimizer_parameter_groups)

    training_history = []
    best_model_state = None
    best_validation_metrics = None
    best_validation_mrr = -float("inf")
    best_epoch = None
    best_final_lambda = None
    best_training_loss = float("inf")
    epochs_without_loss_improvement = 0
    epochs_ran = 0
    early_stopped = False
    resume_phase = "training"
    pending_epoch_diagnostics = None

    if resume_state is not None:
        supported_version = 1 if legacy_v1_resume else 2
        if resume_state.get("version") != supported_version:
            raise ValueError(
                f"Unsupported training checkpoint version in "
                f"{resume_checkpoint_path}: {resume_state.get('version')!r}"
            )
        if resume_state.get("run_variant") != run_variant:
            raise ValueError(
                "Resume checkpoint model variant does not match this entry point."
            )
        saved_args = resume_state.get("args", {})
        ignored_resume_args = {
            "candidate_batch_size",
            "device",
            "evaluate_checkpoint",
            "eval_checkpoint_every",
            "num_workers",
            "output_dir",
            "resume_legacy_v1",
            "resume_from_checkpoint",
        }
        mismatches = [
            name
            for name, value in vars(args).items()
            if name not in ignored_resume_args
            and name in saved_args
            and saved_args[name] != value
        ]
        if mismatches:
            details = ", ".join(
                f"{name}: saved={saved_args[name]!r}, current="
                f"{getattr(args, name)!r}"
                for name in mismatches
            )
            raise ValueError(
                f"Resume arguments do not match the checkpoint ({details})."
            )

        load_joint_model_state(
            model,
            resume_state["model_state_dict"],
            allow_legacy_calibration=legacy_v1_resume,
        )
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        training_history = resume_state["training_history"]
        best_model_state = resume_state["best_model_state"]
        best_validation_metrics = resume_state["best_validation_metrics"]
        best_validation_mrr = resume_state["best_validation_mrr"]
        best_epoch = resume_state["best_epoch"]
        best_final_lambda = resume_state["best_final_lambda"]
        best_training_loss = float(
            resume_state.get(
                "best_training_loss",
                min(
                    (
                        record["loss"]
                        for record in training_history
                        if "loss" in record
                    ),
                    default=float("inf"),
                ),
            )
        )
        epochs_without_loss_improvement = int(
            resume_state.get("epochs_without_loss_improvement", 0)
        )
        epochs_ran = resume_state["epoch_completed"]
        early_stopped = resume_state["early_stopped"]
        resume_phase = resume_state["phase"]
        pending_epoch_diagnostics = resume_state.get(
            "pending_epoch_diagnostics"
        )
        if best_epoch is not None and best_model_state is None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    "Training state references a best epoch, but the best "
                    f"checkpoint is missing: {checkpoint_path}"
                )
            best_model_state = load_torch_checkpoint(
                checkpoint_path
            )["model_state_dict"]
        random.setstate(resume_state["python_random_state"])
        torch.set_rng_state(resume_state["torch_random_state"])
        if (
            torch.cuda.is_available()
            and resume_state.get("cuda_random_state") is not None
        ):
            torch.cuda.set_rng_state_all(resume_state["cuda_random_state"])
        print(
            f"resumed_training_checkpoint={resume_checkpoint_path} "
            f"phase={resume_phase} epoch_completed={epochs_ran}"
        )
    elif evaluation_checkpoint_state is not None:
        saved_args = evaluation_checkpoint_state.get("args", {})
        ignored_evaluation_args = {
            "candidate_batch_size",
            "device",
            "early_stopping_min_delta",
            "early_stopping_patience",
            "early_stopping_warmup_epochs",
            "eval_split",
            "evaluate_checkpoint",
            "eval_checkpoint_every",
            "num_workers",
            "output_dir",
            "resume_legacy_v1",
            "resume_from_checkpoint",
        }
        mismatches = [
            name
            for name, value in vars(args).items()
            if name not in ignored_evaluation_args
            and name in saved_args
            and saved_args[name] != value
        ]
        if mismatches:
            details = ", ".join(
                f"{name}: saved={saved_args[name]!r}, current="
                f"{getattr(args, name)!r}"
                for name in mismatches
            )
            raise ValueError(
                "Evaluation arguments do not match the checkpoint "
                f"({details})."
            )
        saved_lambda_mode = evaluation_checkpoint_state.get("lambda_mode")
        if (
            saved_lambda_mode is not None
            and saved_lambda_mode != lambda_mode
        ):
            raise ValueError(
                f"Checkpoint lambda mode is {saved_lambda_mode!r}, but this "
                f"entry point uses {lambda_mode!r}."
            )

        best_model_state = evaluation_checkpoint_state["model_state_dict"]
        loaded_legacy_calibration = load_joint_model_state(
            model,
            best_model_state,
            allow_legacy_calibration=True,
        )
        if loaded_legacy_calibration:
            print("loaded_legacy_checkpoint_with_identity_calibration=true")
            # Upgrade the in-memory best state so subsequent saves contain the
            # explicit identity calibrators instead of relabeling legacy data.
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        best_validation_metrics = evaluation_checkpoint_state.get(
            "best_validation_metrics"
        )
        if (
            best_validation_metrics is None
            and evaluation_checkpoint_state.get("selection_split") == "valid"
        ):
            best_validation_metrics = evaluation_checkpoint_state.get(
                "metrics"
            )
        if best_validation_metrics is None:
            raise ValueError(
                "Evaluation checkpoint does not contain validation metrics."
            )
        best_validation_mrr = float(best_validation_metrics["MRR"])
        best_epoch = evaluation_checkpoint_state.get("best_epoch")
        best_final_lambda = evaluation_checkpoint_state.get(
            "best_final_lambda",
            evaluation_checkpoint_state.get("final_lambda"),
        )
        epochs_ran = int(
            evaluation_checkpoint_state.get("epochs_ran", args.num_epochs)
        )
        early_stopped = bool(
            evaluation_checkpoint_state.get("early_stopped", False)
        )
        training_history = evaluation_checkpoint_state.get(
            "training_history",
            [],
        )
        best_training_loss = float(
            evaluation_checkpoint_state.get(
                "best_training_loss",
                min(
                    (
                        record["loss"]
                        for record in training_history
                        if "loss" in record
                    ),
                    default=float("inf"),
                ),
            )
        )
        resume_phase = "training_complete"
        print(
            f"loaded_evaluation_checkpoint={evaluation_checkpoint_path} "
            f"best_epoch={best_epoch}"
        )

    tokenizer.save_pretrained(output_dir)

    def checkpoint_model_state(clone=False):
        state = {}
        for name, value in model.state_dict().items():
            if legacy_v1_resume and name in CALIBRATION_STATE_KEYS:
                continue
            value = value.detach().cpu()
            state[name] = value.clone() if clone else value
        return state

    def save_training_state(phase, pending_diagnostics=None):
        atomic_torch_save(
            training_state_path,
            {
                "version": 1 if legacy_v1_resume else 2,
                "fusion_calibration": fusion_calibration_label,
                "negative_batching": negative_batching_label,
                "negative_loss_weighting": (
                    effective_negative_loss_weighting
                ),
                "run_variant": run_variant,
                "phase": phase,
                "epoch_completed": epochs_ran,
                "pending_epoch_diagnostics": pending_diagnostics,
                "model_state_dict": checkpoint_model_state(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_history": training_history,
                # Best weights live in checkpoint_path. Avoid duplicating
                # hundreds of MB in every resumable training-state write.
                "best_model_state": None,
                "best_validation_metrics": best_validation_metrics,
                "best_validation_mrr": best_validation_mrr,
                "best_epoch": best_epoch,
                "best_final_lambda": best_final_lambda,
                "best_training_loss": best_training_loss,
                "epochs_without_loss_improvement": (
                    epochs_without_loss_improvement
                ),
                "early_stopped": early_stopped,
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
            f"phase={phase} epoch={epochs_ran}"
        )

    def validate_final_epoch(epoch, epoch_diagnostics):
        nonlocal best_model_state
        nonlocal best_validation_metrics
        nonlocal best_validation_mrr
        nonlocal best_epoch
        nonlocal best_final_lambda

        lambda_value = float(epoch_diagnostics["lambda"])
        validation_metrics = evaluate_link_prediction(
            model=model,
            tokenizer=tokenizer,
            eval_triples=valid_triples,
            all_true_triples=all_true_triples,
            entities=entities,
            relations=relations,
            entity_to_idx=entity_to_idx,
            relation_to_idx=relation_to_idx,
            device=device,
            max_seq_length=args.max_seq_length,
            candidate_batch_size=args.candidate_batch_size,
            progress_description=f"Final validation after epoch {epoch}",
            checkpoint_path=validation_progress_path,
            checkpoint_every=args.eval_checkpoint_every,
        )
        epoch_diagnostics["validation_metrics"] = validation_metrics
        validation_mrr = float(validation_metrics["MRR"])
        best_validation_mrr = validation_mrr
        best_validation_metrics = dict(validation_metrics)
        best_epoch = epoch
        best_final_lambda = lambda_value
        best_model_state = checkpoint_model_state(clone=True)
        epoch_diagnostics["is_best"] = True
        print(
            f"final_validation_epoch={epoch} "
            f"valid_mrr={validation_mrr:.6f} "
            "stopping_strategy=training_loss_patience"
        )

        training_history.append(epoch_diagnostics)
        atomic_torch_save(
            checkpoint_path,
            {
                "model_state_dict": best_model_state,
                "metrics": best_validation_metrics,
                "args": vars(args),
                "lambda_mode": lambda_mode,
                "fusion_calibration": fusion_calibration_label,
                "negative_batching": negative_batching_label,
                "negative_loss_weighting": effective_negative_loss_weighting,
                "calibration": model.calibration_state(),
                "lambda_requires_grad": model.lambda_logit.requires_grad,
                "configured_lambda": configured_lambda,
                "final_lambda": best_final_lambda,
                "selection_split": None,
                "selection_metric": None,
                "stopping_strategy": "training_loss_patience",
                "validation_filter_scope": "train_valid_test",
                "best_epoch": best_epoch,
                "epochs_ran": epochs_ran,
                "best_validation_metrics": best_validation_metrics,
                "best_final_lambda": best_final_lambda,
                "training_history": training_history,
            },
        )
        print(
            f"saved_final_checkpoint={checkpoint_path} epoch={best_epoch}"
        )
        save_training_state("training")
        return bool(epoch_diagnostics.get("stop_training", True))

    training_should_stop = (
        early_stopped or resume_phase == "training_complete"
    )
    if resume_phase == "validation_pending":
        if pending_epoch_diagnostics is None:
            raise ValueError(
                "Validation-pending checkpoint is missing epoch diagnostics."
            )
        print(f"resuming_validation_epoch={epochs_ran}")
        training_should_stop = validate_final_epoch(
            epochs_ran,
            pending_epoch_diagnostics,
        )

    first_epoch = epochs_ran + 1
    for epoch in range(first_epoch, args.num_epochs + 1):
        if training_should_stop:
            break
        epochs_ran = epoch
        model.train()
        losses = []
        kge_parameters = [
            parameter for parameter in model.kge_model.parameters()
            if parameter.requires_grad
        ]
        kge_parameters_at_epoch_start = [
            parameter.detach().clone() for parameter in kge_parameters
        ]
        bert_positive_logit_sum = 0.0
        bert_negative_logit_sum = 0.0
        kge_positive_logit_sum = 0.0
        kge_negative_logit_sum = 0.0
        joint_positive_logit_sum = 0.0
        joint_negative_logit_sum = 0.0
        joint_positive_bce_sum = 0.0
        joint_negative_bce_sum = 0.0
        num_positive_examples = 0
        num_negative_examples = 0
        kge_gradient_norm_sum = 0.0
        num_kge_gradient_norms = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            labels = batch.pop("labels").to(device)
            indexed_triples = batch.pop("indexed_triples").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            joint_logits, bert_logits, kge_logits = model(
                indexed_triples=indexed_triples,
                return_components=True,
                **batch,
            )
            example_losses = F.binary_cross_entropy_with_logits(
                joint_logits,
                labels,
                reduction="none",
            )
            if legacy_v1_resume:
                loss = F.binary_cross_entropy_with_logits(
                    joint_logits,
                    labels,
                )
            else:
                loss = negative_sampling_bce_loss(
                    joint_logits,
                    labels,
                    args.negative_ratio,
                    weighting=args.negative_loss_weighting,
                )
            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                positive_mask = labels > 0.5
                negative_mask = ~positive_mask
                bert_positive_logit_sum += float(
                    bert_logits[positive_mask].detach().sum().item()
                )
                bert_negative_logit_sum += float(
                    bert_logits[negative_mask].detach().sum().item()
                )
                kge_positive_logit_sum += float(
                    kge_logits[positive_mask].detach().sum().item()
                )
                kge_negative_logit_sum += float(
                    kge_logits[negative_mask].detach().sum().item()
                )
                joint_positive_logit_sum += float(
                    joint_logits[positive_mask].detach().sum().item()
                )
                joint_negative_logit_sum += float(
                    joint_logits[negative_mask].detach().sum().item()
                )
                joint_positive_bce_sum += float(
                    example_losses[positive_mask].detach().sum().item()
                )
                joint_negative_bce_sum += float(
                    example_losses[negative_mask].detach().sum().item()
                )
                num_positive_examples += int(positive_mask.sum().item())
                num_negative_examples += int(negative_mask.sum().item())

                squared_gradient_norm = sum(
                    float(parameter.grad.detach().float().square().sum())
                    for parameter in kge_parameters
                    if parameter.grad is not None
                )
                kge_gradient_norm_sum += math.sqrt(squared_gradient_norm)
                num_kge_gradient_norms += 1

            optimizer.step()
            if not legacy_v1_resume:
                post_kge_parameter_update(model.kge_model)
            losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            squared_kge_update_norm = sum(
                float(
                    (parameter.detach() - parameter_at_start)
                    .float()
                    .square()
                    .sum()
                )
                for parameter, parameter_at_start in zip(
                    kge_parameters,
                    kge_parameters_at_epoch_start,
                )
            )
        kge_update_norm = math.sqrt(squared_kge_update_norm)
        bert_positive_logit_mean = (
            bert_positive_logit_sum / max(num_positive_examples, 1)
        )
        bert_negative_logit_mean = (
            bert_negative_logit_sum / max(num_negative_examples, 1)
        )
        kge_positive_logit_mean = (
            kge_positive_logit_sum / max(num_positive_examples, 1)
        )
        kge_negative_logit_mean = (
            kge_negative_logit_sum / max(num_negative_examples, 1)
        )
        joint_positive_logit_mean = (
            joint_positive_logit_sum / max(num_positive_examples, 1)
        )
        joint_negative_logit_mean = (
            joint_negative_logit_sum / max(num_negative_examples, 1)
        )
        joint_positive_bce_mean = (
            joint_positive_bce_sum / max(num_positive_examples, 1)
        )
        joint_negative_bce_mean = (
            joint_negative_bce_sum / max(num_negative_examples, 1)
        )
        kge_gradient_norm_mean = (
            kge_gradient_norm_sum / max(num_kge_gradient_norms, 1)
        )
        lambda_value = float(model.mixing_weight.detach().cpu())
        calibration_state = model.calibration_state()
        mean_loss = sum(losses) / max(len(losses), 1)
        epoch_diagnostics = {
            "epoch": epoch,
            "loss": mean_loss,
            "lambda": lambda_value,
            **calibration_state,
            "bert_pos_logit": bert_positive_logit_mean,
            "bert_neg_logit": bert_negative_logit_mean,
            "bert_logit_gap": (
                bert_positive_logit_mean - bert_negative_logit_mean
            ),
            "kge_pos_logit": kge_positive_logit_mean,
            "kge_neg_logit": kge_negative_logit_mean,
            "kge_logit_gap": (
                kge_positive_logit_mean - kge_negative_logit_mean
            ),
            "joint_pos_logit": joint_positive_logit_mean,
            "joint_neg_logit": joint_negative_logit_mean,
            "joint_logit_gap": (
                joint_positive_logit_mean - joint_negative_logit_mean
            ),
            "joint_pos_bce": joint_positive_bce_mean,
            "joint_neg_bce": joint_negative_bce_mean,
            "kge_grad_norm": kge_gradient_norm_mean,
            "kge_update_norm": kge_update_norm,
            "validation_metrics": None,
        }
        loss_improved = (
            mean_loss
            < best_training_loss - args.early_stopping_min_delta
        )
        if loss_improved:
            best_training_loss = mean_loss
            epochs_without_loss_improvement = 0
        elif epoch >= args.early_stopping_warmup_epochs:
            epochs_without_loss_improvement += 1
        should_stop_early = (
            epoch >= args.early_stopping_warmup_epochs
            and epochs_without_loss_improvement
            >= args.early_stopping_patience
            and epoch < args.num_epochs
        )
        if should_stop_early:
            early_stopped = True
        epoch_diagnostics["training_loss_improved"] = loss_improved
        epoch_diagnostics["loss_patience"] = (
            epochs_without_loss_improvement
        )
        epoch_diagnostics["stop_training"] = (
            should_stop_early or epoch == args.num_epochs
        )
        print(
            f"epoch={epoch} "
            f"loss={mean_loss:.6f} "
            f"lambda={lambda_value:.6f} "
            f"bert_scale={calibration_state['bert_scale']:.6f} "
            f"bert_bias={calibration_state['bert_bias']:.6f} "
            f"kge_scale={calibration_state['kge_scale']:.6f} "
            f"kge_bias={calibration_state['kge_bias']:.6f}"
        )
        print(
            f"bert_pos_logit={bert_positive_logit_mean:.6f} "
            f"bert_neg_logit={bert_negative_logit_mean:.6f} "
            f"bert_logit_gap="
            f"{bert_positive_logit_mean - bert_negative_logit_mean:.6f}"
        )
        print(
            f"kge_model={args.kge_model} "
            f"kge_pos_logit={kge_positive_logit_mean:.6f} "
            f"kge_neg_logit={kge_negative_logit_mean:.6f} "
            f"kge_logit_gap="
            f"{kge_positive_logit_mean - kge_negative_logit_mean:.6f} "
            f"kge_grad_norm={kge_gradient_norm_mean:.6f} "
            f"kge_update_norm={kge_update_norm:.6f}"
        )
        print(
            f"joint_pos_logit={joint_positive_logit_mean:.6f} "
            f"joint_neg_logit={joint_negative_logit_mean:.6f} "
            f"joint_logit_gap="
            f"{joint_positive_logit_mean - joint_negative_logit_mean:.6f} "
            f"joint_pos_bce={joint_positive_bce_mean:.6f} "
            f"joint_neg_bce={joint_negative_bce_mean:.6f}"
        )
        print(
            f"training_loss_patience={epochs_without_loss_improvement}/"
            f"{args.early_stopping_patience} "
            f"best_training_loss={best_training_loss:.6f}"
        )
        if should_stop_early:
            print(
                f"early_stopping_epoch={epoch} "
                f"best_training_loss={best_training_loss:.6f}"
            )

        should_evaluate = should_stop_early or epoch == args.num_epochs
        if should_evaluate:
            save_training_state(
                "validation_pending",
                pending_diagnostics=epoch_diagnostics,
            )
            training_should_stop = validate_final_epoch(
                epoch,
                epoch_diagnostics,
            )
            if training_should_stop:
                break
        else:
            training_history.append(epoch_diagnostics)
            save_training_state("training")

    if best_model_state is None:
        raise RuntimeError(
            "Training completed without a validation evaluation."
        )

    load_joint_model_state(
        model,
        best_model_state,
        allow_legacy_calibration=True,
    )
    lambda_value = float(model.mixing_weight.detach().cpu())
    final_calibration = model.calibration_state()
    save_training_state("training_complete")
    print(
        f"final_epoch={best_epoch} "
        f"final_valid_mrr={best_validation_mrr:.6f} "
        f"lambda={lambda_value:.6f}"
    )

    if args.eval_split == "valid":
        metrics = best_validation_metrics
    else:
        metrics = evaluate_link_prediction(
            model=model,
            tokenizer=tokenizer,
            eval_triples=test_triples,
            all_true_triples=all_true_triples,
            entities=entities,
            relations=relations,
            entity_to_idx=entity_to_idx,
            relation_to_idx=relation_to_idx,
            device=device,
            max_seq_length=args.max_seq_length,
            candidate_batch_size=args.candidate_batch_size,
            progress_description="Evaluating final checkpoint on test",
            checkpoint_path=test_progress_path,
            checkpoint_every=args.eval_checkpoint_every,
        )
    print(
        f"final_{args.eval_split}_metrics="
        f"{json.dumps(metrics, sort_keys=True)}"
    )

    runtim_sec = time.perf_counter() - run_start_time
    runtime_min = runtim_sec / 60.0
    results = {
        "metrics": metrics,
        "args": vars(args),
        "dataset_path": dataset_path,
        "bert_model_path": bert_model_path,
        "num_train_triples": len(train_triples),
        "num_valid_triples": len(valid_triples),
        "num_test_triples": len(test_triples),
        "num_entities": len(entities),
        "num_relations": len(relations),
        "lambda_mode": lambda_mode,
        "fusion_calibration": fusion_calibration_label,
        "negative_batching": negative_batching_label,
        "negative_loss_weighting": effective_negative_loss_weighting,
        "lambda_requires_grad": model.lambda_logit.requires_grad,
        "configured_lambda": configured_lambda,
        "final_lambda": lambda_value,
        "selection_split": None,
        "selection_metric": None,
        "stopping_strategy": "training_loss_patience",
        "validation_filter_scope": "train_valid_test",
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "early_stopped": early_stopped,
        "best_training_loss": best_training_loss,
        "epochs_without_loss_improvement": (
            epochs_without_loss_improvement
        ),
        "best_validation_metrics": best_validation_metrics,
        "best_final_lambda": best_final_lambda,
        "final_calibration": final_calibration,
        "num_negative_entities": len(negative_entity_ids),
        "negative_filter_size": len(training_negative_filter),
        "training_history": training_history,
        "output_dir": output_dir,
        "runtime_min": runtime_min,
    }
    atomic_json_save(
        os.path.join(output_dir, "results.json"),
        results,
    )

    atomic_torch_save(
        checkpoint_path,
        {
            "model_state_dict": best_model_state,
            "metrics": metrics,
            "args": vars(args),
            "lambda_mode": lambda_mode,
            "fusion_calibration": fusion_calibration_label,
            "negative_batching": negative_batching_label,
            "negative_loss_weighting": effective_negative_loss_weighting,
            "calibration": final_calibration,
            "lambda_requires_grad": model.lambda_logit.requires_grad,
            "configured_lambda": configured_lambda,
            "final_lambda": lambda_value,
            "selection_split": None,
            "selection_metric": None,
            "stopping_strategy": "training_loss_patience",
            "validation_filter_scope": "train_valid_test",
            "best_epoch": best_epoch,
            "epochs_ran": epochs_ran,
            "early_stopped": early_stopped,
            "best_training_loss": best_training_loss,
            "epochs_without_loss_improvement": (
                epochs_without_loss_improvement
            ),
            "best_validation_metrics": best_validation_metrics,
            "best_final_lambda": best_final_lambda,
            "final_calibration": final_calibration,
            "num_negative_entities": len(negative_entity_ids),
            "negative_filter_size": len(training_negative_filter),
            "training_history": training_history,
        },
    )
    tokenizer.save_pretrained(output_dir)

    print(f"final_checkpoint_dir={output_dir}")
    print(f"runtime_minutes={runtime_min:.2f}")


if __name__ == "__main__":
    main()