import argparse 
import json 
import os 
import random 
from collections import defaultdict 
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn 
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm 

def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

def read_triples(path):
    triples = [] 
    with open(path, "r", encoding = "utf-8") as f:
        for line in f:
            parts = line.strip().split("\t") 

            if len(parts) != 3: 
                parts = line.strip().split() 
            
            if len(parts) != 3:
                raise ValueError(f"Invalid triple format: {line.strip()}") 
            
            triples.append(tuple(parts)) 
    return triples

def read_support(path):
    support_path = os.path.join(path, "support") 

    with open(os.path.join(support_path, "entity.json"), "r", encoding = "utf-8") as f:
        entities = json.load(f) 

    with open(os.path.join(support_path, "relation.json"), "r", encoding = "utf-8") as f:
        relations = json.load(f)
    
    return entities, relations 


def read_dataset_splits(dataset_path, validation_only=False):
    """Read test.txt only for final, non-tuning runs."""
    train_triples = read_triples(os.path.join(dataset_path, "train.txt"))
    valid_triples = read_triples(os.path.join(dataset_path, "valid.txt"))
    test_triples = None
    if not validation_only:
        test_triples = read_triples(os.path.join(dataset_path, "test.txt"))
    return train_triples, valid_triples, test_triples


def dataset_name_from_path(dataset_path):
    """Return the dataset name, ignoring a trailing numeric subset folder."""
    normalized_path = os.path.normpath(dataset_path)
    leaf = os.path.basename(normalized_path)
    try:
        float(leaf)
    except ValueError:
        return leaf
    return os.path.basename(os.path.dirname(normalized_path))


def dataset_variant_name_from_path(dataset_path):
    """Return a label containing both the dataset and perturbation level."""
    normalized_path = os.path.normpath(dataset_path)
    dataset_name = dataset_name_from_path(normalized_path)
    leaf = os.path.basename(normalized_path)
    try:
        float(leaf)
    except ValueError:
        return dataset_name
    return f"{dataset_name}_{leaf}"


def default_output_paths(dataset_path, timestamp=None):
    dataset_name = dataset_variant_name_from_path(dataset_path)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{dataset_name}_qwen_frozen_we_{timestamp}"
    stem = os.path.join("decoder_runs", run_name, run_name)
    return f"{stem}.pt", f"{stem}.json"


def default_feature_cache_path(dataset_path, model_name, max_length):
    dataset_name = dataset_name_from_path(dataset_path)
    subset = os.path.basename(os.path.normpath(dataset_path))
    model_leaf = model_name.rstrip("/").split("/")[-1]
    cache_name = (
        f"{dataset_name}_{subset}_{model_leaf}_"
        f"{PROMPT_VERSION}_max{max_length}.pt"
    )
    return os.path.join("decoder_feature_cache", cache_name)


#prompt construction 

def truncate_words(text, max_words):
    if not text:
        return "" 
    
    words = str(text).split() 
    return " ".join(words[:max_words]) 

def entity_text(entity_id, entities, description_words=48):
    metadata = entities[entity_id]

    name = metadata.get("name", entity_id) 
    description = truncate_words(metadata.get("desc", ""), description_words) 

    if description:
        return f"{name}. {description}" 
    return name 

def relation_text(relation_id, relations, description_words = 32):
    metadata = relations[relation_id] 

    name = metadata.get("name", relation_id) 
    description = truncate_words(metadata.get("desc", ""), description_words) 

    if description:
        return (
            f"{name}. "
            f"Relation description: {description}"
        )

    return name

PROMPT_VERSION = "qwen_frozen_we_v2_head_tail"


def build_query_prompt(
    source_entity_id,
    relation_id,
    entities,
    relations,
    direction="tail",
):
    """Build a causal prompt for explicit head or tail prediction."""
    entity = entity_text(source_entity_id, entities)
    relation = relation_text(relation_id, relations)

    if direction == "tail":
        return (
            "Knowledge graph link prediction.\n"
            f"Head entity: {entity}\n"
            f"Relation: {relation}\n"
            "Tail entity:"
        )
    if direction == "head":
        return (
            "Knowledge graph link prediction.\n"
            f"Tail entity: {entity}\n"
            f"Relation: {relation}\n"
            "Head entity:"
        )
    raise ValueError(f"Unsupported prediction direction: {direction!r}")


def build_prediction_queries(triples, directions=("tail", "head")):
    """Group triples into multi-target queries for both prediction directions."""
    invalid_directions = set(directions).difference({"head", "tail"})
    if invalid_directions:
        raise ValueError(
            f"Unsupported prediction directions: {sorted(invalid_directions)}"
        )

    queries = []
    if "tail" in directions:
        grouped_tails = defaultdict(set)
        for head, relation, tail in triples:
            grouped_tails[(head, relation)].add(tail)
        queries.extend(
            ("tail", head, relation, sorted(tails))
            for (head, relation), tails in grouped_tails.items()
        )

    if "head" in directions:
        grouped_heads = defaultdict(set)
        for head, relation, tail in triples:
            grouped_heads[(tail, relation)].add(head)
        queries.extend(
            ("head", tail, relation, sorted(heads))
            for (tail, relation), heads in grouped_heads.items()
        )

    return queries


def build_kvsall_queries(triples):
    """Compatibility helper returning the former tail-only tuple layout."""
    return [
        (source, relation, targets)
        for _, source, relation, targets in build_prediction_queries(
            triples,
            directions=("tail",),
        )
    ]


#dataset class 

class QwenKvsAllDataset(Dataset):
    def __init__(
        self,
        triples,
        entities,
        relations,
        entity_to_idx,
        directions=("tail", "head"),
    ):
        self.entities = entities
        self.relations = relations 
        self.entity_to_idx = entity_to_idx
        self.queries = build_prediction_queries(triples, directions=directions)

    def __len__(self):
        return len(self.queries) 
    
    def __getitem__(self, index):
        direction, source_entity, relation, targets = self.queries[index]

        prompt = build_query_prompt(
            source_entity_id=source_entity,
            relation_id=relation,
            entities=self.entities,
            relations=self.relations,
            direction=direction,
        )

        target_indices = [
            self.entity_to_idx[target] for target in targets
        ]

        return {
            "prompt": prompt,
            "target_indices": target_indices,
            "direction": direction,
        }

class KvsAllCollator:
    def __init__(self, tokenizer, num_entities, max_length):

        self.tokenizer = tokenizer 
        self.num_entities = num_entities 
        self.max_length = max_length 

    def __call__(self, examples):
        prompts = [
            example["prompt"] for example in examples
        ]

        encoded = self.tokenizer(
            prompts,
            padding = True,
            truncation = True,
            max_length = self.max_length,
            return_tensors = "pt"
        ) 

        targets = torch.zeros(
            len(examples),
            self.num_entities,
            dtype=torch.float32
        ) 

        for row, example in enumerate(examples):
            targets[row, example["target_indices"]] = 1.0 
        
        encoded["targets"] = targets 
        return encoded 


class FrozenFeatureDataset(Dataset):
    """Training queries represented by cached frozen-Qwen hidden states."""

    def __init__(self, features, queries, entity_to_idx):
        if len(features) != len(queries):
            raise ValueError("Feature and query counts do not match.")
        self.features = features
        self.queries = queries
        self.entity_to_idx = entity_to_idx

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        direction, _, _, target_entities = self.queries[index]
        return {
            "feature": self.features[index],
            "target_indices": [
                self.entity_to_idx[entity]
                for entity in target_entities
            ],
            "direction": direction,
        }


class FrozenFeatureCollator:
    def __init__(self, num_entities):
        self.num_entities = num_entities

    def __call__(self, examples):
        features = torch.stack(
            [example["feature"] for example in examples]
        )
        targets = torch.zeros(
            len(examples),
            self.num_entities,
            dtype=torch.float32,
        )
        for row, example in enumerate(examples):
            targets[row, example["target_indices"]] = 1.0
        return {
            "features": features,
            "targets": targets,
        }

class QwenLinkPredictor(nn.Module):
    def __init__(self, model_name, num_entities, embedding_dim = 256, dtype=torch.bfloat16):

        super().__init__() 

        self.qwen = AutoModel.from_pretrained(model_name, dtype = dtype, local_files_only=False,) 

        self.qwen.requires_grad_(False) 
        self.qwen.eval() 

        hidden_size = self.qwen.config.hidden_size 

        self.proj = nn.Linear(hidden_size, embedding_dim, bias=False) 

        self.entity_embeddings = nn.Embedding(
            num_entities,
            embedding_dim
        )

        nn.init.xavier_uniform_(self.proj.weight) 
        nn.init.xavier_uniform_(self.entity_embeddings.weight) 

        self.register_buffer(
        "logit_scale",
        torch.tensor(embedding_dim**-0.5),
        )

    def train(self, mode = True):
        super().train(mode) 
        if self.qwen is not None:
            self.qwen.eval()
        return self 
    
    def encode_frozen_features(self, input_ids, attention_mask):
        if self.qwen is None:
            raise RuntimeError("The frozen Qwen encoder has already been released.")

        with torch.no_grad():
            output = self.qwen(
                input_ids = input_ids,
                attention_mask = attention_mask,
                use_cache = False
            )

            hidden_states = output.last_hidden_state 

            last_token_positions = attention_mask.sum(dim=1) - 1 

            batch_indices = torch.arange(
                hidden_states.size(0),
                device = hidden_states.device
            ) 

            last_hidden = hidden_states[
                batch_indices,
                last_token_positions
            ]

        return last_hidden

    def score_features(self, features):
        query = self.proj(features.to(self.proj.weight.dtype))
        return (
            query @ self.entity_embeddings.weight.T
        ) * self.logit_scale

    def release_frozen_encoder(self):
        """Release Qwen after all required prompt features are cached."""
        self.qwen = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        features=None,
    ):
        if features is None:
            if input_ids is None or attention_mask is None:
                raise ValueError(
                    "Provide features or both input_ids and attention_mask."
                )
            features = self.encode_frozen_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return self.score_features(features)

def print_trainable_parameters(model):
    trainable = []
    total_trainable = 0

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable.append((name, parameter.numel()))
            total_trainable += parameter.numel()

    print("Trainable parameters:")

    for name, count in trainable:
        print(f"  {name}: {count:,}")

    print(f"Total trainable: {total_trainable:,}")

def loss_func(logits, targets):

    num_positive = targets.sum() 
    num_total = targets.numel() 
    num_negative = num_total - num_positive 

    pos_weight = (
        num_negative / num_positive.clamp_min(1.0)
    ).detach() #Unlinks tensor from the graph.  graph.Stops tracking future gradients.

    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight = pos_weight
    )


def build_true_entity_indexes(*triple_sets):
    """Build filtered-evaluation lookup tables for head and tail queries."""
    true_entities = {
        "tail": defaultdict(set),
        "head": defaultdict(set),
    }
    for triples in triple_sets:
        for head, relation, tail in triples:
            true_entities["tail"][(head, relation)].add(tail)
            true_entities["head"][(tail, relation)].add(head)
    return true_entities


def build_true_tail_index(*triple_sets):
    """Compatibility helper for the former tail-only evaluator."""
    return build_true_entity_indexes(*triple_sets)["tail"]


def ranks_to_metrics(ranks):
    """Compute tail-only filtered link-prediction metrics."""
    if not ranks:
        raise ValueError("Cannot compute ranking metrics without ranks.")

    ranks = torch.tensor(ranks, dtype=torch.float32)
    return {
        "MRR": float((1.0 / ranks).mean().item()),
        "H@1": float((ranks <= 1).float().mean().item()),
        "H@3": float((ranks <= 3).float().mean().item()),
        "H@10": float((ranks <= 10).float().mean().item()),
        "num_ranks": int(ranks.numel()),
    }


def rank_of_target(scores, target_index):
    """Return the optimistic one-based rank used by the existing evaluators."""
    target_score = scores[target_index]
    return int((scores > target_score).sum().item()) + 1


def query_prompt(query, entities, relations):
    direction, source_entity, relation, _ = query
    return build_query_prompt(
        source_entity_id=source_entity,
        relation_id=relation,
        entities=entities,
        relations=relations,
        direction=direction,
    )


def load_feature_cache(path, model_name, max_length, hidden_size):
    if not os.path.isfile(path):
        return {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    expected = {
        "version": 1,
        "model_name": model_name,
        "max_length": max_length,
        "prompt_version": PROMPT_VERSION,
        "hidden_size": hidden_size,
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        print(f"ignoring_incompatible_feature_cache={path}")
        return {}
    return payload.get("features", {})


def cache_frozen_query_features(
    model,
    tokenizer,
    queries,
    entities,
    relations,
    device,
    max_length,
    batch_size,
    cache_path,
    rebuild=False,
):
    """Encode each distinct prompt once and persist its frozen Qwen state."""
    if batch_size < 1:
        raise ValueError("Feature-cache batch size must be at least 1.")

    prompts = [query_prompt(query, entities, relations) for query in queries]
    unique_prompts = list(dict.fromkeys(prompts))
    hidden_size = model.proj.in_features
    cached = {} if rebuild else load_feature_cache(
        cache_path,
        model_name=model.qwen.config.name_or_path,
        max_length=max_length,
        hidden_size=hidden_size,
    )
    missing_prompts = [prompt for prompt in unique_prompts if prompt not in cached]

    model.eval()
    progress = tqdm(
        range(0, len(missing_prompts), batch_size),
        desc="Caching frozen Qwen features",
    )
    with torch.no_grad():
        for start in progress:
            prompt_batch = missing_prompts[start:start + batch_size]
            encoded = tokenizer(
                prompt_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }
            features = model.encode_frozen_features(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            ).detach().to(device="cpu", dtype=torch.float32)
            for prompt, feature in zip(prompt_batch, features):
                cached[prompt] = feature.clone()
            progress.set_postfix(
                cached=len(cached),
                required=len(unique_prompts),
            )

    if missing_prompts or rebuild:
        atomic_torch_save(
            {
                "version": 1,
                "model_name": model.qwen.config.name_or_path,
                "max_length": max_length,
                "prompt_version": PROMPT_VERSION,
                "hidden_size": hidden_size,
                "features": cached,
            },
            cache_path,
        )
    print(
        f"feature_cache={cache_path} unique_prompts={len(unique_prompts)} "
        f"new_prompts={len(missing_prompts)}"
    )
    return torch.stack([cached[prompt] for prompt in prompts])


def evaluate_cached_ranking(
    model,
    queries,
    features,
    all_true_entities,
    entity_to_idx,
    device,
    batch_size,
    description,
):
    """Compute combined and direction-specific filtered ranking metrics."""
    if len(queries) != len(features):
        raise ValueError("Evaluation query and feature counts do not match.")
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be at least 1.")

    ranks_by_direction = {"head": [], "tail": []}
    model.eval()

    progress = tqdm(
        range(0, len(queries), batch_size),
        desc=description,
    )
    with torch.no_grad():
        for start in progress:
            query_batch = queries[start:start + batch_size]
            feature_batch = features[start:start + batch_size].to(device)
            logits = model(
                features=feature_batch,
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

    combined_ranks = (
        ranks_by_direction["head"]
        + ranks_by_direction["tail"]
    )
    metrics = ranks_to_metrics(combined_ranks)
    metrics["head_metrics"] = ranks_to_metrics(ranks_by_direction["head"])
    metrics["tail_metrics"] = ranks_to_metrics(ranks_by_direction["tail"])
    return metrics


def clone_trainable_state(model):
    """Clone only W and E; Qwen is frozen and loaded from Hugging Face."""
    return {
        "projection": {
            key: value.detach().cpu().clone()
            for key, value in model.proj.state_dict().items()
        },
        "entity_embeddings": {
            key: value.detach().cpu().clone()
            for key, value in model.entity_embeddings.state_dict().items()
        },
    }


def load_trainable_state(model, state):
    model.proj.load_state_dict(state["projection"])
    model.entity_embeddings.load_state_dict(state["entity_embeddings"])


def atomic_torch_save(value, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def save_json(value, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary_path, path)


def should_early_stop(
    evaluations_without_improvement,
    patience,
    epoch,
    num_epochs,
):
    return (
        patience > 0
        and evaluations_without_improvement >= patience
        and epoch < num_epochs
    )

#training loop 

def move_batch_to_dev(batch, device):
    return {
        key: value.to(device) for key, value in batch.items()
    }

def train_one_epoch(model, loader, optimizer, device):

    model.train() 

    total_loss = 0.0 
    total_examples = 0 

    progress = tqdm(loader, desc="Training") 

    for batch in progress:
        batch = move_batch_to_dev(batch, device) 
        targets = batch.pop("targets") 

        optimizer.zero_grad(set_to_none = True) 

        logits = model(features=batch["features"])

        loss = loss_func(logits, targets) 
        loss.backward() 

        torch.nn.utils.clip_grad_norm_(
            [
                model.proj.weight,
                model.entity_embeddings.weight
            ],
            max_norm = 1.0
        )

        optimizer.step() 

        batch_size = targets.size(0) 
        total_loss += loss.item() * batch_size 
        total_examples += batch_size 

        progress.set_postfix(
            loss = f"{loss.item():.4f}"
        )
    
    return total_loss / total_examples 

def main():

    parser = argparse.ArgumentParser() 

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
            "decoder_runs/{run_name}/{run_name}.pt."
        ),
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Evaluate filtered head+tail ranking on validation every N epochs.",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Number of cached head/tail queries per evaluation batch.",
    )

    parser.add_argument(
        "--validation_only",
        action="store_true",
        help=(
            "Tune on train/valid only: do not read or evaluate test.txt."
        ),
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
        "--feature_batch_size",
        type=int,
        default=64,
        help="Batch size used only while caching frozen Qwen features.",
    )

    parser.add_argument(
        "--feature_cache_path",
        default=None,
        help="Persistent prompt-to-Qwen-feature cache path.",
    )

    parser.add_argument(
        "--rebuild_feature_cache",
        action="store_true",
        help="Ignore compatible cached features and recompute them.",
    )

    parser.add_argument(
        "--results_path",
        default=None,
        help=(
            "JSON results path; defaults to "
            "decoder_runs/{run_name}/{run_name}.json."
        ),
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args() 
    if args.num_epochs < 1:
        parser.error("--num_epochs must be at least 1")
    if args.eval_every < 1:
        parser.error("--eval_every must be at least 1")
    if args.eval_batch_size < 1:
        parser.error("--eval_batch_size must be at least 1")
    if args.feature_batch_size < 1:
        parser.error("--feature_batch_size must be at least 1")
    if args.early_stopping_patience < 0:
        parser.error("--early_stopping_patience cannot be negative")
    if args.early_stopping_min_delta < 0:
        parser.error("--early_stopping_min_delta cannot be negative")

    default_checkpoint_path, default_results_path = default_output_paths(
        args.dataset_path
    )
    if args.output_path is None:
        args.output_path = default_checkpoint_path
    if args.results_path is None:
        args.results_path = default_results_path
    if args.feature_cache_path is None:
        args.feature_cache_path = default_feature_cache_path(
            args.dataset_path,
            args.model_name,
            args.max_length,
        )

    set_seed(args.seed) 

    device = torch.device(args.device) 

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
        entity_id: index for index, entity_id in enumerate(entity_ids) 
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        local_files_only=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    tokenizer.padding_side = "right" 

    model = QwenLinkPredictor(model_name = args.model_name,
    num_entities = len(entity_ids),
    embedding_dim = args.embedding_dim,
    dtype = torch.bfloat16).to(device)

    train_queries = build_prediction_queries(train_triples)
    valid_queries = build_prediction_queries(valid_triples)
    test_queries = (
        [] if test_triples is None
        else build_prediction_queries(test_triples)
    )
    all_queries = train_queries + valid_queries + test_queries
    all_features = cache_frozen_query_features(
        model=model,
        tokenizer=tokenizer,
        queries=all_queries,
        entities=entities,
        relations=relations,
        device=device,
        max_length=args.max_length,
        batch_size=args.feature_batch_size,
        cache_path=args.feature_cache_path,
        rebuild=args.rebuild_feature_cache,
    )
    train_end = len(train_queries)
    valid_end = train_end + len(valid_queries)
    train_features = all_features[:train_end]
    valid_features = all_features[train_end:valid_end]
    test_features = all_features[valid_end:]

    model.release_frozen_encoder()
    print_trainable_parameters(model)

    train_dataset = FrozenFeatureDataset(
        features=train_features,
        queries=train_queries,
        entity_to_idx=entity_to_idx,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=FrozenFeatureCollator(len(entity_ids)),
    )
    print(
        f"train_queries={len(train_queries)} "
        f"valid_queries={len(valid_queries)} "
        f"test_queries={len(test_queries)} "
        f"updates_per_epoch={len(train_loader)}"
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.proj.parameters(),
                "lr": args.learning_rate
            },
            {
                "params": model.entity_embeddings.parameters(),
                "lr": args.learning_rate
            }
        ],
        weight_decay = args.weight_decay
    )

    best_validation_mrr = -float("inf")
    best_validation_metrics = None
    best_epoch = None
    best_trainable_state = None
    training_history = []
    evaluations_without_improvement = 0
    early_stopped = False

    for epoch in range(1, args.num_epochs + 1):
        loss = train_one_epoch(
            model = model,
            loader = train_loader,
            optimizer = optimizer,
            device = device
        )
    
        epoch_record = {
            "epoch": epoch,
            "train_loss": loss,
        }
        print(f"epoch={epoch} train_loss={loss:.6f}")

        should_evaluate = (
            epoch % args.eval_every == 0
            or epoch == args.num_epochs
        )
        if not should_evaluate:
            training_history.append(epoch_record)
            continue

        validation_metrics = evaluate_cached_ranking(
            model=model,
            queries=valid_queries,
            features=valid_features,
            all_true_entities=validation_true_entities,
            entity_to_idx=entity_to_idx,
            device=device,
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
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        epoch_record["evaluations_without_improvement"] = (
            evaluations_without_improvement
        )
        training_history.append(epoch_record)

        if is_best:
            best_validation_mrr = validation_mrr
            best_validation_metrics = dict(validation_metrics)
            best_epoch = epoch
            best_trainable_state = clone_trainable_state(model)
            checkpoint = {
                **best_trainable_state,
                "entity_ids": entity_ids,
                "entity_to_idx": entity_to_idx,
                "model_name": args.model_name,
                "embedding_dim": args.embedding_dim,
                "max_length": args.max_length,
                "prompt_version": PROMPT_VERSION,
                "prediction_direction": "head_and_tail_no_reciprocals",
                "selection_split": "valid",
                "selection_metric": "MRR",
                "best_epoch": best_epoch,
                "best_validation_metrics": best_validation_metrics,
                "training_history": training_history,
                "args": vars(args),
            }
            atomic_torch_save(checkpoint, args.output_path)
            print(
                f"saved_best_checkpoint={args.output_path} "
                f"best_epoch={best_epoch} "
                f"best_valid_mrr={best_validation_mrr:.6f}"
            )
        should_stop = should_early_stop(
            evaluations_without_improvement=evaluations_without_improvement,
            patience=args.early_stopping_patience,
            epoch=epoch,
            num_epochs=args.num_epochs,
        )
        if should_stop:
            early_stopped = True
            print(
                f"early_stopping_epoch={epoch} "
                f"best_epoch={best_epoch} "
                f"best_valid_mrr={best_validation_mrr:.6f}"
            )
            break

    if best_trainable_state is None:
        raise RuntimeError("Training completed without validation evaluation.")

    load_trainable_state(model, best_trainable_state)
    model.to(device)
    test_metrics = None
    if not args.validation_only:
        test_metrics = evaluate_cached_ranking(
            model=model,
            queries=test_queries,
            features=test_features,
            all_true_entities=test_true_entities,
            entity_to_idx=entity_to_idx,
            device=device,
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
    else:
        print("validation_only=true test_set_not_loaded=true")

    checkpoint = {
        **best_trainable_state,
        "entity_ids": entity_ids,
        "entity_to_idx": entity_to_idx,
        "model_name": args.model_name,
        "embedding_dim": args.embedding_dim,
        "max_length": args.max_length,
        "prompt_version": PROMPT_VERSION,
        "prediction_direction": "head_and_tail_no_reciprocals",
        "selection_split": "valid",
        "selection_metric": "MRR",
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "training_history": training_history,
        "early_stopped": early_stopped,
        "feature_cache_path": os.path.abspath(args.feature_cache_path),
        "args": vars(args),
    }
    atomic_torch_save(checkpoint, args.output_path)

    results = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "model_name": args.model_name,
        "prompt_version": PROMPT_VERSION,
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
        "num_train_queries": len(train_queries),
        "num_valid_queries": len(valid_queries),
        "num_test_queries": len(test_queries),
        "num_entities": len(entity_ids),
        "training_history": training_history,
        "early_stopped": early_stopped,
        "feature_cache_path": os.path.abspath(args.feature_cache_path),
        "args": vars(args),
        "checkpoint_path": os.path.abspath(args.output_path),
    }
    save_json(results, args.results_path)
    print(f"saved_checkpoint={args.output_path}")
    print(f"saved_results={args.results_path}")

if __name__ == "__main__":
    main()








        
