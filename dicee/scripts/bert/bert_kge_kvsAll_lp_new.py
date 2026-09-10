r"""Staged BERT/KGE link prediction, selected by filtered validation MRR.

1. Train KGE on original and reciprocal K-vs-all queries, once each per epoch.
2. Freeze the final KGE; train BERT independently on random head/tail corruptions.
3. Freeze both final checkpoints; select a logit mixture on the full validation set.
4. Evaluate the selected mixture on test once, alongside both individual branches.

Example (run from the repository root)::

    python -m dicee.scripts.bert.bert_kge_kvsAll_lp_new \
        --dataset_path bert_datasets/umls-ext-sems/0.0 \
        --bert_model_path checkpoints/umls-ext-cp/bert-pretrained \
        --kge_model TransE --kge_num_epochs 100 --bert_num_epochs 10

The BERT weight includes endpoints 0 and 1 and is the only fusion search parameter.
No validation/test labels are used for training or negative sampling.
Validation filtering uses train + valid facts; test filtering uses all three splits.
Ties use Dice's default rank rule: one plus the number of strictly higher scores.

Each branch validates only after its last epoch; there is no validation-based
early stopping or selection among training epochs. For checkpoint compatibility,
final branch checkpoints retain the names best_kge.pt and best_bert.pt.

best_kge.pt and best_bert.pt can be reused with --kge_checkpoint/--bert_checkpoint
to skip completed stages. These flags accept this script's checkpoint format and
verify dataset, model, and text mappings. They do not resume optimizer state.
For legacy BERT checkpoints without a saved tokenizer, supply the original
--tokenizer_path if it was not bert-base-cased and preserve support JSON ordering.
"""

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertModel

from dicee.scripts.bert_bce_link_prediction import (
    atomic_json_save,
    atomic_torch_save,
    create_unique_output_dir,
    dataset_run_name,
    load_torch_checkpoint,
    negative_sampling_bce_loss,
    post_kge_parameter_update,
    read_support,
    read_triples,
)
from dicee.scripts.bert_bce_kvsall_link_prediction import (
    create_kvsall_kge_model,
    kvsall_bce_loss,
)


PIPELINE = "staged_bert_kge_kvsall_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def index_triples(triples, entity_to_idx, relation_to_idx):
    try:
        return [(entity_to_idx[h], relation_to_idx[r], entity_to_idx[t])
                for h, r, t in triples]
    except KeyError as error:
        raise ValueError(f"Triple identifier absent from support metadata: {error}") from error


def query_for_triple(triple, direction, num_relations):
    """Head prediction uses the separately trained inverse relation."""
    h, r, t = triple
    if direction == "tail":
        return (h, r), t
    if direction == "head":
        return (t, r + num_relations), h
    raise ValueError(f"Unknown prediction direction: {direction}")


def build_fact_index(triples, num_relations):
    facts = defaultdict(set)
    for triple in triples:
        for direction in ("tail", "head"):
            query, target = query_for_triple(triple, direction, num_relations)
            facts[query].add(target)
    return dict(facts)


class ReciprocalKvsAllDataset(Dataset):
    def __init__(self, triples, num_entities, num_relations):
        self.facts = build_fact_index(triples, num_relations)
        self.queries = list(self.facts)
        self.num_entities = num_entities
        if not self.queries:
            raise ValueError("KGE training requires non-empty training triples.")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        targets = torch.zeros(self.num_entities)
        targets[list(self.facts[query])] = 1.0
        return torch.tensor(query, dtype=torch.long), targets


class CalibratedKGE(nn.Module):
    """A positive scale and bias make distance scores usable with dense BCE."""
    def __init__(self, kge_model, num_entities):
        super().__init__()
        self.kge_model = kge_model
        self.num_entities = num_entities
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, queries):
        scores = self.kge_model.forward_k_vs_all(queries)
        if scores.ndim == 3 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        if scores.shape != (len(queries), self.num_entities):
            raise ValueError(f"Unexpected KGE score shape: {tuple(scores.shape)}")
        return self.log_scale.exp() * scores + self.bias

    def post_update(self):
        post_kge_parameter_update(self.kge_model)
        with torch.no_grad():
            self.log_scale.clamp_(-8.0, 8.0)


def load_tokenizer_and_config(args, entities, relations):
    """Reject architecture mismatches before starting an expensive KGE stage."""
    model_path = str(resolve_path(args.bert_model_path))
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type != "bert":
        raise ValueError(
            f"This script requires a BERT checkpoint; found {config.model_type!r} "
            f"in {model_path}. Use matching BERT weights and tokenizer."
        )
    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError("--max_seq_length exceeds the BERT position embedding limit.")
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        tokenizer_path = (model_path if (Path(model_path) / "tokenizer_config.json").exists()
                          else "bert-base-cased")
    elif resolve_path(tokenizer_path).exists():
        tokenizer_path = str(resolve_path(tokenizer_path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)
    if not type(tokenizer).__name__.startswith("BertTokenizer"):
        raise ValueError(f"Expected a BERT tokenizer, got {type(tokenizer).__name__}.")
    tokens = [e["name"] for e in entities.values()]
    tokens += [r[f"sep{i}"] for r in relations.values() for i in range(1, 6)]
    # One call preserves both the entity and relation special-token lists.
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if len(set(tokenizer.convert_tokens_to_ids(tokens))) != len(tokens):
        raise ValueError("Dataset special tokens do not have unique tokenizer IDs.")
    print(f"bert_checkpoint_vocab={config.vocab_size} tokenizer_vocab={len(tokenizer)} "
          f"tokenizer_source={tokenizer_path}")
    return tokenizer, config


class BudgetedTripleEncoder:
    """Tokenize metadata once and share the remaining token budget fairly.

    Entity/relation IDs keep their N-BERT prompt positions. Raw entity names are
    included with descriptions, so useful names are not replaced solely by IDs.
    Both training and candidate evaluation use exactly this encoder.
    """
    def __init__(self, tokenizer, entities, relations, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.entities = list(entities.values())
        self.relations = list(relations.values())
        if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
            raise ValueError("BERT tokenizer must define CLS and SEP tokens.")
        self.entity_tokens = [tokenizer.convert_tokens_to_ids(e["name"])
                              for e in self.entities]
        self.separators = [[tokenizer.convert_tokens_to_ids(r[f"sep{i}"])
                            for i in range(1, 6)] for r in self.relations]
        self.descriptions = [self.tokenize(
            f"{e.get('raw_name', '')}: {e['desc']}" if e.get("raw_name") else str(e["desc"])
        ) for e in self.entities]
        self.relation_names = [self.tokenize(r["name"]) for r in self.relations]
        # Seven prompt markers/entity IDs, two BERT boundary tokens, and room
        # for at least one token on each description side.
        if any(len(name) + 11 > max_seq_length for name in self.relation_names):
            raise ValueError("--max_seq_length leaves no room for both descriptions.")

    def tokenize(self, text):
        return self.tokenizer.encode(str(text), add_special_tokens=False,
                                     truncation=True, max_length=self.max_seq_length)

    def encode(self, triple):
        h, r, t = (int(value) for value in triple)
        s1, s2, s3, s4, s5 = self.separators[r]
        prefix = [s1, self.entity_tokens[h], s2] + self.relation_names[r]
        prefix += [s3, self.entity_tokens[t], s4]
        budget = self.max_seq_length - len(prefix) - 3  # CLS, SEP, and s5
        head, tail = self.descriptions[h], self.descriptions[t]
        head_count = min(len(head), budget // 2)
        tail_count = min(len(tail), budget - head_count)
        head_count = min(len(head), budget - tail_count)  # redistribute unused room
        ids = [self.tokenizer.cls_token_id] + prefix + head[:head_count]
        ids += [s5] + tail[:tail_count] + [self.tokenizer.sep_token_id]
        return {"input_ids": ids, "attention_mask": [1] * len(ids),
                "token_type_ids": [0] * len(ids)}

    def batch(self, triples, device):
        features = [self.encode(triple) for triple in triples]
        encoded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        return {key: value.to(device) for key, value in encoded.items()}


class BertTripleScorer(nn.Module):
    def __init__(self, model_path, tokenizer):
        super().__init__()
        # MLM checkpoints have no trained pooler. Use the contextual CLS vector.
        self.bert = BertModel.from_pretrained(str(model_path), add_pooling_layer=False)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, **encoded):
        output = self.bert(**encoded)
        return self.classifier(output.last_hidden_state[:, 0]).squeeze(-1)


class RandomNegativeSampler:
    """Randomly replace a head or tail, excluding known training positives."""
    def __init__(self, train_facts, candidate_ids, num_relations, negative_ratio, seed):
        self.facts = train_facts
        self.candidates = list(candidate_ids)
        self.num_relations = num_relations
        self.negative_ratio = negative_ratio
        self.rng = random.Random(seed)

    def corrupt(self, triple):
        directions = ["tail", "head"]
        self.rng.shuffle(directions)
        # Rejection sampling, followed by an exhaustive fallback for dense rows.
        for direction in directions:
            query, _ = query_for_triple(triple, direction, self.num_relations)
            true = self.facts.get(query, set())
            for _ in range(100):
                candidate = self.rng.choice(self.candidates)
                if candidate not in true:
                    return self.replace(triple, direction, candidate)
            remaining = [candidate for candidate in self.candidates if candidate not in true]
            if remaining:
                return self.replace(triple, direction, self.rng.choice(remaining))
        raise ValueError(f"No false head or tail corruption is available for {triple}.")

    @staticmethod
    def replace(triple, direction, candidate):
        h, r, t = triple
        return (h, r, candidate) if direction == "tail" else (candidate, r, t)

    def batch(self, positives):
        triples, labels = [], []
        for positive in positives:
            positive = tuple(int(value) for value in positive)
            triples.append(positive)
            labels.append(1.0)
            for _ in range(self.negative_ratio):
                triples.append(self.corrupt(positive))
                labels.append(0.0)
        return triples, torch.tensor(labels)


def filtered_rank(scores, target, known_targets):
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite candidate scores; refusing to report misleading MRR.")
    target_score = scores[target]
    scores = scores.clone()
    other_true = [index for index in known_targets if index != target]
    scores[other_true] = -torch.inf
    greater = int((scores > target_score).sum())
    return 1.0 + greater


class RankingMetrics:
    def __init__(self):
        self.count = 0
        self.rr = 0.0
        self.hits = {k: 0 for k in (1, 3, 10)}
        self.direction_rr = {"head": 0.0, "tail": 0.0}
        self.direction_count = {"head": 0, "tail": 0}

    def add(self, rank, direction):
        self.count += 1
        self.rr += 1.0 / rank
        self.direction_rr[direction] += 1.0 / rank
        self.direction_count[direction] += 1
        for k in self.hits:
            self.hits[k] += rank <= k

    def result(self):
        if not self.count:
            raise ValueError("Cannot evaluate an empty split.")
        return {"MRR": self.rr / self.count,
                **{f"H@{k}": value / self.count for k, value in self.hits.items()},
                **{f"{d}_MRR": self.direction_rr[d] / self.direction_count[d]
                   for d in self.direction_rr}, "num_queries": self.count}


def fusion_grid(lambdas):
    """Endpoints are unconditional; ties in validation MRR favor pure KGE."""
    configs = [{"bert_weight": 0.0}]
    for weight in sorted(set(lambdas)):
        if 0 < weight < 1:
            configs.append({"bert_weight": weight})
    configs.append({"bert_weight": 1.0})
    return configs


def fuse_scores(kge_scores, bert_scores, config):
    weight = config["bert_weight"]
    # Explicit endpoints avoid 0 * inf/NaN and unnecessary branch dependencies.
    if weight == 0:
        return kge_scores
    if weight == 1:
        return bert_scores
    return (1 - weight) * kge_scores + weight * bert_scores


@torch.inference_mode()
def score_bert_query(bert, encoder, triple, direction, num_entities, batch_size, device):
    scores = []
    for start in range(0, num_entities, batch_size):
        candidates = [RandomNegativeSampler.replace(triple, direction, candidate)
                      for candidate in range(start, min(start + batch_size, num_entities))]
        scores.append(bert(**encoder.batch(candidates, device)).float().cpu())
    return torch.cat(scores)


@torch.inference_mode()
def evaluate_rankings(triples, facts, num_entities, num_relations, device,
                      candidate_batch_size, kge_eval_batch_size,
                      configs, kge=None, bert=None, encoder=None, description="Validation"):
    """Stream each query's scores through every mixture; never store Q x E logits."""
    if kge is not None:
        kge.eval()
    if bert is not None:
        bert.eval()
    need_kge = any(c["bert_weight"] < 1 for c in configs)
    need_bert = any(c["bert_weight"] > 0 for c in configs)
    if (need_kge and kge is None) or (need_bert and (bert is None or encoder is None)):
        raise ValueError("Missing model/encoder required by the evaluation mixtures.")
    queries = [(triple, direction, *query_for_triple(triple, direction, num_relations))
               for triple in triples for direction in ("tail", "head")]
    metrics = [RankingMetrics() for _ in configs]
    for start in tqdm(range(0, len(queries), kge_eval_batch_size), desc=description):
        batch = queries[start:start + kge_eval_batch_size]
        kge_batch = None
        if need_kge:
            kge_batch = kge(torch.tensor([q for _, _, q, _ in batch],
                                         dtype=torch.long, device=device)).float().cpu()
        for row, (triple, direction, query, target) in enumerate(batch):
            kge_scores = kge_batch[row] if need_kge else None
            bert_scores = (score_bert_query(bert, encoder, triple, direction, num_entities,
                                            candidate_batch_size, device) if need_bert else None)
            for config, accumulator in zip(configs, metrics):
                rank = filtered_rank(fuse_scores(kge_scores, bert_scores, config), target,
                                     facts.get(query, set()))
                accumulator.add(rank, direction)
    return [{**config, "metrics": accumulator.result()}
            for config, accumulator in zip(configs, metrics)]


def restore_branch(model, path, expected_metadata):
    checkpoint = load_torch_checkpoint(str(path))
    if checkpoint.get("metadata") != expected_metadata:
        raise ValueError(
            f"Checkpoint {path} does not match this stage's dataset/model/token mapping. "
            "Use a matching checkpoint produced by this script."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def fit_branch(model, loader, optimizer, loss_for_batch, evaluate, num_epochs,
               output_dir, branch, metadata,
               post_update=None):
    """Train all requested epochs, then validate and freeze the final model."""
    history = []
    checkpoint_path = output_dir / f"best_{branch}.pt"
    for epoch in range(1, num_epochs + 1):
        model.train()
        loss_sum, example_count = 0.0, 0
        for batch in tqdm(loader, desc=f"{branch} epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            loss, count = loss_for_batch(batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite {branch} training loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            if post_update is not None:
                post_update()
            loss_sum += float(loss.detach()) * count
            example_count += count
        record = {"epoch": epoch, "loss": loss_sum / example_count,
                  "optimizer_steps": len(loader), "validation_metrics": None}
        if epoch == num_epochs:
            metrics = evaluate()
            record["validation_metrics"] = metrics
            record["checkpoint_policy"] = "final_epoch"
            atomic_torch_save(str(checkpoint_path), {
                "metadata": metadata,
                "model_state_dict": {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()},
                # Retain this key for compatibility with checkpoint reuse.
                "best_epoch": epoch, "validation_metrics": metrics,
                "checkpoint_policy": "final_epoch",
            })
        history.append(record)
        atomic_json_save(str(output_dir / f"{branch}_history.json"), history)
        print(json.dumps({"stage": branch, **record}), flush=True)
    model.eval()
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return {"checkpoint": str(checkpoint_path), "best_epoch": num_epochs,
            "epochs_ran": len(history), "checkpoint_policy": "final_epoch",
            "validation_metrics": metrics}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--support_path")
    parser.add_argument("--bert_model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="Defaults to the checkpoint tokenizer, or bert-base-cased for legacy checkpoints.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="bert_kge_staged_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_split", choices=["valid", "test"], default="test")
    parser.add_argument("--kge_model", choices=["TransE", "RotatE", "MuRE"], default="TransE")
    parser.add_argument("--kge_embedding_dim", type=int, default=128)
    parser.add_argument("--kge_lr", type=float, default=0.01)
    parser.add_argument("--calibration_lr", type=float, default=0.001)
    parser.add_argument("--kge_loss_weight", type=float, default=1.0)
    parser.add_argument("--kge_batch_size", type=int, default=1024)
    parser.add_argument("--kge_eval_batch_size", type=int, default=32)
    parser.add_argument("--kge_num_epochs", type=int, default=100)
    parser.add_argument("--kge_kvsall_candidate_scope", choices=["train", "all"], default="all")
    parser.add_argument("--bert_lr", "--lr", dest="bert_lr", type=float, default=2e-5)
    parser.add_argument("--bert_batch_size", "--batch_size", dest="bert_batch_size", type=int, default=128,
                        help="Total examples including negatives; rounded down to complete positive groups.")
    parser.add_argument("--bert_num_epochs", type=int, default=10)
    parser.add_argument("--negative_ratio", type=int, default=5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--candidate_batch_size", type=int, default=128)
    parser.add_argument("--fusion_lambdas", type=float, nargs="+",
                        default=[i / 10 for i in range(11)])
    parser.add_argument("--kge_checkpoint", help="Skip KGE training using a matching best_kge.pt from this script.")
    parser.add_argument("--bert_checkpoint", help="Skip BERT training using a matching best_bert.pt from this script.")
    return parser


def validate_args(args):
    positive_ints = ("kge_embedding_dim", "kge_batch_size", "kge_eval_batch_size",
                    "kge_num_epochs", "bert_batch_size", "bert_num_epochs",
                    "negative_ratio", "candidate_batch_size")
    for name in positive_ints:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive.")
    for name in ("bert_lr", "kge_lr", "calibration_lr", "kge_loss_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name} must be finite and positive.")
    if args.num_workers < 0 or args.max_seq_length < 12:
        raise ValueError("--num_workers must be nonnegative and --max_seq_length at least 12.")
    if args.bert_batch_size < 1 + args.negative_ratio:
        raise ValueError("--bert_batch_size must fit a positive and all its negatives.")
    if any(not 0 <= weight <= 1 for weight in args.fusion_lambdas):
        raise ValueError("All --fusion_lambdas must be between 0 and 1.")


def main(args=None):
    args = build_parser().parse_args() if args is None else args
    validate_args(args)
    started = time.perf_counter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")
        device = torch.device("cpu")
    dataset_path = resolve_path(args.dataset_path)
    entities, relations = read_support(str(dataset_path),
                                      str(resolve_path(args.support_path)) if args.support_path else None)
    entity_ids, relation_ids = list(entities), list(relations)
    ei, ri = ({key: i for i, key in enumerate(ids)} for ids in (entity_ids, relation_ids))
    train = index_triples(read_triples(str(dataset_path / "train.txt")), ei, ri)
    valid = index_triples(read_triples(str(dataset_path / "valid.txt")), ei, ri)
    if not train or not valid or len(entities) < 2:
        raise ValueError("Nonempty train/valid splits and at least two entities are required.")
    # Test is intentionally read only after all checkpoint and fusion decisions.
    n_entities, n_relations = len(ei), len(ri)
    tokenizer, _ = load_tokenizer_and_config(args, entities, relations)
    encoder = BudgetedTripleEncoder(tokenizer, entities, relations, args.max_seq_length)
    train_facts = build_fact_index(train, n_relations)
    valid_facts = build_fact_index(train + valid, n_relations)
    train_entities = sorted({entity for h, _, t in train for entity in (h, t)})
    run_name = (f"{dataset_run_name(args.dataset_path)}_{args.kge_model}_staged_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir = Path(create_unique_output_dir(str(resolve_path(args.output_dir)), run_name))
    print(f"run_output_dir={output_dir}", flush=True)
    atomic_json_save(str(output_dir / "args.json"), vars(args))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    atomic_json_save(str(output_dir / "token_mapping.json"), {
        "entity_ids": entity_ids, "relation_ids": relation_ids,
        "entity_token_ids": encoder.entity_tokens, "relation_token_ids": encoder.separators,
        "inverse_relation_offset": n_relations,
    })
    common_metadata = {"pipeline": PIPELINE, "entity_ids": entity_ids, "relation_ids": relation_ids,
                       "train_fingerprint": fingerprint(train), "valid_fingerprint": fingerprint(valid),
                       "rank_policy": "dice_default_strictly_greater"}
    kge_metadata = {**common_metadata, "branch": "kge", "model": args.kge_model,
                    "embedding_dim": args.kge_embedding_dim, "reciprocal": True,
                    "candidate_scope": args.kge_kvsall_candidate_scope}
    kge = CalibratedKGE(create_kvsall_kge_model(
        args.kge_model, n_entities, 2 * n_relations, embedding_dim=args.kge_embedding_dim,
        random_seed=args.seed, learning_rate=args.kge_lr), n_entities).to(device)
    eval_kwargs = dict(num_entities=n_entities, num_relations=n_relations, device=device,
                       candidate_batch_size=args.candidate_batch_size,
                       kge_eval_batch_size=args.kge_eval_batch_size)
    kge_config = {"bert_weight": 0.0}
    bert_config = {"bert_weight": 1.0}

    if args.kge_checkpoint:
        path = resolve_path(args.kge_checkpoint)
        checkpoint = restore_branch(kge, path, kge_metadata)
        kge_stage = {"checkpoint": str(path), "best_epoch": checkpoint["best_epoch"],
                     "epochs_ran": 0, "validation_metrics": checkpoint["validation_metrics"]}
        kge.requires_grad_(False).eval()
    else:
        kge_dataset = ReciprocalKvsAllDataset(train, n_entities, n_relations)
        kge_loader = DataLoader(kge_dataset, batch_size=args.kge_batch_size, shuffle=True,
                                num_workers=args.num_workers,
                                generator=torch.Generator().manual_seed(args.seed))
        candidates = (train_entities if args.kge_kvsall_candidate_scope == "train"
                      else list(range(n_entities)))
        candidates = torch.tensor(candidates, dtype=torch.long, device=device)
        optimizer = torch.optim.Adam([
            {"params": kge.kge_model.parameters(), "lr": args.kge_lr},
            {"params": [kge.log_scale, kge.bias], "lr": args.calibration_lr},
        ])

        def kge_loss(batch):
            queries, targets = (value.to(device) for value in batch)
            loss = kvsall_bce_loss(kge(queries), targets, candidates)
            return args.kge_loss_weight * loss, len(queries)

        def evaluate_kge():
            return evaluate_rankings(valid, valid_facts, configs=[kge_config], kge=kge,
                                     description="KGE validation", **eval_kwargs)[0]["metrics"]

        print(f"kge_rows_per_epoch={len(kge_dataset)} kge_steps_per_epoch={len(kge_loader)} "
              "kge_training_directions=tail_and_reciprocal_head")
        kge_stage = fit_branch(kge, kge_loader, optimizer, kge_loss, evaluate_kge,
                               args.kge_num_epochs, output_dir, "kge", kge_metadata, kge.post_update)
        del optimizer

    # KGE is frozen for all subsequent stages, including BERT training.
    torch.manual_seed(args.seed + 1)
    bert = BertTripleScorer(resolve_path(args.bert_model_path), tokenizer).to(device)
    bert_metadata = {**common_metadata, "branch": "bert", "max_seq_length": args.max_seq_length,
                     "text_fingerprint": fingerprint([entities, relations]),
                     "tokenizer_fingerprint": fingerprint(tokenizer.get_vocab())}
    if args.bert_checkpoint:
        path = resolve_path(args.bert_checkpoint)
        checkpoint = restore_branch(bert, path, bert_metadata)
        bert_stage = {"checkpoint": str(path), "best_epoch": checkpoint["best_epoch"],
                      "epochs_ran": 0, "validation_metrics": checkpoint["validation_metrics"]}
        bert.requires_grad_(False).eval()
    else:
        sampler = RandomNegativeSampler(train_facts, train_entities, n_relations,
                                         args.negative_ratio, args.seed + 2)
        positive_batch_size = args.bert_batch_size // (1 + args.negative_ratio)
        bert_loader = DataLoader(TensorDataset(torch.tensor(train, dtype=torch.long)),
                                 batch_size=positive_batch_size, shuffle=True,
                                 num_workers=args.num_workers,
                                 generator=torch.Generator().manual_seed(args.seed + 1))
        optimizer = torch.optim.AdamW(bert.parameters(), lr=args.bert_lr, weight_decay=0.01)

        def bert_loss(batch):
            triples, labels = sampler.batch(batch[0].tolist())
            logits = bert(**encoder.batch(triples, device))
            loss = negative_sampling_bce_loss(logits, labels.to(device), args.negative_ratio,
                                              weighting="balanced")
            return loss, len(labels)

        def evaluate_bert():
            return evaluate_rankings(valid, valid_facts, configs=[bert_config], bert=bert,
                                     encoder=encoder, description="BERT validation",
                                     **eval_kwargs)[0]["metrics"]

        print(f"bert_steps_per_epoch={len(bert_loader)} "
              f"bert_effective_batch_size={positive_batch_size * (1 + args.negative_ratio)} "
              f"random_negatives_per_positive={args.negative_ratio}")
        bert_stage = fit_branch(bert, bert_loader, optimizer, bert_loss, evaluate_bert,
                                args.bert_num_epochs, output_dir, "bert", bert_metadata)
        del optimizer, sampler

    configs = fusion_grid(args.fusion_lambdas)
    validation = evaluate_rankings(valid, valid_facts, configs=configs, kge=kge, bert=bert,
                                   encoder=encoder, description="Selecting frozen fusion",
                                   **eval_kwargs)
    selected = max(validation, key=lambda row: row["metrics"]["MRR"])
    selected_config = {"bert_weight": selected["bert_weight"]}
    fusion = {"formula": "(1 - bert_weight) * kge_logit + bert_weight * bert_logit",
              "selected": selected, "validation_grid": validation,
              "selection_split": "valid", "selection_metric": "MRR",
              "kge_checkpoint": kge_stage["checkpoint"], "bert_checkpoint": bert_stage["checkpoint"],
              "tokenizer_path": str(output_dir / "tokenizer"),
              "token_mapping_path": str(output_dir / "token_mapping.json"),
              "rank_policy": "dice_default_strictly_greater", "max_seq_length": args.max_seq_length}
    atomic_json_save(str(output_dir / "fusion.json"), fusion)
    print(f"selected_fusion={json.dumps(selected)}", flush=True)
    if args.eval_split == "test":
        test = index_triples(read_triples(str(dataset_path / "test.txt")), ei, ri)
        test_facts = build_fact_index(train + valid + test, n_relations)
        # Score each test query once per branch, reporting all three configurations
        # without using test metrics to revise the chosen mixture.
        final_rows = evaluate_rankings(test, test_facts,
                                       configs=[kge_config, bert_config, selected_config],
                                       kge=kge, bert=bert, encoder=encoder,
                                       description="Final test", **eval_kwargs)
    else:
        final_rows = [validation[0], validation[-1], selected]
    results = {"pipeline": PIPELINE, "args": vars(args), "output_dir": str(output_dir),
               "num_entities": n_entities, "num_relations": n_relations,
               "kge_stage": kge_stage, "bert_stage": bert_stage,
               "selected_fusion": selected, "eval_split": args.eval_split,
               "kge_metrics": final_rows[0]["metrics"], "bert_metrics": final_rows[1]["metrics"],
               "metrics": final_rows[2]["metrics"], "runtime_min": (time.perf_counter() - started) / 60}
    atomic_json_save(str(output_dir / "results.json"), results)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
