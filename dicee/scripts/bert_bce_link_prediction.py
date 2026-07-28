import argparse
import json
import math
import os
import random
from datetime import datetime
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import time
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
    
    @property
    def mixing_weight(self): 
        return torch.sigmoid(self.lambda_logit)
    
    def forward(
        self, 
        indexed_triples,
        input_ids,
        attention_mask,
        token_type_ids=None,
        return_components=False,
    ):

        bert_logits = self.bert_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )

        kge_logits = self.kge_model.forward_triples(indexed_triples)
        kge_logits = kge_logits.reshape_as(bert_logits)
        
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
        relation_to_idx, max_seq_length, negative_ratio=1):
        # Store triples and metadata needed to create positive and corrupted examples.
        self.triples = triples
        self.entities = entities
        self.relations = relations
        self.entity_ids = list(entities.keys())
        self.true_triples = set(triples)
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
):
    # Evaluate filtered head and tail prediction ranks for every held-out triple.
    entity_ids = list(entities.keys())
    entity_to_pos = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
    ranks = []

    for head, relation, tail in tqdm(
        eval_triples,
        desc=progress_description,
    ):

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
        for idx, candidate_tail in enumerate(entity_ids):
            candidate = (head, relation, candidate_tail)
            if candidate_tail != tail and candidate in all_true_triples:
                tail_scores[idx] = -float("inf")
        ranks.append(rank_of_target(tail_scores, entity_to_pos[tail]))

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
        for idx, candidate_head in enumerate(entity_ids):
            candidate = (candidate_head, relation, tail)
            if candidate_head != head and candidate in all_true_triples:
                head_scores[idx] = -float("inf")
        ranks.append(rank_of_target(head_scores, entity_to_pos[head]))

    ranks = torch.tensor(ranks, dtype=torch.float)
    return {
        "MRR": float((1.0 / ranks).mean().item()),
        "H@1": float((ranks <= 1).float().mean().item()),
        "H@3": float((ranks <= 3).float().mean().item()),
        "H@10": float((ranks <= 10).float().mean().item()),
    }


def parse_args(use_fixed_lambda=False):
    # Define CLI options for training and evaluating the BERT BCE baseline.
    parser = argparse.ArgumentParser(description="Train/evaluate a BERT BCE link-prediction baseline.")
    parser.add_argument("--dataset_path", type=str, default="KGs/UMLS")
    parser.add_argument("--support_path", type=str, default=None)
    parser.add_argument("--bert_model_path", type=str, default="checkpoints/umls/bert-pretrained")
    parser.add_argument("--tokenizer_path", type=str, default="bert-base-cased")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--candidate_batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--negative_ratio", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_split", type=str, default="test", choices=["valid", "test"])
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
        help="Evaluate validation MRR every N epochs.",
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
    if args.early_stopping_patience < 1:
        raise ValueError("--early_stopping_patience must be at least 1.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early_stopping_min_delta cannot be negative.")
    if args.early_stopping_warmup_epochs < 0:
        raise ValueError("--early_stopping_warmup_epochs cannot be negative.")
    if args.eval_every < 1:
        raise ValueError("--eval_every must be at least 1.")

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
    run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_variant = (
        "joint"
        if fixed_lambda is None
        else f"joint_fixed_lambda_{fixed_lambda:g}"
    )
    run_name = (
        f"{dataset_name}_{run_variant}_{args.kge_model}_"
        f"{args.eval_split}_{run_datetime}"
    )

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
    model.to(device)

    train_dataset = TripleBCEDataset(
        triples=train_triples,
        entities=entities,
        relations=relations,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        max_seq_length=args.max_seq_length,
        negative_ratio=args.negative_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_text(batch, tokenizer, args.max_seq_length),
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
    patience_reference_mrr = -float("inf")
    best_epoch = None
    best_final_lambda = None
    evaluations_without_improvement = 0
    epochs_ran = 0
    early_stopped = False
    output_dir = None
    checkpoint_path = None
    for epoch in range(1, args.num_epochs + 1):
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
            loss = example_losses.mean()
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
        mean_loss = sum(losses) / max(len(losses), 1)
        epoch_diagnostics = {
            "epoch": epoch,
            "loss": mean_loss,
            "lambda": lambda_value,
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
        print(
            f"epoch={epoch} "
            f"loss={mean_loss:.6f} "
            f"lambda={lambda_value:.6f}"
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

        should_evaluate = (
            epoch % args.eval_every == 0
            or epoch == args.num_epochs
        )
        if should_evaluate:
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
                progress_description=f"Validating epoch {epoch}",
            )
            epoch_diagnostics["validation_metrics"] = validation_metrics
            validation_mrr = float(validation_metrics["MRR"])
            is_best = validation_mrr > best_validation_mrr

            if is_best:
                best_validation_mrr = validation_mrr
                best_validation_metrics = dict(validation_metrics)
                best_epoch = epoch
                best_final_lambda = lambda_value
                best_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

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

            epoch_diagnostics["is_best"] = is_best
            epoch_diagnostics["patience"] = (
                evaluations_without_improvement
            )
            print(
                f"validation_epoch={epoch} "
                f"valid_mrr={validation_mrr:.6f} "
                f"best_valid_mrr={best_validation_mrr:.6f} "
                f"best_epoch={best_epoch} "
                f"patience={evaluations_without_improvement}/"
                f"{args.early_stopping_patience}"
            )

        training_history.append(epoch_diagnostics)
        if should_evaluate and is_best:
            if output_dir is None:
                output_dir = create_unique_output_dir(
                    base_output_dir,
                    run_name,
                )
                checkpoint_path = os.path.join(
                    output_dir,
                    f"joint_bert_{args.kge_model}_"
                    "bce_link_prediction.pt",
                )
                tokenizer.save_pretrained(output_dir)
            atomic_torch_save(
                checkpoint_path,
                {
                    "model_state_dict": best_model_state,
                    "metrics": best_validation_metrics,
                    "args": vars(args),
                    "lambda_mode": lambda_mode,
                    "lambda_requires_grad": (
                        model.lambda_logit.requires_grad
                    ),
                    "configured_lambda": configured_lambda,
                    "final_lambda": best_final_lambda,
                    "selection_split": "valid",
                    "selection_metric": "MRR",
                    "validation_filter_scope": (
                        "train_valid_test"
                    ),
                    "best_epoch": best_epoch,
                    "epochs_ran": epochs_ran,
                    "best_validation_metrics": (
                        best_validation_metrics
                    ),
                    "best_final_lambda": best_final_lambda,
                    "training_history": training_history,
                },
            )
            print(
                f"saved_best_checkpoint={checkpoint_path} "
                f"best_epoch={best_epoch}"
            )
        if (
            should_evaluate
            and epoch >= args.early_stopping_warmup_epochs
            and evaluations_without_improvement
            >= args.early_stopping_patience
            and epoch < args.num_epochs
        ):
            early_stopped = True
            print(
                f"early_stopping_epoch={epoch} "
                f"best_epoch={best_epoch} "
                f"best_valid_mrr={best_validation_mrr:.6f}"
            )
            break

    if best_model_state is None:
        raise RuntimeError(
            "Training completed without a validation evaluation."
        )

    model.load_state_dict(best_model_state)
    lambda_value = float(model.mixing_weight.detach().cpu())
    print(
        f"restored_best_epoch={best_epoch} "
        f"best_valid_mrr={best_validation_mrr:.6f} "
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
            progress_description="Evaluating best checkpoint on test",
        )
    print(
        f"final_{args.eval_split}_metrics="
        f"{json.dumps(metrics, sort_keys=True)}"
    )

    runtim_sec = time.perf_counter() - run_start_time
    runtime_min = runtim_sec / 60.0
    if output_dir is None or checkpoint_path is None:
        raise RuntimeError("The best-checkpoint output path was not created.")

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
        "lambda_requires_grad": model.lambda_logit.requires_grad,
        "configured_lambda": configured_lambda,
        "final_lambda": lambda_value,
        "selection_split": "valid",
        "selection_metric": "MRR",
        "validation_filter_scope": "train_valid_test",
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "early_stopped": early_stopped,
        "best_validation_metrics": best_validation_metrics,
        "best_final_lambda": best_final_lambda,
        "training_history": training_history,
        "output_dir": output_dir,
        "runtime_min": runtime_min,
    }
    with open(os.path.join(output_dir, "results.json"), "x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    atomic_torch_save(
        checkpoint_path,
        {
            "model_state_dict": best_model_state,
            "metrics": metrics,
            "args": vars(args),
            "lambda_mode": lambda_mode,
            "lambda_requires_grad": model.lambda_logit.requires_grad,
            "configured_lambda": configured_lambda,
            "final_lambda": lambda_value,
            "selection_split": "valid",
            "selection_metric": "MRR",
            "validation_filter_scope": "train_valid_test",
            "best_epoch": best_epoch,
            "epochs_ran": epochs_ran,
            "early_stopped": early_stopped,
            "best_validation_metrics": best_validation_metrics,
            "best_final_lambda": best_final_lambda,
            "training_history": training_history,
        },
    )
    tokenizer.save_pretrained(output_dir)

    print(f"best_checkpoint_dir={output_dir}")
    print(f"runtime_minutes={runtime_min:.2f}")


if __name__ == "__main__":
    main()
