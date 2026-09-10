"""Train a joint BERT/KGE link predictor with KGE K-vs-all supervision.

BERT still trains on the positive-plus-corruption batches used by
``bert_bce_link_prediction.py``.  KGE instead receives a dense K-vs-all
tail-prediction loss over unique ``(head, relation)`` pairs.  The sampled
fusion loss also backpropagates into both BERT and KGE, so KGE learns from
both the K-vs-all objective and their combined sampled predictions.

K-vs-all rows are intentionally original-relation tail queries only.  The
script does not silently add reciprocal KGE relations, because doing so would
also require inverse-relation handling in the joint head-ranking evaluator.

Use ``bert_bce_kvsall_link_prediction_random_search.py`` to search KGE
learning rates, embedding dimensions, and maximum epochs using this same
training pipeline.
"""

import argparse
import json
import math
import os
import random
import time
from datetime import datetime

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from dicee.scripts.bert_bce_link_prediction import (
    BertTripleClassifier,
    GroupedNegativeBatchSampler,
    JointBCEModel,
    TripleBCEDataset,
    add_nbert_tokens,
    atomic_json_save,
    atomic_torch_save,
    collate_text,
    create_unique_output_dir,
    dataset_run_name,
    evaluate_link_prediction,
    load_torch_checkpoint,
    negative_filter_triples,
    negative_sampling_bce_loss,
    post_kge_parameter_update,
    read_support,
    read_triples,
    resolve_path,
    training_entity_ids,
)
from dicee.static_funcs import intialize_model


def create_kvsall_kge_model(
    model_name,
    num_entities,
    num_relations,
    embedding_dim=32,
    random_seed=42,
    learning_rate=0.1,
    negative_ratio=1,
    optimizer_name="Adam",
    eval_model="test",
):
    """Create one of the supported KGE models configured for K-vs-all."""
    model_name = {
        "TransE": "Pykeen_TransE",
        "RotatE": "Pykeen_RotatE",
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
        "scoring_technique": "KvsAll",
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


class KvsAllTailDataset(Dataset):
    """Map each observed ``(head, relation)`` pair to all of its train tails."""

    def __init__(self, triples, entity_to_idx, relation_to_idx):
        if not triples:
            raise ValueError("K-vs-all training requires at least one triple.")
        expected_entity_indices = set(range(len(entity_to_idx)))
        if set(entity_to_idx.values()) != expected_entity_indices:
            raise ValueError("entity_to_idx must use contiguous indices from 0.")
        expected_relation_indices = set(range(len(relation_to_idx)))
        if set(relation_to_idx.values()) != expected_relation_indices:
            raise ValueError("relation_to_idx must use contiguous indices from 0.")

        tails_by_head_relation = {}
        for head, relation, tail in triples:
            try:
                pair = (entity_to_idx[head], relation_to_idx[relation])
                tail_index = entity_to_idx[tail]
            except KeyError as error:
                raise ValueError(
                    "Training triples must be covered by entity/relation mappings."
                ) from error
            tails_by_head_relation.setdefault(pair, set()).add(tail_index)

        self.num_entities = len(entity_to_idx)
        self.head_relations = tuple(tails_by_head_relation)
        self.tail_indices = tuple(
            tuple(sorted(tails_by_head_relation[pair]))
            for pair in self.head_relations
        )

    def __len__(self):
        return len(self.head_relations)

    def __getitem__(self, index):
        target = torch.zeros(self.num_entities, dtype=torch.float)
        target[list(self.tail_indices[index])] = 1.0
        return (
            torch.tensor(self.head_relations[index], dtype=torch.long),
            target,
        )


def kge_candidate_indices(scope, train_triples, entities, entity_to_idx):
    """Select entities included in the dense KGE BCE reduction.

    ``all`` is the usual closed-world K-vs-all reduction over every support
    entity. ``train`` matches the BERT negative pool and avoids explicitly
    teaching the KGE that support-only evaluation entities are false.
    """
    if scope == "train":
        entity_ids = training_entity_ids(train_triples, entities)
    elif scope == "all":
        entity_ids = list(entities)
    else:
        raise ValueError(f"Unknown K-vs-all candidate scope: {scope!r}")
    return torch.tensor(
        [entity_to_idx[entity_id] for entity_id in entity_ids],
        dtype=torch.long,
    )


def _validate_kvsall_logits(logits, batch_size, num_entities):
    """Normalize model-specific all-tail score shapes to ``[batch, entities]``."""
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    expected_shape = (batch_size, num_entities)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            "KGE forward_k_vs_all returned shape "
            f"{tuple(logits.shape)}, expected {expected_shape}."
        )
    return logits


def forward_kvsall_logits(model, head_relations, num_entities):
    """Score every tail entity and apply the KGE branch calibration."""
    raw_logits = model.kge_model.forward_k_vs_all(head_relations)
    raw_logits = _validate_kvsall_logits(
        raw_logits,
        batch_size=head_relations.shape[0],
        num_entities=num_entities,
    )
    return model.kge_scale * raw_logits + model.kge_bias


def kvsall_bce_loss(logits, targets, candidate_indices=None):
    """Compute dense K-vs-all BCE, optionally masking evaluation-only entities."""
    if logits.shape != targets.shape:
        raise ValueError(
            "K-vs-all logits and targets must have the same shape; got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    if candidate_indices is not None:
        logits = logits.index_select(1, candidate_indices)
        targets = targets.index_select(1, candidate_indices)
    return F.binary_cross_entropy_with_logits(logits, targets)


def sampled_fusion_logits(
    model,
    indexed_triples,
    input_ids,
    attention_mask,
    token_type_ids=None,
):
    """Return sampled fusion logits with gradients through both branches."""
    raw_bert_logits = model.bert_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    bert_logits = model.bert_scale * raw_bert_logits + model.bert_bias
    raw_kge_logits = model.kge_model.forward_triples(indexed_triples)
    raw_kge_logits = raw_kge_logits.reshape_as(raw_bert_logits)
    sampled_kge_logits = model.kge_scale * raw_kge_logits + model.kge_bias
    lambda_value = model.mixing_weight
    joint_logits = (
        lambda_value * bert_logits
        + (1.0 - lambda_value) * sampled_kge_logits
    )
    return joint_logits, bert_logits, sampled_kge_logits


def paired_training_batches(bert_loader, kge_loader):
    """Yield ``max(len(...))`` paired batches, cycling only the shorter loader."""
    bert_steps = len(bert_loader)
    kge_steps = len(kge_loader)
    if bert_steps < 1 or kge_steps < 1:
        raise ValueError("Both BERT and KGE training loaders must be non-empty.")

    bert_iterator = iter(bert_loader)
    kge_iterator = iter(kge_loader)
    for _ in range(max(bert_steps, kge_steps)):
        try:
            bert_batch = next(bert_iterator)
        except StopIteration:
            bert_iterator = iter(bert_loader)
            bert_batch = next(bert_iterator)
        try:
            kge_batch = next(kge_iterator)
        except StopIteration:
            kge_iterator = iter(kge_loader)
            kge_batch = next(kge_iterator)
        yield bert_batch, kge_batch


def gradient_norm(parameters):
    """Return an L2 norm over the available gradients in one parameter branch."""
    squared_norm = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(squared_norm)


def clone_model_state(model):
    """Copy model tensors to CPU so the best validation checkpoint is stable."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def build_parser():
    """Build the shared CLI for single runs and hyperparameter search."""
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate a joint BERT + KGE link predictor with BERT "
            "negative sampling and KGE K-vs-all scoring."
        )
    )
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
            "Target BERT examples per optimizer step. It is rounded down to "
            "a multiple of 1 + --negative_ratio."
        ),
    )
    parser.add_argument(
        "--kge_batch_size",
        type=int,
        default=None,
        help=(
            "Number of dense K-vs-all (head, relation) rows per step. "
            "Defaults to the number of BERT positive groups; lower it when "
            "the entity count makes B x |E| expensive."
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
    )
    parser.add_argument(
        "--negative_filter_scope",
        choices=["train", "train_valid", "all"],
        default="train",
        help="Facts excluded from BERT negative corruption.",
    )
    parser.add_argument(
        "--kge_kvsall_candidate_scope",
        choices=["train", "all"],
        default="all",
        help=(
            "Reduce dense KGE BCE over all support entities (the default) "
            "or only train-observed entities to match BERT's pool."
        ),
    )
    parser.add_argument(
        "--kge_loss_weight",
        type=float,
        default=1.0,
        help="Multiplier for the dense KGE K-vs-all loss.",
    )
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["valid", "test"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Parent directory for unique run subdirectories; omitted uses "
            "the script's default run parent."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Continue a K-vs-all run from its run directory, training_state.pt, "
            "or final-model checkpoint."
        ),
    )
    parser.add_argument("--eval_checkpoint_every", type=int, default=1)
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
    parser.add_argument("--kge_embedding_dim", type=int, default=32) #64, 128
    parser.add_argument("--kge_lr", type=float, default=0.1) #0.01
    parser.add_argument("--calibration_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_lr", type=float, default=1e-3)
    parser.add_argument("--initial_lambda", type=float, default=0.5)
    parser.add_argument(
        "--lambda_val",
        type=float,
        default=None,
        help="Freeze the BERT fusion weight at this value instead of learning it.",
    )
    return parser


def parse_args():
    return build_parser().parse_args()


def validate_args(args):
    if args.num_epochs < 1:
        raise ValueError("--num_epochs must be at least 1.")
    if args.negative_ratio < 1:
        raise ValueError("--negative_ratio must be at least 1.")
    if args.batch_size < 1 + args.negative_ratio:
        raise ValueError(
            "--batch_size must be at least 1 + --negative_ratio for grouped "
            "BERT negative batches."
        )
    if args.kge_batch_size is not None and args.kge_batch_size < 1:
        raise ValueError("--kge_batch_size must be positive when provided.")
    if args.kge_loss_weight <= 0.0:
        raise ValueError("--kge_loss_weight must be positive.")
    if args.calibration_lr <= 0.0:
        raise ValueError("--calibration_lr must be positive.")
    if args.lambda_lr <= 0.0:
        raise ValueError("--lambda_lr must be positive.")
    if not 0.0 < args.initial_lambda < 1.0:
        raise ValueError("--initial_lambda must be strictly between 0 and 1.")
    if args.lambda_val is not None and not 0.0 < args.lambda_val < 1.0:
        raise ValueError("--lambda_val must be strictly between 0 and 1.")
    if args.eval_checkpoint_every < 1:
        raise ValueError("--eval_checkpoint_every must be at least 1.")
    if args.early_stopping_patience < 1:
        raise ValueError("--early_stopping_patience must be at least 1.")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early_stopping_min_delta cannot be negative.")
    if args.early_stopping_warmup_epochs < 0:
        raise ValueError("--early_stopping_warmup_epochs cannot be negative.")


def prepare_output_dir(repo_root, args, run_name):
    """Create a unique run directory inside the requested output parent."""
    if args.resume_from_checkpoint is not None:
        resume_path = resolve_path(repo_root, args.resume_from_checkpoint)
        if os.path.isdir(resume_path):
            return resume_path
        if os.path.isfile(resume_path):
            return os.path.dirname(resume_path)
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    base_output_dir = (
        os.path.join(repo_root, "bert_bce_kvsall_runs")
        if args.output_dir is None
        else resolve_path(repo_root, args.output_dir)
    )
    return create_unique_output_dir(base_output_dir, run_name)


def resolve_resume_checkpoint(repo_root, args, output_dir):
    """Resolve an exact training state or a legacy best-model checkpoint."""
    if args.resume_from_checkpoint is None:
        return None, None
    requested_path = resolve_path(repo_root, args.resume_from_checkpoint)
    if os.path.isdir(requested_path):
        training_state_path = os.path.join(requested_path, "training_state.pt")
        if os.path.isfile(training_state_path):
            return training_state_path, "training_state"
        requested_path = os.path.join(
            requested_path,
            f"joint_bert_{args.kge_model}_kvsall_link_prediction.pt",
        )
    if not os.path.isfile(requested_path):
        raise FileNotFoundError(
            "No training_state.pt or matching best-model checkpoint found in "
            f"{output_dir}."
        )
    checkpoint = load_torch_checkpoint(requested_path)
    if "optimizer_state_dict" in checkpoint:
        return requested_path, "training_state"
    return requested_path, "legacy_best"


def main(args=None):
    """Run training and return the saved results for search orchestration."""
    if args is None:
        args = parse_args()
    validate_args(args)
    run_start_time = time.perf_counter()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = resolve_path(repo_root, args.dataset_path)
    support_path = (
        resolve_path(repo_root, args.support_path)
        if args.support_path
        else None
    )
    bert_model_path = resolve_path(repo_root, args.bert_model_path)
    configured_lambda = (
        args.lambda_val if args.lambda_val is not None else args.initial_lambda
    )
    lambda_mode = "fixed" if args.lambda_val is not None else "learned"
    dataset_name = dataset_run_name(args.dataset_path)
    run_datetime = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{dataset_name}_joint_kvsall_{args.kge_model}_"
        f"{args.eval_split}_{run_datetime}"
    )
    output_dir = prepare_output_dir(repo_root, args, run_name)
    resume_checkpoint_path, resume_kind = resolve_resume_checkpoint(
        repo_root,
        args,
        output_dir,
    )
    resume_state = (
        load_torch_checkpoint(resume_checkpoint_path)
        if resume_checkpoint_path is not None
        else None
    )
    training_state_path = os.path.join(output_dir, "training_state.pt")
    checkpoint_path = os.path.join(
        output_dir,
        f"joint_bert_{args.kge_model}_kvsall_link_prediction.pt",
    )
    test_progress_path = os.path.join(output_dir, "test_evaluation_progress.pt")
    print(f"run_output_dir={output_dir}")
    if resume_checkpoint_path is not None:
        print(
            f"resume_checkpoint={resume_checkpoint_path} "
            f"resume_kind={resume_kind}"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_triples = read_triples(os.path.join(dataset_path, "train.txt"))
    valid_triples = read_triples(os.path.join(dataset_path, "valid.txt"))
    test_triples = read_triples(os.path.join(dataset_path, "test.txt"))
    if not valid_triples:
        raise ValueError(
            "Validation triples are required for checkpoint selection and "
            "early stopping."
        )
    all_true_triples = set(train_triples) | set(valid_triples) | set(test_triples)
    entities, relations = read_support(dataset_path, support_path)
    entity_to_idx = {
        entity_id: index for index, entity_id in enumerate(entities)
    }
    relation_to_idx = {
        relation_id: index for index, relation_id in enumerate(relations)
    }
    negative_entity_ids = training_entity_ids(train_triples, entities)
    training_negative_filter = negative_filter_triples(
        args.negative_filter_scope,
        train_triples,
        valid_triples,
        test_triples,
    )
    candidate_indices = kge_candidate_indices(
        args.kge_kvsall_candidate_scope,
        train_triples,
        entities,
        entity_to_idx,
    )
    print(
        "bert_scoring_technique=NegSample "
        "kge_scoring_technique=KvsAll "
        "kge_gradient_protocol=kvsall_and_sampled_fusion "
        "kge_kvsall_training_direction=tail_only_no_reciprocals"
    )
    print(
        f"bert_negative_entity_pool=train ({len(negative_entity_ids)}/"
        f"{len(entities)} entities) "
        f"negative_filter_scope={args.negative_filter_scope} "
        f"negative_loss_weighting={args.negative_loss_weighting}"
    )
    print(
        f"kge_kvsall_candidate_scope={args.kge_kvsall_candidate_scope} "
        f"({len(candidate_indices)}/{len(entities)} entities)"
    )

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_path,
        do_basic_tokenize=False,
    )
    add_nbert_tokens(tokenizer, entities, relations)
    bert_model = BertTripleClassifier(bert_model_path, tokenizer)
    kge_model = create_kvsall_kge_model(
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
    if args.lambda_val is not None:
        model.lambda_logit.requires_grad_(False)
    model.to(device)
    candidate_indices = candidate_indices.to(device)

    bert_dataset = TripleBCEDataset(
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
    bert_generator = torch.Generator().manual_seed(args.seed)
    bert_sampler = GroupedNegativeBatchSampler(
        bert_dataset,
        batch_size=args.batch_size,
        generator=bert_generator,
    )
    bert_loader = DataLoader(
        bert_dataset,
        batch_sampler=bert_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_text(
            batch,
            tokenizer,
            args.max_seq_length,
        ),
    )
    kge_dataset = KvsAllTailDataset(
        train_triples,
        entity_to_idx,
        relation_to_idx,
    )
    kge_batch_size = args.kge_batch_size or bert_sampler.groups_per_batch
    kge_generator = torch.Generator().manual_seed(args.seed + 1)
    kge_loader = DataLoader(
        kge_dataset,
        batch_size=kge_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=kge_generator,
    )
    print(
        f"bert_negative_group_size={bert_sampler.group_size} "
        f"bert_positive_groups_per_batch={bert_sampler.groups_per_batch} "
        f"bert_effective_batch_size="
        f"{bert_sampler.group_size * bert_sampler.groups_per_batch}"
    )
    print(
        f"kge_kvsall_rows={len(kge_dataset)} "
        f"kge_batch_size={kge_batch_size} "
        f"optimizer_steps_per_epoch={max(len(bert_loader), len(kge_loader))}"
    )

    optimizer_parameter_groups = [
        {"params": model.bert_model.parameters(), "lr": args.lr},
        {"params": model.kge_model.parameters(), "lr": args.kge_lr},
        {"params": model.calibration_parameters(), "lr": args.calibration_lr},
    ]
    if model.lambda_logit.requires_grad:
        optimizer_parameter_groups.append(
            {"params": [model.lambda_logit], "lr": args.lambda_lr}
        )
    optimizer = torch.optim.Adam(optimizer_parameter_groups)

    best_model_state = None
    best_validation_metrics = None
    best_validation_mrr = -float("inf")
    best_epoch = None
    best_final_lambda = None
    best_training_loss = float("inf")
    epochs_without_loss_improvement = 0
    early_stopped = False
    training_history = []
    epochs_ran = 0
    resumed_with_fresh_optimizer = False

    if resume_state is not None:
        if resume_state.get("run_variant") != "joint_kvsall":
            raise ValueError(
                "Resume checkpoint is not a joint K-vs-all training run."
            )
        saved_args = resume_state.get("args", {})
        ignored_resume_args = {
            "candidate_batch_size",
            "device",
            "eval_checkpoint_every",
            "num_epochs",
            "num_workers",
            "output_dir",
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

        model.load_state_dict(resume_state["model_state_dict"])
        best_validation_metrics = resume_state.get("best_validation_metrics")
        best_validation_mrr = (
            -float("inf")
            if best_validation_metrics is None
            else float(best_validation_metrics["MRR"])
        )
        best_epoch = resume_state.get("best_epoch")
        best_final_lambda = resume_state.get(
            "best_final_lambda",
            resume_state.get("final_lambda"),
        )
        training_history = list(resume_state.get("training_history", []))
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
        early_stopped = bool(resume_state.get("early_stopped", False))

        if resume_kind == "training_state":
            if resume_state.get("version") != 2:
                raise ValueError(
                    "Unsupported K-vs-all training-state version: "
                    f"{resume_state.get('version')!r}."
                )
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            epochs_ran = int(resume_state["epoch_completed"])
            random.setstate(resume_state["python_random_state"])
            torch.set_rng_state(resume_state["torch_random_state"])
            if (
                torch.cuda.is_available()
                and resume_state.get("cuda_random_state") is not None
            ):
                torch.cuda.set_rng_state_all(resume_state["cuda_random_state"])
            bert_generator.set_state(resume_state["bert_generator_state"])
            kge_generator.set_state(resume_state["kge_generator_state"])
            if best_epoch is not None:
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(
                        f"Final-model checkpoint is missing: {checkpoint_path}"
                    )
                best_model_state = load_torch_checkpoint(checkpoint_path)[
                    "model_state_dict"
                ]
            print(
                f"resumed_exact_training_state=true epoch_completed={epochs_ran}"
            )
        else:
            epochs_ran = int(best_epoch or 0)
            best_model_state = clone_model_state(model)
            epochs_without_loss_improvement = 0
            early_stopped = False
            resumed_with_fresh_optimizer = True
            print(
                "resumed_exact_training_state=false "
                f"restart_from_best_epoch={epochs_ran} optimizer=fresh"
            )

        if args.num_epochs <= epochs_ran:
            raise ValueError(
                f"--num_epochs ({args.num_epochs}) must be greater than the "
                f"resumed epoch ({epochs_ran})."
            )

    def save_training_state(epoch_completed):
        atomic_torch_save(
            training_state_path,
            {
                "version": 2,
                "run_variant": "joint_kvsall",
                "epoch_completed": epoch_completed,
                "model_state_dict": clone_model_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_metrics": best_validation_metrics,
                "best_validation_mrr": best_validation_mrr,
                "best_epoch": best_epoch,
                "best_final_lambda": best_final_lambda,
                "best_training_loss": best_training_loss,
                "epochs_without_loss_improvement": (
                    epochs_without_loss_improvement
                ),
                "early_stopped": early_stopped,
                "training_history": training_history,
                "args": vars(args),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "bert_generator_state": bert_generator.get_state(),
                "kge_generator_state": kge_generator.get_state(),
            },
        )
        print(
            f"saved_training_checkpoint={training_state_path} "
            f"epoch={epoch_completed}"
        )

    def validation_progress_path(epoch):
        base_path = os.path.join(
            output_dir,
            f"validation_epoch_{epoch}_progress.pt",
        )
        if not resumed_with_fresh_optimizer or not os.path.exists(base_path):
            return base_path
        suffix = 1
        while True:
            candidate = os.path.join(
                output_dir,
                f"validation_epoch_{epoch}_resumed_{suffix}_progress.pt",
            )
            if not os.path.exists(candidate):
                return candidate
            suffix += 1

    bert_parameters = [
        parameter for parameter in model.bert_model.parameters() if parameter.requires_grad
    ]
    kge_parameters = [
        parameter for parameter in model.kge_model.parameters() if parameter.requires_grad
    ]

    for epoch in range(epochs_ran + 1, args.num_epochs + 1):
        model.train()
        total_losses = []
        sampled_fusion_losses = []
        kvsall_losses = []
        bert_gradient_norms = []
        kge_gradient_norms = []
        sampled_positive_logits = 0.0
        sampled_negative_logits = 0.0
        num_positive_examples = 0
        num_negative_examples = 0

        for bert_batch, kge_batch in tqdm(
            paired_training_batches(bert_loader, kge_loader),
            total=max(len(bert_loader), len(kge_loader)),
            desc=f"Epoch {epoch}",
        ):
            labels = bert_batch.pop("labels").to(device)
            indexed_triples = bert_batch.pop("indexed_triples").to(device)
            bert_batch = {
                key: value.to(device) for key, value in bert_batch.items()
            }
            sampled_logits, _, _ = sampled_fusion_logits(
                model,
                indexed_triples=indexed_triples,
                **bert_batch,
            )
            sampled_loss = negative_sampling_bce_loss(
                sampled_logits,
                labels,
                args.negative_ratio,
                weighting=args.negative_loss_weighting,
            )

            head_relations, kge_targets = kge_batch
            head_relations = head_relations.to(device)
            kge_targets = kge_targets.to(device)
            all_tail_logits = forward_kvsall_logits(
                model,
                head_relations,
                num_entities=len(entity_to_idx),
            )
            kge_loss = kvsall_bce_loss(
                all_tail_logits,
                kge_targets,
                candidate_indices=candidate_indices,
            )
            total_loss = sampled_loss + kge_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            bert_gradient_norms.append(gradient_norm(bert_parameters))
            kge_gradient_norms.append(gradient_norm(kge_parameters))
            optimizer.step()
            post_kge_parameter_update(model.kge_model)

            with torch.no_grad():
                positive_mask = labels > 0.5
                negative_mask = ~positive_mask
                sampled_positive_logits += float(
                    sampled_logits[positive_mask].sum().item()
                )
                sampled_negative_logits += float(
                    sampled_logits[negative_mask].sum().item()
                )
                num_positive_examples += int(positive_mask.sum().item())
                num_negative_examples += int(negative_mask.sum().item())
            total_losses.append(float(total_loss.detach().cpu()))
            sampled_fusion_losses.append(float(sampled_loss.detach().cpu()))
            kvsall_losses.append(float(kge_loss.detach().cpu()))

        mean_total_loss = sum(total_losses) / max(len(total_losses), 1)
        mean_sampled_loss = sum(sampled_fusion_losses) / max(
            len(sampled_fusion_losses),
            1,
        )
        mean_kvsall_loss = sum(kvsall_losses) / max(len(kvsall_losses), 1)
        epoch_diagnostics = {
            "epoch": epoch,
            "loss": mean_total_loss, #kge + bert losses
            "sampled_fusion_loss": mean_sampled_loss,
            "kge_kvsall_loss": mean_kvsall_loss,
            "kge_loss_weight": args.kge_loss_weight,
            "lambda": float(model.mixing_weight.detach().cpu()),
            **model.calibration_state(),
            "bert_grad_norm": sum(bert_gradient_norms) / max(
                len(bert_gradient_norms),
                1,
            ),
            "kge_grad_norm": sum(kge_gradient_norms) / max(
                len(kge_gradient_norms),
                1,
            ),
            "sampled_pos_logit": sampled_positive_logits / max(
                num_positive_examples,
                1,
            ),
            "sampled_neg_logit": sampled_negative_logits / max(
                num_negative_examples,
                1,
            ),
            "validation_metrics": None,
        }
        print(
            f"epoch={epoch} loss={mean_total_loss:.6f} "
            f"sampled_fusion_loss={mean_sampled_loss:.6f} "
            f"kge_kvsall_loss={mean_kvsall_loss:.6f} "
            f"lambda={epoch_diagnostics['lambda']:.6f} "
            f"bert_grad_norm={epoch_diagnostics['bert_grad_norm']:.6f} "
            f"kge_grad_norm={epoch_diagnostics['kge_grad_norm']:.6f}"
        )

        loss_improved = (
            mean_total_loss
            < best_training_loss - args.early_stopping_min_delta
        )
        if loss_improved:
            best_training_loss = mean_total_loss
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
        if not should_evaluate:
            training_history.append(epoch_diagnostics)
            save_training_state(epoch)
            continue

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
            checkpoint_path=validation_progress_path(epoch),
            checkpoint_every=args.eval_checkpoint_every,
        )
        
        epoch_diagnostics["validation_metrics"] = validation_metrics
        validation_mrr = float(validation_metrics["MRR"])
        best_validation_mrr = validation_mrr
        best_validation_metrics = dict(validation_metrics)
        best_epoch = epoch
        best_final_lambda = epoch_diagnostics["lambda"]
        best_model_state = clone_model_state(model)
        epoch_diagnostics["is_best"] = True
        training_history.append(epoch_diagnostics)
        print(
            f"final_validation_epoch={epoch} valid_mrr={validation_mrr:.6f} "
            "stopping_strategy=training_loss_patience"
        )
        atomic_torch_save(
            checkpoint_path,
            {
                "version": 1,
                "run_variant": "joint_kvsall",
                "model_state_dict": best_model_state,
                "metrics": best_validation_metrics,
                "best_validation_metrics": best_validation_metrics,
                "args": vars(args),
                "bert_scoring_technique": "NegSample",
                "kge_scoring_technique": "KvsAll",
                "kge_gradient_protocol": "kvsall_and_sampled_fusion",
                "sampled_fusion_kge_gradients": "enabled",
                "kge_kvsall_training_direction": "tail_only_no_reciprocals",
                "lambda_mode": lambda_mode,
                "configured_lambda": configured_lambda,
                "final_lambda": best_final_lambda,
                "selection_split": None,
                "selection_metric": None,
                "stopping_strategy": "training_loss_patience",
                "best_epoch": best_epoch,
                "best_training_loss": best_training_loss,
                "epochs_without_loss_improvement": (
                    epochs_without_loss_improvement
                ),
                "early_stopped": early_stopped,
                "training_history": training_history,
            },
        )
        print(f"saved_final_checkpoint={checkpoint_path} epoch={best_epoch}")
        save_training_state(epoch)
        break

    if best_model_state is None:
        raise RuntimeError("Training completed without a validation evaluation.")
    model.load_state_dict(best_model_state)
    model.to(device)
    final_lambda = float(model.mixing_weight.detach().cpu())
    final_calibration = model.calibration_state()
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
    runtime_minutes = (time.perf_counter() - run_start_time) / 60.0
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
        "run_variant": "joint_kvsall",
        "bert_scoring_technique": "NegSample",
        "kge_scoring_technique": "KvsAll",
        "kge_gradient_protocol": "kvsall_and_sampled_fusion",
        "sampled_fusion_kge_gradients": "enabled",
        "kge_kvsall_training_direction": "tail_only_no_reciprocals",
        "kge_kvsall_candidate_scope": args.kge_kvsall_candidate_scope,
        "kge_kvsall_candidate_count": len(candidate_indices),
        "lambda_mode": lambda_mode,
        "configured_lambda": configured_lambda,
        "final_lambda": final_lambda,
        "final_calibration": final_calibration,
        "selection_split": None,
        "selection_metric": None,
        "stopping_strategy": "training_loss_patience",
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "best_training_loss": best_training_loss,
        "epochs_without_loss_improvement": (
            epochs_without_loss_improvement
        ),
        "best_validation_metrics": best_validation_metrics,
        "training_history": training_history,
        "output_dir": output_dir,
        "runtime_min": runtime_minutes,
    }
    atomic_json_save(os.path.join(output_dir, "results.json"), results)
    atomic_torch_save(
        checkpoint_path,
        {
            "version": 1,
            "run_variant": "joint_kvsall",
            "model_state_dict": best_model_state,
            "metrics": metrics,
            "best_validation_metrics": best_validation_metrics,
            "args": vars(args),
            "bert_scoring_technique": "NegSample",
            "kge_scoring_technique": "KvsAll",
            "kge_gradient_protocol": "kvsall_and_sampled_fusion",
            "sampled_fusion_kge_gradients": "enabled",
            "kge_kvsall_training_direction": "tail_only_no_reciprocals",
            "kge_kvsall_candidate_scope": args.kge_kvsall_candidate_scope,
            "lambda_mode": lambda_mode,
            "configured_lambda": configured_lambda,
            "final_lambda": final_lambda,
            "final_calibration": final_calibration,
            "selection_split": None,
            "selection_metric": None,
            "stopping_strategy": "training_loss_patience",
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
            "best_training_loss": best_training_loss,
            "epochs_without_loss_improvement": (
                epochs_without_loss_improvement
            ),
            "training_history": training_history,
        },
    )
    tokenizer.save_pretrained(output_dir)
    print(f"final_{args.eval_split}_metrics={json.dumps(metrics, sort_keys=True)}")
    print(f"final_checkpoint_dir={output_dir}")
    print(f"runtime_minutes={runtime_minutes:.2f}")
    return results


if __name__ == "__main__":
    main()