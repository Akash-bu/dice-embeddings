"""Train and evaluate BERT alone for BCE link prediction.

This entry point intentionally keeps the BERT, data, training, and evaluation
hyperparameters from ``bert_bce_link_prediction.py`` while omitting the KGE
model and the learned BERT/KGE mixing weight.
"""

import argparse
import json
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
    add_nbert_tokens,
    atomic_json_save,
    atomic_torch_save,
    create_unique_output_dir,
    dataset_run_name,
    evaluation_fingerprint,
    load_evaluation_progress,
    load_torch_checkpoint,
    rank_of_target,
    ranks_to_metrics,
    read_support,
    read_triples,
    resolve_path,
    save_evaluation_progress,
    triple_prompt,
)


class BertTripleBCEDataset(Dataset):
    """Create positive and corrupted triple prompts for BERT-only training."""

    def __init__(
        self,
        triples,
        entities,
        relations,
        max_seq_length,
        negative_ratio=1,
    ):
        self.triples = triples
        self.entities = entities
        self.relations = relations
        self.entity_ids = list(entities.keys())
        self.true_triples = set(triples)
        self.max_seq_length = max_seq_length
        self.negative_ratio = negative_ratio

    def __len__(self):
        return len(self.triples) * (1 + self.negative_ratio)

    def __getitem__(self, index):
        positive_index = index // (1 + self.negative_ratio)
        offset = index % (1 + self.negative_ratio)
        triple = self.triples[positive_index]
        label = 1.0

        if offset != 0:
            triple = self.corrupt_triple(triple)
            label = 0.0

        prompt = triple_prompt(
            triple,
            self.entities,
            self.relations,
            self.max_seq_length,
        )
        return prompt, label

    def corrupt_triple(self, triple):
        head, relation, tail = triple
        corrupt_head_first = random.random() < 0.5

        for _ in range(100):
            replacement = random.choice(self.entity_ids)
            if corrupt_head_first:
                candidate = (replacement, relation, tail)
            else:
                candidate = (head, relation, replacement)
            if candidate not in self.true_triples:
                return candidate

        for corrupt_head in (corrupt_head_first, not corrupt_head_first):
            start = random.randrange(len(self.entity_ids))
            for offset in range(len(self.entity_ids)):
                replacement = self.entity_ids[
                    (start + offset) % len(self.entity_ids)
                ]
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
    """Tokenize triple prompts and attach their BCE labels."""
    prompts, labels = zip(*batch)
    encoded = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(labels, dtype=torch.float)
    return encoded


def score_prompts(
    model,
    tokenizer,
    prompts,
    device,
    max_seq_length,
    batch_size,
):
    """Score candidate triple prompts using only BERT."""
    scores = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }
            logits = model(**encoded)
            scores.append(logits.detach().cpu())

    return torch.cat(scores, dim=0)


def evaluate_link_prediction(
    model,
    tokenizer,
    eval_triples,
    all_true_triples,
    entities,
    relations,
    device,
    max_seq_length,
    candidate_batch_size,
    progress_description="Evaluating BERT-only BCE MRR",
    checkpoint_path=None,
    checkpoint_every=1,
):
    """Evaluate filtered head and tail prediction with BERT-only scores."""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1.")
    entity_ids = list(entities.keys())
    entity_to_position = {
        entity_id: index
        for index, entity_id in enumerate(entity_ids)
    }
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
        tail_candidates = [
            (head, relation, candidate_tail)
            for candidate_tail in entity_ids
        ]
        tail_prompts = [
            triple_prompt(
                triple,
                entities,
                relations,
                max_seq_length,
            )
            for triple in tail_candidates
        ]
        tail_scores = score_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=tail_prompts,
            device=device,
            max_seq_length=max_seq_length,
            batch_size=candidate_batch_size,
        )
        for index, candidate_tail in enumerate(entity_ids):
            candidate = (head, relation, candidate_tail)
            if (
                candidate_tail != tail
                and candidate in all_true_triples
            ):
                tail_scores[index] = -float("inf")
        ranks.append(
            rank_of_target(
                tail_scores,
                entity_to_position[tail],
            )
        )

        head_candidates = [
            (candidate_head, relation, tail)
            for candidate_head in entity_ids
        ]
        head_prompts = [
            triple_prompt(
                triple,
                entities,
                relations,
                max_seq_length,
            )
            for triple in head_candidates
        ]
        head_scores = score_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=head_prompts,
            device=device,
            max_seq_length=max_seq_length,
            batch_size=candidate_batch_size,
        )
        for index, candidate_head in enumerate(entity_ids):
            candidate = (candidate_head, relation, tail)
            if (
                candidate_head != head
                and candidate in all_true_triples
            ):
                head_scores[index] = -float("inf")
        ranks.append(
            rank_of_target(
                head_scores,
                entity_to_position[head],
            )
        )

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

    return ranks_to_metrics(ranks)


def parse_args():
    """Define the shared BERT BCE training and evaluation options."""
    parser = argparse.ArgumentParser(
        description="Train/evaluate BERT-only BCE link prediction."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="KGs/UMLS",
    )
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--candidate_batch_size",
        type=int,
        default=256,
    )
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--negative_ratio", type=int, default=1)
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
        "--eval_checkpoint_every",
        type=int,
        default=1,
        help="Save evaluation ranks after this many completed triples.",
    )
    return parser.parse_args()


def main():
    """Train BERT with BCE, evaluate link prediction, and save the run."""
    args = parse_args()
    if args.num_epochs < 1:
        raise ValueError("--num_epochs must be at least 1.")
    if args.eval_checkpoint_every < 1:
        raise ValueError("--eval_checkpoint_every must be at least 1.")
    run_start_time = time.perf_counter()
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    dataset_path = resolve_path(repo_root, args.dataset_path)
    support_path = (
        resolve_path(repo_root, args.support_path)
        if args.support_path
        else None
    )
    bert_model_path = resolve_path(repo_root, args.bert_model_path)
    dataset_name = dataset_run_name(args.dataset_path)
    run_datetime = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(repo_root, "bert_bce_runs")
    run_name = (
        f"{dataset_name}_only_bert_{args.eval_split}_{run_datetime}"
    )
    resume_checkpoint_path = None
    resume_state = None
    if args.resume_from_checkpoint:
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

    training_state_path = os.path.join(output_dir, "training_state.pt")
    model_checkpoint_path = os.path.join(
        output_dir,
        "bert_bce_link_prediction.pt",
    )
    evaluation_progress_path = os.path.join(
        output_dir,
        f"{args.eval_split}_evaluation_progress.pt",
    )
    print(f"run_output_dir={output_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    train_triples = read_triples(
        os.path.join(dataset_path, "train.txt")
    )
    valid_triples = read_triples(
        os.path.join(dataset_path, "valid.txt")
    )
    test_triples = read_triples(
        os.path.join(dataset_path, "test.txt")
    )
    entities, relations = read_support(dataset_path, support_path)

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_path,
        do_basic_tokenize=False,
    )
    add_nbert_tokens(tokenizer, entities, relations)
    model = BertTripleClassifier(bert_model_path, tokenizer)
    model.to(device)

    train_dataset = BertTripleBCEDataset(
        triples=train_triples,
        entities=entities,
        relations=relations,
        max_seq_length=args.max_seq_length,
        negative_ratio=args.negative_ratio,
    )
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_completed = 0
    training_history = []
    if resume_state is not None:
        if resume_state.get("version") != 1:
            raise ValueError(
                f"Unsupported training checkpoint version in "
                f"{resume_checkpoint_path}: {resume_state.get('version')!r}"
            )
        saved_args = resume_state.get("args", {})
        ignored_resume_args = {
            "candidate_batch_size",
            "device",
            "eval_checkpoint_every",
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
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        epoch_completed = resume_state["epoch_completed"]
        training_history = resume_state["training_history"]
        random.setstate(resume_state["python_random_state"])
        torch.set_rng_state(resume_state["torch_random_state"])
        if (
            torch.cuda.is_available()
            and resume_state.get("cuda_random_state") is not None
        ):
            torch.cuda.set_rng_state_all(resume_state["cuda_random_state"])
        print(
            f"resumed_training_checkpoint={resume_checkpoint_path} "
            f"phase={resume_state['phase']} "
            f"epoch_completed={epoch_completed}"
        )

    tokenizer.save_pretrained(output_dir)

    def save_training_state(phase):
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
                "training_history": training_history,
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

    for epoch in range(epoch_completed + 1, args.num_epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            labels = batch.pop("labels").to(device)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            logits = model(**batch)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"epoch={epoch} loss={mean_loss:.6f}")
        epoch_completed = epoch
        training_history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
            }
        )
        save_training_state("training")

    save_training_state("training_complete")

    eval_triples = (
        valid_triples
        if args.eval_split == "valid"
        else test_triples
    )
    all_true_triples = (
        set(train_triples)
        | set(valid_triples)
        | set(test_triples)
    )
    metrics = evaluate_link_prediction(
        model=model,
        tokenizer=tokenizer,
        eval_triples=eval_triples,
        all_true_triples=all_true_triples,
        entities=entities,
        relations=relations,
        device=device,
        max_seq_length=args.max_seq_length,
        candidate_batch_size=args.candidate_batch_size,
        checkpoint_path=evaluation_progress_path,
        checkpoint_every=args.eval_checkpoint_every,
    )
    print(json.dumps(metrics, indent=2))

    runtime_min = (
        time.perf_counter() - run_start_time
    ) / 60.0
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
        "training_history": training_history,
        "output_dir": output_dir,
        "runtime_min": runtime_min,
    }
    atomic_json_save(
        os.path.join(output_dir, "results.json"),
        results,
    )

    atomic_torch_save(
        model_checkpoint_path,
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "training_history": training_history,
        },
    )

    print(f"checkpoint_dir={output_dir}")
    print(f"runtime_minutes={runtime_min:.2f}")


if __name__ == "__main__":
    main()
