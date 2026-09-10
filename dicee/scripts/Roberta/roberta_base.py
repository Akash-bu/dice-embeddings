"""Train and evaluate RoBERTa-base alone for BCE link prediction."""
import argparse 
import json 
import os
import random 
import time 
from datetime import datetime 

import torch 
from torch import nn 
from torch.nn import functional as F 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from transformers import RobertaModel, RobertaTokenizer 

from dicee.scripts.bert_bce_link_prediction import (
    add_nbert_tokens,
    atomic_json_save,
    atomic_torch_save,
    create_unique_output_dir,
    dataset_run_name,
    load_torch_checkpoint,
    read_support,
    read_triples,
    resolve_path,
)

from dicee.scripts.bert_only_bce_link_prediction import (
    BertTripleBCEDataset as RobertaTripleBCEDataset,
    collate_text,
    evaluate_link_prediction,
)


class RobertaTripleClassifier(nn.Module):

    def __init__(self, model_path, tokenizer):
        super().__init__() 
        self.roberta = RobertaModel.from_pretrained(model_path) 
        self.roberta.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Linear(
            self.roberta.config.hidden_size,
            1
        )
    
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids = None
    ):
        output = self.roberta(
            input_ids=input_ids,
            attention_mask = attention_mask
        )

        cls_embedding = output.last_hidden_state[:, 0]
        return self.classifier(cls_embedding).squeeze(-1) 
    
def resolve_pretrained_source(repo_root, source):
    if os.path.isabs(source):
        return source 
    
    local_source = os.path.join(repo_root, source)
    if os.path.exists(local_source):
        return local_source
    
    return source 
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate RoBERTa-base BCE link prediction."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="KGs/UMLS",
    )
    parser.add_argument(
        "--support_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--roberta_model_path",
        type=str,
        default="roberta-base",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="roberta-base",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--candidate_batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--negative_ratio",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="valid",
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
    args = parse_args() 

    if args.num_epochs < 1:
        raise ValueError("--num_epochs must be at least 1.")
    if args.negative_ratio < 1:
        raise ValueError("--negative_ratio must be at least 1.")
    if args.eval_checkpoint_every < 1:
        raise ValueError("--eval_checkpoint_every must be at least 1.")

    start_time = time.perf_counter() 

    repo_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
        )
    )

    dataset_path = resolve_path(
        repo_root,
        args.dataset_path
    )

    support_path = (
        resolve_path(repo_root, args.support_path) if args.support_path else None
    )

    roberta_model_source = resolve_pretrained_source(
        repo_root,
        args.roberta_model_path
    )

    tokenizer_source = resolve_pretrained_source(
        repo_root, 
        args.tokenizer_path
    )

    dataset_name = dataset_run_name(args.dataset_path) 

    run_datetime = datetime.now().astimezone().strftime(
        "%Y%m%d_%H%M%S"
    )

    base_output_dir = os.path.join(
        repo_root,
        "roberta_runs"
    )

    run_name = (
        f"{dataset_name}_only_roberta_"
        f"{args.eval_split}_{run_datetime}"
    )

    resume_checkpoint_path = None 
    resume_state = None 

    if args.resume_from_checkpoint:
        resume_checkpoint_path = resolve_path(
            repo_root,
            args.resume_from_checkpoint
        )

        if os.path.isdir(resume_checkpoint_path):
            resume_checkpoint_path = os.path.join(
                resume_checkpoint_path,
                "training_state.pt",
            )

        if not os.path.isfile(resume_checkpoint_path):
            raise FileNotFoundError(
                "Resume checkpoint not found: "
                f"{resume_checkpoint_path}"
            )

        resume_state = load_torch_checkpoint(
            resume_checkpoint_path
        )

        output_dir = os.path.dirname(
            resume_checkpoint_path
        )

        if args.output_dir:
            requested_output_dir = resolve_path(
                repo_root,
                args.output_dir,
            )
            if os.path.abspath(
                requested_output_dir
            ) != os.path.abspath(output_dir):
                raise ValueError(
                    "--output_dir must match the directory containing "
                    "--resume_from_checkpoint."
                )

    elif args.output_dir:
        output_dir = resolve_path(
            repo_root,
            args.output_dir,
        )
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(
            os.path.join(output_dir, "training_state.pt")
        ):
            raise FileExistsError(
                f"{output_dir} already contains training_state.pt; "
                "pass --resume_from_checkpoint to resume it."
            )

    else:
        output_dir = create_unique_output_dir(
            base_output_dir,
            run_name,
        )
    
    training_state_path = os.path.join(
        output_dir,
        "training_state.pt",
    )

    model_checkpoint_path = os.path.join(
        output_dir,
        "roberta_bce_link_prediction.pt"
    )

    evaluation_progress_path = os.path.join(
        output_dir,
        f"{args.eval_split}_eval_progress.pt"
    )

    print(f"run_output_dir = {output_dir}")

    random.seed(args.seed) 
    torch.manual_seed(args.seed) 

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"device = {device}") 

    train_triples = read_triples(
        os.path.join(dataset_path, "train.txt")
    )

    valid_triples = read_triples(
        os.path.join(dataset_path, "valid.txt")
    )

    test_triples = read_triples(
        os.path.join(dataset_path, "test.txt")
    )

    entities, relations = read_support(
        dataset_path, support_path
    )

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_source) 
    add_nbert_tokens(tokenizer, entities, relations) 

    model = RobertaTripleClassifier(
        model_path = roberta_model_source,
        tokenizer = tokenizer
    )

    model.to(device)

    train_dataset = RobertaTripleBCEDataset(
        triples = train_triples,
        entities = entities,
        relations = relations,
        max_seq_length = args.max_seq_length,
        negative_ratio = args.negative_ratio
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        collate_fn = lambda batch: collate_text(
            batch,
            tokenizer,
            args.max_seq_length
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    epoch_completed = 0
    training_history = []

    if resume_state is not None:
        if resume_state.get("version") != 1:
            raise ValueError(
                "Unsupported training checkpoint version in "
                f"{resume_checkpoint_path}: "
                f"{resume_state.get('version')!r}"
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
                f"{name}: saved={saved_args[name]!r}, "
                f"current={getattr(args, name)!r}"
                for name in mismatches
            )
            raise ValueError(
                "Resume arguments do not match the checkpoint "
                f"({details})."
            )

        model.load_state_dict(
            resume_state["model_state_dict"]
        )
        optimizer.load_state_dict(
            resume_state["optimizer_state_dict"]
        )

        epoch_completed = int(
            resume_state["epoch_completed"]
        )
        training_history = resume_state[
            "training_history"
        ]

        random.setstate(
            resume_state["python_random_state"]
        )
        torch.set_rng_state(
            resume_state["torch_random_state"]
        )

        if (
            torch.cuda.is_available()
            and resume_state.get("cuda_random_state") is not None
        ):
            torch.cuda.set_rng_state_all(
                resume_state["cuda_random_state"]
            )

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

    for epoch in range(
        epoch_completed + 1,
        args.num_epochs + 1,
    ):
        model.train()
        losses = []

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
        ):
            labels = batch.pop("labels").to(device)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            logits = model(**batch)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
            )

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
        progress_description="Evaluating RoBERTa baseline MRR",
        checkpoint_path=evaluation_progress_path,
        checkpoint_every=args.eval_checkpoint_every,
    )

    print(json.dumps(metrics, indent=2))

    runtime_min = (
        time.perf_counter() - start_time
    ) / 60.0

    results = {
        "metrics": metrics,
        "args": vars(args),
        "dataset_path": dataset_path,
        "roberta_model_path": roberta_model_source,
        "tokenizer_path": tokenizer_source,
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