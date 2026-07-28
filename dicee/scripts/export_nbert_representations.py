import argparse
import csv
import importlib.util
import os

import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer


def get_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(base_path, path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(base_path, path)


def load_class_from_file(module_name, path, class_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def load_nbert_class():
    nbert_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "nbert.py"))
    return load_class_from_file("dicee_nbert_standalone", nbert_path, "NBert")


def load_prompt_data_class():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nbert_prompt_data.py"))
    return load_class_from_file("dicee_nbert_prompt_data_standalone", data_path, "NBertPromptDataModule")


def resolve_dataset_path(repo_root, dataset, dataset_path):
    if dataset_path is not None:
        return resolve_path(repo_root, dataset_path)
    if dataset is None:
        raise ValueError("Pass either --dataset_path or --dataset.")

    candidates = [
        resolve_path(repo_root, dataset),
        os.path.join(repo_root, "KGs", dataset),
        os.path.join(repo_root, "KGs", dataset.upper()),
        os.path.join(repo_root, "KGs", dataset.lower()),
        os.path.join(repo_root, "datasets", dataset),
        os.path.join(repo_root, "datasets", dataset.lower()),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not resolve dataset {dataset!r}. Pass --dataset_path explicitly."
    )


def build_nbert_config(args, repo_root, text_offsets):
    config = {
        "device": args.device,
        "model_path": resolve_path(repo_root, args.bert_model_path),
        "tokenizer_path": args.tokenizer_path,
        "max_seq_length": args.max_seq_length,
    }
    config.update(text_offsets)
    return config


def load_dice_mapping(mapping_dir, kind):
    path = os.path.join(mapping_dir, f"{kind}_to_idx.csv")
    name_column = "entity" if kind == "entity" else "relation"
    mapping = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        index_column = reader.fieldnames[0]
        for row in reader:
            mapping[row[name_column]] = int(row[index_column])
    return mapping


def encode_dice_keys(keys_text, mapping_dir):
    entity_to_idx = load_dice_mapping(mapping_dir, "entity")
    relation_to_idx = load_dice_mapping(mapping_dir, "relation")
    keys = []
    missing = []
    for head, relation, tail in keys_text:
        if head not in entity_to_idx or tail not in entity_to_idx or relation not in relation_to_idx:
            missing.append((head, relation, tail))
            continue
        keys.append([entity_to_idx[head], relation_to_idx[relation], entity_to_idx[tail]])
    if missing:
        preview = ", ".join(map(str, missing[:3]))
        raise ValueError(
            f"{len(missing)} triples could not be encoded with Dice mappings. "
            f"First missing triples: {preview}"
        )
    return torch.tensor(keys, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export frozen N-BERT triple representations for Dice CCA alignment."
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Dice dataset directory containing train.txt/valid.txt/test.txt and support/.",
    )
    parser.add_argument(
        "--support_path",
        type=str,
        default=None,
        help="Optional directory containing entity.json and relation.json. Defaults to <dataset_path>/support.",
    )
    parser.add_argument("--bert_model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="bert-base-cased")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--dice_mapping_dir",
        type=str,
        default=None,
        help="Optional Dice run/preprocess directory containing entity_to_idx.csv and relation_to_idx.csv.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--anomaly_folder", type=str, default=None)
    parser.add_argument("--anomaly_ratio", type=float, default=None)
    parser.add_argument(
        "--include_splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to export. Use valid or dev depending on the dataset files.",
    )
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include_score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save scalar N-BERT log-probability scores.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = get_repo_root()
    NBert = load_nbert_class()
    NBertPromptDataModule = load_prompt_data_class()

    dataset_path = resolve_dataset_path(repo_root, args.dataset, args.dataset_path)
    support_path = resolve_path(repo_root, args.support_path) if args.support_path else None
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_basic_tokenize=False)
    data_module = NBertPromptDataModule(
        dataset_path=dataset_path,
        support_path=support_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_seq_length=args.max_seq_length,
        anomaly_folder=args.anomaly_folder,
        anomaly_ratio=args.anomaly_ratio,
        include_splits=tuple(args.include_splits),
    )
    tokenizer = data_module.get_tokenizer()
    dataloader = data_module.get_train_dataloader()

    config = build_nbert_config(args, repo_root, data_module.text_offsets)
    bert_encoder = BertForMaskedLM.from_pretrained(config["model_path"])
    scorer = NBert(config, tokenizer=tokenizer, bert_encoder=bert_encoder).to(config["device"])
    scorer.eval()

    keys_text = []
    repr_chunks = []
    score_chunks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Exporting N-BERT representations"):
            output = scorer.link_prediction(batch)
            bert_repr = torch.cat([output["head_repr"], output["tail_repr"]], dim=-1)
            repr_chunks.append(bert_repr.detach().cpu())
            if args.include_score:
                score_chunks.append(output["bert_score"].detach().cpu())
            keys_text.extend([tuple(triple) for triple in batch["data"]])

    payload = {
        "dataset": args.dataset,
        "keys_text": keys_text,
        "bert_repr": torch.cat(repr_chunks, dim=0),
        "repr_kind": "concat(head_repr, tail_repr)",
        "source": {
            "dataset_path": dataset_path,
            "support_path": data_module.support_path,
            "bert_model_path": config["model_path"],
            "tokenizer_path": config["tokenizer_path"],
            "max_seq_length": args.max_seq_length,
            "anomaly_folder": args.anomaly_folder,
            "anomaly_ratio": args.anomaly_ratio,
            "include_splits": args.include_splits,
        },
    }
    if args.include_score:
        payload["bert_score"] = torch.cat(score_chunks, dim=0)
    if args.dice_mapping_dir is not None:
        mapping_dir = os.path.abspath(args.dice_mapping_dir)
        payload["keys"] = encode_dice_keys(keys_text, mapping_dir)
        payload["key_kind"] = "dice_int_triples"
        payload["source"]["dice_mapping_dir"] = mapping_dir

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"Saved {len(keys_text)} N-BERT representations to {args.output_path}")


if __name__ == "__main__":
    main()
