import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class NBertPromptDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        for code, example in enumerate(self.examples):
            example["code"] = code

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class NBertPromptDataModule:
    """Build the frozen N-BERT prompts used for head/tail link prediction."""

    def __init__(
        self,
        dataset_path,
        tokenizer,
        batch_size,
        num_workers,
        pin_memory,
        max_seq_length,
        support_path=None,
        anomaly_folder=None,
        anomaly_ratio=None,
        include_splits=("train", "valid", "test"),
    ):
        self.dataset_path = os.path.abspath(dataset_path)
        self.support_path = os.path.abspath(support_path or os.path.join(self.dataset_path, "support"))
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq_length = max_seq_length
        self.anomaly_folder = anomaly_folder
        self.anomaly_ratio = anomaly_ratio
        self.include_splits = include_splits

        self.entities, self.relations = self.read_support()
        self.text_offsets = self.resize_tokenizer()
        self.lines = self.read_lines()
        examples = self.create_examples()
        random.shuffle(examples)
        self.train_ds = NBertPromptDataset(examples)

    def read_support(self):
        entity_path = os.path.join(self.support_path, "entity.json")
        relation_path = os.path.join(self.support_path, "relation.json")
        if not os.path.exists(entity_path) or not os.path.exists(relation_path):
            raise FileNotFoundError(
                "N-BERT export requires support/entity.json and support/relation.json. "
                f"Looked in: {self.support_path}"
            )

        with open(entity_path, "r", encoding="utf-8") as handle:
            entities = json.load(handle)
        for idx, entity_id in enumerate(entities):
            raw_name = entities[entity_id]["name"]
            entities[entity_id] = {
                "token_id": idx,
                "name": f"[E_{idx}]",
                "desc": entities[entity_id]["desc"],
                "raw_name": raw_name,
            }

        with open(relation_path, "r", encoding="utf-8") as handle:
            relations = json.load(handle)
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

    def resize_tokenizer(self):
        entity_begin_idx = len(self.tokenizer)
        entity_names = [entity["name"] for entity in self.entities.values()]
        self.tokenizer.add_special_tokens({"additional_special_tokens": entity_names})
        entity_end_idx = len(self.tokenizer)

        relation_begin_idx = len(self.tokenizer)
        relation_names = []
        for relation in self.relations.values():
            relation_names.extend(
                [
                    relation["sep1"],
                    relation["sep2"],
                    relation["sep3"],
                    relation["sep4"],
                    relation["sep5"],
                ]
            )
        self.tokenizer.add_special_tokens({"additional_special_tokens": relation_names})
        relation_end_idx = len(self.tokenizer)

        return {
            "text_entity_begin_idx": entity_begin_idx,
            "text_entity_end_idx": entity_end_idx,
            "text_relation_begin_idx": relation_begin_idx,
            "text_relation_end_idx": relation_end_idx,
        }

    def read_lines(self):
        split_files = {
            "train": "train.txt",
            "valid": "valid.txt",
            "dev": "dev.txt",
            "test": "test.txt",
        }
        lines = {}
        for split in self.include_splits:
            file_name = split_files.get(split, f"{split}.txt")
            path = os.path.join(self.dataset_path, file_name)
            if split == "valid" and not os.path.exists(path):
                path = os.path.join(self.dataset_path, "dev.txt")
            if not os.path.exists(path):
                continue
            lines[split] = self.read_triples(path)

        if self.anomaly_folder and self.anomaly_ratio is not None:
            anomaly_dir = os.path.join(
                self.dataset_path,
                self.anomaly_folder,
                str(int(self.anomaly_ratio * 100)),
            )
            anomaly_path = os.path.join(anomaly_dir, "anomaly_triples.txt")
            if os.path.exists(anomaly_path):
                lines["anomaly"] = self.read_triples(anomaly_path)

        if not lines:
            raise FileNotFoundError(
                f"No split files found in {self.dataset_path}. Expected train.txt, valid.txt/dev.txt, or test.txt."
            )
        return lines

    def read_triples(self, path):
        triples = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    parts = line.strip().split()
                if len(parts) != 3:
                    raise ValueError(f"Could not parse triple line in {path}: {line!r}")
                head, relation, tail = parts
                if head not in self.entities or tail not in self.entities or relation not in self.relations:
                    raise ValueError(
                        "Triple is missing N-BERT support metadata: "
                        f"{(head, relation, tail)} from {path}"
                    )
                triples.append((head, relation, tail))
        return triples

    def create_examples(self):
        examples = []
        for split, triples in self.lines.items():
            for head, relation, tail in tqdm(triples, desc=f"[{split}] N-BERT prompts"):
                examples.append(self.create_one_example(head, relation, tail))
        return examples

    def create_one_example(self, head_id, relation_id, tail_id):
        mask_token = self.tokenizer.mask_token
        head = self.entities[head_id]
        relation = self.relations[relation_id]
        tail = self.entities[tail_id]

        head_desc = self.truncate_desc(head["desc"])
        tail_desc = self.truncate_desc(tail["desc"])
        relation_name = relation["name"]

        text_head_prompt = " ".join(
            [
                relation["sep1"],
                mask_token,
                relation["sep2"],
                relation_name,
                relation["sep3"],
                tail["name"],
                relation["sep4"],
                tail_desc,
            ]
        )
        text_tail_prompt = " ".join(
            [
                relation["sep1"],
                head["name"],
                relation["sep2"],
                relation_name,
                relation["sep3"],
                mask_token,
                relation["sep4"],
                head_desc,
            ]
        )

        return {
            "data_triple": (head_id, relation_id, tail_id),
            "data_text": (head["raw_name"], relation_name, tail["raw_name"]),
            "text_head_prompt": text_head_prompt,
            "text_tail_prompt": text_tail_prompt,
            "head_label": head["token_id"],
            "tail_label": tail["token_id"],
        }

    def truncate_desc(self, desc):
        tokens = str(desc).split()
        return " ".join(tokens[: min(self.max_seq_length - 7, len(tokens))])

    def text_batch_encoding(self, inputs):
        encoded_data = self.tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids = torch.tensor(encoded_data["input_ids"])
        token_type_ids = torch.tensor(encoded_data["token_type_ids"])
        attention_mask = torch.tensor(encoded_data["attention_mask"])
        mask_pos = torch.nonzero(torch.eq(input_ids, self.tokenizer.mask_token_id))
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "mask_pos": mask_pos,
        }

    def collate_fn(self, batch_data):
        return {
            "data": [example["data_triple"] for example in batch_data],
            "data_text": [example["data_text"] for example in batch_data],
            "code": [example["code"] for example in batch_data],
            "head_labels": torch.tensor([example["head_label"] for example in batch_data]),
            "tail_labels": torch.tensor([example["tail_label"] for example in batch_data]),
            "text_head_prompts": self.text_batch_encoding(
                [example["text_head_prompt"] for example in batch_data]
            ),
            "text_tail_prompts": self.text_batch_encoding(
                [example["text_tail_prompt"] for example in batch_data]
            ),
        }

    def get_tokenizer(self):
        return self.tokenizer

    def get_train_dataloader(self):
        return DataLoader(
            self.train_ds,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

