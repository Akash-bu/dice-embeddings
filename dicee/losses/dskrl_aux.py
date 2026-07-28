import os
from typing import Dict, Optional


def _to_index_dict(mapping, key_name: str):
    if isinstance(mapping, dict):
        if mapping and all(isinstance(k, int) for k in mapping.keys()):
            return {v: k for k, v in mapping.items()}
        return mapping
    if hasattr(mapping, "to_dict") and not hasattr(mapping, "columns"):
        series_dict = mapping.to_dict()
        if series_dict and all(isinstance(k, int) for k in series_dict.keys()):
            return {v: k for k, v in series_dict.items()}
        return series_dict
    df = mapping
    if hasattr(mapping, "to_pandas"):
        df = mapping.to_pandas()
    if hasattr(df, "columns"):
        if key_name in df.columns and len(df.columns) == 1:
            return dict(zip(df[key_name], df.index))
        if key_name in df.columns and "index" in df.columns:
            return dict(zip(df[key_name], df["index"]))
        if len(df.columns) >= 2:
            return dict(zip(df[df.columns[0]], df[df.columns[1]]))
    raise TypeError(f"Unsupported mapping type for {key_name}: {type(mapping)}")


def _read_label_id_file(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        return {}
    mapping = {}
    with open(path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                mapping[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return mapping


def load_dskrl_aux_data(
    dataset_dir: Optional[str],
    relation_to_idx,
    entity_to_idx=None,
    relation_type_file: str = "relationType.txt",
    relation_domain_file: str = "relationDomain.txt",
    type_id_file: str = "type2id.txt",
    domain_id_file: str = "domain2id.txt",
    entity_types_file: str = "entityTypes.txt",
):
    if not dataset_dir:
        return {}

    relation_to_idx = _to_index_dict(relation_to_idx, "relation")
    if not relation_to_idx:
        return {}

    type_to_idx = _read_label_id_file(os.path.join(dataset_dir, type_id_file))
    domain_to_idx = _read_label_id_file(os.path.join(dataset_dir, domain_id_file))
    relation_type_path = os.path.join(dataset_dir, relation_type_file)
    relation_domain_path = os.path.join(dataset_dir, relation_domain_file)

    num_relations = max(int(idx) for idx in relation_to_idx.values()) + 1
    head_type_ids = [-1] * num_relations
    tail_type_ids = [-1] * num_relations
    head_domain_ids = [-1] * num_relations
    tail_domain_ids = [-1] * num_relations

    if os.path.exists(relation_type_path):
        with open(relation_type_path, "r") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                rel_name, head_type, tail_type = parts[:3]
                rel_id = relation_to_idx.get(rel_name)
                if rel_id is None:
                    continue
                head_type_ids[int(rel_id)] = type_to_idx.get(head_type, -1)
                tail_type_ids[int(rel_id)] = type_to_idx.get(tail_type, -1)

    if os.path.exists(relation_domain_path):
        with open(relation_domain_path, "r") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                rel_name, head_domain, tail_domain = parts[:3]
                rel_id = relation_to_idx.get(rel_name)
                if rel_id is None:
                    continue
                head_domain_ids[int(rel_id)] = domain_to_idx.get(head_domain, -1)
                tail_domain_ids[int(rel_id)] = domain_to_idx.get(tail_domain, -1)

    result = {
        "num_types": len(type_to_idx),
        "num_domains": len(domain_to_idx),
        "head_type_ids": head_type_ids,
        "tail_type_ids": tail_type_ids,
        "head_domain_ids": head_domain_ids,
        "tail_domain_ids": tail_domain_ids,
    }

    # Per-entity type list for the paper's full Eq. 2 + Eq. 3 encoder.
    # entity_types[ent_id] = [(type_id, domain_id, weight), ...]
    entity_types_path = os.path.join(dataset_dir, entity_types_file)
    if entity_to_idx is not None and os.path.exists(entity_types_path):
        ent_to_idx = _to_index_dict(entity_to_idx, "entity")
        num_entities = max(int(idx) for idx in ent_to_idx.values()) + 1
        entity_types = [[] for _ in range(num_entities)]
        with open(entity_types_path, "r") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                ent_name, type_label, domain_label, weight_s = parts[:4]
                ent_id = ent_to_idx.get(ent_name)
                if ent_id is None:
                    continue
                type_id = type_to_idx.get(type_label, -1)
                domain_id = domain_to_idx.get(domain_label, -1)
                if type_id < 0 and domain_id < 0:
                    continue
                try:
                    weight = float(weight_s)
                except ValueError:
                    continue
                entity_types[int(ent_id)].append((type_id, domain_id, weight))
        result["entity_types"] = entity_types
    return result
