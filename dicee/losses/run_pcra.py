"""
One-time script to generate PCRA path files and optional heuristic DSKRL
auxiliary metadata files.
"""
import argparse
import os
import sys
import time
import random
from collections import defaultdict

def map_add(mp, key1, key2, value):
    if key1 not in mp:
        mp[key1] = {}
    if key2 not in mp[key1]:
        mp[key1][key2] = 0.0
    mp[key1][key2] += value


def parse_triple(line, order):
    seg = line.strip().split()
    if len(seg) < 3:
        return None
    if order == "s r o":
        return seg[0], seg[1], seg[2]
    if order == "s o r":
        return seg[0], seg[2], seg[1]
    raise ValueError(f"Unsupported triple order: {order}")


def read_triples(path, order):
    triples = []
    if not os.path.exists(path):
        return triples
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parsed = parse_triple(line, order)
            if parsed:
                h, r, t = parsed
                triples.append((h, r, t))
    return triples


def build_relation_mapping(triples):
    relation2id = {}
    id2relation = {}
    for _, r, _ in triples:
        if r not in relation2id:
            idx = len(relation2id)
            relation2id[r] = idx
            id2relation[idx] = r
    return relation2id, id2relation


def build_entity_mapping(triples):
    entity2id = {}
    for h, _, t in triples:
        if h not in entity2id:
            idx = len(entity2id)
            entity2id[h] = idx
        if t not in entity2id:
            idx = len(entity2id)
            entity2id[t] = idx
    return entity2id


def jaccard_similarity(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def build_entity_relation_signatures(triples):
    signatures = defaultdict(set)
    for h, r, t in triples:
        signatures[h].add(f"H::{r}")
        signatures[t].add(f"T::{r}")
    return signatures


def cluster_relation_role_sets(role_to_items, prefix, threshold):
    """
    Cluster relation-role item sets with greedy Jaccard matching.

    This is a heuristic fallback for datasets that do not ship explicit TKRL/DSKRL
    type-domain metadata. It produces reusable labels across relations when their
    observed argument sets look similar.
    """
    assignments = {}
    cluster_unions = []
    cluster_labels = []

    for relation in sorted(role_to_items):
        items = set(role_to_items[relation])
        best_idx = None
        best_score = -1.0
        for idx, rep_items in enumerate(cluster_unions):
            score = jaccard_similarity(items, rep_items)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= threshold:
            assignments[relation] = cluster_labels[best_idx]
            cluster_unions[best_idx].update(items)
        else:
            cluster_idx = len(cluster_unions)
            cluster_unions.append(set(items))
            cluster_labels.append(f"{prefix}_{cluster_idx}")
            assignments[relation] = cluster_labels[cluster_idx]
    return assignments


def write_label_id_file(path, labels):
    with open(path, "w") as handle:
        for idx, label in enumerate(labels):
            handle.write(f"{label} {idx}\n")
    print(f"Wrote: {path}")


def generate_dskrl_aux_files(
    dataset_dir,
    train_triples,
    relation2id,
    domain_jaccard_threshold=0.2,
    type_jaccard_threshold=0.5,
):
    """
    Generate heuristic relationType.txt / relationDomain.txt plus their id files.

    Domain labels are clustered from the observed sets of entities that appear in
    a relation's head/tail position. Type labels are clustered from the structural
    signatures of those entities, where a signature is the set of incident
    relation-role tokens such as H::r or T::r.
    """
    relation_order = [rel for rel, _ in sorted(relation2id.items(), key=lambda x: x[1])]
    head_entities = defaultdict(set)
    tail_entities = defaultdict(set)
    for h, r, t in train_triples:
        head_entities[r].add(h)
        tail_entities[r].add(t)

    entity_signatures = build_entity_relation_signatures(train_triples)
    head_signature_sets = {}
    tail_signature_sets = {}
    for rel in relation_order:
        head_signature_sets[rel] = {
            "|".join(sorted(entity_signatures[e])) if entity_signatures[e] else "__EMPTY__"
            for e in head_entities.get(rel, set())
        }
        tail_signature_sets[rel] = {
            "|".join(sorted(entity_signatures[e])) if entity_signatures[e] else "__EMPTY__"
            for e in tail_entities.get(rel, set())
        }

    head_domain_labels = cluster_relation_role_sets(
        head_entities,
        prefix="head_domain",
        threshold=domain_jaccard_threshold,
    )
    tail_domain_labels = cluster_relation_role_sets(
        tail_entities,
        prefix="tail_domain",
        threshold=domain_jaccard_threshold,
    )
    head_type_labels = cluster_relation_role_sets(
        head_signature_sets,
        prefix="head_type",
        threshold=type_jaccard_threshold,
    )
    tail_type_labels = cluster_relation_role_sets(
        tail_signature_sets,
        prefix="tail_type",
        threshold=type_jaccard_threshold,
    )

    type_labels = sorted(set(head_type_labels.values()) | set(tail_type_labels.values()))
    domain_labels = sorted(set(head_domain_labels.values()) | set(tail_domain_labels.values()))

    relation_type_path = os.path.join(dataset_dir, "relationType.txt")
    relation_domain_path = os.path.join(dataset_dir, "relationDomain.txt")
    type_id_path = os.path.join(dataset_dir, "type2id.txt")
    domain_id_path = os.path.join(dataset_dir, "domain2id.txt")

    with open(relation_type_path, "w") as handle:
        for rel in relation_order:
            handle.write(f"{rel} {head_type_labels[rel]} {tail_type_labels[rel]}\n")
    print(f"Wrote: {relation_type_path}")

    with open(relation_domain_path, "w") as handle:
        for rel in relation_order:
            handle.write(f"{rel} {head_domain_labels[rel]} {tail_domain_labels[rel]}\n")
    print(f"Wrote: {relation_domain_path}")

    write_label_id_file(type_id_path, type_labels)
    write_label_id_file(domain_id_path, domain_labels)

    # Per-entity type list, supporting the paper's Eq. 2:
    #   T_e = Σ_i α_i · T_{c_i}
    # where i ranges over all (type, domain) pairs entity e participates in.
    # The weight α_i is set to the normalised frequency of that pair across
    # the entity's appearances (as head or tail) in the training graph.
    entity_pair_counts = defaultdict(lambda: defaultdict(float))
    for h, r, t in train_triples:
        entity_pair_counts[h][(head_type_labels[r], head_domain_labels[r])] += 1.0
        entity_pair_counts[t][(tail_type_labels[r], tail_domain_labels[r])] += 1.0

    entity_types_path = os.path.join(dataset_dir, "entityTypes.txt")
    with open(entity_types_path, "w") as handle:
        for entity in sorted(entity_pair_counts.keys()):
            pair_counts = entity_pair_counts[entity]
            total = sum(pair_counts.values())
            if total <= 0:
                continue
            for (type_label, domain_label), count in sorted(pair_counts.items()):
                weight = count / total
                handle.write(f"{entity} {type_label} {domain_label} {weight:.6f}\n")
    print(f"Wrote: {entity_types_path}")


def resolve_batch_dataset_dirs(batch_root, datasets, max_perturbation):
    dirs = []
    for dataset_name in datasets:
        base = os.path.join(batch_root, dataset_name)
        if not os.path.isdir(base):
            continue
        for subdir in os.listdir(base):
            full_path = os.path.join(base, subdir)
            if not os.path.isdir(full_path):
                continue
            try:
                perturbation = float(subdir)
            except ValueError:
                continue
            if max_perturbation is not None and perturbation > max_perturbation:
                continue
            dirs.append((dataset_name, perturbation, full_path))
    dirs.sort(key=lambda x: (x[0], x[1]))
    return [p for _, _, p in dirs]


def generate_pra_for_dataset(args, dataset_dir):
    print(f"[PCRA] Processing dataset_dir={dataset_dir}")
    train_path = os.path.join(dataset_dir, "train.txt")
    test_path = os.path.join(dataset_dir, "test.txt")

    train_triples = read_triples(train_path, args.triple_order)
    test_triples = read_triples(test_path, args.triple_order)

    if not train_triples:
        raise FileNotFoundError(f"train.txt not found or empty in {dataset_dir}")

    relation2id, id2relation = build_relation_mapping(train_triples)
    relation_num = len(relation2id)
    for rid, rname in list(id2relation.items()):
        id2relation[rid + relation_num] = "~" + rname

    entity2id = build_entity_mapping(train_triples)

    ok = {}
    a = {}

    for h, r, t in train_triples:
        rel_id = relation2id[r]
        key_ht = f"{h} {t}"
        key_th = f"{t} {h}"
        if key_ht not in ok:
            ok[key_ht] = {}
        ok[key_ht][rel_id] = 1
        if key_th not in ok:
            ok[key_th] = {}
        ok[key_th][rel_id + relation_num] = 1

        if h not in a:
            a[h] = {}
        if rel_id not in a[h]:
            a[h][rel_id] = {}
        a[h][rel_id][t] = 1

        if t not in a:
            a[t] = {}
        if (rel_id + relation_num) not in a[t]:
            a[t][rel_id + relation_num] = {}
        a[t][rel_id + relation_num][h] = 1

    for h, _, t in test_triples:
        ok.setdefault(f"{h} {t}", {})
        ok.setdefault(f"{t} {h}", {})

    h_e_p = {}

    step = 0
    time1 = time.time()
    path_num = 0

    for e1 in a:
        step += 1
        print(step, end=" ")
        for rel1 in a[e1]:
            e2_set = a[e1][rel1]
            for e2 in e2_set:
                map_add(h_e_p, f"{e1} {e2}", str(rel1), 1.0 / len(e2_set))

        for rel1 in a[e1]:
            e2_set = a[e1][rel1]
            for e2 in e2_set:
                if e2 in a:
                    for rel2 in a[e2]:
                        e3_set = a[e2][rel2]
                        for e3 in e3_set:
                            if f"{e1} {e3}" in ok:
                                map_add(
                                    h_e_p,
                                    f"{e1} {e3}",
                                    f"{rel1} {rel2}",
                                    h_e_p[f"{e1} {e2}"][str(rel1)] * 1.0 / len(e3_set),
                                )

        for e2 in a:
            if f"{e1} {e2}" in h_e_p:
                path_num += len(h_e_p[f"{e1} {e2}"])
                bb = {}
                aa = {}
                sum_val = 0.0
                for rel_path in h_e_p[f"{e1} {e2}"]:
                    bb[rel_path] = h_e_p[f"{e1} {e2}"][rel_path]
                    sum_val += bb[rel_path]
                for rel_path in bb:
                    bb[rel_path] /= sum_val
                    if bb[rel_path] > args.min_prob:
                        aa[rel_path] = bb[rel_path]
        print(path_num, time.time() - time1)
        sys.stdout.flush()

    def write_pos_pra(name, triples):
        out_path = os.path.join(dataset_dir, f"{name}_pra.txt")
        if not triples:
            return
        with open(out_path, "w") as f_out:
            for e1, rel, e2 in triples:
                rel_id = relation2id[rel]
                f_out.write(f"{e1} {e2} {rel_id}\n")
                b = {}
                a_local = {}
                if f"{e1} {e2}" in h_e_p:
                    sum_val = 0.0
                    for rel_path in h_e_p[f"{e1} {e2}"]:
                        b[rel_path] = h_e_p[f"{e1} {e2}"][rel_path]
                        sum_val += b[rel_path]
                    for rel_path in b:
                        b[rel_path] /= sum_val
                        if b[rel_path] > args.min_prob:
                            a_local[rel_path] = b[rel_path]
                f_out.write(str(len(a_local)))
                for rel_path in a_local:
                    f_out.write(f" {len(rel_path.split())} {rel_path} {a_local[rel_path]}")
                f_out.write("\n")

                # reverse triple
                f_out.write(f"{e2} {e1} {rel_id + relation_num}\n")
                b = {}
                a_local = {}
                if f"{e2} {e1}" in h_e_p:
                    sum_val = 0.0
                    for rel_path in h_e_p[f"{e2} {e1}"]:
                        b[rel_path] = h_e_p[f"{e2} {e1}"][rel_path]
                        sum_val += b[rel_path]
                    for rel_path in b:
                        b[rel_path] /= sum_val
                        if b[rel_path] > args.min_prob:
                            a_local[rel_path] = b[rel_path]
                f_out.write(str(len(a_local)))
                for rel_path in a_local:
                    f_out.write(f" {len(rel_path.split())} {rel_path} {a_local[rel_path]}")
                f_out.write("\n")
        print(f"Wrote: {out_path}")

    def write_neg_pra(name, triples):
        out_path = os.path.join(dataset_dir, f"neg_{name}_pra.txt")
        if not triples:
            return
        rng = random.Random(args.seed)
        entities = list(entity2id.keys())
        with open(out_path, "w") as f_out:
            for h, r, t in triples:
                rel_id = relation2id[r]
                for _ in range(args.neg_ratio):
                    if rng.random() < 0.5:
                        h_neg = rng.choice(entities)
                        e1, e2 = h_neg, t
                    else:
                        t_neg = rng.choice(entities)
                        e1, e2 = h, t_neg
                    f_out.write(f"{e1} {e2} {rel_id}\n")
                    b = {}
                    a_local = {}
                    if f"{e1} {e2}" in h_e_p:
                        sum_val = 0.0
                        for rel_path in h_e_p[f"{e1} {e2}"]:
                            b[rel_path] = h_e_p[f"{e1} {e2}"][rel_path]
                            sum_val += b[rel_path]
                        for rel_path in b:
                            b[rel_path] /= sum_val
                            if b[rel_path] > args.min_prob:
                                a_local[rel_path] = b[rel_path]
                    f_out.write(str(len(a_local)))
                    for rel_path in a_local:
                        f_out.write(f" {len(rel_path.split())} {rel_path} {a_local[rel_path]}")
                    f_out.write("\n")
        print(f"Wrote: {out_path}")

    write_pos_pra("train", train_triples)
    write_neg_pra("train", train_triples)
    if args.generate_dskrl_aux:
        generate_dskrl_aux_files(
            dataset_dir=dataset_dir,
            train_triples=train_triples,
            relation2id=relation2id,
            domain_jaccard_threshold=args.dskrl_domain_jaccard_threshold,
            type_jaccard_threshold=args.dskrl_type_jaccard_threshold,
        )


def main():
    parser = argparse.ArgumentParser(description="PCRA generator")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Single dataset dir containing train.txt/valid.txt/test.txt")
    parser.add_argument("--batch_root", type=str, default=None,
                        help="Root like Datasets_Perturbed for batch processing")
    parser.add_argument("--batch_datasets", type=str, nargs="+", default=["KINSHIP", "UMLS"],
                        help="Dataset names under --batch_root to process")
    parser.add_argument("--max_perturbation", type=float, default=None,
                        help="Only process perturbation folders <= this value (e.g., 0.32)")
    parser.add_argument("--triple_order", type=str, default="s r o",
                        help='"s r o" or "s o r"')
    parser.add_argument("--min_prob", type=float, default=0.01)
    parser.add_argument("--neg_ratio", type=int, default=1,
                        help="Number of negative triples per positive for neg_train_pra.txt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--generate_dskrl_aux",
        action="store_true",
        help="Also generate heuristic relationType.txt/relationDomain.txt/type2id.txt/domain2id.txt",
    )
    parser.add_argument(
        "--dskrl_domain_jaccard_threshold",
        type=float,
        default=0.2,
        help="Jaccard threshold for clustering relation-role entity sets into domain labels",
    )
    parser.add_argument(
        "--dskrl_type_jaccard_threshold",
        type=float,
        default=0.5,
        help="Jaccard threshold for clustering entity-signature sets into type labels",
    )
    args = parser.parse_args()

    dataset_dirs = []
    if args.batch_root:
        dataset_dirs.extend(
            resolve_batch_dataset_dirs(
                batch_root=args.batch_root,
                datasets=args.batch_datasets,
                max_perturbation=args.max_perturbation,
            )
        )
    if args.dataset_dir:
        dataset_dirs.append(args.dataset_dir)

    # Keep order, remove duplicates.
    seen = set()
    dataset_dirs = [d for d in dataset_dirs if not (d in seen or seen.add(d))]

    if not dataset_dirs:
        raise ValueError("Provide --dataset_dir, or --batch_root with matching folders.")

    for dataset_dir in dataset_dirs:
        generate_pra_for_dataset(args=args, dataset_dir=dataset_dir)


if __name__ == "__main__":
    main()
