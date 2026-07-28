import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch


DBS = ["UMLS"] #, "KINSHIP" , "NELL-995-h100", "FB15k-237", "WN18RR"] 

RECIPRIOCAL = "without_recipriocal" 

PERCENTAGES = [0.02, 0.04, 0.06]  

NUM_EXPERIMENTS = 2

SAVED_DATASETS_ROOT = Path(f"./saved_perturbed_datasets/{RECIPRIOCAL}/")
SAVED_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 12345
seed_src = random.Random(MASTER_SEED)
EXPERIMENT_SEEDS = [seed_src.randrange(2**32) for _ in range(NUM_EXPERIMENTS)]

def save_triples(triple_list, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triple_list:
            f.write(f"{h}\t{r}\t{t}\n")

def set_seeds(seed):
    try:
        s = int(np.uint32(seed))
    except Exception:
        s = int(np.uint32(abs(hash(str(seed)))))

    os.environ["PYTHONHASHSEED"] = str(s)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def load_triples(path):
    f = open(path, 'r')
    try:
        triples = [tuple(line.strip().split()[:3]) for line in f]
    finally:
        f.close()
    return triples

def save_perturbed_dataset(
    original_triples,
    perturbed_triples,
    feature_tag,
    DB,
    top_k,
    experiment_seed,
    test_path,
    valid_path,
):
    out_dir = (SAVED_DATASETS_ROOT/ DB/ "perturb"/ feature_tag/ str(top_k)/ str(experiment_seed))
    out_dir.mkdir(parents=True, exist_ok=True)

    original_set = set(original_triples)
    perturbed_set = set(perturbed_triples)

    removed_ordered = [x for x in original_triples if x not in perturbed_set]
    added_ordered = [x for x in perturbed_triples if x not in original_set]

    save_triples(perturbed_triples, str(out_dir / "train.txt"))
    save_triples(removed_ordered, str(out_dir / "removed.txt"))
    save_triples(added_ordered, str(out_dir / "added.txt"))

    shutil.copy2(test_path, str(out_dir / "test.txt"))
    shutil.copy2(valid_path, str(out_dir / "valid.txt"))


def perturb_random(triples, k, seed):
    rng = random.Random(seed)
    n = len(triples)

    heads = [h for h, r, t in triples]
    rels = [r for h, r, t in triples]
    tails = [t for h, r, t in triples]

    idx = list(range(n))
    rng.shuffle(idx)
    pick = set(idx[:k])

    out = []
    for i, (h, r, t) in enumerate(triples):
        if i not in pick:
            out.append((h, r, t))
            continue

        which = rng.randint(0, 2)

        if which == 0:
            new_h = rng.choice(heads)
            while new_h == h and len(set(heads)) > 1:
                new_h = rng.choice(heads)
            out.append((new_h, r, t))

        elif which == 1:
            new_r = rng.choice(rels)
            while new_r == r and len(set(rels)) > 1:
                new_r = rng.choice(rels)
            out.append((h, new_r, t))

        else:
            new_t = rng.choice(tails)
            while new_t == t and len(set(tails)) > 1:
                new_t = rng.choice(tails)
            out.append((h, r, new_t))

    return [], out


def main():

    for DB in DBS:
        TRIPLES_PATH = f"../KGs/{DB}/train.txt"
        VALID_PATH = f"../KGs/{DB}/valid.txt"
        TEST_PATH = f"../KGs/{DB}/test.txt"

        train_triples = load_triples(TRIPLES_PATH)
        val_triples = load_triples(VALID_PATH)
        test_triples = load_triples(TEST_PATH)

        n_train = len(train_triples)
        budgets = [max(1, int(n_train * p)) for p in PERCENTAGES]

        for exp_seed in EXPERIMENT_SEEDS:
            set_seeds(exp_seed)

            for top_k in budgets:
                print(f"\\n=== PERTURB | {DB} | seed={exp_seed} | budget={top_k} ===")

                _, perturbed_triples = perturb_random(train_triples, top_k, seed=exp_seed)

                save_perturbed_dataset(train_triples, perturbed_triples, "random", DB, top_k, exp_seed, TEST_PATH, VALID_PATH)


if __name__ == "__main__":
    main()
