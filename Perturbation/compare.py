from collections import Counter, defaultdict
import argparse

def _read_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples

def compare_triple_files(path_a, path_b, *, report_order=True, max_order_examples=10):

    A = _read_triples(path_a)
    B = _read_triples(path_b)

    ca = Counter(A)
    cb = Counter(B)

    shared = sum((ca & cb).values())
    only_a = sum((ca - cb).values())
    only_b = sum((cb - ca).values())

    out = {
        #"file_a": path_a,
        #"file_b": path_b,
        "num_lines_a": len(A),
        "num_lines_b": len(B),
        "shared_triples_count": shared,
        "only_in_a_count": only_a,
        "only_in_b_count": only_b,
    }

    if report_order:
        pos_a = defaultdict(list)
        pos_b = defaultdict(list)
        for i, tr in enumerate(A):
            pos_a[tr].append(i)
        for i, tr in enumerate(B):
            pos_b[tr].append(i)

        common = list((ca & cb).keys())
        moved = 0
        checked = 0
        examples = []

        for tr in common:
            checked += 1
            if pos_a[tr][0] != pos_b[tr][0]:
                moved += 1
                if len(examples) < max_order_examples:
                    examples.append((tr, pos_a[tr][0], pos_b[tr][0]))

        out.update({
            "unique_shared_with_different_first_position": moved,
        })


    return out

def main():
    parser = argparse.ArgumentParser(
        description="Compare two text files line-by-line and count same/different rows."
    )
    parser.add_argument("file_a", help="First text file")
    parser.add_argument("file_b", help="Second text file")
    args = parser.parse_args()

    print(compare_triple_files(args.file_a, args.file_b))

if __name__ == "__main__":
    main()


