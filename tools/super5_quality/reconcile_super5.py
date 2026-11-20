"""Reconcile a Super5 result with another Super5 run using a sequence-only alignment guide."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import os
from pathlib import Path
import shutil
import tempfile


GAPS = b"-."


def read_fasta(path):
    rows = {}
    label = None
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(b">"):
                label = line[1:]
                if not label or label in rows:
                    raise ValueError(f"Empty or duplicate FASTA label in {path}")
                rows[label] = bytearray()
            elif label is None:
                raise ValueError(f"Missing FASTA label in {path}")
            else:
                rows[label].extend(b"".join(line.split()))
    if len({len(row) for row in rows.values()}) != 1 or not all(rows.values()):
        raise ValueError(f"Expected a nonempty rectangular alignment in {path}")
    return rows


def prefix_counts(matrix):
    counts = [0] * len(matrix)
    total = 0
    for j, column in enumerate(zip(*matrix), 1):
        for i, letter in enumerate(column):
            if letter not in GAPS:
                counts[i] += 1
                total += 1
        yield total, j, counts


def shared_boundaries(base, alternate):
    left, right = prefix_counts(base), prefix_counts(alternate)
    a, b = next(left, None), next(right, None)
    yield 0, 0, [0] * len(base)
    while a and b:
        if a[0] == b[0]:
            if a[2] == b[2]:
                yield a[1], b[1], list(a[2])
            a, b = next(left, None), next(right, None)
        elif a[0] < b[0]:
            a = next(left, None)
        else:
            b = next(right, None)


def positions(matrix):
    return [[j for j, letter in enumerate(row) if letter not in GAPS] for row in matrix]


def count_pairs(groups):
    return sum(n * (n - 1) // 2 for n in groups.values())


def agreement(columns, guide, start, end):
    all_columns = Counter()
    joint = Counter()
    for i, (lo, hi) in enumerate(zip(start, end)):
        for j in range(lo, hi):
            c = columns[i][j]
            all_columns[c] += 1
            joint[c, guide[i][j]] += 1
    supported = count_pairs(joint)
    return supported, count_pairs(all_columns) - supported


def pairwise_agreement(columns, guide, start, end):
    by_column = defaultdict(list)
    for i,(lo,hi) in enumerate(zip(start,end)):
        for j in range(lo,hi):
            by_column[columns[i][j]].append((i,guide[i][j]))
    pairs = defaultdict(lambda: [0, 0])
    for group in by_column.values():
        for (i,x),(j,y) in combinations(group,2):
            pairs[i,j][0 if x==y else 1] += 1
    return pairs


def favors_alternative(base, alternate, guide, start, end, local):
    true_a, false_a = agreement(base, guide, start, end)
    true_b, false_b = agreement(alternate, guide, start, end)
    if true_b < true_a or false_b > false_a or (true_a, false_a) == (true_b, false_b):
        return False
    if local:
        return True
    old = pairwise_agreement(base, guide, start, end)
    new = pairwise_agreement(alternate, guide, start, end)
    for pair in old.keys() | new.keys():
        good_a, bad_a = old.get(pair, (0, 0))
        good_b, bad_b = new.get(pair, (0, 0))
        if good_b < good_a or bad_b > bad_a:
            return False
    return True


def select(base, alternate, guides, policy):
    names = list(base)
    raw = [row.translate(None, GAPS) for row in base.values()]
    for matrix in [alternate, *guides]:
        if matrix.keys() != base.keys():
            raise ValueError("Alignment labels differ")
        if any(matrix[n].translate(None, GAPS).upper() != s.upper() for n,s in zip(names,raw)):
            raise ValueError("Ungapped residues differ")
    first = list(base.values())
    second = [alternate[n] for n in names]
    if first == second:
        return []
    first_pos, second_pos = positions(first), positions(second)
    guide_pos = [positions([g[n] for n in names]) for g in guides]
    total = sum(len(s) for s in raw)
    boundaries = iter(shared_boundaries(first,second))
    a0,b0,p0 = next(boundaries)
    selected = []
    for a1,b1,p1 in boundaries:
        if any(x[a0:a1] != y[b0:b1] for x,y in zip(first,second)):
            # Larger substitutions need individual support for every affected sequence pair.
            local = 2*(sum(p1)-sum(p0)) < total and policy != "pair"
            votes = [favors_alternative(first_pos, second_pos, g, p0, p1, local) for g in guide_pos]
            if policy == "pair-or-unanimous":
                accept = votes[0] and (not local or all(votes) or
                    favors_alternative(first_pos, second_pos, guide_pos[0], p0, p1, False))
            elif policy == "corroborate":
                accept = votes[0] and (votes[1] or votes[2])
            elif policy == "primary-or-two":
                accept = votes[0] or (votes[1] and votes[2])
            elif policy == "majority":
                accept = sum(votes) >= 2
            else:
                accept = votes[0]
            if accept:
                selected.append((a0,a1,b0,b1))
        a0,b0,p0 = a1,b1,p1
    if any(p < len(r) for p,r in zip(p0,raw)):
        raise ValueError("Trailing residues lack a common boundary")
    return selected


def reconcile(base_path, alternate_path, guide_paths, out, policy):
    if policy not in ("local", "pair", "majority", "primary-or-two", "corroborate", "pair-or-unanimous"):
        raise ValueError(f"Unknown comparison rule: {policy}")
    required_guides = 3 if policy in ("majority","primary-or-two","corroborate","pair-or-unanimous") else 1
    if len(guide_paths) != required_guides:
        raise ValueError(f"Expected {required_guides} guide alignments")
    if any(p.resolve() == out.resolve() or (out.exists() and out.samefile(p))
           for p in [base_path, alternate_path, *guide_paths]):
        raise ValueError("Choose an output path distinct from the inputs")
    base = read_fasta(base_path)
    alternate = read_fasta(alternate_path)
    selected = select(base, alternate, [read_fasta(p) for p in guide_paths], policy)
    fd, name = tempfile.mkstemp(dir=out.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd,"wb") as stream:
            if not selected:
                with base_path.open("rb") as source:
                    shutil.copyfileobj(source,stream)
            else:
                for label,row in base.items():
                    seq = bytearray()
                    start = 0
                    for lo,hi,left,right in selected:
                        seq.extend(row[start:lo])
                        # Keep original residue case even if the alternative uppercased its FASTA.
                        it = iter(row[lo:hi].translate(None,GAPS))
                        for c in alternate[label][left:right]:
                            seq.append(c if c in GAPS else next(it))
                        start = hi
                    seq.extend(row[start:])
                    stream.write(b">" + label + b"\n")
                    for j in range(0,len(seq),80):
                        stream.write(seq[j:j+80] + b"\n")
        os.replace(temp,out)
    finally:
        if temp.exists():
            temp.unlink()
    return len(selected)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,required=True,help="development Super5 FASTA")
    ap.add_argument("--alternative",type=Path,required=True,help="second Super5 FASTA for the same input")
    ap.add_argument("--guide",type=Path,action="append",required=True,help="sequence-only MSA for the same input")
    ap.add_argument("--policy",choices=["local","pair","majority","primary-or-two","corroborate","pair-or-unanimous"],default="local")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    count=reconcile(args.base,args.alternative,args.guide,args.output,args.policy)
    print(f"Accepted {count} disagreement blocks")
