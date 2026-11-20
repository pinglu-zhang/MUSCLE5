"""Coalesce compatible Super5 columns supported by an independently computed MSA."""
import argparse
from collections import Counter, defaultdict
from heapq import heapify, heappop, heappush
from pathlib import Path
import os
import shutil
import tempfile


def read_alignment(path):
    rows = {}
    name = None
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(b">"):
                name = line[1:]
                if not name or name in rows:
                    raise ValueError(f"Empty or duplicate row label in {path}: {name!r}")
                rows[name] = bytearray()
            elif name is None:
                raise ValueError(f"Missing FASTA header in {path}")
            else:
                rows[name].extend(b"".join(line.split()))
    widths = {len(seq) for seq in rows.values()}
    if len(widths) != 1 or 0 in widths:
        raise ValueError(f"Expected a nonempty rectangular alignment in {path}")
    return rows


def residue_positions(seq):
    return [i for i,c in enumerate(seq) if c not in b"-."]


def reachable(edges, start, end):
    pending = [start]
    seen = {start}
    while pending:
        for nxt in edges[pending.pop()]:
            if nxt == end:
                return True
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return False


def refine_columns(base, guide):
    if base.keys() != guide.keys():
        raise ValueError("The two alignments must contain the same sequence identifiers")
    width = len(next(iter(base.values())))
    before = [set() for _ in range(width)]
    after = [set() for _ in range(width)]
    # Counts of residues in each (original column, guide column) intersection.
    votes = [Counter() for _ in range(width)]
    guide_columns = defaultdict(Counter)
    row_bits = [0] * width
    for i, (name, seq) in enumerate(base.items()):
        other = guide[name]
        if seq.translate(None, b"-.").upper() != other.translate(None, b"-.").upper():
            raise ValueError(f"Residues differ for row {name!r}")
        positions = residue_positions(seq)
        for left, right in zip(positions, positions[1:]):
            after[left].add(right)
            before[right].add(left)
        bit = 1 << i
        for old, new in zip(positions, residue_positions(other)):
            votes[old][new] += 1
            guide_columns[new][old] += 1
            row_bits[old] |= bit
    size = [sum(col.values()) for col in votes]
    support = Counter()
    for column in guide_columns.values():
        groups = sorted(column.items())
        for i, (a, na) in enumerate(groups):
            for b, nb in groups[i+1:]:
                if not (row_bits[a] & row_bits[b]):
                    support[a,b] += na * nb
    # A positive gain strictly reduces pair disagreement with the guide.
    ranked = []
    for (a,b), n in support.items():
        gain = 2*n - size[a]*size[b]
        if gain > 0:
            ranked.append((-gain, b-a, a, b))
    parent = list(range(width))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    merges = added = confirmed = 0
    for _, _, first, second in sorted(ranked):
        a, b = sorted((root(first), root(second)))
        if a == b or row_bits[a] & row_bits[b]:
            continue
        n = sum(v * votes[b][k] for k,v in votes[a].items())
        cross = size[a] * size[b]
        if 2*n <= cross or reachable(after, a, b) or reachable(after, b, a):
            continue
        # Contract incomparable nodes in the residue-order graph.
        for pred in before[b]:
            after[pred].remove(b)
            after[pred].add(a)
        for succ in after[b]:
            before[succ].remove(b)
            before[succ].add(a)
        before[a].update(before[b])
        after[a].update(after[b])
        parent[b] = a
        votes[a].update(votes[b])
        size[a] += size[b]
        row_bits[a] |= row_bits[b]
        merges += 1
        added += cross
        confirmed += n
    stats = dict(merges=merges, new_pairs=added, guide_supported_pairs=confirmed)
    if not merges:
        return base, stats
    pending = [i for i in range(width) if parent[i] == i and not before[i]]
    heapify(pending)
    new_column = {}
    while pending:
        node = heappop(pending)
        new_column[node] = len(new_column)
        for nxt in after[node]:
            before[nxt].remove(node)
            if not before[nxt]:
                heappush(pending, nxt)
    if len(new_column) != width-merges:
        raise RuntimeError("Cycle after contracting alignment columns")
    old_to_new = [new_column[root(i)] for i in range(width)]
    result = {}
    for name, seq in base.items():
        row = bytearray(b"-" * len(new_column))
        prev = -1
        for i in residue_positions(seq):
            col = old_to_new[i]
            if col <= prev:
                raise RuntimeError("Inconsistent residue order")
            row[col] = seq[i]
            prev = col
        result[name] = row
    return result, stats


def refine_file(base_path, guide_path, output):
    for source in (base_path, guide_path):
        if output.resolve() == source.resolve() or (output.exists() and output.samefile(source)):
            raise ValueError("Choose an output path distinct from both input alignments")
    base, guide = read_alignment(base_path), read_alignment(guide_path)
    result, stats = refine_columns(base, guide)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        if stats["merges"] == 0:
            shutil.copyfile(base_path, temp)
        else:
            with temp.open("wb") as stream:
                for name, seq in result.items():
                    stream.write(b">" + name + b"\n")
                    for j in range(0,len(seq),80):
                        stream.write(seq[j:j+80] + b"\n")
        os.replace(temp, output)
    finally:
        if temp.exists():
            temp.unlink()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("super5", type=Path)
    parser.add_argument("guide", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(refine_file(args.super5, args.guide, args.output))
