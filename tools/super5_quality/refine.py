"""Coalesce compatible Super5 columns supported by an independently computed MSA."""
import argparse
from collections import Counter, defaultdict
from heapq import heapify, heappop, heappush
from pathlib import Path
import os
import shutil
import tempfile


def read_alignment(path, *, rectangular=True):
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
    if not widths or 0 in widths or (rectangular and len(widths) != 1):
        raise ValueError(f"Invalid FASTA row lengths in {path}")
    return rows


def residue_positions(seq):
    return [i for i,c in enumerate(seq) if c not in b"-."]


OCCUPIED = bytes(0 if c in b"-." else 1 for c in range(256))


def fully_ordered(rows):
    """Every adjacent pair of columns is ordered by two residues in some row."""
    width = len(next(iter(rows.values())))
    needed = int.from_bytes(b"\x01" * (width - 1), "little")
    observed = 0
    for row in rows.values():
        sites = int.from_bytes(row.translate(OCCUPIED), "little")
        observed |= sites & (sites >> 8)
        if observed == needed:
            return True
    return False


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


def refine_columns(base, guide, *, other_guides=(), require_all=False):
    guides = (guide, *other_guides)
    require_all = require_all or bool(other_guides)
    if any(base.keys() != rows.keys() for rows in guides):
        raise ValueError("All alignments must contain the same sequence identifiers")
    if fully_ordered(base):
        for name, seq in base.items():
            raw = seq.translate(None, b"-.").upper()
            if any(raw != rows[name].translate(None, b"-.").upper() for rows in guides):
                raise ValueError(f"Residues differ for row {name!r}")
        return base, dict(merges=0, new_pairs=0, guide_supported_pairs=0)
    width = len(next(iter(base.values())))
    before = [set() for _ in range(width)]
    after = [set() for _ in range(width)]
    # Counts of residues in each (original column, guide column) intersection.
    votes = [Counter() for _ in range(width)]
    guide_columns = defaultdict(Counter)
    row_bits = [0] * width
    for i, (name, seq) in enumerate(base.items()):
        raw = seq.translate(None, b"-.").upper()
        if any(raw != rows[name].translate(None, b"-.").upper() for rows in guides):
            raise ValueError(f"Residues differ for row {name!r}")
        positions = residue_positions(seq)
        for left, right in zip(positions, positions[1:]):
            after[left].add(right)
            before[right].add(left)
        bit = 1 << i
        # Two residues share a vote only if all supplied alignments put them together.
        sites = (zip(*(residue_positions(rows[name]) for rows in guides))
                 if other_guides else residue_positions(guide[name]))
        for old, new in zip(positions, sites):
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
        if gain > 0 and (not require_all or n == size[a]*size[b]):
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
        if 2*n <= cross or (require_all and n != cross):
            continue
        if reachable(after, a, b) or reachable(after, b, a):
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


def write_alignment(path, rows):
    with path.open("wb") as stream:
        for name, seq in rows.items():
            stream.write(b">" + name + b"\n")
            for j in range(0, len(seq), 80):
                stream.write(seq[j:j+80] + b"\n")


def refine_file(base_path, guide_path, output, *, other_guides=(), require_all=False):
    for source in (base_path, guide_path, *other_guides):
        if output.resolve() == source.resolve() or (output.exists() and output.samefile(source)):
            raise ValueError("Choose an output path distinct from both input alignments")
    base, guide = read_alignment(base_path), read_alignment(guide_path)
    result, stats = refine_columns(base, guide, other_guides=[read_alignment(p) for p in other_guides],
                                   require_all=require_all)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        if stats["merges"] == 0:
            shutil.copyfile(base_path, temp)
        else:
            write_alignment(temp, result)
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
    parser.add_argument("--unanimous", action="store_true", help="require support for every new residue pair")
    parser.add_argument("--witness", action="append", default=[], type=Path,
                        help="additional guide; require every new pair to agree in all guides")
    args = parser.parse_args()
    print(refine_file(args.super5, args.guide, args.output, other_guides=args.witness, require_all=args.unanimous))
