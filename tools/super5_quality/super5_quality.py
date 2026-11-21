"""Refine a Super5 alignment using an independently computed, sequence-only guide."""
import argparse
import filecmp
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from reconcile_super5 import reconcile
from refine import fully_ordered, read_alignment, refine_file


def refine_with_guides(base, guide, alternatives, output):
    for src in (base, guide, *alternatives):
        if output.resolve() == src.resolve() or (output.exists() and output.samefile(src)):
            raise ValueError("Choose an output path distinct from all input files")
    blocks = 0
    with TemporaryDirectory(prefix="pair-check-", dir=output.parent) as tmpdir:
        for index, alternate in enumerate(alternatives):
            if filecmp.cmp(base, alternate, shallow=False):
                continue
            interim = Path(tmpdir) / f"stage-{index}.afa"
            accepted = reconcile(base, alternate, [guide], interim, "pair")
            blocks += accepted
            if accepted:
                base = interim
        final = Path(tmpdir) / "refined.afa"
        stats = refine_file(base, guide, final)
        os.replace(final, output)
    return dict(accepted_blocks=blocks, **stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="unaligned input FASTA")
    source.add_argument("--base", type=Path, help="previously computed Super5 alignment")
    parser.add_argument("--guide", type=Path, help="ordinary -align output for the same input, used with --base")
    parser.add_argument("--muscle", default="muscle", help="development CPU executable used with --input")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--policy", choices=("preserve", "pair"), default="preserve",
                        help="keep existing pairs or also compare whole disagreement blocks")
    parser.add_argument("--alternative", type=Path, action="append", default=[],
                        help="another computed Super5 MSA; repeat to apply candidates in order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.base) != bool(args.guide):
        parser.error("supply --base together with --guide, or use --input")
    if (args.policy == "pair") != bool(args.alternative):
        parser.error("pair requires --alternative; preserve uses only the guide")
    if args.threads < 1:
        parser.error("--threads must be positive")
    for src in (args.input, args.base, args.guide, *args.alternative):
        if src is not None:
            if not src.is_file():
                parser.error(f"missing input file: {src}")
            if args.output.resolve() == src.resolve() or (args.output.exists() and args.output.samefile(src)):
                parser.error("choose an output path distinct from all input files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="super5-quality-", dir=args.output.parent) as tmpdir:
        tmp = Path(tmpdir)
        base, guide = args.base, args.guide
        if args.input:
            base, guide = tmp / "super5.afa", tmp / "align.afa"
            def align(mode, outfile):
                subprocess.run([args.muscle, mode, str(args.input), "-threads", str(args.threads),
                                "-output", str(outfile)], check=True)
            align("-super5", base)
            if (all(filecmp.cmp(base, alt, shallow=False) for alt in args.alternative)
                    and fully_ordered(read_alignment(base))):
                # A chain of mandatory residue-order edges forbids every merge.
                # Byte-identical alternative files also forbid any replacement.
                os.replace(base, args.output)
                stats = dict(accepted_blocks=0, merges=0, new_pairs=0, guide_supported_pairs=0)
                print(json.dumps(dict(policy=args.policy, **stats)))
                return
            align("-align", guide)
        stats = refine_with_guides(base, guide, args.alternative, args.output)
    print(json.dumps(dict(policy=args.policy, **stats)))


if __name__ == "__main__":
    main()
