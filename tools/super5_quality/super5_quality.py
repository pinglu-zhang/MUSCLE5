"""Refine a Super5 alignment using an independently computed, sequence-only guide."""
import argparse
import filecmp
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from reconcile_super5 import reconcile
from refine import refine_file


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
            blocks += reconcile(base, alternate, [guide], interim, "pair")
            base = interim
        stats = refine_file(base, guide, output)
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
                        help="preserve keeps existing pairs; pair compares a second run using every sequence pair")
    parser.add_argument("--alternative", type=Path, action="append", default=[],
                        help="another computed Super5 MSA; repeat to apply candidates in order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.base) != bool(args.guide):
        parser.error("supply --base together with --guide, or use --input")
    if (args.policy == "pair") != bool(args.alternative):
        parser.error("--alternative is required for pair mode and only applies to that mode")
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
            for mode, outfile in (("-super5", base), ("-align", guide)):
                subprocess.run([args.muscle, mode, str(args.input), "-threads", str(args.threads),
                                "-output", str(outfile)], check=True)
        stats = refine_with_guides(base, guide, args.alternative, args.output)
    print(json.dumps(dict(policy=args.policy, **stats)))


if __name__ == "__main__":
    main()
