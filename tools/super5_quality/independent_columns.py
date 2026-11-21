"""Merge compatible columns when every new residue pair is supported by an independent MSA."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from refine import fully_ordered, read_alignment, refine_columns, write_alignment


def supported_columns(base_path, output, threads=32, source=None):
    if threads < 1:
        raise ValueError("The thread count must be positive")
    for path in (base_path, source):
        if path and (output.resolve() == path.resolve() or (output.exists() and output.samefile(path))):
            raise ValueError("Choose an output distinct from each input file")
    base = read_alignment(base_path)
    sequences = read_alignment(source, rectangular=False) if source else base
    if source and (base.keys() != sequences.keys() or any(
            base[k].translate(None, b"-.").upper() != row.translate(None, b"-.").upper()
            for k, row in sequences.items())):
        raise ValueError("The input FASTA and alignment must contain the same sequences")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="independent-columns-", dir=output.parent) as tmpdir:
        tmp = Path(tmpdir)
        final = tmp / "final.afa"
        if fully_ordered(base):
            shutil.copyfile(base_path, final)
            stats = dict(merges=0, new_pairs=0, guide_supported_pairs=0, guides_computed=0)
        else:
            # External aligners can abbreviate labels and sort rows. Map by opaque ID.
            names = {f"S{i:06d}".encode(): name for i, name in enumerate(sequences)}
            raw = tmp / "ungapped.fa"
            write_alignment(raw, {key: sequences[name].translate(None, b"-.") for key, name in names.items()})

            def restore(path):
                alignment = read_alignment(path)
                if alignment.keys() != names.keys():
                    raise ValueError("Guide generator returned different sequence identifiers")
                return {name: alignment[key] for key, name in names.items()}

            msa = tmp / "mafft.afa"
            with msa.open("wb") as stream:
                subprocess.run(["mafft", "--amino", "--localpair", "--maxiterate", "1000",
                                "--thread", str(threads), "--threadit", "0", raw], stdout=stream, check=True)
            guide = restore(msa)
            result, stats = refine_columns(base, guide, require_all=True)
            if stats["merges"]:
                write_alignment(final, result)
            else:
                shutil.copyfile(base_path, final)
            stats["guides_computed"] = 1
        os.replace(final, output)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", type=Path, help="ordinary or Super5 alignment")
    source.add_argument("--mode", choices=("align", "super5"), help="compute the starting alignment from --input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="original input FASTA, to use its row order for guide generation")
    parser.add_argument("--muscle", default="muscle", help="CPU executable used when --mode is supplied")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    if args.base:
        stats = supported_columns(args.base, args.output, args.threads, args.input)
    else:
        if not args.input or not args.input.is_file() or args.threads < 1:
            parser.error("--mode requires an input FASTA and a positive thread count")
        if args.output.resolve() == args.input.resolve() or (args.output.exists() and args.output.samefile(args.input)):
            parser.error("choose an output distinct from the input FASTA")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="muscle-quality-", dir=args.output.parent) as folder:
            base = Path(folder) / "base.afa"
            if args.mode == "super5":
                tool = Path(__file__).with_name("super5_quality.py")
                cmd = [sys.executable, tool, "--input", args.input, "--muscle", args.muscle,
                       "--threads", str(args.threads), "--output", base]
            else:
                cmd = [args.muscle, "-align", args.input, "-threads", str(args.threads), "-output", base]
            subprocess.run(cmd, check=True, stdout=sys.stderr)
            stats = supported_columns(base, args.output, args.threads, args.input)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
