"""Check two byte-level regressions against the recorded Linux x86-64 v5.3 release."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


# (case, mode, input FASTA SHA-256, saved upstream output FASTA SHA-256)
CASES = (
    ("twi_111", "align",
     "9afa98e06cc1400566ebe9adfdb813ba6690fbb7066e6068c02a143893e05a57",
     "4f0b3efb24493b008266dc08e9ec4a6664899873a7b736a3500eb3dbd6d8637f"),
    ("sup_383", "super5",
     "8d181af85d56a9dc771af38312b1b3247cb0948afe807fe3cffda13396a979e0",
     "695fa8394578a93eec691dc718b74ee53a4c747a89fa206659e26a8839ed7f44"),
)


def check_hash(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"{path.name}: expected {expected}, got {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="path to the development MUSCLE binary")
    parser.add_argument("inputs", type=Path, help="SABRE/consistent_msa input FASTA directory")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        for case, mode, input_sha, output_sha in CASES:
            source = (args.inputs / (case + ".fa")).resolve()
            check_hash(source, input_sha)
            output = Path(tmp) / (case + "." + mode + ".afa")
            proc = subprocess.run([str(args.binary.resolve()), "-" + mode, str(source),
                                   "-threads", "32", "-output", str(output)],
                                  cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise SystemExit(proc.stdout.decode(errors="replace"))
            check_hash(output, output_sha)
            print(f"{case} -{mode}: matches saved upstream FASTA")


if __name__ == "__main__":
    main()
