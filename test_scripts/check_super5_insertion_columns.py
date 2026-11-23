"""Run a small Super5 alignment with 0, 1, and 3 inserts at one backbone boundary."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def read_fasta(path):
    rows = {}
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
            if name in rows:
                raise ValueError(f"Duplicate output label {name}")
            rows[name] = ""
        elif line:
            rows[name] += line
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    left = "AIVDRYQLTGFNSPKMEWHCVQ"
    right = "DYIRKAGFETLMNPSQ"
    expected = {
        "longest": "GVKFWHE" + left + "---" + right,
        "no_insert": "-------" + left + "---" + right,
        "one_insert": "-------" + left + "P--" + right,
        "three_inserts": "-------" + left + "PWY" + right,
    }
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        source = folder / "input.fa"
        output = folder / "out.afa"
        source.write_text("".join(f">{k}\n{v.replace('-', '')}\n" for k, v in expected.items()))
        # A single center makes the projection independent of clustering cutoffs.
        subprocess.run([str(args.binary.resolve()), "-super5", str(source),
                        "-super5_minea1", "0.8", "-threads", "1", "-output", str(output)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        observed = read_fasta(output)
        if observed != expected:
            failures = [k for k in expected if observed.get(k) != expected[k]]
            raise SystemExit(f"Insertion columns differ for {failures}; observed {observed}")
    print("Super5 preserves all four backbone rows at the insertion boundary")


if __name__ == "__main__":
    main()
