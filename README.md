# MUSCLE5 — CPU-optimized fork

**This project continues development of MUSCLE5, the multiple sequence alignment software by Robert C. Edgar.** Our work focuses on CPU performance for protein sequence alignment. [Original MUSCLE5 repository](https://github.com/rcedgar/muscle) · [Upstream commit used as our starting point](https://github.com/rcedgar/muscle/commit/29aa0671d0e46c862457749c7f2d87f29007b8eb)

## What this fork changes

This fork uses SIMD to accelerate pair-HMM calculations, reuses data in the sparse probability consistency step, and parallelizes parts of iterative refinement and the Super5 cluster-center search.

## Align protein sequences

Build this fork from source to use the CPU optimizations. [MUSCLE build instructions](https://github.com/rcedgar/muscle/wiki/Building-MUSCLE)

Use `-align` for an input FASTA file containing unaligned protein sequences.

```bash
muscle -align proteins.fa -output aligned.afa
```

For a large protein family, use Super5. This example uses 24 CPU threads.

```bash
muscle -super5 proteins.fa -threads 24 -output aligned.afa
```

MUSCLE5 can also generate ensembles of alternative alignments to assess how alignment variation affects downstream analyses. [MUSCLE5 manual](https://drive5.com/muscle5/manual)

## Optional Super5 refinement

The Python tool can merge compatible Super5 columns using a second alignment of the same sequences as a guide. Its default policy keeps every existing residue pair. Run it with a Super5 output and an ordinary `-align` output.

```bash
python3 tools/super5_quality/super5_quality.py --base super5.afa --guide align.afa --output refined.afa
```

The `--policy pair` option also evaluates disagreement blocks in additional Super5 runs. Pass one `--alternative file.afa` per run. Each affected sequence pair must retain or increase its agreement with the guide. This check uses sequence alignments only. Reference Q and TC can be measured separately on a benchmark.

## Reference

Edgar, R. C. (2022). Muscle5: High-accuracy alignment ensembles enable unbiased assessments of sequence homology and phylogeny. *Nature Communications*, **13**, 6968. [Paper](https://doi.org/10.1038/s41467-022-34630-w)
