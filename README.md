# FaST-ER-LMM

PyTorch port of FaST-LMM (Lippert et al. 2011, Nat. Methods).

## How to install install the environment?

```bash
pip install -e .
```

deps: torch, pysnptools, scipy, pandas, numpy, tqdm, rich, pyarrow

## Quick run

single pheno, 100 perms:

```bash
fasterlmm gwas \
  --geno path/to/plink_prefix \
  --pheno path/to/wide_phen.tsv \
  --outdir results/ \
  --pheno-idx 0 \
  --n-perm 100
```

Pheno tsv format: first column `Strain` (matching plink IIDs), the rest are pheno value columns

Multi-pheno range, optional covariates, parquet output:

```bash
fasterlmm gwas \
  --geno data/yeast \
  --pheno phen.tsv \
  --covar aneuploidies.cov \
  --outdir runs/all/ \
  --pheno-start 0 --pheno-end 200 \
  --bundle
```

Multi-GPU sharding, slurm-style (one shard per GPU):

```bash
fasterlmm gwas ... --shard 0/4 --device cuda:0 &
fasterlmm gwas ... --shard 1/4 --device cuda:1 &
fasterlmm gwas ... --shard 2/4 --device cuda:2 &
fasterlmm gwas ... --shard 3/4 --device cuda:3 &
wait
```

## Watch a live run (it's fun!)

```bash
fasterlmm watch results/.../status.json
```
TUI that polls every 0.5s, shows the current state, pheno, shape, perm count, threshold, n_signif!

## Outputs

Per pheno under `outdir/<pheno_name>/`:

- `gwas.tsv` -> SNP / Chr / Pos / F / PValue
- `perms.tsv` -> per-permutation min p across the genome
- `threshold.txt` -> 5% quantile of perm min p (the genome-wide significance threshold)
- `status.json` -> live progress for `fasterlmm watch`

with `--bundle`, an extra `gwas_bundle.parquet` at the outdir root concats every per-pheno gwas.tsv plus a `pheno` column.

## How FaSTLMM does things:

Mostly so I don't get lost as I port pieces over

`single_snp()` (single_snp.py) loads geno/pheno, builds `K = Z Zᵀ / M`,
then for each chromosome-LOCO calls `LMM` (lmm.py):

  - `eig(K)` once
  - `findH2` -> golden-section over h2, each step calls `nLLeval` (log-likelihood)
  - per-SNP Wald after rotating Y, X, S by Uᵀ

The port keeps `single_snp`'s outer shape and swaps the `LMM` internals out for torch -> the eigendecomp, the rotations, and the per-SNP scan are the parts that move to GPU
