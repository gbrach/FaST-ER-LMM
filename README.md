# FaST-ER-LMM

PyTorch port of FaST-LMM ([Lippert et al. 2011, Nat. Methods](https://doi.org/10.1038/nmeth.1681), [github](https://github.com/fastlmm/FaST-LMM)).

## How to install install the environment?
Fresh mamba/conda environment recommended!

```bash
mamba create -n fasterlmm python=3.11 -y
mamba activate fasterlmm
pip install -e .
```

Dependencies: pytorch, pysnptools, scipy, pandas, numpy, tqdm, rich, pyarrow

## Try it on the shipped example

A tiny yeast subset sits under `data/example/`: 150 strains × 1500 SNPs (16 chromosomes) + 5 nuclear glucose phenos (YAL001C / YBR001C / YGR001C / YLR001C / YPR001W), wide TSV. Total ~136 KB.

End-to-end smoke run on CPU in a few seconds:

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --outdir runs/example/ \
  --pheno-idx 0 --n-perm 20 --device cpu
```

## Quick run

Single pheno, 100 perms:

```bash
fasterlmm gwas \
  --geno path/to/plink_prefix \
  --pheno path/to/wide_phen.tsv \
  --outdir results/ \
  --pheno-idx 0 \
  --n-perm 100
```

Pheno TSV format: first column `Strain` (matching plink IIDs), the rest are pheno value columns

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

Example of a run on 2 V100S, one shard per GPU, with the final bundle step at the end.
`fasterlmm gwas` is single-process per call (one GPU per call), so for 2 GPUs we ask for both in one allocation and launch two background processes that share it:

```bash
mamba activate fasterlmm

srun -p gpu --gres=gpu:tesla:2 -c 16 --mem=64G -t 1:00:00 bash -c '
  fasterlmm gwas --geno data/yeast --pheno phen.tsv --outdir runs/all/ --shard 0/2 --device cuda:0 &
  fasterlmm gwas --geno data/yeast --pheno phen.tsv --outdir runs/all/ --shard 1/2 --device cuda:1 &
  wait
  # bundle once at the end so the two shards dont race on the parquet
  python -c "from fasterlmm.bundle import bundle_outdir; bundle_outdir(\"runs/all/\")"
'
```

## Watch a live run (it's fun!)

```bash
fasterlmm watch results/.../status.json
```
TUI that polls every 0.5s, shows the current state, pheno, shape, perm count, threshold, n_signif!

## CLI options

`fasterlmm gwas`:

```
--geno            required          PLINK BED prefix (.bed/.bim/.fam)
--pheno           required          Wide phen TSV, first col Strain
--covar           none              PLINK-style .cov, optional
--outdir          required          Putput directory, created if missing
--pheno-idx       none              0-based single pheno column, to run a super small test
--pheno-start     0                 0-based start of a pheno range (INCLUSIVE)
--pheno-end       all phenos        0-based end of the range (EXCLUSIVE)
--n-perm          100               Permutations count for the threshold determination
--seed            19930909          RNG seed for permutations
--device          cuda              cuda, cuda:N, or cpu
--shard           none              X/N round-robin slice across the pheno list, one shard per GPU
--bundle          off               Concatenating per-phenotype results gwas.tsv into one parquet at outdir root
```

if neither `--pheno-idx` nor `--pheno-start/--pheno-end` is set, every pheno in the TSV is scanned.

`fasterlmm watch`:

```
status_file     required (pos.)   path to the status.json to tail
--poll-sec      0.5               seconds between polls
```

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

## TODO

- [ ] benchmarks!!
- [ ] `gwas-gxe` (GxE / single-K interaction scan)
- [ ] `gwas-epi` (tier-2 pairwise epistasis, anchor × all-SNPs)
- [ ] multi-cluster epi-hub orchestration: daemon + per-cluster workers
- [ ] richer `fasterlmm watch` panes for the multi-shard case (gxe-watch / epi-watch parity)
