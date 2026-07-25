<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".assets/logo_dark.svg">
    <img src=".assets/logo.svg" width="640" alt="FaST-ER-LMM">
  </picture>
</p>

# FaST-ER-LMM

A PyTorch port of [FaST-LMM](https://github.com/fastlmm/FaST-LMM) for genome-wide association scans on CPUs and GPUs. Scan many phenotypes, estimate significance thresholds with permutations, and split work across multiple GPUs.

Based on [Lippert et al. (2011), Nature Methods](https://doi.org/10.1038/nmeth.1681).

[Code map](docs/HOW_IT_WORKS.md)

<p align="center">
  <img src=".assets/gif_truth_1g_vs_2g.gif" width="780" alt="Live progress dashboard for scans on one and two GPUs">
</p>

## Benchmark

In the v1.2.0 benchmark, a simulated dataset with 1,000 samples, 100,000 variants, and 6,484 phenotypes took **9.6 minutes on one NVIDIA V100S (32 GB)** and **5.2 minutes on two**.

## Install

Requires Python 3.10 or newer. From a clone of this repository:

```bash
mamba create -n fasterlmm python=3.11 -y
mamba activate fasterlmm
pip install -e .
```

## Quick start

Run the bundled yeast example (150 strains, 1,500 variants, 20 phenotypes) on CPU:

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --covar data/example/example_covar.tab \
  --outdir runs/example/ \
  --bundle --device cpu
```

By default, each chromosome is tested using relatedness estimated from the other chromosomes (LOCO). Phenotype values are transformed using their ranks to follow a normal distribution, and each phenotype gets 100 permutations. Use `--no-loco`, `--no-rint`, or `--n-perm` to change these settings.

Watch progress from another terminal:

```bash
fasterlmm watch runs/example/
```

## Inputs

| Flag | Format |
|------|--------|
| `--geno` | PLINK `.bed` / `.bim` / `.fam` prefix, without an extension |
| `--pheno` | TSV with header `Strain<TAB>pheno1<TAB>pheno2...`; strain IDs match PLINK IIDs |
| `--covar` | Optional whitespace-delimited file: `FID IID c1 c2...`, without a header |
| `--outdir` | Output directory, created if needed |

All phenotype columns are scanned by default. Use `--pheno-idx I` to select one (zero-based), or `--pheno-start S --pheno-end E` for a range with an exclusive end.

## GPUs and distributed runs

Replace `--device cpu` in the example with:

- `--device cuda` for NVIDIA GPUs. Multiple visible GPUs automatically split the phenotypes across workers; add `--no-multi-gpu` to use only the first.
- `--device mps` for Apple Silicon. This uses float32, with CPU fallbacks for some linear algebra operations.

For Slurm arrays or multiple nodes, give each task a zero-based `--shard X/N` and the same output directory. For example, in an eight-task Slurm array:

```bash
fasterlmm gwas \
  --geno data/yeast --pheno phen.tsv --covar aneuploidies.cov \
  --outdir runs/all/ --device cuda --bundle \
  --shard "${SLURM_ARRAY_TASK_ID}/8"
```

After all tasks finish, gather their bundles with `fasterlmm concat runs/all/`. A single-job multi-GPU run gathers its bundles automatically.

## Large datasets

`extreme` estimates relatedness from a subset of variants and can read the test variants in blocks to reduce memory use:

```bash
fasterlmm extreme \
  --geno data/bigN --pheno phen.tsv \
  --outdir runs/bigN/ \
  --grm-k 5000 --block-size 8192 \
  --resident off --device cuda --bundle
```

`--grm-k` sets the target number of variants for estimating relatedness; `--grm PREFIX` supplies your own PLINK subset instead. `--block-size` sets how many test variants to read at a time. By default, genotypes stay in memory if they fit; `--resident off` always reads them in blocks. This command always uses LOCO and defaults to float32; add `--float64` for higher numerical precision.

## Outputs

Each phenotype gets a folder under the output directory containing:

- `gwas.tsv`: per-variant association results in the FaST-LMM `single_snp` column format, sorted by p-value.
- `perms.tsv`: the minimum genome-wide p-value for each permutation.
- `threshold.txt`: the permutation significance threshold (5th percentile by default; set with `--perm-quantile`).

Progress files (`status.json` or `status.shard*.json`) live at the output directory root and feed `fasterlmm watch`.

With `--bundle`, results also go into `gwas_bundle.parquet`, a **directory of Parquet parts**, with additional `threshold` and `significant` columns. Add `--no-per-pheno-dirs` to keep only the bundle and progress files.

```python
import pandas as pd

df = pd.read_parquet("runs/example/gwas_bundle.parquet")
```

In Snakemake, declare the bundle with `directory("runs/example/gwas_bundle.parquet")`.

## Reference and tests

See `fasterlmm gwas --help` and `fasterlmm extreme --help` for all options.

```bash
pip install -e ".[test]"
pytest
```

The tests run on CPU; checks requiring optional dependencies, GPUs, or external data are skipped when unavailable.
