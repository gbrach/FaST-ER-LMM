# LUX — LUdicrously eXtra

LUX adds research scans on top of FaST-ER-LMM. It ships with the normal installation, while keeping its own `fasterlmm_lux` namespace and commands. LUX imports the core; the core never imports LUX.

This package currently contains **pairwise epistasis scanning**: select anchor SNPs from a marginal GWAS, then scan each anchor against the other variants. GxE and cross-cluster orchestration are not included.

## Install

From the FaST-ER-LMM repository root, in your Python environment:

```bash
pip install -e .
```

This single installation provides `fasterlmm`, `gwas-epi`, and `epi-watch`. The separate `fasterlmm-lux` distribution from the original sibling repository also provides the `fasterlmm_lux` namespace; use a fresh environment for this bundled version if that distribution is already installed.

## From marginal GWAS to pairs

First run the standard scan. Keep the individual phenotype folders: LUX reads each phenotype's `gwas.tsv` and optional `threshold.txt`.

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --covar data/example/example_covar.tab \
  --outdir runs/marginals/ \
  --device cpu
```

Then use its top marginal SNPs as pairwise anchors:

```bash
gwas-epi \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --covar data/example/example_covar.tab \
  --marginal-dir runs/marginals/ \
  --marginal-top-k 5 --top-k 100 \
  --rint --n-perm 100 \
  --outdir runs/pairs/ \
  --device cpu --bundle
```

Use the same input files and phenotype transformation in both stages. Core GWAS applies RINT by default; LUX requires `--rint` explicitly. The small anchor count above is a starting example; increase `--marginal-top-k` to scan more anchors.

For NVIDIA GPUs, use `--device cuda`; multiple visible GPUs split phenotypes across workers. `--no-multi-gpu` keeps the scan on one GPU. This pairwise command supports CPU and CUDA, not MPS.

Follow the scan from another terminal:

```bash
epi-watch runs/pairs/.status.json
```

## What the scan does

For each phenotype, LUX selects the top marginal SNPs and tests their standardized genotype products with the other SNPs. The fixed effects contain the supplied covariates, an intercept, and the anchor's main effect. Kinship excludes both SNPs' chromosomes (leave-double-chromosome-out, LDCO). Input genotypes must cover at least three chromosomes.

This is the existing Lux anchor-conditioned scan. Its numerical behavior is preserved here; it does not add both SNP main effects to each pair's fixed-effect model. Treat the resulting pairs as candidates for follow-up under the model appropriate to your analysis.

| Option | Default / behavior |
|---|---|
| `--marginal-top-k` | 100 anchor SNPs per phenotype, selected before the MAF filter. |
| `--top-k` | Retain the best 1,000 pairs per phenotype. |
| `--maf-floor` | Exclude SNPs with MAF below 0.05 from anchors, tests, and kinship. |
| `--exclude-window-kb` | Exclude pairs within 25 kb on the same chromosome. |
| `--n-perm` | 0: no permutation threshold unless explicitly requested. |
| `--perm-quantile` | 0.05: quantile of permutation minimum pair p-values. |
| `--dtype` | `float32`; use `float64` for higher precision. |
| `--anchor-batch`, `--snp-chunk`, `--pheno-chunk` | Control scan tile sizes and memory use. |
| `--dry-run` | Load inputs and report the planned scan without fitting pairs. |

`--all-anchors` skips the marginal-GWAS requirement and visits each unordered pair once. Run this mode on one device. `--p-threshold P` retains every tested pair below P instead of only the best K; its result buffer can grow large. This retention cutoff is separate from the permutation significance threshold.

Permutations shuffle phenotype values while retaining the selected anchors; they do not repeat marginal anchor selection. The inherited scan tracks permutation maxima across the union of anchors for the phenotypes in that worker, so thresholds can depend on which phenotypes are grouped together.

## Results

`--bundle` writes **one file**, `all_pairs.parquet`, with the retained pairs from every phenotype. It includes `SNP_i`, `SNP_j`, their chromosome positions, `PValue`, `SnpWeight`, `SnpWeightSE`, marginal anchor statistics, `threshold`, and `signif`.

```python
import pandas as pd

pairs = pd.read_parquet("runs/pairs/all_pairs.parquet")
hits = pairs.loc[pairs["signif"]]
```

Without permutations, `threshold` is NaN and `signif` is false; that means significance was not assessed. The bundle contains retained pairs, not every pair tested.

By default, each phenotype also gets `<phenotype>.tier2.first_assoc.txt.gz`. Add `--output-format parquet` for per-phenotype Parquet files, or `--bundle --no-per-pheno-dirs` for only the combined bundle and progress files. When permutations are enabled, phenotype folders also contain threshold and significant-pair files.

## Package boundary

```text
FaST-ER-LMM core                 LUX
src/fasterlmm/          <──────   lux/src/fasterlmm_lux/
  LMM fitting, input loading      pairwise scan, pair kernel
  gwas, extreme, watch            pair output, pair dashboard
```

Both namespaces are installed together, but the dependency goes one way: LUX imports the core. Pairwise-specific kernels and progress helpers live here. The existing `fasterlmm extreme` command stays in the core for compatibility; moving or exposing it through LUX is a separate change. GxE can follow this same code boundary later.

Run `gwas-epi --help` for all options. Implementation provenance and verification notes are in [DEVELOPMENT.md](DEVELOPMENT.md).
