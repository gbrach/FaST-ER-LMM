<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".assets/logo_dark.svg">
    <img src=".assets/logo.svg" width="640" alt="FaST-ER-LMM">
  </picture>
</p>

# FaST-ER-LMM

**Linear mixed-model GWAS for thousands of phenotypes, on CPUs and GPUs.**

FaST-ER-LMM is a PyTorch port of [FaST-LMM](https://github.com/fastlmm/FaST-LMM). Give it PLINK genotypes, a table of phenotypes, and optional covariates. It runs an association scan for each phenotype, accounts for relatedness, and estimates genome-wide significance thresholds with permutations.

- **Many phenotypes in one run.** Batch traits and their permutations together; distribute phenotypes across available NVIDIA GPUs automatically.
- **LOCO by default.** Leave-one-chromosome-out scans estimate relatedness from the other chromosomes.
- **A path for larger cohorts.** The `extreme` command uses low-rank kinship and optional genotype streaming to reduce memory use.
- **Results ready to use.** Per-phenotype TSVs, a combined Parquet dataset, and a live terminal dashboard.

[Quick start](#quick-start) · [Input formats](#input-formats) · [Results](#results) · [GPUs and clusters](#gpus-and-clusters) · [Large datasets](#large-datasets) · [LUX](#lux-research-extensions)

<p align="center">
  <img src=".assets/gif_truth_1g_vs_2g.gif" width="780" alt="Live progress dashboard for scans on one and two GPUs">
</p>

## Performance

The recorded v1.2.0 benchmark for **1,000 samples × 100,000 variants × 6,484 phenotypes** took:

| Hardware | Wall time |
|---|---:|
| 1 × NVIDIA V100S (32 GB) | 9.6 minutes |
| 2 × NVIDIA V100S (32 GB) | 5.2 minutes |

These are single-run timings recorded in [the figure script](generate_figures.py), not averages across repeated runs. Runtime depends on dataset size, permutation count, hardware, and output settings.

## Install

Requires **Python 3.10+**. Install directly from GitHub inside a mamba environment:

```bash
mamba create -n fasterlmm python=3.11 pip git -y
mamba activate fasterlmm
python -m pip install git+https://github.com/gbrach/FaST-ER-LMM.git
```

Mamba creates the environment; pip installs FaST-ER-LMM from GitHub. No Conda channel package is needed. This installs the core `fasterlmm` command and LUX's pairwise commands, `gwas-epi` and `epi-watch`, together.

For an editable installation and the bundled example data, clone the repository. You can use the mamba environment above or create a Python virtual environment:

```bash
git clone https://github.com/gbrach/FaST-ER-LMM.git
cd FaST-ER-LMM
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For NVIDIA GPUs, your PyTorch installation must support CUDA. CPU runs work with `--device cpu`; Apple Silicon runs use `--device mps`.

## Quick start

From the repository checkout above, run the bundled yeast example: **150 strains, 1,500 variants, and 20 phenotypes**. The example files require a checkout, even if you installed directly from GitHub.

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --covar data/example/example_covar.tab \
  --outdir runs/example/ \
  --device cpu \
  --bundle
```

This scans every phenotype with LOCO, applies a rank-based inverse normal transform (RINT), and runs 100 permutations per phenotype. Results are written to `runs/example/`.

While the scan runs, open a second terminal in the same environment to follow progress:

```bash
fasterlmm watch runs/example/
```

To run your own data, replace the three input paths and choose an output directory. Omit `--covar` if you have no covariates, or change `--device cpu` to `--device cuda` for NVIDIA GPUs.

## Input formats

| Input | Expected format |
|---|---|
| `--geno` | PLINK `.bed`, `.bim`, and `.fam` files; pass their shared prefix without an extension. |
| `--pheno` | Tab-separated table with a `Strain` column followed by numeric phenotype columns. IDs match PLINK IIDs. |
| `--covar` | Optional whitespace-delimited table with `FID IID c1 c2 ...`, **without a header**. |
| `--outdir` | Output directory, created if needed. |

For example, a phenotype table with two traits looks like this (columns are separated by tabs):

```text
Strain	growth_rate	gene_expression
sample_01	0.82	12.4
sample_02	0.95	10.8
sample_03	0.71	15.1
```

All phenotype columns are scanned by default. Select one with `--pheno-idx 0`, or a range with `--pheno-start 0 --pheno-end 20`. Indices are zero-based; the end is exclusive.

### Common options

| Option | What it changes |
|---|---|
| `--n-perm 1000` | Run 1,000 permutations per phenotype; default: 100. |
| `--perm-quantile 0.05` | Set the quantile of permutation minimum p-values used as the significance threshold; default: 0.05. |
| `--no-rint` | Use phenotype values without the default rank-based inverse normal transform. |
| `--no-loco` | Use a shared relatedness model across chromosomes (`gwas` only). |
| `--phenos-per-job 32` | Set how many phenotypes are processed per batch; lower this to reduce batch memory use. |
| `--bundle --no-per-pheno-dirs` | Write a combined Parquet dataset without individual phenotype folders. |
| `--dry-run` | Load inputs and print the planned work before scanning (`gwas` only). |

## Results

With `--bundle`, an output directory contains:

```text
runs/example/
├── <phenotype>/
│   ├── gwas.tsv
│   ├── perms.tsv
│   └── threshold.txt
├── gwas_bundle.parquet/
└── status.json
```

Each phenotype gets:

| File | Contents |
|---|---|
| `gwas.tsv` | Per-variant results sorted by p-value, using the FaST-LMM `single_snp` column format. |
| `perms.tsv` | The minimum genome-wide p-value from each permutation. |
| `threshold.txt` | The significance threshold: the 5th percentile of permutation minimum p-values by default. |

The combined bundle adds `threshold` and `significant` columns. A variant is marked significant when its p-value is below its phenotype's threshold.

Read the bundle and select hits in Python:

```python
import pandas as pd

results = pd.read_parquet("runs/example/gwas_bundle.parquet")
hits = results.loc[results["significant"]]
print(hits[["Pheno", "SNP", "Chr", "ChrPos", "PValue", "threshold"]])
```

Despite its suffix, `gwas_bundle.parquet` is a **directory of Parquet parts**. In Snakemake, declare it with `directory("runs/example/gwas_bundle.parquet")`.

Progress is stored in `status.json`, or `status.shard*.json` for sharded runs, and displayed by `fasterlmm watch`.

## GPUs and clusters

Choose a device with `--device`:

| Device | Behavior |
|---|---|
| `cpu` | Run on CPU. |
| `cuda` | Use visible NVIDIA GPUs, automatically splitting phenotypes across multiple GPUs. |
| `cuda:0` | Use a specific NVIDIA GPU. |
| `mps` | Use Apple Silicon with float32 and CPU fallbacks for some linear algebra operations. |

The default is `cuda`. Add `--no-multi-gpu` to use only the first GPU when selecting `cuda`. A single-job multi-GPU run gathers its result bundles automatically.

For Slurm arrays or multiple nodes, assign each job a zero-based `--shard X/N` and use the same output directory. For an eight-task Slurm array with task IDs **0–7**:

```bash
fasterlmm gwas \
  --geno data/yeast \
  --pheno phen.tsv \
  --covar aneuploidies.cov \
  --outdir runs/all/ \
  --device cuda --bundle \
  --shard "${SLURM_ARRAY_TASK_ID}/8"
```

Once every task has finished, gather the bundles:

```bash
fasterlmm concat runs/all/
```

## Large datasets

Use `extreme` when the standard scan's memory requirements become limiting. It estimates relatedness from a subset of variants and can stream the test variants from disk in blocks.

```bash
fasterlmm extreme \
  --geno data/bigN \
  --pheno phen.tsv \
  --outdir runs/bigN/ \
  --grm-k 5000 \
  --block-size 8192 \
  --resident off \
  --device cuda --bundle
```

| Option | Purpose |
|---|---|
| `--grm-k 5000` | Target 5,000 variants for estimating relatedness, selected by striding through the input. |
| `--grm PREFIX` | Supply your own PLINK kinship-marker subset instead of automatic selection. |
| `--block-size 8192` | Read 8,192 test variants per block. |
| `--resident off` | Always stream genotypes; the default, `auto`, keeps them in memory when they fit. |
| `--float64` | Use float64 instead of the default float32. |

`extreme` always uses LOCO and supports the same phenotype selection, permutation, bundle, and sharding options as `gwas`. Using fewer kinship markers changes the relatedness estimate; it does not reduce the set of variants tested for association.

## LUX: research extensions

**LUX — LUdicrously eXtra** builds on the FaST-ER-LMM core through a separate Python namespace in [`lux/`](lux/README.md). Both are included in the same install. The core stays focused on the FaST-LMM reimplementation; LUX has its own code and commands, and the core never imports it.

The first integration is the **pairwise epistasis scan** from [fasterlmm-lux](https://github.com/gbrach/fasterlmm-lux): top marginal SNPs × other SNPs, with leave-double-chromosome-out kinship and optional permutation thresholds.

```bash
# Available after the normal installation:
gwas-epi --help
```

Use `gwas-epi` to scan pairs and `epi-watch` to follow progress. The [LUX guide](lux/README.md) walks through using ordinary `fasterlmm gwas` results as anchors. GxE and cross-cluster orchestration remain separate future integrations. The existing `fasterlmm extreme` command remains available in the core.

## Documentation and reference

For the complete command-line options:

```bash
fasterlmm gwas --help
fasterlmm extreme --help
```

See the [code map](docs/HOW_IT_WORKS.md) for how loading, model fitting, permutations, and output fit together. Report problems through [GitHub issues](https://github.com/gbrach/FaST-ER-LMM/issues).

FaST-ER-LMM builds on FaST-LMM, described in [Lippert et al. (2011), *Nature Methods*](https://doi.org/10.1038/nmeth.1681).

## TODO

- [x] Benchmarks!!
- [ ] LUX: integrate `gwas-gxe`, the GxE / single-K interaction scan.
- [x] LUX: integrate `gwas-epi`, tier-2 pairwise epistasis, bundled with its own namespace and commands.
- [ ] LUX: integrate multi-cluster epi-hub orchestration, daemon + per-cluster workers.
- [x] Richer `fasterlmm watch` dashboard: one panel per GPU shard with progress, rate, ETA, LOCO sweep, and GPU memory.
- [x] `fasterlmm extreme`: streamed genotypes + capped low-rank K for big N, scaling past the dense `gwas` path.
- [x] Portable, CPU-only pytest suite covering the package; FaST-LMM parity, GPU, and external checks skip when their requirements are unavailable.
- [ ] Simulations for GxE and epistasis, to validate once implemented.
- [ ] Manhattan and QQ plots, perhaps through a `fasterlmm plot` entry point that reads the Parquet bundle. Port the existing R code to Python.
- [ ] Benchmark on H100 and H200, just for fun!
- [x] MPS support: `--device mps` runs on Apple Silicon GPUs in float32.
- [ ] Binary phenotypes?
