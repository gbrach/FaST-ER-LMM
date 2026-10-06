<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".assets/logo_dark.svg">
    <img src=".assets/logo.svg" width="640" alt="FaST-ER-LMM">
  </picture>
</p>

# FaST-ER-LMM

**Linear mixed-model GWAS for thousands of phenotypes, on CPUs and GPUs.**

FaST-ER-LMM is a PyTorch port of [FaST-LMM](https://github.com/fastlmm/FaST-LMM). Give it PLINK genotypes, a table of phenotypes, and optional covariates. It runs an association scan for each phenotype, accounts for relatedness, and estimates genome-wide significance thresholds with permutations.

- Batch phenotypes and their permutations together, with automatic distribution across available NVIDIA GPUs.
- Leave-one-chromosome-out (LOCO) is the default: kinship for each chromosome is estimated from the other chromosomes.
- The `extreme` command uses low-rank kinship and optional genotype streaming to reduce memory use for larger cohorts.
- Results include per-phenotype TSVs and a combined Parquet dataset. Follow progress with the live terminal dashboard.

[Quick start](#quick-start) · [Input formats](#input-formats) · [Results](#results) · [Manhattan plots](#manhattan-plots) · [GPUs and clusters](#gpus-and-clusters) · [Large datasets](#large-datasets)

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

Requires Python 3.10+. Install directly from GitHub inside a mamba environment:

```bash
mamba create -n fasterlmm python=3.11 pip git -y
mamba activate fasterlmm
python -m pip install git+https://github.com/gbrach/FaST-ER-LMM.git
```

Mamba creates the environment; pip installs FaST-ER-LMM from GitHub. No Conda channel package is needed.

For NVIDIA GPU runs, check that `python -c "import torch; print(torch.cuda.is_available())"` prints `True`. If it prints `False`, PyTorch cannot access CUDA in this environment. Check that the installed PyTorch build supports CUDA and that the GPU is available.

CPU runs work with `--device cpu`; Apple Silicon runs use `--device mps`.

## Quick start

The commands above install FaST-ER-LMM. Clone the repository to get the example data:

```bash
git clone https://github.com/gbrach/FaST-ER-LMM.git
cd FaST-ER-LMM
```

Run the bundled yeast example (**150 strains, 1,500 variants, and 20 phenotypes**) on an NVIDIA GPU:

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --covar data/example/example_covar.tab \
  --outdir runs/example/ \
  --device cuda \
  --bundle --manhattan
```

This scans every phenotype with LOCO, applies a rank-based inverse normal transform (RINT), and runs 100 permutations per phenotype. Results are written to `runs/example/`, with one PDF per phenotype in `runs/example/manhattan/`.

While the scan runs, open a second terminal in the same environment to follow progress:

```bash
fasterlmm watch runs/example/
```

To run your own data, replace the three input paths and choose an output directory. Omit `--covar` if you have no covariates. Use `--device cpu` for CPU runs or `--device mps` on Apple Silicon.

To build the kinship from a different set of variants than the ones tested, add `--kinship-geno` with a second PLINK prefix:

```bash
fasterlmm gwas \
  --geno data/example/example \
  --kinship-geno data/my_kinship_panel \
  --pheno data/example/example_pheno.tsv \
  --outdir runs/example_kinship/
```

`--geno` is still the panel being tested. Both PLINK sets need the same individuals and matching chromosome labels, since LOCO drops the tested chromosome from the kinship panel.

## Recommended setup

For a typical run over many phenotypes:

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --covar data/example/example_covar.tab \
  --outdir runs/example/ \
  --bundle --no-per-pheno-dirs --clump
```

- The scan applies a rank-based inverse normal transform (RINT) to every phenotype by default, so raw values can be supplied. `pheno_rint.tsv` in the output directory holds the transformed table. Use `--no-rint` to keep values as supplied, for example if they are already normalised.
- `--bundle --no-per-pheno-dirs` writes one Parquet dataset in place of one folder per phenotype, which matters at thousands of phenotypes. `lambda_gc.tsv` holds the inflation factors in that mode.
- `--clump` adds an `LDGroup` column to the bundle, grouping significant variants by LD (defaults: 50 kb window, r2 0.5).
- Missing values in the phenotype table are fine: phenotypes missing values for the same individuals are scanned together using the remaining individuals.

## Input formats

| Input | Expected format |
|---|---|
| `--geno` | PLINK `.bed`, `.bim`, and `.fam` files; pass their shared prefix without an extension. |
| `--pheno` | Tab-separated table with a `Strain` column followed by numeric phenotype columns. IDs match PLINK IIDs. |
| `--covar` | Optional whitespace-delimited table with `FID IID c1 c2 ...`, **without a header**. |
| `--kinship-geno` | Optional second PLINK BED prefix for estimating kinship (`gwas` only). Use the same individuals and chromosome labels as `--geno`; LOCO excludes the tested chromosome from this panel. |
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
| `--clump` | Add an `LDGroup` column for variants below each phenotype's permutation threshold (`gwas` only). |
| `--clump-window-kb 50` | Maximum distance in kb between two variants for them to be linked in the same group; default: 50. |
| `--clump-r2 0.5` | r2 at or above which two variants are linked in the same group; default: 0.5. |
| `--clump-p 1e-5` | Fixed p-value cutoff for the variants that get a group, instead of each phenotype's permutation threshold. |
| `--no-rint` | Use phenotype values without the default rank-based inverse normal transform. |
| `--no-loco` | Use a shared relatedness model across chromosomes (`gwas` only). |
| `--phenos-per-job 32` | Set how many phenotypes are processed per batch; lower this to reduce batch memory use. |
| `--bundle --no-per-pheno-dirs` | Write a combined Parquet dataset without individual phenotype folders. |
| `--manhattan` | Generate one Manhattan PDF per phenotype, including when results use `--bundle`. |
| `--dry-run` | Load inputs and print the planned work before scanning (`gwas` only). |

## Results

With `--bundle --manhattan`, an output directory contains:

```text
runs/example/
├── <phenotype>/
│   ├── gwas.tsv
│   ├── perms.tsv
│   └── threshold.txt
├── gwas_bundle.parquet/
│   └── gwas-results-part-00000.parquet
├── manhattan/
│   └── <phenotype>.pdf
├── lambda_gc.tsv
├── pheno_rint.tsv
├── run_info.json
├── run.log
└── status.json
```

Each phenotype gets:

| File | Contents |
|---|---|
| `gwas.tsv` | Per-variant results sorted by p-value, using the FaST-LMM `single_snp` column format. |
| `perms.tsv` | The minimum genome-wide p-value from each permutation. |
| `threshold.txt` | The significance threshold: the 5th percentile of permutation minimum p-values by default. |
| `lambda_gc.txt` | Genomic inflation factor (lambda GC) of the phenotype, written only without `--bundle`. |

With `--clump`, `gwas.tsv` and the bundle gain an `LDGroup` column. Only variants below the cutoff get a group; the rest stay empty.

Two variants are linked if they are on the same chromosome, within the distance window, and meet the r² cutoff. Groups are connected components, so a chain of linked variants can extend beyond the window. Groups are numbered by their smallest p-value; group 1 contains the top hit.

LD uses squared Pearson correlation over individuals called at both variants, as in PLINK `--r2`. It is calculated from the full aligned genotype panel, independently of phenotype missingness. The grouping follows `addLinkageGroups.py` from the [1086 yeast genomes GWAS](https://github.com/HaploTeam/1086YeastGenomes/tree/main/GWAS).

The combined bundle adds `threshold` and `significant` columns. A variant is marked significant when its p-value is below its phenotype's threshold.

Read the bundle and select hits in Python:

```python
import pandas as pd

results = pd.read_parquet("runs/example/gwas_bundle.parquet")
hits = results.loc[results["significant"]]
print(hits[["Pheno", "SNP", "Chr", "ChrPos", "PValue", "threshold"]])
```

In R, read the bundle with `arrow` and the tidyverse:

```r
pak::pak(c("arrow", "tidyverse"))

gwas_results <- dplyr::collect(arrow::open_dataset("~/GWAS_Carmen/gwas_carmen_mexagave/gwas_bundle.parquet", format = "parquet"))
```

`gwas_bundle.parquet` is a directory of Parquet parts. In Snakemake, declare it with `directory("runs/example/gwas_bundle.parquet")`.

Parts are named `gwas-results-part-00000.parquet`, `gwas-results-part-00001.parquet`, and so on. Gathered GPU or cluster shards add a prefix, such as `shard0-gwas-results-part-00000.parquet`. Each file can contain several phenotypes; read the whole directory to load all results.

`lambda_gc.tsv` lists the genomic inflation factor of every scanned phenotype in phenotype column order, with columns `Pheno`, `PhenoIndex`, `LambdaGC`, and `NVariants`. It is written even with `--no-per-pheno-dirs`. With `--bundle`, it is the only lambda output; without it, each phenotype folder also gets `lambda_gc.txt`.

Lambda GC is computed from the `PValue` column as the median of `qchisq(1 - p, 1)` divided by `qchisq(0.5, 1)`, with missing p-values dropped. This is the same definition as `calc_GIF.R` in the [1086 yeast genomes repository](https://github.com/HaploTeam/1086YeastGenomes/blob/main/GWAS/src/calc_GIF.R). A `--shard` array writes one `lambda_gc.shardX.tsv` per task, and `fasterlmm concat` merges them.

With RINT on (the default), `pheno_rint.tsv` holds the transformed phenotype table in the input layout (a `Strain` column, then one column per phenotype, input individual order), so the values the scan used can be inspected or fed back with `--pheno ... --no-rint`. It covers every individual in the phenotype file, before matching to the genotype. A `--shard` array writes it from task 0 only, and `--no-rint` skips it.

For `gwas`, missing values (NA or empty cells) in the phenotype table are allowed. Phenotypes missing values for the same individuals are scanned together using the remaining individuals, with one eigendecomposition per chromosome. A missing value removes an individual only from the phenotypes where it is missing. This matches running FaST-LMM on each phenotype after removing individuals with missing values.

Phenotypes that are entirely NA, or have too few individuals left to fit the model, are skipped with a message. With RINT on, ranks use only the observed values of each phenotype. Each distinct missing-value pattern needs its own decompositions, so scattered NAs make the scan slower than a complete table.

`run_info.json` records the exact command line, parameter values (including defaults), the input files with their sizes and modification times, software versions, the host and GPU, and the final outcome. `run.log` keeps everything printed during the run. Sharded runs write `run_info.shardX.json` and `run.shardX.log` per task. Keep these files with the results to reproduce a run; `status.json` only tracks live progress and is reduced to a short record when the run ends.

Progress is stored in `status.json`, or `status.shard*.json` for sharded runs, and displayed by `fasterlmm watch`.

## Manhattan plots

Plotting is included in the normal installation. Add `--manhattan` to `gwas` or `extreme`:

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --covar data/example/example_covar.tab \
  --outdir runs/example/ \
  --device cuda --bundle --manhattan
```

With `--bundle`, this writes **one PDF per phenotype** in `runs/example/manhattan/`, alongside `gwas_bundle.parquet/`. It also works with `--no-per-pheno-dirs`. Without `--bundle`, each phenotype folder gets its own `manhattan.pdf`.

To plot saved results without rerunning GWAS:

```bash
fasterlmm plot runs/example/gwas_bundle.parquet --manhattan
```

The input can also be a run directory, a phenotype directory, a `gwas.tsv`, or a single Parquet file. Parquet data is read by phenotype using its row groups; PDFs are written outside the Parquet dataset. A missing threshold produces an unthresholded plot, explicitly marked as such.

Select phenotypes with `--pheno`. To combine selected phenotypes into a single multipage PDF, explicitly pass `--out`:

```bash
fasterlmm plot runs/example/ --manhattan \
  --pheno YAL001C --pheno YAL002W --out runs/example/selected.pdf
```

Use `--label-top 20` to label more hits, or `--label-top 0` to turn labels off. Chromosome widths default to the largest observed position on each chromosome, in natural chromosome order. Supply `--chrom-sizes genome.sizes` for reference lengths and ordering: two whitespace-separated columns, chromosome and length, without a header. FASTA `.fai` indexes also work.

For explicit `--shard X/N --bundle --manhattan` runs, each task writes its phenotypes' PDFs to the shared `manhattan/` directory. After all tasks finish, `fasterlmm concat runs/all/ --manhattan` gathers the data and generates the individual PDFs. Automatic multi-GPU runs generate one PDF per phenotype after gathering their data.

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

Use `extreme` when the standard scan needs too much memory. It estimates relatedness from a subset of variants and can stream the test variants from disk in blocks.

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

## Documentation and reference

For the complete command-line options:

```bash
fasterlmm gwas --help
fasterlmm extreme --help
fasterlmm plot --help
```

A single-file reference with every flag, the input and output formats, and examples is in [if_you_are_a_LLM_read_this.help](https://github.com/gbrach/FaST-ER-LMM/blob/main/if_you_are_a_LLM_read_this.help).

See the [code map](docs/HOW_IT_WORKS.md) for how loading, model fitting, permutations, and output fit together.

FaST-ER-LMM builds on FaST-LMM, described in [Lippert et al. (2011), *Nature Methods*](https://doi.org/10.1038/nmeth.1681).

## TODO

- [x] Benchmarks!!
- [x] Richer `fasterlmm watch` dashboard: one panel per GPU shard with progress, rate, ETA, LOCO sweep, and GPU memory.
- [x] `fasterlmm extreme`: streamed genotypes + capped low-rank K for big N, scaling past the dense `gwas` path.
- [x] Portable, CPU-only pytest suite covering the package; FaST-LMM parity, GPU, and external checks skip when their requirements are unavailable.
- [x] Manhattan PDFs in the existing R plot style, including multipage output and plotting from Parquet bundles with `fasterlmm plot`.
- [ ] QQ plots.
- [ ] Benchmark on H100 and H200, just for fun!
- [x] MPS support: `--device mps` runs on Apple Silicon GPUs in float32.
- [ ] Binary phenotypes?
