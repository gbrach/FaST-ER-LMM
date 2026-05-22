<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".assets/logo_dark.svg">
    <img src=".assets/logo.svg" width="640" alt="FaST-ER-LMM">
  </picture>
</p>

# FaST-ER-LMM

PyTorch port of [FaST-LMM](https://github.com/fastlmm/FaST-LMM) ([Lippert et al. 2011](https://doi.org/10.1038/nmeth.1681), Nat. Methods).

<p align="center">
  <img src=".assets/gif_truth_1g_vs_2g.gif" width="780" alt="side-by-side watcher replay of the same scan on 1 vs 2 GPUs">
</p>

## How to install install the environment?
Fresh mamba/conda environment recommended!

```bash
mamba create -n fasterlmm python=3.11 -y
mamba activate fasterlmm
pip install -e .
```

Dependencies: pytorch, pysnptools, scipy, pandas, numpy, tqdm, rich, pyarrow

## Examples

Pheno TSV format: first column `Strain` (matching plink IIDs), the rest are pheno value columns.
A tiny yeast subset lives under `data/example/`: 150 strains × 1500 SNPs (16 chromosomes), 5 nuclear glucose phenos (YAL001C, YBR001C, YGR001C, YLR001C, YPR001W), and a small covariate file `example_covar.tab` with 43 aneuploidy covariates
Covariate file format: PLINK whitespace, `FID IID c1 c2 c3 ...`, no header.

### 1. CPU run on the shipped data, all 5 phenos with covariates, bundled parquet

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --covar data/example/example_covar.tab \
  --outdir runs/example/ \
  --bundle --device cpu --n-perm 100 --rint
```

### 2. Multi-GPU on a single node

Pass `--device cuda` on multi-GPU machines. FaST-ER-LMM will spawn one worker per device and round-robin the pheno list across them. Each worker drops its own `status.shard{i}.json` as it goes; with `--bundle` each worker streams its own parts and the job gathers them into the `gwas_bundle.parquet` dataset before it exits (more on that dataset in Outputs):

```bash
srun -p gpu --gres=gpu:2 -c 16 --mem=64G -t 1:00:00 \
  fasterlmm gwas \
    --geno data/yeast --pheno phen.tsv --covar aneuploidies.cov \
    --outdir runs/all/ --device cuda --bundle --n-perm 100 --rint
```

Add `--no-multi-gpu` to use only the first GPU.

### 3. Slurm-array / cross-node sharding

Each task gets one GPU and one round-robin pheno slice via `--shard X/N`. Saved here as `scan.sbatch`:

```bash
#!/bin/bash
#SBATCH --array=0-7
#SBATCH --partition=gpu --gres=gpu:2
#SBATCH -c 8 --mem=32G -t 4:00:00
#SBATCH --output=logs/scan_%A_%a.out
mamba activate fasterlmm
fasterlmm gwas \
  --geno data/yeast --pheno phen.tsv --covar aneuploidies.cov \
  --outdir runs/all/ \
  --shard ${SLURM_ARRAY_TASK_ID}/8 --device cuda --bundle --n-perm 100 --rint
```

Submit it, then gather the 8 shards once the array finishes. Each `--shard` task streams its own bundle part, but a slurm array has no parent process to gather them, so that is what `fasterlmm concat` is for (CPU-only, instant, it just moves the parts into the bundle dataset):

```bash
sbatch scan.sbatch
# once the array is done:
fasterlmm concat runs/all/
```

### Apple Silicon (M-series GPU)

On a Mac, `--device mps` runs the scan on the M-chip GPU through Metal:

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --outdir runs/example/ \
  --bundle --device mps --n-perm 100 --rint
```

One caveat worth knowing: Metal has no float64, so the `mps` path runs in float32. That makes it no longer bit-for-bit with fastlmm, but the difference stays well inside float noise and it's perfectly fine for a real scan (the cluster `cuda` runs stay float64, so the parity path is untouched). A couple of linear-algebra bits that Metal doesn't implement, the eigendecomposition above all, quietly detour through the CPU. It's a single device so there's no multi-GPU dispatch here, just one worker.

## Watch a live run (it's fun!)

```bash
fasterlmm watch runs/all/ # one panel per GPU
```

The TUI refreshes once a second. It discovers the `status.shard*.json` files on its own and draws a dashboard: an overall panel (progress bar, phenos done, scan rate, ETA, RAM) plus one panel per GPU shard with its own progress bar, rate, a live LOCO chromosome strip and GPU memory. Once the scan is done and the trailing bundle writes are still draining, each shard panel switches to a write-drain readout (throughput in MB/s plus a CPU-busy figure), which makes it easy to tell at a glance whether the drain is compute-bound or stuck waiting on the filesystem.

## How does it compare?

Same input PLINK + pheno TSV, LOCO, 100 permutations. FastLMM baseline is ran on 12 parallel workers, 10CPUs each. FaST-ER-LMM is running on one or two old V100S-32GB GPUs.

### Per-pheno speedup according to the number of samples N

<p align="center">
  <img src=".assets/fig_speedup_vs_N.png" width="780" alt="per-pheno speedup over fastlmm: 1 and 2 GPUs across N from 500 to 10000">
</p>

Compared to our previous pipeline.
At N=1000 it's already ~1.5k× per pheno on 1 GPU!

###  Correlations against FaSTLMM on the real data

<p align="center">
  <img src=".assets/fig_correlation_realdata.png" width="780" alt="−log10 p-value scatter, fasterlmm vs fastlmm, on a real yeast scan">
</p>

### On simulated data:

<p align="center">
  <img src=".assets/fig_full_usecase.png" width="780" alt="full transcriptome walltime: 1 vs 2 V100S, N from 500 to 10000, M=100k, P=6484">
</p>

A full yeast-transcriptome scan (P=6484 phenos, M=100k SNPs, N=1000) runs in about 6 minutes on 2 V100S, where the original fastlmm needs ~130 CPU-hours on the same data.


## CLI

required:

- `--geno PREFIX` PLINK BED prefix (no extension)
- `--pheno PHENOS.tsv` phenotype TSV, header `Strain<TAB>pheno1<TAB>...`, one row per strain
- `--outdir DIR` per-pheno outputs land here

optional input:

- `--covar COVAR.tab` no header covariate tab file

main options:

- `--loco` / `--no-loco` leave-one-chromosome-out, on by default
- `--n-perm 100` perms per pheno for the threshold
- `--perm-quantile 0.05` quantile of per-perm min-p used as threshold
- `--rint` / `--no-rint` Blom rank-based inverse normal transform on each pheno column. on by default (matches Victor's R helper in the starlight pipeline)
- `--seed 19930909` rng seed for the permutations

pheno selection (default scans every column of the TSV):

- `--pheno-idx I` single pheno column (0-based) for a quick sanity scan
- `--pheno-start S` / `--pheno-end E` 0-based half-open range, e.g. `--pheno-start 0 --pheno-end 200`

output:

- `--bundle` stream every per-pheno table into a `gwas_bundle.parquet` dataset at the outdir root (a directory of parquet parts, see Outputs)
- `--no-per-pheno-dirs` skip the per-pheno folder tree altogether, write only the bundle parquet (needs `--bundle`). much kinder to the filesystem once there are thousands of phenos

tuning:

- `--phenos-per-job 256` real phenos packed into one gpu scan, gpu columns = this × (1 + n_perm)
- `--write-workers` writer threads for the per-pheno output, off the gpu thread so the next batch scans while this one writes. left unset it fills the core allocation: the cores the job can see, split across the GPU workers sharing the node (so `-c 16` on 2 GPUs lands 8 threads per worker). more cores there means a faster bundle write, snappy compression scales with them until the filesystem becomes the limit

the gpu tile sizes sort themselves out from whatever vram is free and the writer pool from whatever cores the job got, so there's nothing to tune there anymore (it used to be `--snp-chunk` / `--pheno-chunk`)

dispatch:

- `--device cuda|mps|cpu` force a device, defaults to cuda if available. `mps` runs the Apple Silicon GPU in float32 (see the Apple Silicon note above)
- `--no-multi-gpu` use the first visible GPU only
- `--shard X/N` process the X-th of N round-robin pheno slices (slurm-array / cross-node use)

misc:

- `--dry-run` print the planned work and exit

`fasterlmm watch <PATH>`: live TUI dashboard. PATH is an outdir (one panel per GPU shard) or a single `status.json` (one-pane view). It refreshes once a second.

`fasterlmm concat <OUTDIR>`: gather the per-shard bundle parts from a slurm-array run into the `gwas_bundle.parquet` dataset. CPU-only and instant. A single-job multi-GPU run already does this itself, so concat is only needed after a `--shard` array.

## Outputs

Per pheno under `outdir/<pheno_name>/`:

- `gwas.tsv` -> the per-variant association table, the same columns FaSTLMM's `single_snp` writes (`sid_index`, `SNP`, `Chr`, `GenDist`, `ChrPos`, `PValue`, `SnpWeight`, `SnpWeightSE`, `EffectSize`, `SnpFractVarExpl`, `Mixing`, `Nullh2`, `Pheno`, `PhenoCount`), sorted by PValue. So it drops straight into anything that already reads fastlmm output
- `perms.tsv` -> per-permutation min p across the genome
- `threshold.txt` -> 5% quantile of perm min p (the genome-wide significance threshold)
- `status.json` -> live progress for `fasterlmm watch`

### The `--bundle` parquet

With `--bundle`, every per-pheno table also gets streamed into a `gwas_bundle.parquet` at the outdir root, with two extra columns on top of the per-variant table, `threshold` and `significant`. The streaming runs during the scan, one row group per pheno, so the bundle is basically done the moment the GPUs are. Pair it with `--no-per-pheno-dirs` to skip the per-pheno folder tree and keep only the bundle, which is a lot kinder to the filesystem than a tree of thousands of folders.

One thing worth highlighting: **`gwas_bundle.parquet` is a directory, not a single file.** It holds a set of parquet parts (`part0.parquet`, `part1.parquet`, ...). That is on purpose, and it is what makes the writing fast: each writer thread streams its own part, so the per-pheno compression runs fully in parallel instead of queueing through one shared file. A directory of parquet parts is itself a perfectly standard parquet dataset, so it changes nothing about how the bundle gets read: pandas, pyarrow, duckdb and R's arrow all open a directory of parts exactly like a single file.

```python
import pandas as pd
df = pd.read_parquet("runs/all/gwas_bundle.parquet")  # a file or a directory, same call
```

For a single-job run (one GPU, or the multi-GPU auto-dispatch) the parts are gathered by the job itself, so the finished bundle is just there when the scan ends. A slurm-array run has no parent process to do that, so a single `fasterlmm concat <outdir>` gathers the parts once the array is done (see the slurm-array example above). Either way the gather is instant, all it does is move the parts into the bundle directory.

If the bundle feeds a Snakemake rule, the directory has to be declared as a directory output: `output: directory("runs/all/gwas_bundle.parquet")`, otherwise Snakemake assumes a plain file and complains.


## TODO

- [x] benchmarks!!
- [ ] `gwas-gxe` (GxE / single-K interaction scan)
- [ ] `gwas-epi` (tier-2 pairwise epistasis, anchor × all-SNPs)
- [ ] multi-cluster epi-hub orchestration: daemon + per-cluster workers
- [x] richer `fasterlmm watch` dashboard, one panel per GPU shard with progress / rate / ETA / LOCO sweep / GPU memory
- [ ] wire `--extreme` back into the CLI (randomised low-rank K for big N, the code's in `extreme.py` but not plugged in)
- [ ] unit tests? if I ever find some time...
- [ ] simulations for GxE and epistasis, to validate once implemented
- [ ] manhattan and qqplots, maybe a `fasterlmm plot` entrypoint that can take the bundle parquet as input. Just need to port my R code to python
- [ ] bench on H100 and H200! just for fun
- [x] MPS support, `--device mps` runs the M-chip GPU in float32
- [ ] binary phenotypes?



