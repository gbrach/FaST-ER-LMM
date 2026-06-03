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

## How to install the environment?
I recommend a fresh mamba/conda env...

```bash
mamba create -n fasterlmm python=3.11 -y
mamba activate fasterlmm
pip install -e .
```

Dependencies: pytorch, pysnptools, scipy, pandas, numpy, tqdm, rich, pyarrow

To run the tests: `pip install -e ".[test]"` then `pytest`. The suite is cpu-only and skips the fastlmm parity, GPU, and external checks when those arent around, so it passes on a fresh clone with nothing extra.

## Examples

Input files:

| flag      | format                                                                       |
|-----------|------------------------------------------------------------------------------|
| `--geno`  | standard PLINK (`.bed` / `.bim` / `.fam`), just the prefix NO extension! |
| `--pheno` | TSV, header `Strain<TAB>p1<TAB>p2 ...`, one row per strain (first col matches plink IIDs) |
| `--covar` | PLINK whitespace, `FID IID c1 c2 c3 ...`, no header (optional)               |

A tiny yeast subset is in `data/example/`: 150 strains × 1500 variants (16 chromosomes), 20 nuclear glucose phenos (4 per chrom on chroms 1, 2, 7, 12, 16) and a small covariate file `example_covar.tab` with 43 aneuploidy covariates.

### 1. CPU run on the shipped data, all 20 phenos with covariates, bundled parquet

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --covar data/example/example_covar.tab \
  --outdir runs/example/ \
  --bundle --device cpu --n-perm 100 --rint
```

### 2. Multi-GPU on a single node

Pass `--device cuda` on multi-GPU machines. FaST-ER-LMM will spawn one worker per device and round-robin the pheno list across them. Each worker drops its own `status.shard{i}.json` as it goes; with `--bundle` each worker streams its own parts and the job gathers them into the `gwas_bundle.parquet` dataset before it exits:

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

Submit it, then gather the 8 shards once the array finishes. Each `--shard` task streams its own bundle part. Hence `fasterlmm concat` is here to concat everything at the end:

```bash
sbatch scan.sbatch
# once the array is done:
fasterlmm concat runs/all/
```

### 4. Very large N: the `extreme` path

When N gets big enough that the dense N×N kinship stops fiting (think tens of thousands of strains), `fasterlmm extreme` swaps how the kinship gets built without touching the association math. The kinship comes from a pruned marker set (auto-strided down to `--grm-k` markers, or a pre-pruned `--grm` bed) so K stays low-rank and never lands as a whole N×N matrix, and the test variants stream off the BED a block at a time so the N×M genotype never lands resident either. The scan math underneath is fastlmm's own k<N branch, so for a given kinship marker set it reproduces the dense path to float tol, the only aproximation is the pruning itself.

```bash
fasterlmm extreme \
  --geno data/bigN --pheno phen.tsv \
  --outdir runs/bigN/ \
  --grm-k 5000 --block-size 8192 \
  --device cuda --bundle --n-perm 100 --rint
```

Every `gwas` flag works here too, plus the kinship-pruning and streaming ones (`--grm-k`, `--grm`, `--block-size`, `--resident`). At 100k strains × 100k variants a single pheno with its 100 perms lands in about 2.9 seconds, and a full transcriptome (~6500 phenos) in about 5.4 hours on one V100S, where the dense path would want a 40 GB genotype slab in ram before a single scan tensor even exists (see the comparison below).

### Apple Silicon (M-series GPU)

On a Mac, `--device mps` runs the scan on the M-chip GPU through Metal:

```bash
fasterlmm gwas \
  --geno data/example/example \
  --pheno data/example/example_pheno.tsv \
  --outdir runs/example/ \
  --bundle --device mps --n-perm 100 --rint
```

Caveat: Metal has no float64, so the `mps` path runs in float32. That makes it no longer bit-for-bit with fastlmm... A couple of linear-algebra bits that Metal doesn't implement detour through the CPU. Also it is single device, one worker, no multi-GPU dispatch.

## Watch a live run (it's fun!)

```bash
fasterlmm watch runs/all/ # one panel per GPU
```

The TUI refreshes once a second. It discovers (hopefully) the `status.shard*.json` files on its own and draws a dashboard: an overall panel (progress bar, phenos done, scan rate, ETA, RAM) plus one panel per GPU shard with its own progress bar.

## How does it compare?

Same input PLINK + pheno TSV, LOCO, 100 permutations. FastLMM baseline ran on 10 parallel jobs, 10 CPUs each. FaST-ER-LMM is running on one or two old V100S-32GB GPUs.

### Per-pheno speedup according to the number of samples N

<p align="center">
  <img src=".assets/fig_speedup_vs_N.png" width="780" alt="per-pheno speedup over fastlmm: 1 and 2 GPUs across N from 500 to 10000">
</p>

Compared to our previous pipeline.
At N=1000 it's ~870× faster per pheno on 1 GPU and ~1.25k× on 2 GPUs (V100S).

###  Correlations against FaSTLMM on the real data

<p align="center">
  <img src=".assets/fig_correlation_realdata.png" width="780" alt="−log10 p-value scatter, fasterlmm vs fastlmm, on a real yeast scan">
</p>

### On simulated data:

<p align="center">
  <img src=".assets/fig_full_usecase.png" width="780" alt="full transcriptome walltime: 1 vs 2 V100S, N from 500 to 10000, M=100k, P=6484">
</p>

A full yeast-transcriptome scan (P=6484 phenos, M=100k variants, N=1000) runs in about 5 minutes on 2 V100S, where the original fastlmm needs ~130 wall-hours (~13k CPU-hours) on the same data.

### At the extreme scale

The speedup figure above stops at N=10000, wich is about where the dense kinship path gives up. The `extreme` path keeps going, so I benchmarked it at the full 100k × 100k geometry against fastlmm doing the exact same deliverable: one real pheno plus its 100 permutations, low-rank LOCO, same k=5000 kinship markers on both sides.

- fastlmm (low-rank LOCO, the lab one-job-per-pheno pattern we'd actualy run): about 31.8 hours per pheno
- fasterlmm extreme: 2.9 seconds per pheno (its 100 perms ride the same eigendecomposition, so they come almost for free)

That's roughly 39,500× per pheno. The whole story is the eigendecomposition: fastlmm pays the O(N³) decompose over and over, once per pheno, and it's 86% of its wall. extreme decomposes a tiny k×k Gram once and reuses it across every pheno and every perm. Scaled to a full transcriptome (~6500 phenos × 100 perms) that's 5.4 hours on one V100S (2.7 on two), against something like 4.7 years for fastlmm on five parallel jobs (yes, years).


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
- `--rint` / `--no-rint` Blom rank-based inverse normal transform on each pheno column. on by default (matches Victor's R helper)

pheno selection (default scans every column of the TSV):

- `--pheno-idx I` single pheno column (0-based) for a quick sanity scan
- `--pheno-start S` / `--pheno-end E` 0-based half-open range, e.g. `--pheno-start 0 --pheno-end 200`

output:

- `--bundle` stream every per-pheno table into a `gwas_bundle.parquet` dataset at the outdir root (a directory of parquet parts, see Outputs)
- `--no-per-pheno-dirs` skip the per-pheno folder tree altogether, write only the bundle parquet (needs `--bundle`)

tuning:

- `--phenos-per-job 256` real phenos packed into one GPU scan, GPU columns = this × (1 + n_perm)
- `--write-workers` writer threads for the per-pheno output. It fills the core allocation by default
- `--seed 19930909` RNG seed for the permutations

dispatch:

- `--device cuda|mps|cpu` force a device, defaults to cuda if available. `mps` runs the Apple Silicon GPU in float32
- `--no-multi-gpu` use the first visible GPU only
- `--shard X/N` process the X-th of N round-robin pheno slices (slurm-array / cross-node use)

misc:

- `--dry-run` print the planned work and exit

`fasterlmm watch <PATH>`: live TUI dashboard. PATH is an outdir (one panel per GPU shard) or a single `status.json` (one-pane view). It refreshes once a second.

`fasterlmm concat <OUTDIR>`: gather the per-shard bundle parts into the `gwas_bundle.parquet` dataset. A single-job multi-GPU run already does this itself, so concat is only needed after a `--shard` array.

`fasterlmm extreme <same flags as gwas>`: the big-N variant. Same outputs and same association math as `gwas`, but it streams the genotypes off the BED and builds the kinship from a pruned marker set insted of the dense N×N one, so it keeps going well past where `gwas` runs out of memory. On top of everything `gwas` takes, it adds:

- `--grm-k 5000` target kinship marker count for the auto-stride when no `--grm` is given (lower it and the rank of K drops with it)
- `--grm PREFIX` a pre-pruned kinship BED to use insted of the auto-stride
- `--block-size 8192` test variants held resident per streamed block, the lever that keeps the N×M genotype off the heap
- `--resident auto|on|off` hold the standardised genotype resident when the N×M slab fits ram (auto, the default), else re-stream it every pheno batch
- `--float64` run in float64, the default is float32 at this scale

## Outputs

Per pheno under `outdir/<pheno_name>/`:

- `gwas.tsv` -> the per-variant association table, the same columns FaSTLMM's `single_snp` writes (`sid_index`, `SNP`, `Chr`, `GenDist`, `ChrPos`, `PValue`, `SnpWeight`, `SnpWeightSE`, `EffectSize`, `SnpFractVarExpl`, `Mixing`, `Nullh2`, `Pheno`, `PhenoCount`), sorted by PValue. So it drops straight into anything that already reads fastlmm output
- `perms.tsv` -> per-permutation min p across the genome
- `threshold.txt` -> 5% quantile of perm min p (the genome-wide significance threshold)
- `status.json` -> live progress for `fasterlmm watch`

### The `--bundle` parquet

With `--bundle`, every per-pheno table also gets streamed into a `gwas_bundle.parquet` at the outdir root, with two extra columns on top of the per-variant table, `threshold` and `significant`. The streaming runs during the scan, one row group per pheno, so the bundle is done the moment the GPUs are. Pair it with `--no-per-pheno-dirs` to skip the per-pheno folder tree and keep only the bundle.

**`gwas_bundle.parquet` is a directory, not a single file.** It holds a set of parquet parts (`part0.parquet`, `part1.parquet`, ...). That is on purpose, and it is what makes the writing fast. Pandas, pyarrow, duckdb and R's arrow all open a directory of parts exactly like a single file so it's transparent. 

```python
import pandas as pd
df = pd.read_parquet("runs/all/gwas_bundle.parquet")  # a file or a directory, same call
```

```R
library(arrow)
df <- read_parquet("runs/all/gwas_bundle.parquet")  # a file or a directory, same call
``` 

For a single-job run (one GPU, or the multi-GPU auto-dispatch) the parts are gathered by the job itself, so the finished bundle is just there when the scan ends.

If the bundle feeds a Snakemake rule, the directory has to be declared as a directory output: `output: directory("runs/all/gwas_bundle.parquet")`, otherwise Snakemake assumes a plain file and complains.


## TODO

- [x] benchmarks!!
- [ ] `gwas-gxe` (GxE / single-K interaction scan)
- [ ] `gwas-epi` (tier-2 pairwise epistasis, anchor × all-SNPs)
- [ ] multi-cluster epi-hub orchestration: daemon + per-cluster workers
- [x] richer `fasterlmm watch` dashboard, one panel per GPU shard with progress / rate / ETA / LOCO sweep / GPU memory
- [x] `fasterlmm extreme` subcommand: streamed genotypes + capped low-rank K for big N (scales past the dense `gwas` path)
- [x] unit tests: a portable pytest suite covering the whole package, cpu-only so it runs on a fresh clone (fastlmm parity / GPU / external checks skip-gate when unavailable)
- [ ] simulations for GxE and epistasis, to validate once implemented
- [ ] manhattan and qqplots, maybe a `fasterlmm plot` entrypoint that can take the bundle parquet as input. Just need to port my R code to python
- [ ] bench on H100 and H200! just for fun
- [x] MPS support, `--device mps` runs the M-chip GPU in float32
- [ ] binary phenotypes?



