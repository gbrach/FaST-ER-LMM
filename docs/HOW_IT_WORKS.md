# code map

`src/fasterlmm/`

## gwas

```text
cli_main.py -> cli.py
                -> io.py          PLINK + phenos + optional covars
                -> normalize.py   pheno ranks (RINT)
                -> io.py          same strains, same row order
                -> perms.py       adding shuffled copies to each batch
                -> core.py        model + variant tests
                -> cli.py         scores -> p-values
                -> bundle.py      results out

LOCO: testing chr1 -> relatedness from the other chroms
      testing chr2 -> same idea
      ...

shuffle -> move pheno values between strains, keep genotypes fixed

real phenos     -> results for every variant
shuffled phenos -> one minimum p-value per shuffle -> 5th percentile (default) -> threshold
p-value below threshold -> significant
```

## extreme

```text
cli_main.py -> cli_extreme.py
                -> io_stream.py      genotype blocks from disk, or memory if they fit
                -> extreme_scan.py   chromosome loop
                -> lowrank.py        model + variant tests
                -> cli_extreme.py    scores -> p-values
                -> cli.py + bundle.py   results out

--grm-k -> fewer variants for estimating relatedness
--grm   -> chosen variants from a separate PLINK file

extreme.py -> separate experiment, unused by the command
```

## outputs + progress

```text
per pheno -> gwas.tsv + perms.tsv + threshold.txt
--bundle  -> gwas_bundle.parquet/  (directory of parts)

workers -> progress.py -> status files -> watch.py -> dashboard

multiple GPUs -> split the pheno list -> gather results automatically
--shard jobs  -> split the pheno list -> all finished -> fasterlmm concat
```

## LUX layer

`lux/src/fasterlmm_lux/` ships in the same distribution as `src/fasterlmm/`, with
its own namespace and commands. The dependency goes from LUX to the core;
no `fasterlmm` module imports LUX.

```text
fasterlmm gwas -> per-phenotype gwas.tsv
                         |
gwas-epi -> epi.marginals -> epi.scan -> epi.writer
                               |
                       fasterlmm.core + Lux's batched pair kernel

epi-watch <- Lux's progress snapshots
```

The pairwise scan uses LDCO kinship, excluding both SNP chromosomes. GxE and
orchestration are not part of this package. See the [LUX guide](../lux/README.md).
