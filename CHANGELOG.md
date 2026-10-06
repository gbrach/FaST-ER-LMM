# Changelog

## 1.5.0

- Add LD clumping to `gwas`: `--clump` adds an `LDGroup` column (per-folder `gwas.tsv` and the bundle) with greedy per-phenotype groups of the variants under the permutation threshold, options `--clump-window-kb` (default 50), `--clump-r2` (default 0.5) and `--clump-p` (fixed cutoff instead of the threshold).

## 1.4.3

- Seed the permutations of a phenotype with missing values on its column in the phenotype table, so its threshold no longer depends on which NA group it lands in. Tables without missing values are unchanged.

## 1.4.0

- Allow missing values in the phenotype table of `gwas`. Phenotypes are grouped by NA pattern and each group scans on its own strain subset with one eigendecomposition per chromosome; a table without missing values is unchanged. Previously one NA turned every result of that phenotype into NA.
- Report the genomic inflation factor: `lambda_gc.txt` per phenotype folder (classic output, not with `--bundle`) and a `lambda_gc.tsv` summary (`Pheno`, `PhenoIndex`, `LambdaGC`, `NVariants`), same definition as `calc_GIF.R`; shard arrays write `lambda_gc.shardX.tsv`, merged by `fasterlmm concat`.

## 1.3.2

- Write a run record next to every `gwas` and `extreme` result: `run_info.json` (command line, resolved options, input files with size and modification time, versions, host, GPU, outcome) and `run.log` (stdout and stderr); sharded runs write `run_info.shardX.json` and `run.shardX.log`.
- Write the RINT-transformed phenotype table as `pheno_rint.tsv` in the output directory (skipped with `--no-rint`, written by shard 0 only).
- Add a GPU check to the README install section.
- Document the `--kinship-geno PREFIX` option of `gwas` (kinship from a second PLINK panel, `--geno` stays the tested panel) in the README and the command-line reference.

## 1.3.1

- Write one Manhattan PDF per phenotype by default, including Parquet bundles, bundle-only scans, and sharded runs. Bundled results put PDFs in `manhattan/<phenotype>.pdf` beside the dataset; `fasterlmm plot --out combined.pdf` still combines selected phenotypes explicitly.
- Give mixed-phenotype Parquet parts descriptive names (`gwas-results-part-00000.parquet`), prefixed by shard when gathered. Existing bundles remain readable.
- Simplify the README installation instructions, use CUDA and Manhattan plots in the quick start, and shorten the LUX section heading.
- Move the core Python package to top-level `fasterlmm/`; keep imports, commands, and the separate Lux package unchanged.

## 1.3.0

- Add `--manhattan` to `gwas` and `extreme`, generating PDFs in the skyblue/navy plotting style with permutation thresholds and labelled significant hits.
- Combine plots into a multipage PDF with `--bundle`; support bundle-only and sharded runs.
- Add `fasterlmm plot` for saved TSVs and Parquet datasets, with phenotype selection, chromosome sizes, and label controls. Add `concat --manhattan` for completed shard arrays.
- Include plotting dependencies in the normal installation.
- Version the previously integrated Lux pairwise scan and watcher in the same distribution, retaining the separate `fasterlmm_lux` namespace.
