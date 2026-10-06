# Changelog

## Unreleased

- Allow missing values in the phenotype table of `gwas`. Phenotypes are grouped by NA pattern and each group scans on its own strain subset with one eigendecomposition per chromosome; a table without missing values is unchanged. Previously one NA turned every result of that phenotype into NA.

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
