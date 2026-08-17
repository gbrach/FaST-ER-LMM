# Changelog

## 1.3.0

- Add `--manhattan` to `gwas` and `extreme`, generating PDFs in the skyblue/navy plotting style with permutation thresholds and labelled significant hits.
- Combine plots into a multipage PDF with `--bundle`; support bundle-only and sharded runs.
- Add `fasterlmm plot` for saved TSVs and Parquet datasets, with phenotype selection, chromosome sizes, and label controls. Add `concat --manhattan` for completed shard arrays.
- Include plotting dependencies in the normal installation.
- Version the previously integrated Lux pairwise scan and watcher in the same distribution, retaining the separate `fasterlmm_lux` namespace.
