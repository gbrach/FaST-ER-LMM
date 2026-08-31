# Changelog

## 1.3.1

- Write one Manhattan PDF per phenotype by default, including Parquet bundles, bundle-only scans, and sharded runs. Bundled results put PDFs in `manhattan/<phenotype>.pdf` beside the dataset; `fasterlmm plot --out combined.pdf` still combines selected phenotypes explicitly.
- Give mixed-phenotype Parquet parts descriptive names (`gwas-results-part-00000.parquet`), prefixed by shard when gathered. Existing bundles remain readable.
- Simplify the README installation instructions, use CUDA and Manhattan plots in the quick start, and shorten the LUX section heading.

## 1.3.0

- Add `--manhattan` to `gwas` and `extreme`, generating PDFs in the skyblue/navy plotting style with permutation thresholds and labelled significant hits.
- Combine plots into a multipage PDF with `--bundle`; support bundle-only and sharded runs.
- Add `fasterlmm plot` for saved TSVs and Parquet datasets, with phenotype selection, chromosome sizes, and label controls. Add `concat --manhattan` for completed shard arrays.
- Include plotting dependencies in the normal installation.
- Version the previously integrated Lux pairwise scan and watcher in the same distribution, retaining the separate `fasterlmm_lux` namespace.
