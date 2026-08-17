# Developing LUX

The pairwise implementation was imported from [fasterlmm-lux](https://github.com/gbrach/fasterlmm-lux) at commit `c4bf320e09c2350f657ea32b7918866b9cc50d48`, using the local sibling checkout. The import includes the top-level `epi` modules and their `_status`, `_watchkit`, and bundle helpers. It excludes `gxe` and `epi/orch`.

The fitting math, batched Wald kernel, permutation helper, and retained-pair buffers preserve the source implementation. Integration fixes exclude duplicate same-chromosome pairs in all-anchor mode and apply pair masks to the fallback path as well as the batched path. CLI integration adapts command names, progress defaults, and worker handling to the bundled LUX layer. No core modules are copied or modified.

The root `pyproject.toml` builds one `fasterlmm` distribution containing both `fasterlmm` and `fasterlmm_lux`. LUX has no separate installer here. It imports internal fitting and dashboard helpers from the bundled core, so rerun the frozen scan comparison and CLI integration checks when those interfaces change.

From the repository root:

```bash
pip install -e '.[test]'
python -m pytest lux/tests
```

`tests/data/epi_parity_ref.npz` is the unchanged frozen reference from the source repository, originally captured from the development scan. It contains fixed simulated inputs, pair identities, effect estimates, p-values, and permutation thresholds. The parity checks exercise the batched path and the fallback through the public core. Do not regenerate expected values to accommodate an unexplained change.

Keep new research features within `fasterlmm_lux`. Core imports and commands must not import LUX, even though both ship together. Never add GxE or orchestration as an incidental dependency of the pairwise scan.
