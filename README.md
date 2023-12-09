# fasterlmm

torch port of FaST-LMM (Lippert et al. 2011, Nat. Methods). wip.

## How fastlmm works

Short call graph, mostly so I don't get lost as I port pieces over

`single_snp()` (single_snp.py) loads geno/pheno, builds `K = Z Zᵀ / M`,
then for each chromosome-LOCO calls `LMM` (lmm.py):

  - `eig(K)` once
  - `findH2` -> golden-section over h2, each step calls `nLLeval` (log-likelihood)
  - per-SNP Wald after rotating Y, X, S by Uᵀ

The port keeps `single_snp`'s outer shape and swaps the `LMM` internals
out for torch -> the eigendecomp, the rotations, and the per-SNP scan are
the parts that move to gpu
