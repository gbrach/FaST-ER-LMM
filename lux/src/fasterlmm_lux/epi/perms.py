"""phenotype permutation for the tier-2 perm-derived threshold — VENDORED.

`permute_phenotypes` lives in the dev `fasterlmm.perms` (SAVEfasterlmm
perms.py:33) but is ABSENT from the public `fasterlmm.perms` (which only ships
`perm_threshold`). The dev `perms.py` is otherwise a heavy GWAS-orchestration
module, so lux lifts only this one pure-numpy function. It sits on the tier-2
parity surface (it feeds the perm threshold), so it is kept byte-faithful and
parity-tested; re-sync if the dev version changes.
"""

from __future__ import annotations

import numpy as np


def permute_phenotypes(Y: np.ndarray, n_perm: int, seed: int = 0) -> np.ndarray:
    """packing real + n_perm shuffles into one (N, P*(1+n_perm)) block.

    independent shuffle per (perm, pheno), matching fastlmm's
    matrix_permut. the python loop version was the dominant cpu cost at
    P=64, n_perm=1000; argsort of one (n_perm, N, P) random matrix drops
    it to a few hundred ms
    """
    N, P = Y.shape
    rng = np.random.default_rng(seed)
    if n_perm == 0:
        return Y.copy()
    # one independent permutation per (k, p) via argsort of a (n_perm, N, P) random matrix along axis=1; take_along_axis then gathers Y[orders[k, n, p], p]
    rand = rng.random(size=(n_perm, N, P), dtype=np.float64)
    orders = rand.argsort(axis=1)  # (n_perm, N, P)
    perm_blocks = np.take_along_axis(Y[None, :, :], orders, axis=1)  # (n_perm, N, P)
    out = np.empty((N, P * (1 + n_perm)), dtype=Y.dtype)
    out[:, :P] = Y
    out[:, P:] = perm_blocks.transpose(1, 0, 2).reshape(N, n_perm * P)
    return out
