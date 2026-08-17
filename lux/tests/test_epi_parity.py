"""parity: the migrated lux tier-2 epistasis scan reproduces the dev version.

the reference fixture (tests/data/epi_parity_ref.npz) was captured from the dev
`fasterlmm.epistasis.scan.scan_tier2` by tests/_gen_epi_parity_ref.py. this test
runs the *lux* scan (fasterlmm_lux.epi.scan, on the public core + the vendored
snp_wald_scan_batched / permute_phenotypes) over the same frozen inputs and
asserts the per-pheno top-K tables match: exact equality on the integer pair
indices (top-K membership + order), allclose on the float stats + threshold.

if this trips, either an import was misrepointed OR a public-core primitive the
scan reuses (eigendecompose / fit_delta_grid / snp_wald_scan) drifted numerically
from the dev version the vendored batched kernel was matched against. those three
are NOT exercised by the gxe parity path, so this is their first parity check —
a failure here is a real core-drift finding, not necessarily a porting bug.

regenerate the reference only when the dev scan output legitimately changes; see
tests/_gen_epi_parity_ref.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REF_PATH = Path(__file__).resolve().parent / "data" / "epi_parity_ref.npz"

# float64 CPU reduction noise (BLAS accumulation order across the public-core
# primitives vs the dev ones the reference was captured with) shows up at ~3e-9
# on near-zero betas / small p-values. the integer top-K indices match dev
# EXACTLY (same pairs, same order — see the exact-match fields below), so a real
# divergence (a misrepointed import or wrong kernel) would be orders of magnitude
# off and still trip this; the tolerance only absorbs reduction jitter.
RTOL = 1e-6
ATOL = 1e-8

# P in the fixture (kept in sync with _gen_epi_parity_ref.py). parametrize needs
# a literal at collection time; test_pheno_count guards it against drift.
_P = 4

# (npz field stem, Tier2Result attribute, exact-match?) — exact for the integer
# top-K membership/order indices, allclose for the float stats
_FIELDS = [
    ("snp_i_idx", "snp_i_idx", True),
    ("snp_j_idx", "snp_j_idx", True),
    ("beta", "beta", False),
    ("se", "se", False),
    ("pvalue", "pvalue", False),
]


@pytest.fixture(scope="module")
def ref():
    if not REF_PATH.exists():
        pytest.skip(f"parity reference missing: {REF_PATH} "
                    f"(regenerate with tests/_gen_epi_parity_ref.py)")
    return np.load(REF_PATH, allow_pickle=True)


@pytest.fixture(scope="module")
def lux_results(ref):
    """run the lux tier-2 scan on the frozen dev inputs."""
    import torch
    from fasterlmm_lux.epi.scan import scan_tier2

    snp_id = [str(s) for s in ref["snp_id"]]
    pheno_names = [str(s) for s in ref["pheno_names"]]
    # rebuild per_pheno_marginals from the two parallel arrays the gen flattened
    per_pheno_marginals: dict[str, list[str]] = {nm: [] for nm in pheno_names}
    for nm, s in zip(ref["marg_pheno"], ref["marg_snp"]):
        per_pheno_marginals[str(nm)].append(str(s))

    return scan_tier2(
        Z_std=torch.from_numpy(ref["Z_std"]),
        Y=torch.from_numpy(ref["Y"]),
        X=torch.from_numpy(ref["X"]),
        chrom=ref["chrom"],
        snp_id=snp_id,
        pheno_names=pheno_names,
        per_pheno_marginals=per_pheno_marginals,
        pos=ref["pos"],
        top_k_pair=int(ref["top_k_pair"]),
        var_floor=float(ref["var_floor"]),
        anchor_batch=int(ref["anchor_batch"]),
        device="cpu",
        dtype=torch.float64,
        n_perm=int(ref["n_perm"]),
        perm_seed=int(ref["perm_seed"]),
        perm_quantile=float(ref["perm_quantile"]),
        exclude_window_kb=0.0)


def test_reference_came_from_dev(ref):
    assert "SAVEfasterlmm" in str(ref["core_path"]), \
        "parity reference was not captured from the dev checkout"


def test_pheno_count(ref, lux_results):
    assert len(lux_results) == int(ref["P"]) == _P


@pytest.mark.parametrize("p", range(_P))
@pytest.mark.parametrize("field,attr,exact", _FIELDS)
def test_tier2_parity(ref, lux_results, p, field, attr, exact):
    expected = ref[f"t{p}_{field}"]
    got = np.asarray(getattr(lux_results[p], attr))
    assert got.shape == expected.shape, \
        f"pheno {p} {field}: shape {got.shape} != dev {expected.shape}"
    if exact:
        assert np.array_equal(got, expected), \
            f"pheno {p} {field}: top-K membership/order differs from dev"
    else:
        _assert_close(expected, got, f"pheno {p} {field}")


@pytest.mark.parametrize("p", range(_P))
def test_threshold_parity(ref, lux_results, p):
    expected = float(ref[f"t{p}_threshold"])
    got = float(lux_results[p].threshold)
    if np.isnan(expected):
        assert np.isnan(got), f"pheno {p} threshold: {got} != dev NaN"
    else:
        assert np.isclose(got, expected, rtol=RTOL, atol=ATOL), \
            f"pheno {p} threshold: {got} != dev {expected}"


def _rebuild_inputs(ref):
    """the scan_tier2 kwargs from the frozen fixture (shared with lux_results)."""
    import torch
    snp_id = [str(s) for s in ref["snp_id"]]
    pheno_names = [str(s) for s in ref["pheno_names"]]
    per_pheno_marginals: dict[str, list[str]] = {nm: [] for nm in pheno_names}
    for nm, s in zip(ref["marg_pheno"], ref["marg_snp"]):
        per_pheno_marginals[str(nm)].append(str(s))
    return dict(
        Z_std=torch.from_numpy(ref["Z_std"]), Y=torch.from_numpy(ref["Y"]),
        X=torch.from_numpy(ref["X"]), chrom=ref["chrom"], snp_id=snp_id,
        pheno_names=pheno_names, per_pheno_marginals=per_pheno_marginals,
        pos=ref["pos"], top_k_pair=int(ref["top_k_pair"]),
        var_floor=float(ref["var_floor"]), anchor_batch=int(ref["anchor_batch"]),
        device="cpu", dtype=torch.float64, n_perm=int(ref["n_perm"]),
        perm_seed=int(ref["perm_seed"]), perm_quantile=float(ref["perm_quantile"]),
        exclude_window_kb=0.0)


@pytest.mark.parametrize("p", range(_P))
@pytest.mark.parametrize("field,attr,exact", _FIELDS)
def test_tier2_fallback_path_parity(ref, monkeypatch, p, field, attr, exact):
    """force EVERY anchor through the per-anchor fallback (public-core
    snp_wald_scan) by making the vendored batched kernel always raise
    LinAlgError — the exact except at scan.py:431. the fixed-seed parity fixture
    never trips a batched cholesky, so this is the only coverage of that branch,
    which a live cluster run surfaced as a real bug: the migrated call used the
    dev ScanResult schema (compute_pvalue kwarg + .chi2/.min_pvalue) the public
    core doesn't have. asserts the fallback both completes and reproduces the dev
    top-K, proving the public-ScanResult wiring (.f / .max_F) is numerically right
    (the per-anchor and batched paths compute the same Wald test)."""
    import torch
    import fasterlmm_lux.epi.scan as scan_mod

    def _force_fallback(*a, **k):
        raise torch._C._LinAlgError("forced batched failure for fallback coverage")
    monkeypatch.setattr(scan_mod, "snp_wald_scan_batched", _force_fallback)

    results = scan_mod.scan_tier2(**_rebuild_inputs(ref))
    expected = ref[f"t{p}_{field}"]
    got = np.asarray(getattr(results[p], attr))
    assert got.shape == expected.shape, \
        f"pheno {p} {field}: shape {got.shape} != dev {expected.shape} (fallback path)"
    if exact:
        assert np.array_equal(got, expected), \
            f"pheno {p} {field}: fallback top-K membership/order differs from dev"
    else:
        _assert_close(expected, got, f"pheno {p} {field} (fallback path)")


def _assert_close(expected, got, label):
    expected = np.asarray(expected)
    got = np.asarray(got)
    close = np.isclose(expected, got, rtol=RTOL, atol=ATOL, equal_nan=True)
    if not close.all():
        diff = np.abs(expected - got)
        finite = diff[np.isfinite(diff)]
        worst = float(finite.max()) if finite.size else float("nan")
        raise AssertionError(
            f"{label}: {int((~close).sum())}/{got.size} elements diverge from dev "
            f"(max abs diff {worst:.3e}, rtol={RTOL}, atol={ATOL})")
