"""
the cuda-OOM -> cpu lapack eigh fallback in core._eigh_with_cpu_fallback
cuSOLVER's eigh workspace can OOM at large N even when K itself fits, so the fallback ships K to cpu and
retries.  there's no gpu here, but the branch is exercisable on cpu by mocking torch.linalg.eigh to raise
an out-of-memory RuntimeError once and then succeed -- the handler keys only off the 'out of memory' string
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fasterlmm.core import _eigh_with_cpu_fallback

DTYPE = torch.float64


def _spd(n: int, seed: int = 19930909) -> torch.Tensor:
    """a small symmetric positive-definite matrix to eigh"""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n + 5))
    return torch.from_numpy(A @ A.T / (n + 5)).to(DTYPE)


def test_oom_then_cpu_retry_matches_direct(monkeypatch):
    """first eigh raises 'out of memory', the fallback retries and returns the true decomposition"""
    K = _spd(20)
    real = torch.linalg.eigh
    calls = {"n": 0}

    def flaky(A):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return real(A)

    monkeypatch.setattr(torch.linalg, "eigh", flaky)
    s, U = _eigh_with_cpu_fallback(K)
    assert calls["n"] == 2  # one failed attempt, one retry on cpu
    # the retry result is the genuine eigendecomposition
    recon = U @ torch.diag(s) @ U.T
    assert torch.allclose(recon, K, atol=1e-9)
    assert torch.all(s[:-1] <= s[1:])  # ascending


def test_non_oom_runtimeerror_propagates(monkeypatch):
    """a RuntimeError that is NOT an OOM must bubble up, the fallback only catches memory errors"""
    K = _spd(12)

    def boom(A):
        raise RuntimeError("some unrelated linalg failure")

    monkeypatch.setattr(torch.linalg, "eigh", boom)
    with pytest.raises(RuntimeError, match="unrelated"):
        _eigh_with_cpu_fallback(K)


def test_clean_path_no_fallback(monkeypatch):
    """when eigh succeeds first try the helper just returns it, no retry"""
    K = _spd(15)
    real = torch.linalg.eigh
    calls = {"n": 0}

    def counted(A):
        calls["n"] += 1
        return real(A)

    monkeypatch.setattr(torch.linalg, "eigh", counted)
    s, U = _eigh_with_cpu_fallback(K)
    assert calls["n"] == 1
    assert torch.allclose(U @ torch.diag(s) @ U.T, K, atol=1e-9)
