"""per-pheno top-K buffer of pair records.

at starlight scale the unfiltered M^2 x P matrix is hundreds of TB; only
the per-pheno top-K rows survive, plus a perm-derived threshold (v2).

two impls live here:
- TopKHeap: pure-python heapq. simple, used by the unit tests + as a
  reference. per-push overhead is dominated by python-level float casts
  and dataclass alloc, fine when K and n are small
- TopKBuffer: numpy-backed bulk-add. per-cell add_many concatenates new
  candidates with the running top-K, runs argpartition once to keep K,
  no python loops over per-test entries. used by the production tier-2
  scanner where each cell pushes thousands of candidates per pheno

p-value is the actual ranking criterion but F-stat is monotonic in p
within a fixed dof and avoids the scipy round-trip on every push
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(order=True)
class _PairEntry:
    f_stat: float  # heap key (min-heap → smallest at root)
    snp_i: int
    snp_j: int
    beta: float
    se: float
    pvalue: float


class TopKHeap:
    """fixed-K min-heap of (f_stat, snp_i, snp_j, beta, se, pvalue).

    push_many takes parallel arrays from one scan chunk. drain returns the
    list sorted by p ascending (best first)
    """

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._heap: list[_PairEntry] = []

    def __len__(self) -> int:
        return len(self._heap)

    def push(self, f_stat: float, snp_i: int, snp_j: int,
             beta: float, se: float, pvalue: float) -> None:
        entry = _PairEntry(f_stat=f_stat, snp_i=snp_i, snp_j=snp_j,
                           beta=beta, se=se, pvalue=pvalue)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        elif f_stat > self._heap[0].f_stat:
            heapq.heapreplace(self._heap, entry)

    def push_many(self, f_stats, snp_i: int, snp_js,
                  betas, ses, pvalues) -> None:
        """batched push from one scan chunk. snp_i fixed (the marginal
        SNP); snp_js / f_stats / betas / ses / pvalues are per-test-SNP
        arrays of equal length"""
        n = len(f_stats)
        if not (len(snp_js) == n and len(betas) == n and len(ses) == n
                and len(pvalues) == n):
            raise ValueError("push_many: array lengths must all match")
        for i in range(n):
            f = float(f_stats[i])
            if f != f and f != f:  # NaN check, skip masked entries
                continue
            self.push(f, snp_i, int(snp_js[i]), float(betas[i]),
                      float(ses[i]), float(pvalues[i]))

    def drain_sorted(self) -> list[_PairEntry]:
        """returning entries sorted by p ascending. doesn't mutate heap"""
        return sorted(self._heap, key=lambda e: e.pvalue)


class TopKBuffer:
    """numpy top-K. add_many concatenates new candidates with the
    running buffer and runs argpartition once to keep top-K by F-stat.
    no python loop over per-test entries.

    drain_sorted returns _PairEntry list sorted by p ascending. p is
    looked up from the per-test pvalue array passed to add_many; if
    the caller passes nan (compute_pvalue=False on the scanner) p stays
    nan and gets filled by the caller post-drain via a single batched
    F-sf

    memory: 5 numpy arrays of length K each (f_stat, snp_i, snp_j, beta,
    se) + a length-K pvalue. K=1000 default = ~50KB / pheno. negligible
    """

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._n = 0  # current fill count, <= k
        self._f_stat = np.full(k, -np.inf, dtype=np.float64)
        self._snp_i = np.zeros(k, dtype=np.int64)
        self._snp_j = np.zeros(k, dtype=np.int64)
        self._beta = np.zeros(k, dtype=np.float64)
        self._se = np.zeros(k, dtype=np.float64)
        self._pvalue = np.full(k, np.nan, dtype=np.float64)

    def __len__(self) -> int:
        return self._n

    def add_many(self, f_stats: np.ndarray, snp_i: int, snp_js: np.ndarray,
                 betas: np.ndarray, ses: np.ndarray,
                 pvalues: np.ndarray | None = None) -> None:
        """add a batch from one scan chunk. snp_i is fixed (the anchor);
        per-test-SNP arrays must all have the same length. pvalues=None
        signals 'caller will fill p later from f_stat at drain time' and
        keeps the pvalue column NaN"""
        f_stats = np.asarray(f_stats, dtype=np.float64)
        n = f_stats.size
        if snp_js.shape[0] != n or betas.shape[0] != n or ses.shape[0] != n:
            raise ValueError("add_many: array lengths must all match")
        # drop NaN-masked entries up front so argpartition doesn't see them
        valid = ~np.isnan(f_stats)
        if not valid.all():
            f_stats = f_stats[valid]
            snp_js = snp_js[valid]
            betas = betas[valid]
            ses = ses[valid]
            if pvalues is not None:
                pvalues = pvalues[valid]
            n = f_stats.size
        if n == 0:
            return
        # cap the new batch to top-K by F-stat first; this avoids sorting
        # all 6000 entries when only 1000 will survive
        if n > self.k:
            sel = np.argpartition(-f_stats, self.k - 1)[:self.k]
            f_stats = f_stats[sel]
            snp_js = snp_js[sel]
            betas = betas[sel]
            ses = ses[sel]
            if pvalues is not None:
                pvalues = pvalues[sel]
            n = self.k
        # merge batch with current top-K (overwriting the unfilled tail
        # if we haven't hit K yet)
        end = self._n + n
        if end <= self.k:
            sl = slice(self._n, end)
            self._f_stat[sl] = f_stats
            self._snp_i[sl] = snp_i
            self._snp_j[sl] = snp_js
            self._beta[sl] = betas
            self._se[sl] = ses
            self._pvalue[sl] = pvalues if pvalues is not None else np.nan
            self._n = end
            return
        # buffer overfills, do the partition merge
        cat_f = np.concatenate([self._f_stat[:self._n], f_stats])
        cat_i = np.concatenate([self._snp_i[:self._n], np.full(n, snp_i, dtype=np.int64)])
        cat_j = np.concatenate([self._snp_j[:self._n], snp_js.astype(np.int64, copy=False)])
        cat_b = np.concatenate([self._beta[:self._n], betas])
        cat_s = np.concatenate([self._se[:self._n], ses])
        cat_p = np.concatenate([self._pvalue[:self._n],
                                pvalues if pvalues is not None
                                else np.full(n, np.nan, dtype=np.float64)])
        sel = np.argpartition(-cat_f, self.k - 1)[:self.k]
        self._f_stat[:] = cat_f[sel]
        self._snp_i[:] = cat_i[sel]
        self._snp_j[:] = cat_j[sel]
        self._beta[:] = cat_b[sel]
        self._se[:] = cat_s[sel]
        self._pvalue[:] = cat_p[sel]
        self._n = self.k

    def drain_sorted(self) -> list[_PairEntry]:
        """sorted by f_stat descending (= pvalue ascending in F(1, df2)).
        if pvalue column is nan (compute_pvalue=False path), the caller
        is responsible for filling it via a single batched F-sf"""
        n = self._n
        order = np.argsort(-self._f_stat[:n])
        return [_PairEntry(
                    f_stat=float(self._f_stat[i]),
                    snp_i=int(self._snp_i[i]),
                    snp_j=int(self._snp_j[i]),
                    beta=float(self._beta[i]),
                    se=float(self._se[i]),
                    pvalue=float(self._pvalue[i]))
                for i in order]

    def f_stats(self) -> np.ndarray:
        """raw F-stat array of the current top-K (length = len(self)),
        in insertion order. used by the post-drain batched F-sf path"""
        return self._f_stat[:self._n].copy()


class ThresholdBuffer:
    """unbounded per-pheno buffer of pairs whose F-stat clears a preset cut.

    used for the all-pairs / one-pheno path where the user wants every
    suggestive interaction kept for downstream network building, not
    just a top-K. each add_many filters by F-stat > f_cut up front so
    the in-memory footprint scales with how loose the cut is, not with
    the M^2 universe.

    drains via drain_arrays(), which returns parallel numpy arrays
    sorted by F-stat descending (= p-value ascending). avoids the
    O(N) _PairEntry alloc that TopKBuffer.drain_sorted does — at
    p<0.01 with M=95k we're looking at ~45M entries per pheno, the
    python-object round-trip would be ~5 GB of dataclass overhead
    """

    def __init__(self, f_cut: float):
        if not np.isfinite(f_cut):
            raise ValueError(f"f_cut must be finite, got {f_cut}")
        self.f_cut = float(f_cut)
        self._f_stat_chunks: list[np.ndarray] = []
        self._snp_i_chunks: list[np.ndarray] = []
        self._snp_j_chunks: list[np.ndarray] = []
        self._beta_chunks: list[np.ndarray] = []
        self._se_chunks: list[np.ndarray] = []
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def add_many(self, f_stats: np.ndarray, snp_i: int, snp_js: np.ndarray,
                 betas: np.ndarray, ses: np.ndarray,
                 pvalues: np.ndarray | None = None) -> None:
        """append rows with finite F-stat > f_cut. pvalues arg ignored —
        we recompute via one batched F-sf at drain time anyway"""
        f_stats = np.asarray(f_stats, dtype=np.float64)
        n_in = f_stats.size
        if snp_js.shape[0] != n_in or betas.shape[0] != n_in or ses.shape[0] != n_in:
            raise ValueError("add_many: array lengths must all match")
        # filter F > cut and finite in one pass. NaN comparisons are False so np.isfinite excludes both nan and inf
        keep = np.isfinite(f_stats) & (f_stats > self.f_cut)
        if not keep.any():
            return
        f_keep = f_stats[keep]
        n = f_keep.size
        self._f_stat_chunks.append(f_keep)
        self._snp_i_chunks.append(np.full(n, snp_i, dtype=np.int64))
        self._snp_j_chunks.append(np.asarray(snp_js[keep], dtype=np.int64))
        self._beta_chunks.append(np.asarray(betas, dtype=np.float64)[keep])
        self._se_chunks.append(np.asarray(ses, dtype=np.float64)[keep])
        self._n += n

    def drain_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """concat chunks, sort by F-stat descending, return parallel arrays
        (f_stat, snp_i, snp_j, beta, se). caller fills p-values from f via one batched F-sf"""
        if self._n == 0:
            empty_i = np.empty(0, dtype=np.int64)
            empty_f = np.empty(0, dtype=np.float64)
            return empty_f, empty_i, empty_i, empty_f, empty_f
        f = np.concatenate(self._f_stat_chunks)
        i = np.concatenate(self._snp_i_chunks)
        j = np.concatenate(self._snp_j_chunks)
        b = np.concatenate(self._beta_chunks)
        s = np.concatenate(self._se_chunks)
        # drop chunk refs to free the per-chunk arrays before the sort allocates the order array
        self._f_stat_chunks.clear()
        self._snp_i_chunks.clear()
        self._snp_j_chunks.clear()
        self._beta_chunks.clear()
        self._se_chunks.clear()
        order = np.argsort(-f)
        return f[order], i[order], j[order], b[order], s[order]
