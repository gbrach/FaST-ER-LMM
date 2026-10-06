"""
clump.ld_clump against plink --r2 on a synthetic panel with missing calls, monomorphic variants and exact duplicates
skips when no plink binary is on the PATH (or in FASTERLMM_PLINK), so the portable suite stays portable.
the reference is the grouping of HaploTeam/1086YeastGenomes addLinkageGroups.py: plink --r2 pairs, then connected components
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fasterlmm.clump import ld_clump
from fasterlmm.io import read_plink

PLINK = os.environ.get("FASTERLMM_PLINK") or shutil.which("plink")
pytestmark = pytest.mark.skipif(PLINK is None, reason = "no plink binary (set FASTERLMM_PLINK or put plink on the PATH)")


def _panel(tmp_path, miss = 0.03, n = 400, per_chr = 200, seed = 5):
    """three chromosomes on a 5 kb grid, blocks of correlated 0/1/2 genotypes, written as a PLINK bed set"""
    rng = np.random.default_rng(seed)
    cols, chrom, pos, ids = [], [], [], []
    for c in (1, 2, 3):
        i = 0
        while i < per_chr:
            b = min(int(rng.integers(3, 12)), per_chr - i)
            lat = rng.standard_normal(n)
            for k in range(b):
                a = rng.uniform(0.5, 0.95)
                v = a * lat + np.sqrt(1 - a * a) * rng.standard_normal(n)
                cols.append(np.digitize(v, [-0.7, 0.7]).astype(float))
                chrom.append(c)
                pos.append(5000 * (i + k) + 1000)
                ids.append(f"V{c}_{i + k}")
            i += b
    z = np.stack(cols, 1)
    z[:, [7, 300]] = 0  # monomorphic
    z[:, [50, 400]] = z[:, [49, 399]]  # exact duplicates of the left neighbour
    z[rng.random(z.shape) < miss] = np.nan
    al = np.where(np.isnan(z), "0 0", np.where(z == 0, "A A", np.where(z == 1, "A G", "G G")))
    with open(tmp_path / "syn.ped", "w") as fh:
        for i in range(n):
            fh.write(f"s{i} s{i} 0 0 0 -9 " + " ".join(al[i]) + "\n")
    pd.DataFrame({"c": chrom, "id": ids, "g": 0, "p": pos}).to_csv(tmp_path / "syn.map", sep = " ", header = False, index = False)
    subprocess.run([PLINK, "--file", str(tmp_path / "syn"), "--make-bed", "--out", str(tmp_path / "syn")], check = True, capture_output = True)
    return str(tmp_path / "syn"), rng


@pytest.mark.parametrize("window_kb,r2", [(5, 0.8), (10, 0.2), (50, 0.5), (100, 1.0)])
def test_clump_partition_equals_plink_graph(tmp_path, window_kb, r2):
    prefix, rng = _panel(tmp_path)
    g = read_plink(prefix)
    cand_mask = rng.random(len(g.sid)) < 0.5
    cand = np.flatnonzero(cand_mask)
    ids = [g.sid[i] for i in cand]
    pd.Series(ids).to_csv(tmp_path / "ids.txt", header = False, index = False)
    subprocess.run([PLINK, "--bfile", prefix, "--extract", str(tmp_path / "ids.txt"), "--r2", "--ld-window-r2", str(r2), "--ld-window", "1000000",
                    "--ld-window-kb", str(window_kb), "--out", str(tmp_path / "ld")], check = True, capture_output = True)
    node = {s: i for i, s in enumerate(ids)}
    try:
        his = pd.read_csv(tmp_path / "ld.ld", sep = r"\s+")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        his = pd.DataFrame(columns = ["SNP_A", "SNP_B"])
    rows = [node[s] for s in his.SNP_A]
    cols = [node[s] for s in his.SNP_B]
    _, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape = (len(ids), len(ids))), directed = False)
    mine = ld_clump(np.asarray(g.Z, dtype = np.float32), np.asarray(g.chrom), np.asarray(g.pos, dtype = np.float64),
                    np.where(cand_mask, 1e-9, 0.9), cutoff = 0.5, window_bp = window_kb * 1000.0, r2_min = r2)
    part = lambda l: sorted(map(tuple, (np.flatnonzero(np.asarray(l) == u) for u in np.unique(l))))
    assert part(lab) == part(mine[cand])
