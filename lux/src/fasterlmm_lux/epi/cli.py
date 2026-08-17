"""tier-2 pairwise epistasis cli (`gwas-epi`).

usage:
    gwas-epi \\
        --geno          /path/to/plink_prefix \\
        --pheno         /path/to/phenotypes.tsv \\
        --covar         /path/to/covariates.tab \\
        --marginal-dir  results/gwas/ \\
        --marginal-top-k 100 \\
        --outdir        results/epi/ \\
        --top-k         1000

tier-2 pairwise epistasis: top-K marginal SNPs × all SNPs × LDCO
kinship, 1-dof Wald F per pair (Lippert 2018 FaST-LMM-Epi). per-pheno
top-K pair table written to <outdir>/<pheno>/. marginal-dir points at a
prior fasterlmm gwas results dir and provides the per-pheno top-K
marginal SNPs that anchor each pair test

output modes:
- default: per-pheno <outdir>/<pheno>/{<pheno>.tier2.first_assoc.{txt.gz,
  parquet}, <pheno>.threshold.txt, <pheno>.signif_pairs.txt}
- --bundle: also drop <outdir>/all_pairs.parquet streamed at run time
  (Pheno + threshold + signif baked in, one row group per pheno).
  multi-gpu shards land as all_pairs.shard{r}.parquet and the parent
  collapses them after the workers finish.
- --bundle --no-per-pheno-dirs: bundle only, skip the per-pheno-folder
  tree. saves disk + inodes when only the bundle is consumed downstream
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fasterlmm.io import (
    align_inputs, read_covar, read_phen, read_plink, standardise_columns)

# scan kernel + writer + marginals reader repointed onto the lux epi package;
# stage/write_status -> _status (the public progress lacks the snapshot
# state-machine the watcher reads + the stage context manager)
from fasterlmm_lux._status import stage as progress_stage, write_status
from fasterlmm_lux.epi.marginals import (
    check_marginal_manifest, read_marginal_stats, read_marginal_thresholds,
    read_marginals, union_marginal_snps)
from fasterlmm_lux.epi.scan import scan_tier2
from fasterlmm_lux.epi.writer import BundleWriter, _write_one_for_pool


EXAMPLE_INPUTS = """\
example inputs (transcriptomics, trimmed to 3 of ~6000 genes x 5 of 642 strains):

  PHENOS.tsv (--pheno). one column per gene, log-expression:
    Strain  YAL001C   YAL002W   YAL003W
    AAL     5.421     8.137    10.882
    AAP     4.987     7.842    11.103
    ABD     5.336     8.205    10.954
    ADL     4.815     7.561    10.621
    ADM     5.412     8.094    10.937

  COVAR.tab (--covar). PLINK whitespace, FID IID c1 c2 ... no header:
    AAL AAL 1 0 0 0 0 ...
    AAP AAP 0 0 0 0 0 ...

  PREFIX (--geno). PLINK BED trio (no extension):
    /path/to/matrix.SNPs.InDels.SVs.CNVs.plink  ->  reads .{bed,bim,fam}

  MARGINAL_DIR (--marginal-dir). a fasterlmm gwas results dir laid out
  as <marginal_dir>/<pheno>/gwas.tsv (legacy first_assoc files also work). tier-2
  reads the top-K rows of each per-pheno file (already sorted by PValue
  ascending) and uses them as anchors for the pair scan.

invocation:
    gwas-epi \\
        --geno /path/to/matrix.SNPs.InDels.SVs.CNVs.plink \\
        --pheno PHENOS.tsv --covar COVAR.tab \\
        --marginal-dir results/gwas/ --marginal-top-k 100 \\
        --outdir results/epi/ --top-k 1000
"""


class _CliFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gwas-epi",
        description="GPU tier-2 pairwise epistasis (top-K marginal × all SNPs, LDCO)",
        epilog=EXAMPLE_INPUTS,
        formatter_class=_CliFormatter)
    # required
    p.add_argument("--geno", required=True, help="PLINK BED prefix (no extension)")
    p.add_argument("--pheno", required=True, help="phenotype TSV, 'Strain<TAB>pheno1<TAB>...' header")
    p.add_argument("--marginal-dir", default=None,
        help="prior fasterlmm gwas results dir containing <pheno>/gwas.tsv. "
             "required unless --all-anchors is set")
    p.add_argument("--outdir", required=True, help="per-pheno tier-2 output dir")
    # data
    p.add_argument("--covar", default=None, help="PLINK whitespace covariate file")
    p.add_argument("--rint", action="store_true", help="apply Blom RINT to pheno columns")
    p.add_argument("--rint-ties", default="average",
        choices=["average", "random", "min", "max", "ordinal"], help="tie-handling for --rint")
    p.add_argument("--maf-floor", type=float, default=0.05,
        help="drop SNPs with minor allele freq < this from genome universe AND marginals")
    # scan
    p.add_argument("--marginal-top-k", type=int, default=100,
        help="top-K marginal SNPs per pheno used as pair anchors")
    p.add_argument("--top-k", type=int, default=1000,
        help="top-K pairs retained per pheno (min-heap drain). ignored when --p-threshold is set")
    p.add_argument("--var-floor", type=float, default=1e-8,
        help="drop pairs whose raw t_ij = S_i ⊙ S_j has variance below this")
    p.add_argument("--all-anchors", action="store_true",
        help="every test SNP doubles as an anchor (M*(M-1)/2 unique pairs per pheno). "
             "skips --marginal-dir, synthesises a marginals dict covering all SNPs, and "
             "switches the scan loop to the upper-triangle (skips chrom_m > chrom_j tiles "
             "since the elementwise product is symmetric). meant for one-pheno network "
             "runs; pair with --p-threshold to keep the long tail")
    p.add_argument("--p-threshold", type=float, default=None,
        help="write every pair with p-value < this (vs top-K). uses an unbounded "
             "ThresholdBuffer per pheno; expected row count scales with how loose the cut is "
             "(~45M rows at p<0.01, M=95k, N=542). perm-derived FWER threshold is still "
             "computed when --n-perm > 0 and flags rows via the bundle's signif column")
    # perms
    p.add_argument("--n-perm", type=int, default=0,
        help="perms per pheno for the per-pheno significance threshold (0 = skip)")
    p.add_argument("--perm-quantile", type=float, default=0.05,
        help="quantile of per-perm best-pair p across perms")
    p.add_argument("--perm-seed", type=int, default=19930909,
        help="rng seed for perm shuffles")
    # tuning
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"],
        help="scan precision; fp32 is ~2x faster on V100 vs fp64 with negligible "
             "p-value drift (~1e-5 rel) for tier-2's modest dynamic range")
    p.add_argument("--pheno-chunk", type=int, default=256, help="pheno-column tile width on the gpu")
    p.add_argument("--snp-chunk", type=int, default=4096, help="snp tile width on the gpu")
    p.add_argument("--anchor-batch", type=int, default=8,
        help="anchors per tile-batched U.T@T_std projection (1 = unbatched, larger amortizes "
             "the matmul launch overhead; capped by GPU mem at K * Mj * Neff fp32 cells)")
    p.add_argument("--exclude-window-kb", type=float, default=25.0,
        help="exclude test SNPs within N kb of an anchor on the same chromosome "
             "(Bloom 2015 §Methods). Suppresses LD-trapped near-self pairs from the top-K")
    p.add_argument("--fp16-inner", action="store_true",
        help="use fp16 on V100 TensorCore for ~1.6x speedup on inner matmuls. nondeterministic "
             "across runs at the perm-threshold tail; off by default for reproducibility")
    p.add_argument("--write-workers", type=int, default=8, help="processes for per-pheno writes")
    p.add_argument("--output-format", default="csv", choices=["csv", "parquet"],
        help="per-pheno tier-2 first_assoc format")
    # output mode
    p.add_argument("--bundle", action="store_true",
        help="also emit a run-level <outdir>/all_pairs.parquet with every "
             "(pheno, anchor, test_snp) top-K row, Pheno + threshold + signif "
             "baked in. additive by default. multi-gpu shards land as "
             "all_pairs.shard{r}.parquet and the parent collapses them")
    p.add_argument("--no-per-pheno-dirs", dest="per_pheno_dirs",
        action="store_false", default=True,
        help="skip the <outdir>/<pheno>/ tree. requires --bundle (otherwise "
             "nothing is written)")
    # dispatch
    p.add_argument("--device", default=None, choices=[None, "cuda", "cpu"], help="force a device")
    p.add_argument("--require-gpu", action="store_true",
        help="exit with an error if CUDA is unavailable instead of "
             "falling back to CPU")
    p.add_argument("--no-multi-gpu", action="store_true", help="don't auto-shard across GPUs")
    p.add_argument("--shard", default=None, metavar="X/N", help="process shard X of N (used internally)")
    # misc
    p.add_argument("--status-file", default=None, help="JSON progress snapshot (default: <outdir>/.status.json)")
    p.add_argument("--dry-run", action="store_true", help="print the planned work and exit")
    args = p.parse_args(argv)
    if not args.per_pheno_dirs and not args.bundle:
        p.error("--no-per-pheno-dirs requires --bundle (otherwise nothing is written)")
    if args.marginal_dir is None and not args.all_anchors:
        p.error("--marginal-dir is required unless --all-anchors is set")
    if args.p_threshold is not None and not (0.0 < args.p_threshold < 1.0):
        p.error(f"--p-threshold must be in (0, 1), got {args.p_threshold}")
    for name in ("marginal_top_k", "top_k", "pheno_chunk", "snp_chunk", "anchor_batch", "write_workers"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    if not 0.0 <= args.maf_floor <= 0.5:
        p.error("--maf-floor must be in [0, 0.5]")
    if args.n_perm < 0 or not 0.0 < args.perm_quantile < 1.0:
        p.error("--n-perm must be >= 0 and --perm-quantile must be in (0, 1)")
    if args.exclude_window_kb < 0 or args.var_floor < 0:
        p.error("--exclude-window-kb and --var-floor must be >= 0")
    if args.shard:
        try:
            _, shard_n = _parse_shard(args.shard)
        except ValueError as exc:
            p.error(str(exc))
        if args.all_anchors and shard_n > 1:
            p.error("--all-anchors supports single-device runs only; omit --shard")
    return args


def _parse_shard(s: str | None) -> tuple[int, int] | None:
    if s is None:
        return None
    x, n = s.split("/")
    x, n = int(x), int(n)
    if not (0 <= x < n) or n < 1:
        raise ValueError(f"--shard {s!r} must satisfy 0 <= X < N")
    return x, n


def _rank_reduce_X(X: torch.Tensor) -> torch.Tensor:
    """drop dependent covariate columns from X via pivoted QR.

    the public core's align_inputs has no rank_reduce kwarg (the dev core's
    did). tier-2 needs a rank-reduced X: the eigh-hoisted scan path uses the
    standard rotate which assumes full-rank X, and a rank-deficient covar block
    makes fit_delta_grid's cholesky blow up. lifted verbatim from the dev
    fasterlmm.io._drop_dependent_columns; tier-2 has no fastlmm bit-for-bit
    reference, so trimming dependent cov cols is fine. returns X unchanged when
    already full rank."""
    from scipy.linalg import qr
    X_np = X.cpu().numpy()
    Q, R, piv = qr(X_np, mode="economic", pivoting=True)
    diag_R = np.abs(np.diag(R))
    max_diag = diag_R.max() if diag_R.size else 1.0
    keep = np.sort(piv[diag_R > 1e-10 * max_diag])
    if keep.shape[0] == X_np.shape[1]:
        return X
    return torch.from_numpy(np.ascontiguousarray(X_np[:, keep])).to(
        dtype=X.dtype, device=X.device)


def _load_inputs(args, status_file):
    """loading geno + pheno + optional covar with progress stages.
    optional --rint applied to pheno columns before alignment, mirroring
    fasterlmm.cli.load_inputs so the same knob behaves the same way"""
    with progress_stage("read_plink", status_file, prefix=args.geno):
        geno = read_plink(args.geno)
        print(f"  N={len(geno.iid)}  M={len(geno.sid)}", file=sys.stderr, flush=True)
    with progress_stage("read_phen", status_file, path=args.pheno):
        pheno = read_phen(args.pheno)
        print(f"  N={len(pheno.iid)}  P={len(pheno.names)}", file=sys.stderr, flush=True)
    if args.rint:
        from fasterlmm.normalize import rint_columns
        with progress_stage("rint_phen", status_file, ties=args.rint_ties):
            pheno.Y = rint_columns(pheno.Y, ties=args.rint_ties)
            print(f"  applied RINT to {pheno.Y.shape[1]} phenotype columns "
                  f"(ties='{args.rint_ties}', c=3/8)", file=sys.stderr, flush=True)
    covar = None
    if args.covar:
        with progress_stage("read_covar", status_file, path=args.covar):
            covar = read_covar(args.covar)
            print(f"  N={len(covar.iid)}  C_raw={covar.C.shape[1]}",
                  file=sys.stderr, flush=True)
    return geno, pheno, covar


def _run_one(args, *, shard_rank: int = 0, shard_n: int = 1) -> int:
    """running one shard of the tier-2 scan. shard_n=1 = single worker"""
    status_file = None if args.dry_run else args.status_file
    if shard_n > 1 and status_file:
        status_file = f"{status_file}.shard{shard_rank}"

    # nuking any leftover snapshot from an earlier run, otherwise write_status keeps the old started_at and elapsed_s drifts
    if status_file and not args.dry_run:
        for stale in (Path(status_file), Path(f"{status_file}.jsonl")):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prefix = f"[shard {shard_rank}/{shard_n}] " if shard_n > 1 else ""
    print(f"{prefix}device: {device}", file=sys.stderr, flush=True)
    if (device == "cuda" or args.require_gpu) and (
            device != "cuda" or not torch.cuda.is_available()):
        print(f"{prefix}CUDA was requested but is unavailable or --device cpu was selected.",
              file=sys.stderr, flush=True)
        return 2
    if device == "cuda":
        print(f"{prefix}GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr, flush=True)

    geno, pheno, covar = _load_inputs(args, status_file)

    with progress_stage("align_inputs", status_file):
        # the dev align_inputs took rank_reduce=True; the public core's doesn't,
        # so align on the public core then drop dependent covariate columns here.
        # tier-2 needs the rank-reduced X — the eigh-hoisted scan path (scan.py)
        # uses the standard rotate which assumes X is full rank; glucose's
        # aneuploidy block makes X rank-deficient and fit_delta_grid's cholesky
        # blows up. tier-2 has no fastlmm bit-for-bit reference, so dropping
        # dependent cov cols up front is fine.
        aligned = align_inputs(geno, pheno, covar)
        X_rr = _rank_reduce_X(aligned.X)
        if X_rr.shape[1] != aligned.X.shape[1]:
            aligned = type(aligned)(
                iid=aligned.iid, Z=aligned.Z, Y=aligned.Y, X=X_rr,
                chrom=aligned.chrom, pos=aligned.pos,
                snp_id=aligned.snp_id, pheno_names=aligned.pheno_names)
        print(f"  aligned: N={len(aligned.iid)}  P_total={aligned.Y.shape[1]}  "
              f"C={aligned.X.shape[1]}  M={len(aligned.snp_id)}",
              file=sys.stderr, flush=True)

    # slicing shards AFTER alignment so every shard sees the same strain order.
    # round-robin assignment (phenos[r::n] not [r·P/n:(r+1)·P/n]) averages out
    # per-pheno marginal-density variation across shards. with contiguous slices
    # alphabetically-adjacent phenos can have very different anchor-set sizes
    # post-dedup, leading to ~33% inner_total imbalance at K=250 on the bloom
    # smoke. round-robin mostly fixes this without needing a real anchor-balanced
    # partition (which would need per-pheno top-K reads up front).
    # --all-anchors shards by anchor SNP instead of by pheno (handled below in
    # the synthesize step); skip pheno-slicing here when that mode is active
    if shard_n > 1 and not args.all_anchors:
        P_all = aligned.Y.shape[1]
        sliced_idx = list(range(shard_rank, P_all, shard_n))
        aligned = type(aligned)(
            iid=aligned.iid,
            Z=aligned.Z, Y=aligned.Y[:, sliced_idx],
            X=aligned.X, chrom=aligned.chrom, pos=aligned.pos,
            snp_id=aligned.snp_id,
            pheno_names=[aligned.pheno_names[i] for i in sliced_idx])
        print(f"{prefix}shard slice (round-robin): phenotypes "
              f"[{shard_rank}::{shard_n}] = {len(sliced_idx)} of {P_all}",
              file=sys.stderr, flush=True)
        if not sliced_idx:
            if args.bundle and not args.dry_run:
                BundleWriter(Path(args.outdir) / f"all_pairs.shard{shard_rank}.parquet").close()
            write_status(status_file, {"stage": "main", "stage_state": "done",
                                       "phenos_total": 0, "shard": f"{shard_rank}/{shard_n}"})
            return 0

    # MAF filter applied once on the aligned dosages (NaN-aware mean / 2),
    # then propagated to Z, Z_std, chrom, pos, snp_id so the kinship, the
    # test universe and the marginal lookup all see the same SNP set.
    # rare-variant pairs blow up the F-stat denominator (Lippert 2018 §3.1);
    # cheap to drop them upfront
    keep_snp_id = aligned.snp_id
    keep_chrom = aligned.chrom
    keep_pos = aligned.pos
    Z_kept = aligned.Z
    if args.maf_floor > 0.0:
        with progress_stage("maf_filter", status_file,
                            maf_floor=args.maf_floor):
            Z_np = aligned.Z.cpu().numpy()
            freq = np.nanmean(Z_np, axis=0) / 2.0
            maf = np.minimum(freq, 1.0 - freq)
            keep_mask = maf >= args.maf_floor
            n_kept = int(keep_mask.sum())
            n_drop = int((~keep_mask).sum())
            if n_kept == 0:
                raise SystemExit(
                    f"--maf-floor {args.maf_floor} dropped every SNP; "
                    f"loosen the threshold or check the genotype")
            keep_idx = np.where(keep_mask)[0]
            Z_kept = aligned.Z[:, torch.from_numpy(keep_idx).long()]
            keep_snp_id = [aligned.snp_id[i] for i in keep_idx]
            keep_chrom = aligned.chrom[keep_idx]
            keep_pos = aligned.pos[keep_idx]
            print(f"  maf_floor={args.maf_floor}: kept {n_kept} / "
                  f"{n_kept + n_drop} SNPs ({n_drop} below threshold)",
                  file=sys.stderr, flush=True)

    with progress_stage("standardise_genotypes", status_file):
        Z_std = standardise_columns(Z_kept)

    if args.all_anchors:
        # --all-anchors: synthesise per_pheno_marginals so every kept SNP doubles as an anchor for every pheno. round-robin shard-slice happens below so each shard's marginals dict already has its share. paired with symmetric_pairs in scan_tier2 to skip the lower-triangle, giving M*(M-1)/2 unique pairs per pheno
        with progress_stage("synthesize_all_anchors", status_file,
                            n_anchors=len(keep_snp_id)):
            anchors_full = keep_snp_id
            if shard_n > 1:
                anchors_this_shard = anchors_full[shard_rank::shard_n]
                print(f"{prefix}--all-anchors shard slice: anchors "
                      f"[{shard_rank}::{shard_n}] = {len(anchors_this_shard)} / "
                      f"{len(anchors_full)}",
                      file=sys.stderr, flush=True)
            else:
                anchors_this_shard = anchors_full
            marginals = {nm: list(anchors_this_shard)
                          for nm in aligned.pheno_names}
            union = anchors_this_shard
            print(f"  all-anchors: {len(union)} anchor SNPs × "
                  f"{len(aligned.pheno_names)} phenos × {len(keep_snp_id)} test SNPs",
                  file=sys.stderr, flush=True)
    else:
        with progress_stage("read_marginals", status_file,
                            marginal_dir=args.marginal_dir,
                            top_k=args.marginal_top_k):
            marginals = read_marginals(
                args.marginal_dir, aligned.pheno_names, args.marginal_top_k)
            # filtering per-pheno marginal lists down to SNPs that survived the
            # MAF cut. SNPs picked by the marginal scan but flagged here would
            # silently drop inside scan_tier2 anyway, doing it explicitly here
            # so the user sees the count
            if args.maf_floor > 0.0:
                kept_set = set(keep_snp_id)
                n_marg_before = sum(len(v) for v in marginals.values())
                marginals = {nm: [s for s in sids if s in kept_set]
                              for nm, sids in marginals.items()}
                n_marg_after = sum(len(v) for v in marginals.values())
                if n_marg_before != n_marg_after:
                    print(f"  maf-filtered marginals: {n_marg_after} / "
                          f"{n_marg_before} per-pheno entries survived",
                          file=sys.stderr, flush=True)
            union = union_marginal_snps(marginals)
            print(f"  marginals: top-{args.marginal_top_k} per pheno × {len(marginals)} phenos "
                  f"→ {len(union)} unique anchor SNPs",
                  file=sys.stderr, flush=True)
            # comparing the marginal-dir manifest (if present) against the current epi
            # run. comparing pre-MAF n_variants because that's what the upstream gwas
            # saw. mismatches just warn — the user might be intentionally re-using
            # under different settings (different MAF, RINT toggle, swapped covar)
            check_marginal_manifest(
                args.marginal_dir,
                geno_prefix=Path(args.geno),
                n_strains=len(aligned.iid),
                n_variants=len(aligned.snp_id),
                maf_floor=args.maf_floor,
                rint=bool(args.rint),
                covar_path=Path(args.covar) if args.covar else None,
                rank_correct=False)

    # marginal-i enrichment: read PValue+SnpWeight per anchor and per-pheno threshold from the same marginal-dir. graceful degrade when the slim format lacks SnpWeight / threshold.txt — read_marginal_stats fills NaN, read_marginal_thresholds returns NaN per pheno; the tier-2 bundle then writes NaN into marginal_*_i and nothing else changes
    marginal_stats: dict[str, dict[str, tuple[float, float]]] | None = None
    marginal_thresholds: dict[str, float] | None = None
    if not args.all_anchors:
        with progress_stage("read_marginal_stats", status_file,
                            marginal_dir=args.marginal_dir):
            marginal_stats = read_marginal_stats(
                args.marginal_dir, aligned.pheno_names, snp_ids=union)
            marginal_thresholds = read_marginal_thresholds(
                args.marginal_dir, aligned.pheno_names)
            n_stats = sum(len(v) for v in marginal_stats.values())
            n_thr_present = sum(1 for v in marginal_thresholds.values() if v == v)
            n_beta_present = sum(
                1 for v in marginal_stats.values() for _, b in v.values() if b == b)
            print(f"  marginal stats: {n_stats:,} (pheno, anchor) entries, "
                  f"{n_beta_present:,} with beta, "
                  f"{n_thr_present:,} / {len(marginal_thresholds)} phenos have threshold",
                  file=sys.stderr, flush=True)

    if args.dry_run:
        n_phenos = aligned.Y.shape[1]
        buf_desc = (f"p_threshold={args.p_threshold:g} (unbounded)"
                    if args.p_threshold is not None
                    else f"top-{args.top_k} pair heap")
        anchor_desc = ("all-anchors symmetric" if args.all_anchors
                       else f"top-{args.marginal_top_k} marginal")
        print(f"{prefix}[dry run] would scan {n_phenos} phenotypes × "
              f"{len(union)} {anchor_desc} anchors × {len(keep_snp_id)} test SNPs "
              f"on {device} (LDCO, {buf_desc} per pheno, "
              f"n_perm={args.n_perm}).",
              file=sys.stderr, flush=True)
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    P_total = aligned.Y.shape[1]
    # pid + int shard_rank so epi-watch can attach VmHWM polling and pin nvidia-smi -i to the right physical gpu (CUDA_VISIBLE_DEVICES is set in the spawn child to shard_rank, so the rank IS the device idx)
    write_status(status_file, {"stage": "main", "stage_state": "running",
                               "phenos_total": P_total,
                               "n_strains": len(aligned.iid),
                               "n_variants": len(keep_snp_id),
                               "n_variants_pre_maf": len(aligned.snp_id),
                               "maf_floor": args.maf_floor,
                               "rint": bool(args.rint),
                               "n_perm": args.n_perm,
                               "perm_quantile": args.perm_quantile,
                               "n_marginals_unique": len(union),
                               "marginal_top_k": args.marginal_top_k,
                               "top_k_pair": args.top_k,
                               "p_threshold": args.p_threshold,
                               "all_anchors": bool(args.all_anchors),
                               "device": device,
                               "pid": os.getpid(),
                               "shard_rank": shard_rank, "shard_n": shard_n,
                               "shard": f"{shard_rank}/{shard_n}"})

    overall_t0 = time.time()
    requested_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    results = scan_tier2(
        Z_std=Z_std,
        Y=aligned.Y,
        X=aligned.X,
        chrom=keep_chrom,
        snp_id=keep_snp_id,
        pheno_names=aligned.pheno_names,
        per_pheno_marginals=marginals,
        per_pheno_marginal_stats=marginal_stats,
        per_pheno_marginal_threshold=marginal_thresholds,
        pos=keep_pos,
        top_k_pair=args.top_k,
        var_floor=args.var_floor,
        pheno_chunk=args.pheno_chunk,
        snp_chunk=args.snp_chunk,
        anchor_batch=args.anchor_batch,
        device=device,
        dtype=requested_dtype,
        status_file=status_file,
        n_perm=args.n_perm,
        perm_seed=args.perm_seed,
        perm_quantile=args.perm_quantile,
        exclude_window_kb=args.exclude_window_kb,
        use_fp16_inner=args.fp16_inner,
        p_threshold=args.p_threshold,
        symmetric_pairs=args.all_anchors)

    # writer pool: per-pheno gzip / parquet write fans out across processes.
    # bundle appends stay in the parent because pa_parquet.ParquetWriter
    # holds a file handle that doesn't pickle cleanly across the pool boundary
    from concurrent.futures import ProcessPoolExecutor

    bundle_writer: BundleWriter | None = None
    if args.bundle:
        bundle_path = (
            outdir / f"all_pairs.shard{shard_rank}.parquet"
            if shard_n > 1 else outdir / "all_pairs.parquet")
        bundle_writer = BundleWriter(bundle_path)
        print(f"{prefix}[bundle] streaming to {bundle_path}",
              file=sys.stderr, flush=True)

    write_futures: list = []
    writer_pool = ProcessPoolExecutor(max_workers=args.write_workers)

    def _drain_finished():
        nonlocal write_futures
        still_open = []
        for f in write_futures:
            if f.done():
                f.result()  # re-raises if the worker failed
            else:
                still_open.append(f)
        write_futures = still_open

    try:
        write_status(status_file, {"stage": "write", "stage_state": "running",
                                   "phenos_total": P_total,
                                   "writes_pending": len(results)})
        for r in results:
            if bundle_writer is not None:
                bundle_writer.append_pheno(r, keep_snp_id, keep_chrom, keep_pos)
            write_futures.append(writer_pool.submit(
                _write_one_for_pool,
                (r, keep_snp_id, keep_chrom, keep_pos,
                 str(outdir), 1, args.output_format,
                 args.per_pheno_dirs)))
            _drain_finished()

        if write_futures:
            with progress_stage("drain_writes", status_file,
                                writes_pending=len(write_futures)):
                for f in write_futures:
                    f.result()
                write_futures = []
    finally:
        writer_pool.shutdown(wait=True)
        if bundle_writer is not None:
            bundle_writer.close()

    total_elapsed = time.time() - overall_t0
    print(f"\n{prefix}gwas-epi: {P_total} phenotypes "
          f"× {len(union)} unique marginal anchors × {len(keep_snp_id)} test SNPs "
          f"in {total_elapsed:.1f}s on {device}",
          file=sys.stderr, flush=True)
    write_status(status_file, {"stage": "main", "stage_state": "done",
                               "elapsed_s": total_elapsed,
                               "phenos_total": P_total,
                               "shard": f"{shard_rank}/{shard_n}"})
    return 0


def _shard_entrypoint(rank: int, n: int, args_dict: dict) -> None:
    """Select one of the parent's visible GPUs and propagate the scan exit code."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[rank].strip() if visible else str(rank)
    args = argparse.Namespace(**args_dict)
    raise SystemExit(_run_one(args, shard_rank=rank, shard_n=n))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status_file is None and not args.dry_run:
        args.status_file = str(Path(args.outdir) / ".status.json")

    # explicit --shard means this process is a child subprocess, running just one shard
    if args.shard is not None:
        rank, n = _parse_shard(args.shard)
        return _run_one(args, shard_rank=rank, shard_n=n)

    # auto-dispatching across multiple GPUs if more than one is visible
    requested = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if requested == "cuda" else 0
    if n_gpus > 1 and not args.no_multi_gpu and not args.all_anchors:
        print(f"[multi-gpu] detected {n_gpus} CUDA devices; sharding phenotypes across them.",
              file=sys.stderr, flush=True)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        args_dict = vars(args).copy()
        # pinning device='cuda' in children so they don't redetect and break (each only sees one GPU)
        args_dict["device"] = "cuda"
        procs = []
        for r in range(n_gpus):
            p = ctx.Process(target=_shard_entrypoint, args=(r, n_gpus, args_dict))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        if any(p.exitcode != 0 for p in procs):
            return 1

        # Merge the explicit files produced by this run's GPU workers.
        if args.bundle and not args.dry_run:
            from fasterlmm_lux.bundle import merge_bundle_shards
            outdir = Path(args.outdir)
            shard_paths = [outdir / f"all_pairs.shard{r}.parquet" for r in range(n_gpus)]
            final = outdir / "all_pairs.parquet"
            merge_bundle_shards(shard_paths, final, remove_shards=True)
            print(f"[bundle] merged {n_gpus} shards into {final}",
                  file=sys.stderr, flush=True)
        return 0

    return _run_one(args, shard_rank=0, shard_n=1)


if __name__ == "__main__":
    raise SystemExit(main())
