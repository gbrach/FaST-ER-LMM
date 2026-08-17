"""run-fingerprint manifest dropped next to a marginal gwas dir.

so `fasterlmm gwas-epi` can fail fast when the marginal-dir was built
from a different .bed, a different MAF floor, a different covar file,
or a different fasterlmm version. the manifest sits at:

  <marginal_dir>/.run_manifest.json

writer dumps the params + sha256 of .bim and covar; reader / validator
return mismatched-field messages so the caller decides whether to error
or warn. stdlib only — hashlib, json, datetime, pathlib
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_FILENAME = ".run_manifest.json"


def _fasterlmm_version() -> str:
    """fasterlmm version from package metadata, falling back to "unknown"
    when the dist isn't installed (editable-but-not-pip-installed checkouts,
    pytest-from-source, etc)"""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("fasterlmm")
        except PackageNotFoundError:
            return "unknown"
    except ImportError:
        return "unknown"


def fingerprint_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """sha256 hex of a file, streamed in chunk_size blocks. used to fingerprint
    .bim and covar so silent in-place edits get caught"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _bim_path(geno_prefix: Path) -> Path:
    """PLINK trio convention: <prefix>.bim. fingerprint the .bim because
    .bed itself is huge and the genotype universe (snp_id + chrom + pos)
    fully determines the marginal-scan output schema"""
    return Path(str(geno_prefix) + ".bim")


def write_manifest(
    marginal_dir: Path,
    *,
    geno_prefix: Path, n_strains: int,
    n_variants: int, maf_floor: float,
    rint: bool, covar_path: Path | None,
    n_perm: int, perm_seed: int,
    rank_correct: bool, wall_seconds: float) -> Path:
    """dump the manifest to <marginal_dir>/.run_manifest.json, fingerprinting
    the .bim (always) and the covar file (when given). returns the manifest path.
    creates marginal_dir if missing"""
    marginal_dir = Path(marginal_dir)
    marginal_dir.mkdir(parents=True, exist_ok=True)
    geno_prefix = Path(geno_prefix).resolve()
    bim = _bim_path(geno_prefix)
    geno_fp = fingerprint_file(bim)
    if covar_path is not None:
        covar_abs: str | None = str(Path(covar_path).resolve())
        covar_fp: str | None = fingerprint_file(Path(covar_path))
    else:
        covar_abs = None
        covar_fp = None
    payload = {
        "fasterlmm_version": _fasterlmm_version(),
        "geno_prefix": str(geno_prefix),
        "geno_fingerprint": geno_fp,
        "n_strains": int(n_strains),
        "n_variants": int(n_variants),
        "maf_floor": float(maf_floor),
        "rint": bool(rint),
        "covar_path": covar_abs,
        "covar_fingerprint": covar_fp,
        "n_perm": int(n_perm),
        "perm_seed": int(perm_seed),
        "rank_correct": bool(rank_correct),
        "wall_seconds": float(wall_seconds),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = marginal_dir / MANIFEST_FILENAME
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out


def read_manifest(marginal_dir: Path) -> dict:
    """load the manifest. raises FileNotFoundError if absent"""
    path = Path(marginal_dir) / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}")
    with open(path) as f:
        return json.load(f)


def validate_against(
    manifest: dict,
    *,
    geno_prefix: Path, n_strains: int,
    n_variants: int, maf_floor: float,
    rint: bool, covar_path: Path | None,
    rank_correct: bool) -> list[str]:
    """check the current run's params against a manifest. returns mismatched-
    field messages (empty list = ok). caller decides error vs warn.

    fingerprints geno + covar so silent edits to .bim or the covar file
    show up as mismatches even when the path strings agree
    """
    msgs: list[str] = []
    geno_prefix = Path(geno_prefix).resolve()
    cur_geno_fp = fingerprint_file(_bim_path(geno_prefix))
    if manifest.get("geno_fingerprint") != cur_geno_fp:
        msgs.append(
            f"geno .bim fingerprint changed: manifest={manifest.get('geno_fingerprint')!r} "
            f"current={cur_geno_fp!r}")
    if manifest.get("n_strains") != int(n_strains):
        msgs.append(
            f"n_strains mismatch: manifest={manifest.get('n_strains')!r} current={int(n_strains)!r}")
    if manifest.get("n_variants") != int(n_variants):
        msgs.append(
            f"n_variants mismatch: manifest={manifest.get('n_variants')!r} current={int(n_variants)!r}")
    if manifest.get("maf_floor") != float(maf_floor):
        msgs.append(
            f"maf_floor mismatch: manifest={manifest.get('maf_floor')!r} current={float(maf_floor)!r}")
    if manifest.get("rint") != bool(rint):
        msgs.append(
            f"rint mismatch: manifest={manifest.get('rint')!r} current={bool(rint)!r}")
    if manifest.get("rank_correct") != bool(rank_correct):
        msgs.append(
            f"rank_correct mismatch: manifest={manifest.get('rank_correct')!r} "
            f"current={bool(rank_correct)!r}")

    # covar: 4 cases — both none, manifest none + current set, manifest set + current none, both set (compare path + fingerprint)
    manifest_covar = manifest.get("covar_path")
    if covar_path is None and manifest_covar is None:
        pass
    elif covar_path is not None and manifest_covar is None:
        msgs.append(
            f"covar mismatch: manifest had no covar, current run uses {str(Path(covar_path).resolve())!r}")
    elif covar_path is None and manifest_covar is not None:
        msgs.append(
            f"covar mismatch: manifest had {manifest_covar!r}, current run has no covar")
    else:
        cur_covar_abs = str(Path(covar_path).resolve())
        if manifest_covar != cur_covar_abs:
            msgs.append(
                f"covar_path mismatch: manifest={manifest_covar!r} current={cur_covar_abs!r}")
        cur_covar_fp = fingerprint_file(Path(covar_path))
        if manifest.get("covar_fingerprint") != cur_covar_fp:
            msgs.append(
                f"covar fingerprint changed: manifest={manifest.get('covar_fingerprint')!r} "
                f"current={cur_covar_fp!r}")

    return msgs
