"""
Entry: dispatchs to the gwas, watch and concat subcommands
Usage:
  fasterlmm gwas --geno ... --pheno ... --outdir ...
  fasterlmm watch <status.json>
  fasterlmm concat <outdir>
Buidling this as the new single entry so the dashed forms (fasterlmm-gwas, fasterlmm-watch) can go away
"""

from __future__ import annotations

import sys

from fasterlmm import __version__

_BANNER = f"""\
fasterlmm {__version__}  torch port of fastlmm

subcommands:
  gwas -> single-trait LOCO GWAS with permutation threshold
  watch -> live TUI for a running gwas job
  concat -> gather a slurm-array run's bundle shards into the gwas_bundle.parquet dataset

run `fasterlmm <subcommand> --help` for arguments
"""


def _concat() -> None:
    """
    fasterlmm concat <outdir>: gather the per-shard bundle parts into the gwas_bundle.parquet dataset
    A single-job multi-gpu run folds this step in itself, so concat is really for the slurm-array case -- there each --shard task drops its own part under .bundle_parts/ and no parent process is around to gather them.  The merge is just a few renames, instant and cpu-only, so run it whenever, wherever once the array has finished
    """
    import argparse
    import time

    from fasterlmm.bundle import merge_bundle_parts

    p = argparse.ArgumentParser(
        prog="fasterlmm concat",
        description="gather the per-shard bundle parts a slurm-array gwas run streamed under "
                    ".bundle_parts/ into the gwas_bundle.parquet dataset (cpu only, instant)")
    p.add_argument("outdir", help="the gwas --outdir, the one holding .bundle_parts/")
    p.add_argument("--out", default=None,
                   help="bundle dataset path to write (default <outdir>/gwas_bundle.parquet)")
    args = p.parse_args()
    t0 = time.time()
    path = merge_bundle_parts(args.outdir, out_path=args.out)
    print(f"wrote {path} in {time.time() - t0:.0f}s")


def main() -> None:
    """
    Pop the first argv, route to the right subcommand
    Rewritting sys.argv[0] so the subcommand's own argparse prints "fasterlmm gwas" in its usage line rather than the umbrella name, less confusig for someone reading --help
    """
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(_BANNER)
        return
    sub = argv[0]
    sys.argv = [f"fasterlmm {sub}"] + argv[1:]
    if sub == "gwas":
        from fasterlmm.cli import main as gwas_main
        gwas_main()
    elif sub == "watch":
        from fasterlmm.watch import main as watch_main
        watch_main()
    elif sub == "concat":
        _concat()
    else:
        print(f"unknown subcommand: {sub}\n", file=sys.stderr)
        print(_BANNER, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
