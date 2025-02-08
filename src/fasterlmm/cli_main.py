"""
Entry: dispatchs to the gwas and watch subcommands
Usage:
  fasterlmm gwas --geno ... --pheno ... --outdir ...
  fasterlmm watch <status.json>
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

run `fasterlmm <subcommand> --help` for arguments
"""


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
    else:
        print(f"unknown subcommand: {sub}\n", file=sys.stderr)
        print(_BANNER, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
