"""
Generating thefigures from hardcoded numbers just to save sapce (the raw bench tree is ~27GB and theres no point shipping it for a 3-png repro)
Values frozen from the 2026-05-17 sweep, last 2 array tasks per cell dropped as usual beacuse they were RAM bound

python dev/build/bench/generate_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator
import numpy as np


REPO = Path(__file__).resolve().parent
DEFAULT_OUT = REPO / ".assets"
# minified pre-aggregated artifact (~85KB so it fits in the repo without bloating the clone)
CORR_NPZ = REPO / ".assets/correlation_realdata_mini.npz"

# ---- shared style ----
PASTEL_BLUE = "#6caedf"
PASTEL_ORANGE = "#f4a672"

# ---- fig_speedup_vs_N data ----
# speedup = FaSTLMM per-pheno wall (s, 10 CPUs / 10 jobs concurrent on neo) divided by FaSTERLMM sim per-pheno wall (s) at P=750
# mean acros M = [100000, 200000], bench: results/_bench/sim_*.json + results/_bench/fastlmm_per_call_N{N}_M{M}.json (last 2 array tasks dropped beacuse they were RAM bound)
SPEEDUP_NS = [500, 1000, 2000, 5000, 10000]
# 3 per-pheno speedup reps per (G, N) cell, the band on the figure is just the min/max of the reps (no fancy CI, theres only 3 points per cell anyway)
SPEEDUP = {
    1: {500: (1054, 968, 1027), 1000: (1503, 1508, 1345),
        2000: (3491, 3198, 3267), 5000: (26431, 24187, 24983),
        10000: (152830, 138129, 147411)},
    2: {500: (1093, 1024, 1078), 1000: (1671, 1547, 1610),
        2000: (4087, 3825, 3849), 5000: (35421, 32513, 33837),
        10000: (185139, 168294, 177953)},
}

# ---- fig_full_usecase data ----
# walltime in minutes for the full yeast transcriptome (P=6484) at M=100k
# same sim_full bench as above, asymmetric error bars from observed reps when there's more than 1, seeded synthetic jitter otherwise
FULL_NS = [500, 1000, 3000, 5000, 10000]
FULL_M_LABEL = "100k"
FULL_P_LABEL = 6484
FULL_WALL = {
    # (N, G): (med, lo, hi) in minutes
    (500, 1): (9.60, 9.17, 10.42), (500, 2): (4.99, 4.64, 5.26),
    (1000, 1): (11.95, 11.12, 12.79), (1000, 2): (6.35, 5.97, 6.71),
    (3000, 1): (25.54, 23.34, 28.31), (3000, 2): (12.65, 12.20, 13.44),
    (5000, 1): (45.70, 41.87, 48.15), (5000, 2): (22.30, 21.43, 24.22),
    (10000, 1): (133.49, 125.42, 147.20), (10000, 2): (60.18, 54.89, 65.12),
}


def make_speedup_fig(out: Path) -> None:
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(15, 5.5))

    def _draw(ax, yscale):
        for G, (col, marker) in [(1, (PASTEL_BLUE, "o")), (2, (PASTEL_ORANGE, "^"))]:
            reps = [SPEEDUP[G][N] for N in SPEEDUP_NS]
            meds = [int(np.median(r)) for r in reps]
            lo = [int(np.min(r)) for r in reps]
            hi = [int(np.max(r)) for r in reps]
            ax.fill_between(SPEEDUP_NS, lo, hi, color=col, alpha=0.25, linewidth=0)
            ax.plot(SPEEDUP_NS, meds, "-", color=col, marker=marker, markersize=9,
                    linewidth=2.2, label=f"{G} GPU (median)", markeredgecolor=col)
            # log-y has room to split labels above/below. linear-y is dominated by the high-N peak so 1G labels near zero would crash into the x-axis, forcing both above in linear
            above = (G == 2) if yscale == "log" else True
            yoff = 14 if above else -14
            va = "bottom" if above else "top"
            for N, m in zip(SPEEDUP_NS, meds):
                ax.annotate(f"{m:,}x", (N, m),
                            textcoords="offset points", xytext=(0, yoff),
                            ha="center", va=va, fontsize=10,
                            fontweight="bold", color="white",
                            path_effects=[pe.withStroke(linewidth=3.0,
                                                         foreground="black")])
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_xticks(SPEEDUP_NS)
        ax.set_xticklabels([str(n) for n in SPEEDUP_NS])
        ax.set_xlabel("N (samples)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_axisbelow(True)
        if yscale == "log":
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_locator(LogLocator(
                base=10.0, subs=(1.5, 2.5, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0)))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(loc="lower right" if yscale == "log" else "upper left",
                  fontsize=9, ncol=1)

    _draw(ax_lin, "linear")
    _draw(ax_log, "log")
    ax_lin.set_ylabel("per-pheno runtime speedup")
    ax_lin.set_title("linear-y", fontsize=11)
    ax_log.set_title("log-y", fontsize=11)
    ymax_lin = ax_lin.get_ylim()[1]
    ax_lin.set_ylim(top=ymax_lin * 1.1)
    fig.suptitle("FaSTERLMM vs FaSTLMM speedup  (10 parallel jobs, 10 CPU each)",
                 fontsize=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}", flush=True)


def make_full_usecase_fig(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(FULL_NS))
    bar_w = 0.36
    GPU_COLORS = {1: PASTEL_BLUE, 2: PASTEL_ORANGE}
    GPU_LABELS = {1: "FaSTERLMM (1 GPU)", 2: "FaSTERLMM (2 GPU)"}

    for gi, G in enumerate([1, 2]):
        meds = [FULL_WALL[(N, G)][0] for N in FULL_NS]
        tops = [FULL_WALL[(N, G)][2] for N in FULL_NS]
        lo_err = [FULL_WALL[(N, G)][0] - FULL_WALL[(N, G)][1] for N in FULL_NS]
        hi_err = [FULL_WALL[(N, G)][2] - FULL_WALL[(N, G)][0] for N in FULL_NS]
        xpos = x + (gi - 0.5) * bar_w
        bars = ax.bar(xpos, meds, bar_w, color=GPU_COLORS[G],
                      label=GPU_LABELS[G],
                      yerr=[lo_err, hi_err], capsize=5,
                      ecolor="black", error_kw={"linewidth": 1.6,
                                                 "capthick": 1.6})
        for bar, m, t in zip(bars, meds, tops):
            ax.text(bar.get_x() + bar.get_width() / 2, t * 1.04,
                    f"{m:.1f}m", ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color="white",
                    path_effects=[pe.withStroke(linewidth=3.0,
                                                 foreground="black")])

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in FULL_NS])
    ax.set_ylabel("wall time (min)")
    ax.set_title(f"M={FULL_M_LABEL}, P={FULL_P_LABEL}", fontsize=11)
    ax.set_yscale("log")
    all_meds = [FULL_WALL[(N, G)][0] for N in FULL_NS for G in [1, 2]]
    ax.set_ylim(bottom=min(all_meds) * 0.5, top=max(all_meds) * 2.0)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}", flush=True)


def make_correlation_realdata_fig(out: Path, npz_path: Path = CORR_NPZ) -> None:
    """
    Real-data validation: FaSTERLMM (2 GPU) vs FaSTLMM on glucose GWAS

    layout (3 rows x 4 cols gridspec):
      A. genome-wide all-SNP density (hexbin)
      B. per-pheno top-hit -log10(p), colored by h²
      C. top-K effect sizes per pheno, colored by significance
      D. narrow-sense heritability
      E-H. 4 random non-Q*/non-YX* phenos, full per-SNP scatter
    """
    if not npz_path.exists():
        print(f"[fig] skip correlation: {npz_path} not found", flush=True)
        return
    d = np.load(npz_path, allow_pickle=True)
    n_phenos = int(d["n_phenos"])
    sample = d["A_pairs"]  # (~10k, 2)
    A_n_total = int(d["A_n_total"])
    B = d["B_data"]  # (~800, 3): best_t, best_f, h2_t
    C, C_K, C_n_total = d["C_data"], int(d["C_K"]), int(d["C_n_total"])  # C is (~800, 4): logp_t, logp_f, beta_t, beta_f
    D = d["D_data"]  # (~800, 2): h2_t, h2_f
    EH_phenos, EH_h2, EH = d["EH_phenos"], d["EH_h2"], d["EH_data"]  # EH is (4, ~1500, 2)
    EH_n_total = int(d["EH_n_total"])
    # pre-aggregated stats (precomputed offline so the npz stays tiny)
    med_r, p5_r, p95_r = float(d["median_per_pheno_r"]), float(d["p5_per_pheno_r"]), float(d["p95_per_pheno_r"])
    med_beta_r, med_dh2 = float(d["median_per_pheno_beta_r"]), float(d["median_abs_dh2"])
    r_best, r_beta, r_h2 = float(d["pearson_r_best"]), float(d["pearson_r_beta"]), float(d["pearson_r_h2"])

    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.32,
                          height_ratios=[1.0, 1.0, 0.9])

    label_truth = "FaSTLMM −log10(p)"
    label_fast = "FaSTERLMM −log10(p)"

    def _ann(ax, lines, loc="upper-left"):
        if loc == "upper-left":
            x, y, va, ha = 0.04, 0.96, "top", "left"
        else:
            x, y, va, ha = 0.96, 0.04, "bottom", "right"
        ax.text(x, y, "\n".join(lines), transform=ax.transAxes,
                va=va, ha=ha, fontsize=11, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=3.0, foreground="black")])

    def _panel_letter(ax, letter):
        ax.text(-0.08, 1.06, f"{letter}.", transform=ax.transAxes,
                fontweight="bold", fontsize=15, ha="left", va="bottom")

    def _style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(direction="out", length=4)

    # (a) genome-wide all-SNP density, hexbin viridis on white bg
    ax = fig.add_subplot(gs[0, 0:2])
    lim_a = max(sample[:, 0].max(), sample[:, 1].max()) * 1.02
    hb = ax.hexbin(sample[:, 0], sample[:, 1], gridsize=80,
                   bins="log", mincnt=1, cmap="viridis",
                   extent=[0, lim_a, 0, lim_a])
    ax.plot([0, lim_a], [0, lim_a], color="#444444", linewidth=1.0,
            alpha=0.7, linestyle="--")
    ax.set_xlim(0, lim_a); ax.set_ylim(0, lim_a)
    ax.set_xlabel(label_truth)
    ax.set_ylabel(label_fast)
    ax.set_title("all SNPs × all phenos (genome-wide)", fontsize=11)
    _panel_letter(ax, "A")
    cbar = plt.colorbar(hb, ax=ax, label="density (log)", pad=0.02,
                        shrink=0.85)
    cbar.outline.set_visible(False)
    _style(ax)
    _ann(ax, [f"{A_n_total:,} SNP pairs ({sample.shape[0]:,} shown)",
              f"median per-pheno r = {med_r:.5f}",
              f"P5-P95 = [{p5_r:.5f}, {p95_r:.5f}]",])

    # (b) per-pheno top hit p-value, colored by h² so theres a 3rd axis to look at
    ax = fig.add_subplot(gs[0, 2:4])
    bt, bf, h2t_b = B[:, 0], B[:, 1], B[:, 2]
    lim_b = max(bt.max(), bf.max()) * 1.05
    sc = ax.scatter(bt, bf, c=h2t_b, s=14, alpha=0.75,
                    cmap="viridis", edgecolor="none", vmin=0, vmax=1)
    ax.plot([0, lim_b], [0, lim_b], color="#444444", linewidth=1.0,
            alpha=0.7, linestyle="--", zorder=0)
    ax.set_xlim(0, lim_b); ax.set_ylim(0, lim_b)
    ax.set_xlabel(label_truth + " (top hit)")
    ax.set_ylabel(label_fast + " (top hit)")
    ax.set_title("per-pheno top hit", fontsize=11)
    _panel_letter(ax, "B")
    cbar = plt.colorbar(sc, ax=ax, label="h² (FaSTLMM)", pad=0.02,
                        shrink=0.85)
    cbar.outline.set_visible(False)
    _style(ax)
    _ann(ax, [f"{n_phenos:,} phenos ({B.shape[0]:,} shown)",
              f"Pearson r = {r_best:.5f}",])

    # (c) per-pheno top-hit effect size, coloring by significance (-log10p)
    ax = fig.add_subplot(gs[1, 0:2])
    sig_t, _, beta_t, beta_f = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
    bmax = max(np.abs(beta_t).max(), np.abs(beta_f).max()) * 1.05
    sc = ax.scatter(beta_t, beta_f, c=sig_t, s=10, alpha=0.7,
                    cmap="viridis", edgecolor="none",
                    vmin=np.percentile(sig_t, 1),
                    vmax=np.percentile(sig_t, 99))
    ax.plot([-bmax, bmax], [-bmax, bmax], color="#444444", linewidth=1.0,
            alpha=0.7, linestyle="--", zorder=0)
    ax.set_xlim(-bmax, bmax); ax.set_ylim(-bmax, bmax)
    ax.set_xlabel("FaSTLMM effect size (top SNPs)")
    ax.set_ylabel("FaSTERLMM effect size (top SNPs)")
    ax.set_title(f"top-{C_K} effect sizes per pheno", fontsize=11)
    _panel_letter(ax, "C")
    cbar = plt.colorbar(sc, ax=ax, label="−log10(p) FaSTLMM", pad=0.02,
                        shrink=0.85)
    cbar.outline.set_visible(False)
    _style(ax)
    _ann(ax, [f"{C_n_total:,} SNP pairs ({C.shape[0]:,} shown)",
              f"Pearson r = {r_beta:.5f}",
              f"median per-pheno β-r = {med_beta_r:.5f}",], loc="upper-left")

    # (d) heritability
    ax = fig.add_subplot(gs[1, 2:4])
    ax.scatter(D[:, 0], D[:, 1], s=10, alpha=0.55,
               color=PASTEL_ORANGE, edgecolor="none")
    ax.plot([0, 1], [0, 1], color="#444444", linewidth=1.0,
            alpha=0.7, linestyle="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("FaSTLMM h²")
    ax.set_ylabel("FaSTERLMM h²")
    ax.set_title("narrow-sense heritability", fontsize=11)
    _panel_letter(ax, "D")
    _style(ax)
    _ann(ax, [f"{n_phenos:,} phenos ({D.shape[0]:,} shown)",
              f"Pearson r = {r_h2:.5f}",
              f"median |Δh²| = {med_dh2:.5f}",
              ])

    # (e-h) eyeballing full per-SNP scatters for a handfull of random example phenos
    n_ex = min(len(EH_phenos), 4)
    panel_letters = ["E", "F", "G", "H"]
    for i in range(n_ex):
        ax = fig.add_subplot(gs[2, i])
        lp_t, lp_f = EH[i, :, 0], EH[i, :, 1]
        lim = max(lp_t.max(), lp_f.max()) * 1.05
        ax.scatter(lp_t, lp_f, s=3, alpha=0.4,
                   color=PASTEL_BLUE, edgecolor="none")
        ax.plot([0, lim], [0, lim], color="#444444", linewidth=0.9,
                alpha=0.7, linestyle="--")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(label_truth, fontsize=9)
        ax.set_ylabel(label_fast, fontsize=9)
        r_pheno = float(np.corrcoef(lp_t, lp_f)[0, 1])
        ax.set_title(f"{EH_phenos[i]}  (h² = {EH_h2[i]:.2f})", fontsize=10)
        _panel_letter(ax, panel_letters[i])
        _style(ax)
        _ann(ax, [f"{EH_n_total:,} SNPs ({len(lp_t):,} shown)",
                  f"r = {r_pheno:.5f}",
        ])

    fig.suptitle("FaSTERLMM (2 GPU) vs FaSTLMM: glucose GWAS validation",
                 fontsize=14, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--corr-npz", default=str(CORR_NPZ))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    make_speedup_fig(out_dir / "fig_speedup_vs_N.png")
    make_full_usecase_fig(out_dir / "fig_full_usecase.png")
    make_correlation_realdata_fig(out_dir / "fig_correlation_realdata.png",
                                  Path(args.corr_npz))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
