"""
pipeline/15_plot_journal_validation.py
=======================================
Visualise the journal/publisher proxy-validation results (Chapter 23) so the
findings can be eyeballed. Reads the artefacts produced by 14_validate_journals.py:
    <dir>/journal_lcc_map.csv     per-journal: n_papers, dominant_lcc, purity, ...
    <dir>/journal_validation.txt  parsed for the centroid-sim decile curves + AMI

Produces (→ <dir>/plots/):
    1. purity_hist.png        distribution of per-journal top-LCC purity
    2. purity_vs_size.png     purity vs journal size — specialists concentrate,
                              megajournals scatter (the named outliers)
    3. ood_decile_curve.png   journal/publisher purity vs pred_centroid_sim decile
                              — THE validation plot (monotonic ⇒ OOD score works)
    4. ood_quartile_bars.png  high- vs low-sim AMI & purity (the cross-check)
    5. dominant_lcc_bar.png   how journals distribute across LCC subclasses

Usage:
    python pipeline/15_plot_journal_validation.py \
        --dir reports/journal_validation_v3_300k
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── parse the text report ────────────────────────────────────────────────────
def parse_report(txt: str) -> dict:
    """Pull the per-group decile curves, OOD quartiles and AMI out of the report."""
    out = {}
    # split into GROUP = journal / GROUP = publisher blocks
    blocks = re.split(r"GROUP = (\w+)", txt)
    # blocks = [preamble, 'journal', body, 'publisher', body, ...]
    for i in range(1, len(blocks), 2):
        group, body = blocks[i], blocks[i + 1]
        g = {}
        m = re.search(r"AMI\([^)]+\)\s*=\s*([\d.]+)", body)
        g["ami"] = float(m.group(1)) if m else None
        m = re.search(r"LIFT vs random\s*=\s*([\d.]+)", body)
        g["lift"] = float(m.group(1)) if m else None
        # OOD quartiles
        hi = re.search(r"high-sim \(>= ([\d.]+)\): AMI=([\d.]+)\s+purity=([\d.]+)", body)
        lo = re.search(r"low-sim\s+\(<= ([\d.]+)\): AMI=([\d.]+)\s+purity=([\d.]+)", body)
        if hi and lo:
            g["ood"] = dict(
                hi_t=float(hi.group(1)), hi_ami=float(hi.group(2)), hi_pur=float(hi.group(3)),
                lo_t=float(lo.group(1)), lo_ami=float(lo.group(2)), lo_pur=float(lo.group(3)),
            )
        # decile rows: "decile  0  sim~[0.06,0.52]  purity=0.492  n=356,659"
        rows = re.findall(
            r"decile\s+(\d+)\s+sim~\[([\d.]+),([\d.]+)\]\s+purity=([\d.]+)\s+n=([\d,]+)", body)
        if rows:
            g["deciles"] = pd.DataFrame(
                [(int(a), float(b), float(c), float(d), int(e.replace(",", "")))
                 for a, b, c, d, e in rows],
                columns=["decile", "sim_lo", "sim_hi", "purity", "n"])
        out[group] = g
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="reports/journal_validation_* directory")
    args = ap.parse_args()

    d = Path(args.dir)
    plots = d / "plots"; plots.mkdir(exist_ok=True)
    jmap = pd.read_csv(d / "journal_lcc_map.csv")
    rep = parse_report((d / "journal_validation.txt").read_text())

    # ── 1. purity histogram ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(jmap["purity"], bins=40, color="#4c8bf5", alpha=0.85, edgecolor="white")
    med, mean = jmap["purity"].median(), jmap["purity"].mean()
    ax.axvline(med, color="#d9534f", ls="--", label=f"median {med:.2f}")
    ax.axvline(mean, color="#222", ls=":", label=f"mean {mean:.2f}")
    ax.axvline(0.8, color="green", ls="--", alpha=0.5,
               label=f"≥0.8: {(jmap['purity']>=0.8).mean():.0%} of journals")
    ax.set(title=f"Per-journal top-LCC purity  (n={len(jmap):,} journals, ≥20 papers)",
           xlabel="purity = share of journal's papers in its dominant LCC subclass",
           ylabel="# journals")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots / "purity_hist.png", dpi=150); plt.close(fig)

    # ── 2. purity vs size (the specialists-vs-megajournals story) ─────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(jmap["n_papers"], jmap["purity"], s=10, alpha=0.25, color="#4c8bf5")
    ax.set(xscale="log", title="Purity vs journal size — specialists concentrate, generalists scatter",
           xlabel="# papers in journal (log)", ylabel="top-LCC purity", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    # annotate notable low-purity megajournals + a few pure specialists
    notable_low = ["Nature Communications", "Science Advances", "Applied Sciences",
                   "Proceedings of the Royal Society of London"]
    big_pure = jmap[(jmap.n_papers >= 500) & (jmap.purity >= 0.98)].nlargest(4, "n_papers")
    for _, r in jmap[jmap["journal"].isin(notable_low)].iterrows():
        ax.annotate(str(r["journal"])[:28], (r["n_papers"], r["purity"]),
                    fontsize=7, color="#d9534f",
                    xytext=(0, -10), textcoords="offset points")
        ax.scatter([r["n_papers"]], [r["purity"]], s=40, color="#d9534f", zorder=5)
    for _, r in big_pure.iterrows():
        ax.annotate(str(r["journal"])[:24], (r["n_papers"], r["purity"]),
                    fontsize=7, color="green", xytext=(0, 5), textcoords="offset points")
        ax.scatter([r["n_papers"]], [r["purity"]], s=40, color="green", zorder=5)
    fig.tight_layout(); fig.savefig(plots / "purity_vs_size.png", dpi=150); plt.close(fig)

    # ── 3. OOD decile curve (THE validation plot) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for group, color in [("journal", "#4c8bf5"), ("publisher", "#f5a623")]:
        g = rep.get(group, {})
        if "deciles" in g:
            dec = g["deciles"]
            x = (dec["sim_lo"] + dec["sim_hi"]) / 2
            ax.plot(x, dec["purity"], "o-", color=color,
                    label=f"{group} (AMI={g.get('ami')})")
    ax.set(title="Prediction quality rises with the OOD score (pred_centroid_sim)\n"
                 "→ validates BOTH the model and the centroid-sim trust signal",
           xlabel="pred_centroid_sim (decile midpoint)",
           ylabel="paper-weighted top-LCC purity")
    ax.axvspan(0.0, 0.40, color="red", alpha=0.06)
    ax.axvspan(0.40, 0.57, color="orange", alpha=0.06)
    ax.axvspan(0.57, 0.67, color="yellow", alpha=0.06)
    ax.axvspan(0.67, 1.0, color="green", alpha=0.06)
    ax.text(0.62, ax.get_ylim()[0], "review→trust", fontsize=7, alpha=0.6)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots / "ood_decile_curve.png", dpi=150); plt.close(fig)

    # ── 4. OOD quartile bars ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in [(axes[0], "ami", "AMI(group, LCC)"),
                              (axes[1], "pur", "top-LCC purity")]:
        labels, hi_vals, lo_vals = [], [], []
        for group in ("journal", "publisher"):
            ood = rep.get(group, {}).get("ood")
            if ood:
                labels.append(group)
                hi_vals.append(ood[f"hi_{metric}"]); lo_vals.append(ood[f"lo_{metric}"])
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x - w/2, lo_vals, w, label="low-sim (bottom 25%)", color="#d9534f")
        ax.bar(x + w/2, hi_vals, w, label="high-sim (top 25%)", color="green")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set(title=f"{title}: high vs low pred_centroid_sim")
        for xi, (l, h) in enumerate(zip(lo_vals, hi_vals)):
            ax.text(xi - w/2, l, f"{l:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + w/2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Cross-check: high-sim predictions are more group-consistent")
    fig.tight_layout(); fig.savefig(plots / "ood_quartile_bars.png", dpi=150); plt.close(fig)

    # ── 5. dominant-LCC distribution across journals ──────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    vc = jmap["dominant_lcc"].value_counts().head(30)
    ax.bar(range(len(vc)), vc.values, color="#4c8bf5")
    ax.set_xticks(range(len(vc))); ax.set_xticklabels(vc.index, rotation=45, ha="right")
    ax.set(title="LCC subclass each journal is dominantly assigned to (top 30)",
           xlabel="LCC subclass", ylabel="# journals")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(plots / "dominant_lcc_bar.png", dpi=150); plt.close(fig)

    print(f"✅ 5 plots → {plots}/")
    for p in sorted(plots.glob("*.png")):
        print(f"   {p.name}")


if __name__ == "__main__":
    main()
