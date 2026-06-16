"""
figures/fig_trust.py  —  trust-signal figures from the classified corpus.
Run on the SERVER:
    python figures/fig_trust.py --input classified/classified_v3_300k.parquet \
        --out-dir reports/thesis_figures

Produces:
    fig20_centroid_sim_hist.png   distribution of pred_centroid_sim + tier bands
    fig21_conf_vs_sim.png         conf_div vs pred_centroid_sim (2-D density):
                                  shows conf_div is saturated/useless while
                                  centroid_sim spreads — the over-confidence story
    fig22_trust_tier_donut.png    share of the corpus in each trust tier
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC = "#2f6fed"
TIERS = [("reject", 0, .40, "#d9534f"), ("low", .40, .57, "#e6a417"),
         ("medium", .57, .67, "#3b82c4"), ("trust", .67, 1.01, "#2e8b57")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="classified/classified_v3_300k.parquet")
    ap.add_argument("--out-dir", default="reports/thesis_figures")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    pf = pq.ParquetFile(args.input)
    sim_parts, conf_parts = [], []
    for b in pf.iter_batches(batch_size=200000, columns=["pred_centroid_sim", "conf_div"]):
        d = b.to_pandas()
        sim_parts.append(d["pred_centroid_sim"].to_numpy(dtype="float32"))
        conf_parts.append(d["conf_div"].to_numpy(dtype="float32"))
    sim = np.concatenate(sim_parts); conf = np.concatenate(conf_parts)
    ok = ~np.isnan(sim); sim, conf = sim[ok], conf[ok]
    print(f"  {len(sim):,} docs")

    # ── Fig 20 — centroid-sim distribution with tier bands ───────────────────
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.hist(sim, bins=80, color=ACC, alpha=.85)
    for name, lo, hi, col in TIERS:
        ax.axvspan(lo, hi, color=col, alpha=.08)
        frac = ((sim >= lo) & (sim < hi)).mean()
        ax.text((lo + min(hi, 1.0)) / 2, ax.get_ylim()[1] * .92, f"{name}\n{frac:.0%}",
                ha="center", va="top", fontsize=8, color=col)
    ax.axvline(np.median(sim), color="#222", ls="--", lw=1, label=f"median {np.median(sim):.2f}")
    ax.set(xlabel="pred_centroid_sim (trust signal)", ylabel="documents",
           title="Distribution of the trust signal across the corpus")
    ax.legend(loc="upper left"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig20_centroid_sim_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  ✅ fig20_centroid_sim_hist.png")

    # ── Fig 21 — conf_div vs centroid_sim (the over-confidence figure) ───────
    fig, ax = plt.subplots(figsize=(7, 5.2))
    hb = ax.hexbin(sim, conf, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    fig.colorbar(hb, label="documents (log)")
    ax.set(xlabel="pred_centroid_sim (geometric trust)",
           ylabel="conf_div (softmax confidence)",
           title="Softmax confidence is saturated; centroid similarity is informative")
    ax.text(.02, .06, f"conf_div median = {np.median(conf):.3f}\n"
                      f"centroid_sim median = {np.median(sim):.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=.85))
    fig.tight_layout(); fig.savefig(out / "fig21_conf_vs_sim.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  ✅ fig21_conf_vs_sim.png")

    # ── Fig 22 — trust-tier donut ────────────────────────────────────────────
    sizes = [((sim >= lo) & (sim < hi)).sum() for _, lo, hi, _ in TIERS]
    cols = [c for *_, c in TIERS]
    labels = [f"{n}\n{s/len(sim):.0%}" for (n, *_), s in zip(TIERS, sizes)]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pie(sizes, labels=labels, colors=cols, startangle=90,
           wedgeprops=dict(width=.42, edgecolor="white"), textprops=dict(fontsize=10))
    ax.set(title="Corpus by trust tier")
    fig.tight_layout(); fig.savefig(out / "fig22_trust_tier_donut.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  ✅ fig22_trust_tier_donut.png")

    print(f"\n  trust figures → {out}/")


if __name__ == "__main__":
    main()
