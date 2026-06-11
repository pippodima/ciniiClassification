"""
pipeline/17_score_vs_journal_gold.py
=====================================
Score the classifier against the INDEPENDENT journal→LCC gold map produced by
hand-labeling journal_gold_template.csv (16_build_journal_gold.py). This is the
defensible accuracy number for the thesis — unlike the circular val-accuracy,
the gold labels come from journal scope, which the model never saw.

For every paper whose journal is in the gold map (and not left blank / skipped),
we compare the predicted LCC to the journal's true LCC at three levels:
    main      pred_lcc_main vs gold_lcc_main   (derived from gold_lcc_sub[0] if blank)
    subclass  pred_lcc      vs gold_lcc_sub
    division  pred_lcc_div  vs gold_lcc_div     (only where gold_lcc_div is filled)

Reports:
  • overall paper-weighted accuracy at each level
  • accuracy by pred_centroid_sim TIER  ← the payoff: real accuracy of the
    "trustworthy" slice vs the outlier slice (validates using the OOD dial)
  • per-journal accuracy table (which journals the model gets right/wrong)
  • biggest systematic disagreements: (gold_sub → pred_sub) pairs by volume
    — auto-surfaces things like Geophysical Research Letters gold=QE, pred=QC

Outputs (→ <out-dir>):
  gold_score.txt          full report (also printed)
  per_journal_score.csv   journal, n, acc_main, acc_sub, gold/pred dominant
  confusion_sub.csv       gold_sub, pred_sub, n_papers (disagreements first)

Usage:
    python pipeline/17_score_vs_journal_gold.py \
        --input classified/classified_v3_300k.parquet \
        --gold  reports/journal_validation_v3_300k/journal_gold_template.csv \
        --out-dir reports/journal_validation_v3_300k
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# MUST match the journal normalisation in 16_build_journal_gold.py exactly,
# or papers won't join to their gold row.
def _clean_title(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# pred_centroid_sim trust tiers (from Chapter 23 decile analysis)
TIERS = [
    ("reject  (<0.40)", 0.00, 0.40),
    ("low     (0.40-0.57)", 0.40, 0.57),
    ("medium  (0.57-0.67)", 0.57, 0.67),
    ("trust   (>=0.67)", 0.67, 1.01),
]


def acc(pred: pd.Series, gold: pd.Series) -> float:
    m = gold.notna() & (gold.astype(str).str.len() > 0)
    if m.sum() == 0:
        return float("nan")
    return float((pred[m].astype(str) == gold[m].astype(str)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="classified parquet")
    ap.add_argument("--gold", required=True, help="labeled journal_gold_template.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sim-col", default="pred_centroid_sim")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    def log(m=""): print(m); lines.append(m)

    # ── gold map ──────────────────────────────────────────────────────────────
    gold = pd.read_csv(args.gold, dtype=str).fillna("")
    for c in ("gold_lcc_main", "gold_lcc_sub", "gold_lcc_div"):
        if c not in gold.columns:
            gold[c] = ""
        gold[c] = gold[c].str.strip().str.upper()
    gold["journal_key"] = gold["journal"].map(_clean_title)
    # derive main from sub if main left blank
    gold.loc[gold["gold_lcc_main"] == "", "gold_lcc_main"] = \
        gold["gold_lcc_sub"].str[0].fillna("")
    labeled = gold[gold["gold_lcc_sub"] != ""].copy()
    skipped = gold[gold["gold_lcc_sub"] == ""]

    log("=" * 70)
    log("CLASSIFIER vs INDEPENDENT JOURNAL→LCC GOLD MAP")
    log("=" * 70)
    log(f"  gold journals total : {len(gold)}")
    log(f"  labeled (scored)    : {len(labeled)}")
    log(f"  blank/multidisc.    : {len(skipped)}  (excluded)")
    log("")

    gmap = labeled.set_index("journal_key")[
        ["gold_lcc_main", "gold_lcc_sub", "gold_lcc_div"]]

    # ── papers ────────────────────────────────────────────────────────────────
    cols = ["journal", "pred_lcc_main", "pred_lcc", "pred_lcc_div", args.sim_col]
    df = pd.read_parquet(args.input, columns=cols)
    df["journal_key"] = df["journal"].astype(str).map(_clean_title)
    df = df.join(gmap, on="journal_key", how="inner")
    log(f"  papers matched to a labeled journal : {len(df):,}")
    if len(df) == 0:
        log("  ⚠ no matches — check that gold journals exist in the parquet");
        (out_dir / "gold_score.txt").write_text("\n".join(lines)); return
    log("")

    # ── overall accuracy ───────────────────────────────────────────────────────
    a_main = acc(df["pred_lcc_main"], df["gold_lcc_main"])
    a_sub  = acc(df["pred_lcc"],      df["gold_lcc_sub"])
    a_div  = acc(df["pred_lcc_div"],  df["gold_lcc_div"])
    n_div  = (df["gold_lcc_div"] != "").sum()
    log("-" * 70)
    log("OVERALL paper-weighted accuracy (prediction vs journal-gold):")
    log(f"  main class  (Q/R/T/…) : {a_main:.3f}   n={len(df):,}")
    log(f"  subclass    (QD/RC/…) : {a_sub:.3f}   n={len(df):,}")
    log(f"  division    (QD411/…) : {a_div:.3f}   n={n_div:,}  (where gold_div filled)")
    log("")

    # ── accuracy by trust tier (the payoff) ────────────────────────────────────
    log("-" * 70)
    log(f"ACCURACY BY {args.sim_col} TIER  (subclass level)")
    log(f"  {'tier':<22}{'n_papers':>12}{'% corpus':>10}{'acc_sub':>10}{'acc_main':>10}")
    sim = df[args.sim_col]
    for name, lo, hi in TIERS:
        seg = df[(sim >= lo) & (sim < hi)]
        if len(seg) == 0:
            continue
        log(f"  {name:<22}{len(seg):>12,}{len(seg)/len(df):>9.1%}"
            f"{acc(seg['pred_lcc'], seg['gold_lcc_sub']):>10.3f}"
            f"{acc(seg['pred_lcc_main'], seg['gold_lcc_main']):>10.3f}")
    log("  → if acc rises with the tier, pred_centroid_sim is a valid trust dial")
    log("")

    # ── per-journal accuracy ───────────────────────────────────────────────────
    rows = []
    for jk, sub in df.groupby("journal_key"):
        rows.append({
            "journal": labeled.loc[labeled.journal_key == jk, "journal"].iloc[0],
            "n_papers": len(sub),
            "gold_sub": sub["gold_lcc_sub"].iloc[0],
            "pred_dominant_sub": sub["pred_lcc"].value_counts().index[0],
            "acc_sub": (sub["pred_lcc"] == sub["gold_lcc_sub"]).mean(),
            "acc_main": (sub["pred_lcc_main"] == sub["gold_lcc_main"]).mean(),
        })
    pj = pd.DataFrame(rows).sort_values("acc_sub")
    pj.to_csv(out_dir / "per_journal_score.csv", index=False)

    log("-" * 70)
    log("WORST-SCORING labeled journals (acc_sub, n>=100) — model vs gold:")
    for _, r in pj[pj.n_papers >= 100].head(15).iterrows():
        flag = "←disagree" if r.gold_sub != r.pred_dominant_sub else ""
        log(f"  acc={r.acc_sub:.2f}  n={int(r.n_papers):>6,}  "
            f"gold={r.gold_sub:<5} pred={r.pred_dominant_sub:<5} {flag}  "
            f"{str(r.journal)[:42]}")
    log("")

    # ── systematic confusions ──────────────────────────────────────────────────
    conf = (df[df["pred_lcc"] != df["gold_lcc_sub"]]
            .groupby(["gold_lcc_sub", "pred_lcc"]).size()
            .reset_index(name="n_papers").sort_values("n_papers", ascending=False))
    conf.to_csv(out_dir / "confusion_sub.csv", index=False)
    log("-" * 70)
    log("BIGGEST SYSTEMATIC DISAGREEMENTS (gold_sub → pred_sub, by volume):")
    for _, r in conf.head(15).iterrows():
        log(f"  {r.n_papers:>7,}  gold={r.gold_lcc_sub:<5} → pred={r.pred_lcc:<5}")
    log("")

    (out_dir / "gold_score.txt").write_text("\n".join(lines))
    log(f"  report → {out_dir/'gold_score.txt'}")
    log(f"  per-journal → {out_dir/'per_journal_score.csv'}")
    log(f"  confusion   → {out_dir/'confusion_sub.csv'}")


if __name__ == "__main__":
    main()
