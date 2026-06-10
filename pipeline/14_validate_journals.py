"""
pipeline/14_validate_journals.py
=================================
Proxy ground-truth validation of the LCC classifier using journal / publisher.

WHY
---
There is no manual gold set. But most academic journals are topically coherent,
so a *correct* classifier should assign papers from the same journal to the same
(few) LCC categories. We exploit that: journal identity is an external signal
the model never saw, so agreement between journal and predicted-LCC is genuine
(if indirect) evidence the predictions are meaningful — unlike the circular
val-accuracy (which only measures agreement with the clustering it was trained on).

WHAT IT REPORTS
---------------
1. Coverage — how many docs carry a usable journal / publisher.
2. Per-journal PURITY — top-LCC share + normalised entropy, distribution over
   journals (weighted by papers). Intuitive but gameable, so reported with a caveat.
3. Adjusted Mutual Information AMI(group, pred_lcc) — the HEADLINE metric.
   Chance-corrected, so a degenerate "predict RC for everything" model scores ~0.
   High AMI ⇒ journal strongly determines the prediction ⇒ predictions track real
   topical structure.
4. LIFT vs random baseline — observed top-LCC share / share expected if the
   journal's papers were drawn from the global LCC distribution. Another
   degeneracy-proof view.
5. OOD CROSS-CHECK (the validating one) — recompute purity & AMI on the top vs
   bottom centroid-sim docs. If high-`pred_centroid_sim` docs are markedly more
   journal-consistent than low-sim ones, that validates BOTH the model AND the
   OOD score in one shot, and tells us where to threshold it.
6. Worst-offender journals — high-volume, low-purity journals (model struggles
   or genuinely multidisciplinary).

OUTPUTS
-------
  <out-dir>/journal_validation.txt   full report (also printed)
  <out-dir>/journal_lcc_map.csv      journal → dominant LCC, purity, entropy, n
                                     (also useful as a seed for the search UI)

Usage:
    python pipeline/14_validate_journals.py \
        --input classified/classified_v3_300k.parquet \
        --out-dir reports/journal_validation_v3_300k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy normalised to [0,1] (0 = pure, 1 = uniform)."""
    p = counts / counts.sum()
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def per_group_stats(df: pd.DataFrame, group: str, level: str,
                    min_papers: int) -> pd.DataFrame:
    """One row per group (journal/publisher) with purity + entropy + dominant LCC."""
    rows = []
    for name, sub in df.groupby(group, sort=False):
        n = len(sub)
        if n < min_papers:
            continue
        vc = sub[level].value_counts()
        rows.append({
            group: name,
            "n_papers": n,
            "dominant_lcc": vc.index[0],
            "purity": vc.iloc[0] / n,                       # top-1 share
            "n_distinct_lcc": int((vc > 0).sum()),
            "norm_entropy": _entropy(vc.values),
        })
    return pd.DataFrame(rows).sort_values("n_papers", ascending=False)


def weighted(series: pd.Series, weights: pd.Series) -> float:
    return float(np.average(series, weights=weights))


def ami(df: pd.DataFrame, group: str, level: str) -> float:
    from sklearn.metrics import adjusted_mutual_info_score
    return float(adjusted_mutual_info_score(
        df[group].values, df[level].values, average_method="arithmetic"))


def lift(df: pd.DataFrame, stats: pd.DataFrame, group: str, level: str) -> float:
    """
    Mean over groups of (observed top-LCC share / expected top-LCC share under
    the global LCC distribution). ~1 ⇒ no journal signal; >>1 ⇒ strong signal.
    Degeneracy-proof: a constant predictor gives observed≈expected≈1 → lift≈1.
    """
    global_p = df[level].value_counts(normalize=True)
    lifts, weights = [], []
    g_index = {name: sub for name, sub in df.groupby(group, sort=False)}
    for _, r in stats.iterrows():
        name = r[group]
        sub = g_index[name]
        # expected top-1 share = max over LCCs of global prob * n, /n = max global prob
        # but the journal's *own* dominant lcc may differ; use the global prob of
        # the journal's dominant lcc as the chance expectation for that lcc.
        exp = global_p.get(r["dominant_lcc"], 1e-9)
        lifts.append(r["purity"] / exp)
        weights.append(r["n_papers"])
    return float(np.average(lifts, weights=weights))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="classified parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--level", default="pred_lcc",
                    choices=["pred_lcc_div", "pred_lcc", "pred_lcc_main"],
                    help="LCC granularity for AMI/purity (default: pred_lcc subclass)")
    ap.add_argument("--min-papers", type=int, default=20,
                    help="min papers for a journal/publisher to count (default 20)")
    ap.add_argument("--sim-col", default="pred_centroid_sim")
    ap.add_argument("--ami-sample", type=int, default=500_000,
                    help="subsample size for AMI (0 = use all; default 500k)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    def log(m=""): print(m); lines.append(m)

    cols = ["journal", "publisher", args.level]
    if args.sim_col:
        cols.append(args.sim_col)
    df = pd.read_parquet(args.input, columns=cols)
    N = len(df)

    log("=" * 70)
    log("JOURNAL / PUBLISHER PROXY-GROUND-TRUTH VALIDATION")
    log("=" * 70)
    log(f"  input : {args.input}")
    log(f"  docs  : {N:,}   level: {args.level}   min-papers: {args.min_papers}")
    log("")

    # ── Coverage ─────────────────────────────────────────────────────────────
    for g in ("journal", "publisher"):
        s = df[g].astype("string").str.strip()
        df[g] = s.where(s.str.len() > 0)
        cov = df[g].notna().mean()
        nuniq = df[g].nunique()
        log(f"  {g:<10} coverage = {cov:.1%}   distinct = {nuniq:,}")
    log("")

    def run_for(group: str):
        log("-" * 70)
        log(f"GROUP = {group}")
        d = df[df[group].notna()].copy()
        stats = per_group_stats(d, group, args.level, args.min_papers)
        if stats.empty:
            log(f"  no {group} with >= {args.min_papers} papers"); return None
        n_groups = len(stats)
        covered = stats["n_papers"].sum()
        log(f"  {group}s with >= {args.min_papers} papers: {n_groups:,} "
            f"(covering {covered:,} docs = {covered/N:.1%})")

        # purity (paper-weighted)
        wp = weighted(stats["purity"], stats["n_papers"])
        we = weighted(stats["norm_entropy"], stats["n_papers"])
        log(f"  paper-weighted top-LCC PURITY = {wp:.3f}   "
            f"(median per-{group} = {stats['purity'].median():.3f})")
        log(f"  paper-weighted norm. ENTROPY  = {we:.3f}   (0=pure, 1=uniform)")
        log(f"  >=80% pure journals: "
            f"{(stats['purity'] >= 0.8).mean():.1%} of {group}s, "
            f"{stats.loc[stats['purity']>=0.8,'n_papers'].sum()/covered:.1%} of their papers")

        # lift
        lf = lift(d, stats, group, args.level)
        log(f"  paper-weighted LIFT vs random = {lf:.2f}×  "
            f"(>>1 ⇒ predictions track {group}; ~1 ⇒ no signal / degenerate)")

        # AMI (headline, chance-corrected)
        dd = d
        if args.ami_sample and len(d) > args.ami_sample:
            dd = d.sample(args.ami_sample, random_state=42)
        a = ami(dd, group, args.level)
        log(f"  >>> AMI({group}, {args.level}) = {a:.3f}  "
            f"[headline; chance-corrected, degeneracy-proof]")

        # ── OOD cross-check ───────────────────────────────────────────────────
        if args.sim_col in d.columns:
            q = d[args.sim_col]
            hi_t, lo_t = q.quantile(0.75), q.quantile(0.25)
            hi = d[d[args.sim_col] >= hi_t]
            lo = d[d[args.sim_col] <= lo_t]
            a_hi = ami(hi.sample(min(len(hi), args.ami_sample or len(hi)),
                                 random_state=42) if args.ami_sample else hi,
                       group, args.level)
            a_lo = ami(lo.sample(min(len(lo), args.ami_sample or len(lo)),
                                 random_state=42) if args.ami_sample else lo,
                       group, args.level)
            s_hi = per_group_stats(hi, group, args.level, max(5, args.min_papers // 2))
            s_lo = per_group_stats(lo, group, args.level, max(5, args.min_papers // 2))
            log("")
            log(f"  OOD cross-check (split on {args.sim_col} quartiles):")
            log(f"    high-sim (>= {hi_t:.3f}): AMI={a_hi:.3f}  "
                f"purity={weighted(s_hi['purity'], s_hi['n_papers']):.3f}"
                if not s_hi.empty else f"    high-sim: AMI={a_hi:.3f}")
            log(f"    low-sim  (<= {lo_t:.3f}): AMI={a_lo:.3f}  "
                f"purity={weighted(s_lo['purity'], s_lo['n_papers']):.3f}"
                if not s_lo.empty else f"    low-sim:  AMI={a_lo:.3f}")
            log(f"    → {'OOD score VALID: high-sim more consistent' if a_hi > a_lo else 'WARNING: OOD score not discriminating (high-sim not more consistent)'}")

            # purity by centroid-sim decile
            d = d.assign(_bin=pd.qcut(d[args.sim_col], 10, labels=False, duplicates="drop"))
            log("")
            log(f"  purity by {args.sim_col} decile (low→high):")
            for b, g2 in d.groupby("_bin"):
                st = per_group_stats(g2, group, args.level, 5)
                if st.empty:
                    continue
                log(f"    decile {int(b):>2}  sim~[{g2[args.sim_col].min():.2f},"
                    f"{g2[args.sim_col].max():.2f}]  "
                    f"purity={weighted(st['purity'], st['n_papers']):.3f}  "
                    f"n={len(g2):,}")

        # worst offenders
        big = stats[stats["n_papers"] >= max(args.min_papers, 100)]
        worst = big.nsmallest(12, "purity")
        log("")
        log(f"  Least-coherent high-volume {group}s (n>=100, low purity):")
        for _, r in worst.iterrows():
            log(f"    purity={r['purity']:.2f}  n={int(r['n_papers']):>6,}  "
                f"dom={r['dominant_lcc']:<8}  {str(r[group])[:45]}")
        return stats

    j_stats = run_for("journal")
    log("")
    run_for("publisher")

    if j_stats is not None:
        csv_path = out_dir / "journal_lcc_map.csv"
        j_stats.to_csv(csv_path, index=False)
        log("")
        log(f"  journal→LCC map → {csv_path}")

    (out_dir / "journal_validation.txt").write_text("\n".join(lines))
    log(f"  report → {out_dir/'journal_validation.txt'}")


if __name__ == "__main__":
    main()
