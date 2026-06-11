"""
pipeline/16_build_journal_gold.py
==================================
Build a labeling template for an INDEPENDENT journal → LCC reference map, so the
classifier can be scored against external ground truth (not against itself).

WHY THIS IS NOT CIRCULAR
------------------------
`journal_lcc_map.csv` (from 14_validate_journals.py) defines a journal's LCC as
the model's OWN dominant prediction — scoring against it only re-measures purity
and can never catch a systematic model error. This template instead asks YOU to
assign each journal's true LCC from what it actually publishes (titles + content
keywords). Journal scope is well-defined and independent of the model, so the
resulting map is genuine ground truth. Labeling the top ~200 journals already
anchors ~half the corpus.

The model's current guess IS shown — but in a clearly-flagged side column
(`model_guess_DONT_ANCHOR`) — only so you can spot disagreements. Label from the
titles/keywords, not from that column.

OUTPUT
------
  <out>  a CSV with one row per top-N journal:
    journal, n_papers, sample_titles, top_keywords,
    model_guess_DONT_ANCHOR, model_purity,
    gold_lcc_main, gold_lcc_sub, gold_lcc_div   ← YOU fill these (sub is enough)
    notes
  Rows you can't judge: leave gold_* blank → excluded from scoring (17_*).

Usage (server, where the classified parquet lives):
    python pipeline/16_build_journal_gold.py \
        --input classified/classified_v3_300k.parquet \
        --output reports/journal_validation_v3_300k/journal_gold_template.csv \
        --top 200 --samples 8 --keywords 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="classified parquet")
    ap.add_argument("--output", required=True, help="template CSV to write")
    ap.add_argument("--top", type=int, default=200,
                    help="number of largest journals to include (default 200)")
    ap.add_argument("--samples", type=int, default=8,
                    help="sample titles per journal (default 8)")
    ap.add_argument("--keywords", type=int, default=12,
                    help="TF-IDF keywords per journal (default 12)")
    ap.add_argument("--title-col", default="title")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer

    cols = ["journal", args.title_col, "pred_lcc_main", "pred_lcc", "pred_lcc_div"]
    df = pd.read_parquet(args.input, columns=cols)
    df["journal"] = df["journal"].astype("string").str.strip()
    df = df[df["journal"].notna() & (df["journal"].str.len() > 0)]

    counts = df["journal"].value_counts()
    top_journals = counts.head(args.top).index.tolist()
    print(f"  {len(df):,} docs with a journal; building template for top {len(top_journals)}")

    rows = []
    for jn in top_journals:
        sub = df[df["journal"] == jn]
        n = len(sub)

        # model's current dominant prediction (reference only)
        vc_sub = sub["pred_lcc"].value_counts()
        dom_sub = vc_sub.index[0]
        purity = vc_sub.iloc[0] / n

        # sample titles (deterministic)
        titles = (sub[args.title_col].dropna().astype(str)
                  .drop_duplicates().head(args.samples).tolist())
        sample_titles = " | ".join(t[:90] for t in titles)

        # content keywords from titles (independent of the LCC prediction)
        kw = ""
        txt = sub[args.title_col].dropna().astype(str)
        if len(txt) >= 3:
            try:
                vec = TfidfVectorizer(max_features=2000, stop_words="english",
                                      ngram_range=(1, 2), min_df=2)
                X = vec.fit_transform(txt)
                scores = np.asarray(X.mean(axis=0)).ravel()
                terms = np.array(vec.get_feature_names_out())
                kw = ", ".join(terms[scores.argsort()[::-1][:args.keywords]])
            except ValueError:
                kw = ""

        rows.append({
            "journal": jn,
            "n_papers": n,
            "top_keywords": kw,
            "sample_titles": sample_titles,
            "model_guess_DONT_ANCHOR": dom_sub,
            "model_purity": round(purity, 3),
            "gold_lcc_main": "",     # ← fill: e.g. Q, R, T
            "gold_lcc_sub": "",      # ← fill: e.g. QD, RC, TK  (this is enough)
            "gold_lcc_div": "",      # ← optional finer: e.g. QD411
            "notes": "",
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    cov = out["n_papers"].sum()
    print(f"  ✅ template → {out_path}")
    print(f"     {len(out)} journals, covering {cov:,} papers")
    print(f"     fill gold_lcc_sub (and optionally _div) from sample_titles/top_keywords,")
    print(f"     then score with 17_score_vs_journal_gold.py")


if __name__ == "__main__":
    main()
