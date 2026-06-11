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
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# tag/markup tokens that leak from uncleaned titles (<sub>,<sup>,<scp>,JATS …)
# plus generic academic filler that drowns out the topical signal.
_KW_STOP = {
    "sub", "sup", "scp", "inf", "italic", "bold", "jats", "title", "sec",
    "using", "based", "study", "studies", "new", "novel", "report", "reports",
    "effect", "effects", "analysis", "method", "methods", "application",
    "applications", "research", "investigation", "case", "review",
}

def _clean_title(text: str) -> str:
    """Strip HTML/JATS tags and normalise full-width → ASCII for display + TF-IDF."""
    text = unicodedata.normalize("NFKC", str(text))   # ＣＨＥＭ → CHEM, etc.
    text = re.sub(r"<[^>]+>", " ", text)              # drop <sub>…</sub> markup
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
    # NFKC-normalise + strip markup so full-width/variant journal names merge
    # (e.g. ＣＨＥＭＩＣＡＬ　ＢＵＬＬＥＴＩＮ → CHEMICAL BULLETIN). Apply the SAME
    # normalisation in 17_score_vs_journal_gold.py so the gold map keys match.
    df = df[df["journal"].notna()]
    df["journal"] = df["journal"].astype(str).map(_clean_title)
    df = df[df["journal"].str.len() > 0]

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

        # cleaned titles (strip <sub>/<sup>/JATS markup, NFKC-normalise)
        clean = sub[args.title_col].dropna().astype(str).map(_clean_title)
        clean = clean[clean.str.len() > 0]

        # sample titles (deterministic)
        titles = clean.drop_duplicates().head(args.samples).tolist()
        sample_titles = " | ".join(t[:90] for t in titles)

        # content keywords from titles (independent of the LCC prediction)
        # token_pattern keeps ASCII alpha words only → drops digits, 第N報, CJK
        kw = ""
        if len(clean) >= 3:
            try:
                # builtin english stopwords; custom _KW_STOP applied as a
                # post-filter so we keep informative bigrams intact.
                vec = TfidfVectorizer(
                    max_features=2000, stop_words="english",
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                    ngram_range=(1, 2), min_df=2)
                X = vec.fit_transform(clean)
                scores = np.asarray(X.mean(axis=0)).ravel()
                terms = np.array(vec.get_feature_names_out())
                order = scores.argsort()[::-1]
                picked = [t for t in terms[order]
                          if not any(w in _KW_STOP for w in t.split())]
                kw = ", ".join(picked[:args.keywords])
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
