"""
pipeline/12_report_classification.py
======================================
Post-classification quality report. Automatically called by run_server.py
after the classify stage; can also be run standalone.

Produces:
  1. Overview — total docs, unique classes / subclasses / divisions
  2. Main class distribution with LCC names and per-class confidence
  3. Top 20 subclasses with confidence
  4. Top 20 divisions with confidence
  5. Top N TF-IDF terms per main class (sanity-check: do the words match the topic?)
  6. Sample titles per main class (human readability check)
  7. Low-confidence examples (conf < 0.30)

Usage:
    python pipeline/12_report_classification.py \\
        --classified classified/classified_server.parquet

    # Save report to a file:
    python pipeline/12_report_classification.py \\
        --classified classified/classified_server.parquet \\
        --output reports/report_server.txt

    # Adjust top-words and samples per class:
    python pipeline/12_report_classification.py \\
        --classified classified/classified_server.parquet \\
        --top-words 20 --samples 5
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ─── LCC main class names ────────────────────────────────────────────────────

LCC_NAMES: dict[str, str] = {
    "A": "General Works",
    "B": "Philosophy / Psychology / Religion",
    "C": "Auxiliary Sciences of History",
    "D": "World History",
    "E": "American History",
    "F": "Local History",
    "G": "Geography / Anthropology / Recreation",
    "H": "Social Sciences",
    "J": "Political Science",
    "K": "Law",
    "L": "Education",
    "M": "Music",
    "N": "Fine Arts",
    "P": "Language & Literature",
    "Q": "Science",
    "R": "Medicine",
    "S": "Agriculture",
    "T": "Technology",
    "U": "Military Science",
    "V": "Naval Science",
    "Z": "Bibliography / Library Science",
}

TFIDF_SAMPLE = 3_000   # max docs per class for TF-IDF (keeps it fast at 3M scale)

# Document-structure words that add no topical signal in TF-IDF (but are kept
# in embeddings where they provide semantic context).
_BOILERPLATE: frozenset[str] = frozenset({
    "abstract", "article", "paper", "study", "results", "result",
    "using", "used", "use", "based", "method", "methods", "data",
    "research", "analysis", "also", "new", "present", "show", "shown",
    "propose", "proposed", "approach", "work", "works", "effect",
    "effects", "different", "two", "three", "high", "low", "large",
    "significantly", "significant",
})

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", str(text)).strip()


def _is_latin(text: str) -> bool:
    """True if the string contains only ASCII + common punctuation (no CJK etc.)."""
    return bool(re.fullmatch(r"[\x00-\x7FÀ-ɏ\s\-\.\,\(\)\[\]\{\}\/\:\;\'\"\!\?\+\=\*\&\%\#\@\$\°\~\^]+", str(text).strip()))


# ─── TF-IDF helper ───────────────────────────────────────────────────────────

def _top_terms(texts: pd.Series, top_n: int) -> list[str]:
    if len(texts) < 3:
        return ["(too few documents)"]
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = ENGLISH_STOP_WORDS.union(_BOILERPLATE)
    vec = TfidfVectorizer(
        max_features=15_000,
        stop_words=stop,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X      = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).flatten()
    idx    = scores.argsort()[-top_n:][::-1]
    return list(vec.get_feature_names_out()[idx])


# ─── Report builder ───────────────────────────────────────────────────────────

def build_report(df: pd.DataFrame, top_n: int, n_samples: int) -> str:
    out: list[str] = []

    def line(s: str = ""):
        out.append(s)

    def section(title: str):
        line()
        line("=" * 64)
        line(f"  {title}")
        line("=" * 64)

    n      = len(df)
    n_main = df["pred_lcc_main"].nunique()
    n_sub  = df["pred_lcc"].nunique()
    n_div  = df["pred_lcc_div"].nunique()

    # ── 1. Overview ──────────────────────────────────────────────────────────
    section("OVERVIEW")
    line(f"  Total classified   : {n:,}")
    line(f"  LCC main classes   : {n_main}  (of {len(LCC_NAMES)} possible)")
    line(f"  LCC subclasses     : {n_sub}")
    line(f"  LCC divisions      : {n_div}")
    line()
    line(f"  Confidence (conf_div):")
    line(f"    mean             : {df['conf_div'].mean():.3f}")
    line(f"    median           : {df['conf_div'].median():.3f}")
    line(f"    std              : {df['conf_div'].std():.3f}")
    line(f"    conf < 0.50      : {(df['conf_div'] < 0.50).sum():,}  ({(df['conf_div'] < 0.50).mean():.1%})")
    line(f"    conf < 0.30      : {(df['conf_div'] < 0.30).sum():,}  ({(df['conf_div'] < 0.30).mean():.1%})")

    # ── 2. Main class distribution ───────────────────────────────────────────
    section("LCC MAIN CLASS DISTRIBUTION")
    line(f"  {'Cls':<4}  {'LCC Topic':<38}  {'Count':>8}  {'%':>6}  {'MeanConf':>8}")
    line(f"  {'─'*4}  {'─'*38}  {'─'*8}  {'─'*6}  {'─'*8}")
    for cls, cnt in df["pred_lcc_main"].value_counts().items():
        name = LCC_NAMES.get(cls, "Unknown")
        pct  = cnt / n
        conf = df.loc[df["pred_lcc_main"] == cls, "conf_div"].mean()
        line(f"  {cls:<4}  {name:<38}  {cnt:>8,}  {pct:>6.1%}  {conf:>8.3f}")

    # ── 3. Top subclasses ─────────────────────────────────────────────────────
    section("TOP 20 LCC SUBCLASSES")
    line(f"  {'Subclass':<12}  {'Count':>8}  {'%':>6}  {'MeanConf':>8}")
    line(f"  {'─'*12}  {'─'*8}  {'─'*6}  {'─'*8}")
    for sub, cnt in df["pred_lcc"].value_counts().head(20).items():
        conf = df.loc[df["pred_lcc"] == sub, "conf_div"].mean()
        line(f"  {sub:<12}  {cnt:>8,}  {cnt/n:>6.1%}  {conf:>8.3f}")

    # ── 4. Top divisions ──────────────────────────────────────────────────────
    section("TOP 20 LCC DIVISIONS")
    line(f"  {'Division':<16}  {'Count':>8}  {'%':>6}  {'MeanConf':>8}")
    line(f"  {'─'*16}  {'─'*8}  {'─'*6}  {'─'*8}")
    for div, cnt in df["pred_lcc_div"].value_counts().head(20).items():
        conf = df.loc[df["pred_lcc_div"] == div, "conf_div"].mean()
        line(f"  {div:<16}  {cnt:>8,}  {cnt/n:>6.1%}  {conf:>8.3f}")

    # ── 5. TF-IDF top terms per main class ────────────────────────────────────
    text_col = next(
        (c for c in ["clean_abstract", "abstract"] if c in df.columns), None
    )
    if text_col:
        section(f"TOP {top_n} TF-IDF TERMS PER MAIN CLASS  (source: {text_col})")
        line("  (bigrams included; sampled up to 3,000 docs/class for speed)")
        for cls in sorted(df["pred_lcc_main"].unique()):
            name   = LCC_NAMES.get(cls, "")
            mask   = df["pred_lcc_main"] == cls
            texts  = df.loc[mask, text_col].dropna()
            if len(texts) > TFIDF_SAMPLE:
                texts = texts.sample(TFIDF_SAMPLE, random_state=42)
            terms  = _top_terms(texts, top_n)
            line()
            line(f"  [{cls}] {name}  ({mask.sum():,} docs)")
            line(f"    {', '.join(terms)}")

    # ── 6. Sample titles per main class ──────────────────────────────────────
    if "title" in df.columns:
        section(f"{n_samples} SAMPLE TITLES PER MAIN CLASS  (Latin script only; HTML stripped)")
        for cls in sorted(df["pred_lcc_main"].unique()):
            name = LCC_NAMES.get(cls, "")
            pool = df.loc[df["pred_lcc_main"] == cls, "title"].dropna()
            # Prefer Latin-script titles; fall back to all if too few
            latin_pool = pool[pool.apply(_is_latin)]
            draw_pool  = latin_pool if len(latin_pool) >= n_samples else pool
            sample = draw_pool.sample(min(n_samples, len(draw_pool)), random_state=42).tolist()
            line()
            line(f"  [{cls}] {name}")
            for t in sample:
                clean = _strip_html(t)[:120]
                flag  = "  [non-Latin]" if not _is_latin(t) else ""
                line(f"    • {clean}{flag}")

    # ── 7. Low-confidence examples ────────────────────────────────────────────
    low = df[df["conf_div"] < 0.30].head(10)
    if len(low) > 0:
        section(f"LOW-CONFIDENCE EXAMPLES  (conf < 0.30, up to 10 shown)")
        for _, row in low.iterrows():
            label = f"[{row.get('pred_lcc_main','')}→{row.get('pred_lcc','')}→{row.get('pred_lcc_div','')}]"
            line(f"  conf={row['conf_div']:.3f}  {label}")
            if "title" in row and pd.notna(row.get("title")):
                line(f"    {_strip_html(row['title'])[:120]}")
    else:
        section("LOW-CONFIDENCE EXAMPLES  (conf < 0.30)")
        line("  None — all documents classified with conf ≥ 0.30  ✅")

    line()
    return "\n".join(out)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-classification quality report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--classified", required=True,
                        help="Path to the classified parquet file")
    parser.add_argument("--output",     default=None,
                        help="Save report to this .txt file (optional)")
    parser.add_argument("--top-words",  type=int, default=15,
                        help="TF-IDF terms per main class (default: 15)")
    parser.add_argument("--samples",    type=int, default=3,
                        help="Sample titles per main class (default: 3)")
    args = parser.parse_args()

    path = Path(args.classified)
    if not path.exists():
        print(f"  ❌  File not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {path} ...")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} rows  —  columns: {list(df.columns)}\n")

    report = build_report(df, top_n=args.top_words, n_samples=args.samples)
    print(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nReport saved → {out}")


if __name__ == "__main__":
    main()
