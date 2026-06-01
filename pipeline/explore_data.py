"""
pipeline/explore_data.py
========================
Data quality exploration for any pipeline stage output.
Auto-detects what kind of parquet it is and generates the relevant plots.

Outputs PNG charts + a text summary report to --output directory.
Works on the server (no display needed — uses matplotlib Agg backend).

Usage
-----
    # Explore cleaned data (sample 200k for speed)
    python pipeline/explore_data.py \\
        --input  data/cleaned/english_FullDatasetV2.parquet \\
        --sample 200000

    # Explore classified data
    python pipeline/explore_data.py \\
        --input  classified/classified_FullDataset.parquet

    # Custom output directory
    python pipeline/explore_data.py \\
        --input  data/cleaned/english_FullDatasetV2.parquet \\
        --output reports/exploration_clean/

After running, copy to your local machine:
    scp -r server:/path/to/reports/exploration_clean/ ./
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

# ─── colour palette ───────────────────────────────────────────────────────────
BLUE   = "#4C72B0"
GREEN  = "#55A868"
RED    = "#C44E52"
ORANGE = "#DD8452"
PURPLE = "#8172B2"
PALETTE = [BLUE, GREEN, RED, ORANGE, PURPLE,
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

LCC_NAMES = {
    "A": "General Works",        "B": "Philosophy/Religion",
    "C": "Aux. Sciences Hist.",  "D": "World History",
    "E": "American History",     "F": "Local History",
    "G": "Geography/Anthrop.",   "H": "Social Sciences",
    "J": "Political Science",    "K": "Law",
    "L": "Education",            "M": "Music",
    "N": "Fine Arts",            "P": "Language & Literature",
    "Q": "Science",              "R": "Medicine",
    "S": "Agriculture",          "T": "Technology",
    "U": "Military Science",     "V": "Naval Science",
    "Z": "Bibliography",
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _save(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")


def _wordcount(text: str) -> int:
    return len(str(text).split()) if pd.notna(text) else 0


def _extract_year(s):
    if pd.isna(s):
        return None
    m = re.search(r"\b(19[5-9]\d|20[012]\d)\b", str(s))
    return int(m.group()) if m else None


def _load(source: Path, sample: int | None) -> pd.DataFrame:
    pf    = pq.ParquetFile(source)
    n     = pf.metadata.num_rows
    cols  = pf.schema_arrow.names
    # never load the embeddings column for exploration
    keep  = [c for c in cols if c != "embeddings"]
    print(f"  {n:,} rows  —  {len(cols)} columns")
    if "embeddings" in cols:
        print("  (embeddings column excluded from load)")

    if sample and sample < n:
        print(f"  Sampling {sample:,} rows for speed ...")
        rng    = np.random.default_rng(42)
        chosen = np.sort(rng.choice(n, size=sample, replace=False))
        chosen_set = set(chosen.tolist())
        rows, gi = [], 0
        for batch in pf.iter_batches(batch_size=50_000, columns=keep):
            chunk  = batch.to_pandas()
            nc     = len(chunk)
            local  = [i for i in range(nc) if (gi + i) in chosen_set]
            if local:
                rows.append(chunk.iloc[local])
            gi += nc
            if gi > chosen.max():
                break
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.read_parquet(source, columns=keep)

    print(f"  Loaded {len(df):,} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_missing(df: pd.DataFrame, out: Path):
    miss = (df.isna() | (df == "")).mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        print("  no missing values — skipping missing-values chart")
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.35)))
    bars = ax.barh(miss.index[::-1], miss.values[::-1] * 100, color=RED, alpha=0.8)
    ax.set_xlabel("Missing / empty  (%)")
    ax.set_title("Missing values per column")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    for bar, v in zip(bars, miss.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out, "01_missing_values.png")


def plot_abstract_length(df: pd.DataFrame, out: Path):
    col = "clean_abstract" if "clean_abstract" in df.columns else "abstract"
    if col not in df.columns:
        return
    wc = df[col].dropna().apply(_wordcount)
    wc = wc[wc > 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(wc.clip(upper=600), bins=60, color=BLUE, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Word count (capped at 600)")
    axes[0].set_ylabel("Documents")
    axes[0].set_title(f"Abstract length  —  {col}")

    pcts = [10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(wc, pcts)
    axes[1].barh([f"p{p}" for p in pcts], vals, color=BLUE, alpha=0.8)
    axes[1].set_xlabel("Words")
    axes[1].set_title("Percentiles")
    for i, v in enumerate(vals):
        axes[1].text(v + 1, i, f"{v:.0f}", va="center", fontsize=9)

    fig.suptitle(
        f"n={len(wc):,}  |  mean={wc.mean():.0f}  median={wc.median():.0f}  "
        f"<50w: {(wc<50).mean():.1%}  >300w: {(wc>300).mean():.1%}",
        fontsize=9, y=1.01
    )
    fig.tight_layout()
    _save(fig, out, "02_abstract_length.png")


def plot_year(df: pd.DataFrame, out: Path):
    col = next((c for c in ["publication_date", "created_at"] if c in df.columns), None)
    if col is None:
        return
    years = df[col].apply(_extract_year).dropna().astype(int)
    years = years[(years >= 1950) & (years <= 2026)]
    if years.empty:
        return
    counts = years.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(counts.index, counts.values, color=GREEN, alpha=0.85, width=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Documents")
    ax.set_title(f"Publication year distribution  (n={len(years):,})")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.xticks(rotation=45)
    fig.tight_layout()
    _save(fig, out, "03_publication_year.png")


def plot_top_n(df: pd.DataFrame, col: str, out: Path, filename: str,
               title: str, n: int = 25):
    if col not in df.columns:
        return
    counts = df[col].dropna().value_counts().head(n)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.32)))
    ax.barh(counts.index[::-1], counts.values[::-1], color=PURPLE, alpha=0.8)
    ax.set_xlabel("Documents")
    ax.set_title(f"Top {n} {title}")
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v + counts.max() * 0.005, i, f"{v:,}", va="center", fontsize=7)
    fig.tight_layout()
    _save(fig, out, filename)


def plot_language(df: pd.DataFrame, out: Path):
    col = "abstract_lang" if "abstract_lang" in df.columns else "language"
    if col not in df.columns:
        return
    counts = df[col].dropna().value_counts().head(20)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [GREEN if l == "en" else BLUE for l in counts.index]
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Documents")
    ax.set_title("Language distribution (green = English)")
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v + counts.max() * 0.005, i, f"{v:,}", va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out, "06_language.png")


def plot_doc_type(df: pd.DataFrame, out: Path):
    if "type" not in df.columns:
        return
    counts = df["type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        autopct="%1.1f%%", colors=PALETTE[:len(counts)],
        startangle=140, pctdistance=0.82
    )
    ax.set_title("Document type distribution")
    fig.tight_layout()
    _save(fig, out, "07_doc_type.png")


# ── Classification-specific plots ─────────────────────────────────────────────

def plot_lcc_distribution(df: pd.DataFrame, out: Path):
    if "pred_lcc_main" not in df.columns:
        return
    counts = df["pred_lcc_main"].value_counts()
    labels = [f"{c}\n{LCC_NAMES.get(c,'')}" for c in counts.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, counts.values,
                  color=PALETTE[:len(counts)], alpha=0.85, edgecolor="white")
    ax.set_ylabel("Documents")
    ax.set_title("LCC main class distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + counts.max() * 0.01,
                f"{h/len(df):.1%}", ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out, "08_lcc_distribution.png")


def plot_confidence(df: pd.DataFrame, out: Path):
    if "conf_div" not in df.columns:
        return
    conf = df["conf_div"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(conf, bins=50, color=ORANGE, alpha=0.85, edgecolor="white")
    axes[0].axvline(0.50, color=RED, ls="--", lw=1.5, label="0.50")
    axes[0].axvline(0.30, color=RED, ls=":",  lw=1.5, label="0.30")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Documents")
    axes[0].set_title("Confidence distribution")
    axes[0].legend(fontsize=8)

    if "pred_lcc_main" in df.columns:
        cls_order = df.groupby("pred_lcc_main")["conf_div"].median().sort_values().index
        data  = [df.loc[df["pred_lcc_main"] == c, "conf_div"].dropna().values
                 for c in cls_order]
        axes[1].boxplot(data, labels=cls_order, vert=True, patch_artist=True,
                        boxprops=dict(facecolor=ORANGE, alpha=0.6))
        axes[1].set_xlabel("LCC main class")
        axes[1].set_ylabel("Confidence")
        axes[1].set_title("Confidence per class")

    fig.suptitle(
        f"mean={conf.mean():.3f}  median={conf.median():.3f}  "
        f"<0.50: {(conf<0.50).mean():.1%}  <0.30: {(conf<0.30).mean():.1%}",
        fontsize=9
    )
    fig.tight_layout()
    _save(fig, out, "09_confidence.png")


def plot_subclass_top20(df: pd.DataFrame, out: Path):
    if "pred_lcc" not in df.columns:
        return
    counts = df["pred_lcc"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(counts.index[::-1], counts.values[::-1], color=BLUE, alpha=0.8)
    ax.set_xlabel("Documents")
    ax.set_title("Top 20 predicted LCC subclasses")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    fig.tight_layout()
    _save(fig, out, "10_top_subclasses.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def text_summary(df: pd.DataFrame, out: Path, source: Path):
    lines = []
    L = lines.append

    L("=" * 62)
    L(f"  DATA EXPLORATION SUMMARY")
    L(f"  source : {source}")
    L("=" * 62)
    L(f"  Rows      : {len(df):,}")
    L(f"  Columns   : {len(df.columns)}  —  {list(df.columns)}")
    L()

    # Missing values
    miss = (df.isna() | (df == "")).mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    if not miss.empty:
        L("  MISSING / EMPTY VALUES")
        L(f"  {'Column':<25}  {'Missing':>8}")
        L(f"  {'─'*25}  {'─'*8}")
        for col, pct in miss.items():
            L(f"  {col:<25}  {pct:>7.1%}")
        L()

    # Abstract quality
    for col in ["clean_abstract", "abstract"]:
        if col in df.columns:
            wc = df[col].dropna().apply(_wordcount)
            wc = wc[wc > 0]
            L(f"  ABSTRACT LENGTH  ({col})")
            L(f"  n={len(wc):,}  mean={wc.mean():.0f}w  "
              f"median={wc.median():.0f}w  p10={np.percentile(wc,10):.0f}  "
              f"p90={np.percentile(wc,90):.0f}")
            L(f"  < 50 words  : {(wc<50).sum():,}  ({(wc<50).mean():.1%})")
            L(f"  < 20 words  : {(wc<20).sum():,}  ({(wc<20).mean():.1%})  ← likely noise")
            L(f"  > 300 words : {(wc>300).sum():,}  ({(wc>300).mean():.1%})")
            L()
            break

    # Year
    for col in ["publication_date", "created_at"]:
        if col in df.columns:
            years = df[col].apply(_extract_year).dropna()
            if not years.empty:
                L(f"  PUBLICATION YEARS  (from {col})")
                L(f"  range  : {int(years.min())} – {int(years.max())}")
                L(f"  top 5  : {years.value_counts().head(5).to_dict()}")
                L(f"  no year: {df[col].isna().sum():,}  ({df[col].isna().mean():.1%})")
                L()
            break

    # Language
    for col in ["abstract_lang", "language"]:
        if col in df.columns:
            vc = df[col].value_counts().head(10).to_dict()
            L(f"  LANGUAGE  ({col})")
            for lang, cnt in vc.items():
                L(f"    {lang:<8}  {cnt:>10,}  ({cnt/len(df):.1%})")
            L()
            break

    # Top publishers / journals
    for col, label in [("publisher", "PUBLISHERS"), ("journal", "JOURNALS")]:
        if col in df.columns:
            vc = df[col].dropna().value_counts().head(10)
            L(f"  TOP 10 {label}")
            for val, cnt in vc.items():
                L(f"    {str(val)[:45]:<45}  {cnt:>8,}")
            L()

    # Duplicates
    for col in ["title", "doi"]:
        if col in df.columns:
            dupes = df[col].dropna().duplicated().sum()
            L(f"  DUPLICATE {col.upper()}S  : {dupes:,}  ({dupes/len(df):.2%})")
    L()

    # Classification results
    if "pred_lcc_main" in df.columns:
        L("  CLASSIFICATION RESULTS")
        L(f"  Classes : {df['pred_lcc_main'].nunique()}  "
          f"Subclasses : {df['pred_lcc'].nunique() if 'pred_lcc' in df.columns else 'N/A'}")
        vc = df["pred_lcc_main"].value_counts()
        for cls, cnt in vc.items():
            name = LCC_NAMES.get(cls, "")
            L(f"    {cls}  {name:<30}  {cnt:>10,}  ({cnt/len(df):.1%})")
        L()

    if "conf_div" in df.columns:
        c = df["conf_div"].dropna()
        L("  CONFIDENCE (conf_div)")
        L(f"  mean={c.mean():.3f}  median={c.median():.3f}  std={c.std():.3f}")
        L(f"  < 0.50 : {(c<0.50).sum():,}  ({(c<0.50).mean():.1%})")
        L(f"  < 0.30 : {(c<0.30).sum():,}  ({(c<0.30).mean():.1%})")
        L()

    L("=" * 62)

    report = "\n".join(lines)
    print(report)

    rpath = out / "summary.txt"
    rpath.write_text(report, encoding="utf-8")
    print(f"  saved → summary.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Data quality exploration — generates PNG charts + text summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  required=True,
                        help="Parquet file to explore (cleaned, classified, or embedded)")
    parser.add_argument("--output", default=None,
                        help="Output directory for charts (default: reports/exploration_{stem}/)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample size for speed — omit to use all rows. "
                             "200000 is enough for reliable statistics.")
    parser.add_argument("--top-n",  type=int, default=25,
                        help="How many entries to show in top-N charts (default: 25)")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        print(f"Error: file not found: {source}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output) if args.output else \
          Path("reports") / f"exploration_{source.stem}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nExploring: {source}")
    print(f"Output   : {out}")
    print(f"Sample   : {args.sample or 'all rows'}\n")

    df = _load(source, args.sample)

    # ── detect parquet type ───────────────────────────────────────────────────
    has_classification = "pred_lcc_main" in df.columns
    has_embeddings     = "embeddings" in pq.ParquetFile(source).schema_arrow.names
    kind = ("classified" if has_classification
            else "embedded" if has_embeddings
            else "cleaned")
    print(f"\nDetected type: {kind}\n")

    # ── generate plots ────────────────────────────────────────────────────────
    print("Generating charts:")
    plot_missing(df, out)
    plot_abstract_length(df, out)
    plot_year(df, out)
    plot_top_n(df, "publisher", out, "04_top_publishers.png",
               "publishers", n=args.top_n)
    plot_top_n(df, "journal",   out, "05_top_journals.png",
               "journals",   n=args.top_n)
    plot_language(df, out)
    plot_doc_type(df, out)

    if has_classification:
        plot_lcc_distribution(df, out)
        plot_confidence(df, out)
        plot_subclass_top20(df, out)

    # ── text summary ──────────────────────────────────────────────────────────
    print("\nText summary:")
    text_summary(df, out, source)

    print(f"\n✅  All outputs in: {out}")
    print(f"   Copy to local machine:")
    print(f"   scp -r <server>:{out.resolve()} ./")


if __name__ == "__main__":
    main()
