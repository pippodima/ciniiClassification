"""
search/make_sample.py
=====================
Export a small, representative slice of the classified corpus so the search demo
can run entirely on a laptop — no Meilisearch on the server needed.

Keeps only the columns the indexer uses (drops full_text/affiliations/topics/…),
so the output parquet is small and quick to download (scp/rsync).

Run on the SERVER:
    python search/make_sample.py \
        --source classified/classified_v3_300k.parquet \
        --out    cinii_sample.parquet \
        --n 150000 --stratify

Then download cinii_sample.parquet locally and:
    python search/index_meili.py --source cinii_sample.parquet --index cinii
"""
from __future__ import annotations

import argparse

import pandas as pd

KEEP = [
    "title", "clean_abstract", "abstract", "authors", "journal", "publisher",
    "publication_date", "doi", "pred_lcc_main", "pred_lcc", "pred_lcc_div",
    "conf_div", "pred_centroid_sim",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="classified/classified_v3_300k.parquet")
    ap.add_argument("--out", default="cinii_sample.parquet")
    ap.add_argument("--n", type=int, default=150000, help="target sample size")
    ap.add_argument("--stratify", action="store_true",
                    help="proportional sampling per LCC main class with a floor, "
                         "so rare categories still appear in the demo")
    ap.add_argument("--floor", type=int, default=300,
                    help="min rows per main class when --stratify (default 300)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cols = [c for c in KEEP]
    df = pd.read_parquet(args.source, columns=cols)
    df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "")]
    n_total = len(df)
    print(f"  source: {n_total:,} titled docs")

    if not args.stratify or args.n >= n_total:
        out = df.sample(min(args.n, n_total), random_state=args.seed)
    else:
        parts = []
        for _, g in df.groupby("pred_lcc_main", dropna=False):
            take = min(len(g), max(args.floor, round(args.n * len(g) / n_total)))
            parts.append(g.sample(take, random_state=args.seed))
        out = pd.concat(parts)
        if len(out) > args.n:                       # trim back toward target
            out = out.sample(args.n, random_state=args.seed)

    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out.to_parquet(args.out, index=False)
    mb = out.memory_usage(deep=True).sum() / 1e6
    print(f"  wrote {len(out):,} docs → {args.out}  (~{mb:.0f} MB in memory; "
          f"parquet on disk is smaller)")
    print("  main-class spread:")
    for cls, c in out["pred_lcc_main"].value_counts().head(12).items():
        print(f"    {cls:<3} {c:>7,}")


if __name__ == "__main__":
    main()
