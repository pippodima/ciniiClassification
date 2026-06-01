"""
Check real token length distribution of title+abstract on a sample.
Tells you whether max_length=512 truncates a meaningful fraction of documents.

Usage:
    python pipeline/check_token_lengths.py \
        --input data/cleaned/english_FullDatasetV2.parquet \
        --sample 5000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--model",  default="Qwen/Qwen3-Embedding-0.6B")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Sampling {args.sample:,} rows from {args.input} ...")
    pf  = pq.ParquetFile(args.input)
    n   = pf.metadata.num_rows
    rng = np.random.default_rng(42)
    chosen = set(rng.choice(n, min(args.sample, n), replace=False).tolist())

    rows, gi = [], 0
    for batch in pf.iter_batches(batch_size=50_000,
                                  columns=["title", "clean_abstract"]):
        chunk = batch.to_pandas()
        local = [i for i in range(len(chunk)) if (gi + i) in chosen]
        if local:
            rows.append(chunk.iloc[local])
        gi += len(chunk)
        if gi > max(chosen):
            break

    df = pd.concat(rows, ignore_index=True)
    df["text"] = df["title"].fillna("") + ". " + df["clean_abstract"].fillna("")

    print(f"Tokenising {len(df):,} documents ...")
    lengths = df["text"].apply(
        lambda t: len(tokenizer.encode(str(t), add_special_tokens=False))
    )

    print()
    print("=" * 45)
    print("  TOKEN LENGTH DISTRIBUTION")
    print("=" * 45)
    print(f"  mean   : {lengths.mean():.0f}")
    print(f"  median : {lengths.median():.0f}")
    print()
    for pct in [50, 75, 90, 95, 99, 99.9]:
        v = np.percentile(lengths, pct)
        print(f"  p{pct:5.1f}  : {v:>6.0f} tokens")
    print(f"  max    : {lengths.max():>6} tokens")
    print()
    for cap in [256, 512, 768, 1024]:
        truncated = (lengths > cap).mean()
        print(f"  > {cap:4d} tokens (would be truncated) : {truncated:.2%}")
    print("=" * 45)

if __name__ == "__main__":
    main()
