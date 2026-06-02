"""
pipeline/run_reembed.py
========================
Memory-free re-embedding pipeline for 3M+ documents.

Writes the corpus in fixed-size shards so the full embedding matrix is never
in RAM.  Each shard is a self-contained parquet file (metadata + embeddings).
The link between every document and its embedding is preserved inside its shard.

After this script finishes you can inspect, sample, and train without ever
loading the full corpus into memory.

Output layout
-------------
    data/embedded/{run_name}/
        shard_00000.parquet   — up to SHARD_SIZE rows each
        shard_00001.parquet
        ...
        manifest.json         — {n_docs, n_shards, shard_size, emb_dim,
                                  model_name, completed_shards}

Peak RAM during embedding  ≈  batch_size  × emb_dim × 4 bytes
                           ≈  32 × 1024 × 4  =  128 KB   (negligible)
Peak RAM during shard write ≈  shard_size × emb_dim × 4 bytes
                           ≈  50_000 × 1024 × 4  =  200 MB  (fine)

Usage
-----
    # Full re-embed of a cleaned English parquet
    python pipeline/run_reembed.py \\
        --run-name FullDataset_clean \\
        --source data/cleaned/english_FullDataset.parquet \\
        --device cuda \\
        --batch-size 16

    # Resume a crashed run (completed shards are auto-skipped)
    python pipeline/run_reembed.py \\
        --run-name FullDataset_clean \\
        --source data/cleaned/english_FullDataset.parquet \\
        --device cuda

    # Custom shard size (default 50_000)
    python pipeline/run_reembed.py --run-name FullDataset_clean \\
        --source ... --shard-size 100000

    # Dry run — show what would happen without doing anything
    python pipeline/run_reembed.py --run-name FullDataset_clean \\
        --source ... --dry-run

After embedding — sample for training
--------------------------------------
    python pipeline/run_training.py \\
        --run-name v3 \\
        --source data/embedded/FullDataset_clean \\
        --sample-size 150000 \\
        --suggest-lcc models/release
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PIPELINE = Path(__file__).resolve().parent
ROOT     = PIPELINE.parent
sys.path.insert(0, str(PIPELINE))

from config import CLEANED_DIR, EMBEDDED_DIR

MANIFEST_FILE = "manifest.json"
SHARD_FMT     = "shard_{:05d}.parquet"


# ─── Logging ──────────────────────────────────────────────────────────────────

def _log(msg: str = ""):
    print(msg, flush=True)

def _sep(char: str = "─", width: int = 62):
    _log(char * width)

def _section(title: str):
    _log()
    _sep("═")
    _log(f"  {title}")
    _sep("═")


# ─── Dynamic module loader ────────────────────────────────────────────────────

def _load_module(filename: str):
    path = PIPELINE / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Manifest helpers ─────────────────────────────────────────────────────────

def _load_manifest(out_dir: Path) -> dict:
    p = out_dir / MANIFEST_FILE
    return json.loads(p.read_text()) if p.exists() else {}

def _save_manifest(out_dir: Path, data: dict):
    tmp = out_dir / (MANIFEST_FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, out_dir / MANIFEST_FILE)


# ─── Shard writer ─────────────────────────────────────────────────────────────

def _write_shard(df_chunk: pd.DataFrame, embeddings: np.ndarray,
                 out_path: Path) -> None:
    """
    Write one shard atomically.  embeddings has shape (N, D) float32.
    Stored as a list column so it round-trips through pandas/arrow cleanly.
    Write goes to .tmp first, then os.replace() — crash-safe.
    """
    assert len(df_chunk) == len(embeddings), "row count mismatch"

    chunk = df_chunk.copy().reset_index(drop=True)
    chunk["embeddings"] = [embeddings[i].tolist() for i in range(len(embeddings))]

    # Promote null-typed columns → string for schema stability across shards
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    fields = [
        pa.field(f.name, pa.string() if f.type == pa.null() else f.type)
        for f in table.schema
    ]
    table = table.cast(pa.schema(fields), safe=False)

    tmp = out_path.with_suffix(".tmp.parquet")
    try:
        pq.write_table(table, str(tmp), compression="snappy")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out_path)


# ─── Sampling from shards ─────────────────────────────────────────────────────

def sample_from_shards(shard_dir: Path, k: int, seed: int = 42) -> pd.DataFrame:
    """
    Randomly sample k rows from a sharded embedding directory.
    Memory usage: one shard at a time (~200 MB for 50k × 1024 float32).

    This is called by run_training.py when --source points to a directory.
    """
    manifest = _load_manifest(shard_dir)
    if not manifest:
        raise FileNotFoundError(f"No manifest.json found in {shard_dir}")

    n_total     = manifest["n_docs"]
    shard_size  = manifest["shard_size"]
    done_shards = set(manifest.get("completed_shards", []))
    shard_files = sorted(
        p for p in shard_dir.glob("shard_*.parquet")
        if int(p.stem.split("_")[1]) in done_shards
    )

    if not shard_files:
        raise FileNotFoundError(f"No completed shards found in {shard_dir}")

    k = min(k, n_total)
    rng = np.random.default_rng(seed)

    # Proportional allocation: how many rows to draw from each shard
    n_shards = len(shard_files)
    alloc    = np.zeros(n_shards, dtype=int)
    base     = k // n_shards
    alloc[:] = base
    alloc[:k - base * n_shards] += 1   # distribute remainder
    rng.shuffle(alloc)                  # don't bias toward early shards

    parts: list[pd.DataFrame] = []
    for shard_path, n_draw in zip(shard_files, alloc):
        if n_draw == 0:
            continue
        shard = pd.read_parquet(str(shard_path))
        n_draw = min(n_draw, len(shard))
        parts.append(shard.sample(n_draw, random_state=int(rng.integers(1, 2**31))))

    result = pd.concat(parts, ignore_index=True)
    # Final shuffle so the sample isn't ordered by shard
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


# ─── Main embedding loop ──────────────────────────────────────────────────────

def run_reembed(
    source:     Path,
    out_dir:    Path,
    device:     str,
    batch_size: int,
    model_name: str,
    shard_size: int,
    max_length: int = 512,
    dry_run:    bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load source ───────────────────────────────────────────────────────────
    _log(f"  Source       : {source}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    _log("  Reading source schema ...")
    pf       = pq.ParquetFile(source)
    n_total  = pf.metadata.num_rows
    _log(f"  Total docs   : {n_total:,}")
    _log(f"  Shard size   : {shard_size:,}")
    _log(f"  Shards needed: {-(-n_total // shard_size):,}  "
         f"(~{shard_size * 1024 * 4 / 1e6:.0f} MB each)")

    if dry_run:
        _log("  [dry-run] — no files written")
        return

    # ── Check resume state ────────────────────────────────────────────────────
    manifest = _load_manifest(out_dir)
    done_shards: set[int] = set(manifest.get("completed_shards", []))
    if done_shards:
        _log(f"  Resuming — {len(done_shards)} shards already written")

    # ── Load embedding model ──────────────────────────────────────────────────
    _log(f"  Device       : {device}")
    _log(f"  Model        : {model_name}")
    _embed = _load_module("03_embed.py")
    _proc  = _load_module("02_process_text.py")

    model = _embed.getModel(device=device, modelName=model_name)
    # Cap sequence length so attention (O(L²)) doesn't OOM on long abstracts.
    # 512 tokens covers >99% of scientific abstracts; Qwen3 default is 32768.
    model.max_seq_length = max_length
    _log(f"  Max tokens   : {max_length}  (tokenizer truncation)")
    query = (
        "Given a scientific paper title and abstract, "
        "produce an embedding that captures the research topic."
    )

    # ── Stream through source, accumulate shards ───────────────────────────────
    from tqdm import tqdm as _tqdm

    shard_id    = 0
    shard_buf_df:  list[pd.DataFrame] = []
    shard_buf_emb: list[np.ndarray]   = []
    shard_rows  = 0
    total_done  = sum(
        pq.ParquetFile(out_dir / SHARD_FMT.format(i)).metadata.num_rows
        for i in done_shards
        if (out_dir / SHARD_FMT.format(i)).exists()
    ) if done_shards else 0

    t_start = time.time()
    outer_batch = batch_size * 4   # rows read from parquet per iteration

    pbar = _tqdm(
        total   = n_total,
        initial = total_done,
        desc    = "Embedding",
        unit    = "doc",
        dynamic_ncols = True,
    )

    for raw_batch in pf.iter_batches(batch_size=outer_batch):
        chunk = raw_batch.to_pandas()

        # Clean text if needed
        if "clean_abstract" not in chunk.columns:
            chunk = _proc.apply_clean_text_to_df(chunk)

        # Embed this batch — inner tqdm suppressed, outer bar tracks progress
        docs = _embed.get_title_and_abstract(
            chunk, title_col="title", abs_col="clean_abstract"
        )
        embs = _embed.getEmbeddings(
            model=model,
            query=query,
            documents=docs,
            device=device,
            batch_size=batch_size,
            show_progress=False,
        )
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32, copy=False)

        pbar.update(len(chunk))

        # Accumulate into shard buffer
        shard_buf_df.append(chunk.drop(columns=["embeddings"], errors="ignore"))
        shard_buf_emb.append(embs)
        shard_rows += len(chunk)

        # Flush complete shards
        while shard_rows >= shard_size:
            # Build one full shard from the buffer
            df_full  = pd.concat(shard_buf_df,  ignore_index=True)
            emb_full = np.vstack(shard_buf_emb)

            df_shard  = df_full.iloc[:shard_size].copy()
            emb_shard = emb_full[:shard_size]

            # Remainder stays in buffer
            df_rem    = df_full.iloc[shard_size:].copy()
            emb_rem   = emb_full[shard_size:]
            shard_buf_df  = [df_rem]  if len(df_rem)  > 0 else []
            shard_buf_emb = [emb_rem] if len(emb_rem) > 0 else []
            shard_rows    = len(df_rem)

            shard_path = out_dir / SHARD_FMT.format(shard_id)
            if shard_id not in done_shards:
                _write_shard(df_shard, emb_shard, shard_path)
                done_shards.add(shard_id)
                total_done += shard_size
                elapsed = time.time() - t_start
                rate    = total_done / elapsed if elapsed > 0 else 0
                eta     = (n_total - total_done) / rate if rate > 0 else float("inf")
                pbar.write(f"  ✅ shard {shard_id:05d}  "
                           f"{total_done:>8,}/{n_total:,}  "
                           f"{rate:,.0f} doc/s  ETA {eta/60:.0f} min")
                _save_manifest(out_dir, {
                    "n_docs":            total_done,
                    "n_shards":          len(done_shards),
                    "shard_size":        shard_size,
                    "model_name":        model_name,
                    "completed_shards":  sorted(done_shards),
                    "updated_at":        datetime.now().isoformat(),
                })
            else:
                _log(f"  ⏭  shard {shard_id:05d} already done — skipping")

            shard_id += 1

    # ── Flush the last (partial) shard ────────────────────────────────────────
    if shard_buf_df and any(len(x) > 0 for x in shard_buf_df):
        df_last  = pd.concat(shard_buf_df,  ignore_index=True)
        emb_last = np.vstack(shard_buf_emb)
        if len(df_last) > 0 and shard_id not in done_shards:
            shard_path = out_dir / SHARD_FMT.format(shard_id)
            _write_shard(df_last, emb_last, shard_path)
            done_shards.add(shard_id)
            total_done += len(df_last)
            pbar.write(f"  ✅ shard {shard_id:05d}  (final, {len(df_last):,} rows)")

    pbar.close()

    # ── Final manifest ────────────────────────────────────────────────────────
    _save_manifest(out_dir, {
        "n_docs":           total_done,
        "n_shards":         len(done_shards),
        "shard_size":       shard_size,
        "emb_dim":          1024,
        "model_name":       model_name,
        "completed_shards": sorted(done_shards),
        "completed_at":     datetime.now().isoformat(),
    })

    elapsed = time.time() - t_start
    _section("EMBEDDING COMPLETE")
    _log(f"  Documents embedded : {total_done:,}")
    _log(f"  Shards written     : {len(done_shards)}")
    _log(f"  Output directory   : {out_dir}")
    _log(f"  Total time         : {elapsed/60:.1f} min")
    _log(f"  Throughput         : {total_done/elapsed:,.0f} docs/s")
    _log()
    _log("  To sample and train a new model:")
    _log(f"    python pipeline/run_training.py \\")
    _log(f"        --run-name v3 \\")
    _log(f"        --source   {out_dir} \\")
    _log(f"        --sample-size 150000 \\")
    _log(f"        --suggest-lcc models/release")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Memory-free re-embedding for 3M+ documents (sharded output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run-name",   required=True,
                        help="Name for this embedding run — output goes to "
                             "data/embedded/{run_name}/")
    parser.add_argument("--source",     required=True,
                        help="Cleaned English parquet to embed "
                             "(e.g. data/cleaned/english_FullDataset.parquet)")
    parser.add_argument("--device",     default=os.getenv("DEVICE", "cpu"),
                        help="Compute device: cpu / cuda / mps  (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Embedding batch size per GPU pass (default: 32). "
                             "Reduce to 8–16 if CUDA OOM.")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max tokens per document (default: 512). "
                             "Qwen3 supports 32768 but 512 covers >99%% of abstracts "
                             "and dramatically reduces VRAM. Lower this if OOM persists.")
    parser.add_argument("--shard-size", type=int, default=50_000,
                        help="Rows per shard parquet file (default: 50000 ≈ 200 MB)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B",
                        help="HuggingFace embedding model")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print plan without writing anything")
    args = parser.parse_args()

    out_dir = EMBEDDED_DIR / args.run_name

    _section(f"RE-EMBED  run={args.run_name}  {datetime.now():%Y-%m-%d %H:%M}")
    _log(f"  Source     : {args.source}")
    _log(f"  Output     : {out_dir}")
    _log(f"  Device     : {args.device}  batch={args.batch_size}")
    _log(f"  Max tokens : {args.max_length}")
    _log(f"  Shard size : {args.shard_size:,}")
    _log(f"  Model      : {args.model_name}")

    run_reembed(
        source     = Path(args.source),
        out_dir    = out_dir,
        device     = args.device,
        batch_size = args.batch_size,
        model_name = args.model_name,
        shard_size = args.shard_size,
        max_length = args.max_length,
        dry_run    = args.dry_run,
    )


if __name__ == "__main__":
    main()
