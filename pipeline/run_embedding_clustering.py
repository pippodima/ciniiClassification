import os
import sys
import importlib.util
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure pipeline/ is on sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_df, save, _get_device
from clustering_utils import auto_optimize_clustering
from labeling_utils import label_clusters
from plot_utils import plot_static, plot_interactive
from datetime import datetime
import numpy as np
import pandas as pd
from config import ENGLISH, EMBEDDED_DIR, CLUSTERED_DATA, CLUSTER_METADATA, PIPELINE_METADATA, UMAP_PNG, UMAP_HTML, EMBEDDED_26K

# Import from numbered stage files (can't use normal import on digit-prefixed names)
def _load_module(filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_embed = _load_module("03_embed.py")
getEmbeddings = _embed.getEmbeddings
getModel = _embed.getModel
get_title_and_abstract = _embed.get_title_and_abstract
getEmbeddings_checkpointed = _embed.getEmbeddings_checkpointed

_proc = _load_module("02_process_text.py")
apply_clean_text_to_df = _proc.apply_clean_text_to_df


OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(EMBEDDED_DIR))


def _save_chunked(df: pd.DataFrame, embeddings: np.ndarray, output_path: str, chunk_size: int = 2000) -> None:
    """
    Write df + embeddings to Parquet in chunks so the full embedding matrix
    never needs to live as Python objects in RAM simultaneously.

    Atomic write: data goes to a .tmp file and is renamed to output_path only
    on full success.  A crash during save leaves no partial output file, so
    run_stage correctly detects the stage as incomplete on restart.

    Schema coercion: Arrow infers type `null` for all-null columns in a chunk,
    which conflicts with `string` in later chunks.  We promote null→string in
    the file schema so every chunk can be safely cast.
    """
    from pathlib import Path
    out     = Path(output_path)
    tmp     = out.with_suffix(".tmp.parquet")
    n       = len(df)
    df_rest = df.reset_index(drop=True)
    writer  = None
    schema  = None

    try:
        for start in range(0, n, chunk_size):
            end   = min(start + chunk_size, n)
            chunk = df_rest.iloc[start:end].copy()
            chunk["embeddings"] = [embeddings[i].tolist() for i in range(start, end)]
            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                # Promote null-typed fields → string so all chunks share one schema
                fields = [
                    pa.field(f.name, pa.string() if f.type == pa.null() else f.type)
                    for f in table.schema
                ]
                schema = pa.schema(fields)
                writer = pq.ParquetWriter(str(tmp), schema, compression="snappy")

            writer.write_table(table.cast(schema, safe=False))

    except Exception:
        if writer:
            writer.close()
        if tmp.exists():
            tmp.unlink()
        raise

    writer.close()
    import os
    os.replace(tmp, out)   # atomic: output_path is absent or complete, never partial
    print(f"  ✅ Saved {n:,} rows → {output_path}")


INPUT_PATH = os.getenv("INPUT_PATH", str(ENGLISH))

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
DEVICE = os.getenv("DEVICE", "cpu")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
SAVE_EVERY_N_BATCHES = int(os.getenv("SAVE_EVERY_N_BATCHES", "10"))

# You can tune these for scaling:
# - Smaller batch size reduces RAM, increases runtime
# - Higher save cadence reduces I/O, increases work lost on crash
query_prompt = (
    "Given a scientific paper title and abstract, produce an embedding that captures the research topic."
)

def main():
    print("📥 Loading data...")
    df = load_df(INPUT_PATH)

    if "clean_abstract" not in df.columns:
        print("🧹 Cleaning text...")
        df = apply_clean_text_to_df(df)
    else:
        print("🧹 clean_abstract already present — skipping re-clean.")

    print("🧠 Loading embedding model...")
    model = getModel(device=DEVICE, modelName=MODEL_NAME)

    print("📝 Preparing documents...")
    documents = get_title_and_abstract(df, title_col="title", abs_col="clean_abstract")

    print("🔢 Generating embeddings (checkpointed)...")
    embeddings = getEmbeddings_checkpointed(
        model=model,
        query=query_prompt,
        documents=documents,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        checkpoint_dir=f"{OUTPUT_DIR}/checkpoints",
        checkpoint_name="embeddings_ckpt",
        save_every_n_batches=SAVE_EVERY_N_BATCHES,
    )

    # Ensure embeddings are float32 (smaller, faster)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "embedded.parquet")

    print("💾 Saving outputs (chunked write — scalable)...")
    _save_chunked(df, embeddings, output_path, chunk_size=2000)

    # ── Validation report ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  EMBEDDING STAGE REPORT")
    print("=" * 55)
    print(f"  Documents embedded : {len(df):,}")
    print(f"  Embedding dim      : {embeddings.shape[1]}")
    norms = np.linalg.norm(embeddings[:1000], axis=1)
    print(f"  L2 norms (first 1k): mean={norms.mean():.4f}  std={norms.std():.4f}")
    if "full_text" in df.columns:
        jats_ft = df["full_text"].str.contains("<jats:", na=False).sum()
        print(f"  JATS in full_text  : {jats_ft}  (should be 0)")
    if "clean_abstract" in df.columns:
        jats_ca = df["clean_abstract"].str.contains("<jats:", na=False).sum()
        print(f"  JATS in clean_abs  : {jats_ca}  (should be 0)")
    print(f"  Output             : {output_path}")
    print("=" * 55)
    print("✅ Done.")


if __name__ == "__main__":
    main()
