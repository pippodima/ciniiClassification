import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime
import os, json, hashlib, tempfile
from datetime import datetime



# -----------------------------------------------------
# Embedding helpers
# -----------------------------------------------------
def getModel(device="cpu", modelName="Qwen/Qwen3-Embedding-0.6B"):
    return SentenceTransformer(modelName, device=device)


def get_title_and_abstract(df: pd.DataFrame, title_col="title_en", abs_col="abstract_en"):
    df["full_text"] = df[title_col].fillna("") + ". " + df[abs_col].fillna("")
    return df["full_text"].tolist()


def getEmbeddings(model, query, documents, device="cpu", batch_size=32):
    _ = model.encode(query, prompt_name="query", device=device)

    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
        batch = documents[i:i + batch_size]
        batch_embs = model.encode(batch)
        embeddings.extend(batch_embs)

    return np.array(embeddings)
    



def _atomic_write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path), suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _fingerprint_documents(documents):
    """
    Stable fingerprint of inputs to prevent resuming on different data/order.
    """
    h = hashlib.sha256()
    for d in documents:
        if d is None:
            d = ""
        if not isinstance(d, str):
            d = str(d)
        h.update(d.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()

def getEmbeddings_checkpointed(
    model,
    query,
    documents,
    device="cpu",
    batch_size=32,
    checkpoint_dir="output",
    checkpoint_name="embeddings_ckpt",
    save_every_n_batches=10,
):
    """
    Crash-proof embeddings:
      - Writes embeddings to a memmap on disk continuously
      - Atomically updates a JSON state file
      - Resumes safely using an input fingerprint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    state_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.state.json")
    mmap_path  = os.path.join(checkpoint_dir, f"{checkpoint_name}.embs.mmap")

    # Prime query embedding (as you do)
    _ = model.encode(query, prompt_name="query", device=device)

    n_docs = len(documents)
    doc_fp = _fingerprint_documents(documents)

    # -------------------------
    # Resume / initialize state
    # -------------------------
    next_i = 0
    dim = None
    dtype = np.float32

    if os.path.exists(state_path) and os.path.exists(mmap_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            if state.get("n_docs") != n_docs:
                raise ValueError(f"n_docs mismatch (ckpt {state.get('n_docs')} vs current {n_docs})")
            if state.get("doc_fingerprint") != doc_fp:
                raise ValueError("Input fingerprint mismatch (data/order changed). Refusing to resume.")

            dim = int(state["dim"])
            next_i = int(state["next_i"])
            if next_i < 0 or next_i > n_docs:
                raise ValueError("Corrupt checkpoint: next_i out of range")

            embs = np.memmap(mmap_path, mode="r+", dtype=dtype, shape=(n_docs, dim))
            print(f"🔁 Resuming from checkpoint: next_i={next_i}/{n_docs}, dim={dim}")

        except Exception as e:
            print(f"⚠️ Checkpoint exists but failed to load safely: {e}")
            print("   -> Starting fresh (keeping old files is safer than deleting automatically).")
            next_i = 0
            dim = None
            embs = None
    else:
        embs = None

    # -------------------------
    # If starting fresh, infer dim with 1 small batch
    # -------------------------
    if dim is None:
        if n_docs == 0:
            return np.zeros((0, 0), dtype=dtype)

        # Compute first batch to infer embedding dim
        first_batch = documents[0 : min(batch_size, n_docs)]
        first_embs = np.asarray(model.encode(first_batch))
        if first_embs.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {first_embs.shape}")
        dim = int(first_embs.shape[1])

        # Create memmap and write first batch
        embs = np.memmap(mmap_path, mode="w+", dtype=dtype, shape=(n_docs, dim))
        embs[0 : first_embs.shape[0]] = first_embs.astype(dtype, copy=False)
        embs.flush()

        next_i = first_embs.shape[0]

        _atomic_write_json(state_path, {
            "timestamp": datetime.now().isoformat(),
            "next_i": next_i,
            "n_docs": n_docs,
            "dim": dim,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "save_every_n_batches": save_every_n_batches,
            "doc_fingerprint": doc_fp,
            "mmap_path": mmap_path,
        })

        print(f"🆕 Initialized checkpoint: next_i={next_i}/{n_docs}, dim={dim}")

    # Already complete?
    if next_i >= n_docs:
        print("✅ Embeddings already complete.")
        # Return a normal ndarray copy (so caller doesn't depend on mmap file handle)
        return np.array(embs, copy=True)

    # -------------------------
    # Main loop with crash safety
    # -------------------------
    start_range = range(next_i, n_docs, batch_size)
    pbar = tqdm(start_range, desc="Embedding batches (crash-proof)")
    batches_since_save = 0

    try:
        for i in pbar:
            batch = documents[i : i + batch_size]
            batch_embs = np.asarray(model.encode(batch))

            if batch_embs.ndim != 2 or batch_embs.shape[1] != dim:
                raise ValueError(f"Embedding shape changed: got {batch_embs.shape}, expected (*, {dim})")

            batch_len = batch_embs.shape[0]
            embs[i : i + batch_len] = batch_embs.astype(dtype, copy=False)

            next_i = i + batch_len
            batches_since_save += 1
            pbar.set_postfix(done=next_i, total=n_docs)

            # Periodic durable save
            if batches_since_save >= save_every_n_batches or next_i >= n_docs:
                embs.flush()
                _atomic_write_json(state_path, {
                    "timestamp": datetime.now().isoformat(),
                    "next_i": next_i,
                    "n_docs": n_docs,
                    "dim": dim,
                    "dtype": str(dtype),
                    "batch_size": batch_size,
                    "save_every_n_batches": save_every_n_batches,
                    "doc_fingerprint": doc_fp,
                    "mmap_path": mmap_path,
                })
                batches_since_save = 0

    except KeyboardInterrupt:
        # Make Ctrl+C safe
        print("\n🛑 Interrupted. Saving progress...")
        try:
            embs.flush()
        except Exception:
            pass
        _atomic_write_json(state_path, {
            "timestamp": datetime.now().isoformat(),
            "next_i": next_i,
            "n_docs": n_docs,
            "dim": dim,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "save_every_n_batches": save_every_n_batches,
            "doc_fingerprint": doc_fp,
            "mmap_path": mmap_path,
            "note": "Interrupted by user",
        })
        raise

    except Exception as e:
        # Make any crash safe
        print(f"\n💥 Error during embedding: {e}")
        print("   Saving progress before exiting...")
        try:
            embs.flush()
        except Exception:
            pass
        _atomic_write_json(state_path, {
            "timestamp": datetime.now().isoformat(),
            "next_i": next_i,
            "n_docs": n_docs,
            "dim": dim,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "save_every_n_batches": save_every_n_batches,
            "doc_fingerprint": doc_fp,
            "mmap_path": mmap_path,
            "note": f"Exception: {type(e).__name__}",
        })
        raise

    # Done
    print("✅ Done embedding. Final flush...")
    embs.flush()
    _atomic_write_json(state_path, {
        "timestamp": datetime.now().isoformat(),
        "next_i": n_docs,
        "n_docs": n_docs,
        "dim": dim,
        "dtype": str(dtype),
        "batch_size": batch_size,
        "save_every_n_batches": save_every_n_batches,
        "doc_fingerprint": doc_fp,
        "mmap_path": mmap_path,
    })

    return np.array(embs, copy=True)
