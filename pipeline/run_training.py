"""
pipeline/run_training.py
========================
Crashproof retraining pipeline: sample → embed → cluster → [manual LCC] → train.

Stages
------
  1. sample   — Draw N documents from any parquet (cleaned text OR pre-embedded)
  2. embed    — Generate Qwen3 embeddings (checkpointed, fully resumable)
               SKIPPED AUTOMATICALLY if --source already has an embeddings column
  3. cluster  — UMAP + HDBSCAN with silhouette auto-tuning; output lcc_mapping.csv
  [PAUSE]     — Fill in lcc_subclass + lcc_division columns in lcc_mapping.csv
  4. train    — Validate mapping, build training dataset, train Model A & B

Usage
-----
    # From cleaned text parquet (will embed the sample)
    python pipeline/run_training.py --run-name v3 --sample-size 50000

    # From run_server embedded output (SKIPS re-embedding — fastest path)
    python pipeline/run_training.py --run-name v3 \
        --source data/embedded/FullDataset/embedded.parquet \
        --sample-size 150000 \
        --suggest-lcc models/release

    # Resume after filling in lcc_mapping.csv
    python pipeline/run_training.py --run-name v3 --train

    # Skip stages already completed
    python pipeline/run_training.py --run-name v3 --skip sample embed

    # Force redo a stage
    python pipeline/run_training.py --run-name v3 --force cluster

    # Bootstrap LCC suggestions from existing model (dramatically reduces manual work)
    python pipeline/run_training.py --run-name v3 --suggest-lcc models/release

    # Dry run — print what would happen without doing it
    python pipeline/run_training.py --run-name v3 --dry-run

Output layout
-------------
    training_runs/{run_name}/
        sample.parquet           sampled documents (title, abstract, clean_abstract, ...)
        embedded.parquet         sample + embeddings column
        clusters.parquet         embedded + cluster_id column (−1 = outlier)
        cluster_metadata.parquet per-cluster: id, n_papers, label, keywords
        lcc_mapping.csv          ← EDIT THIS: fill lcc_subclass + lcc_division
        lcc_mapping.parquet      validated mapping (written automatically at train stage)
        training_dataset.parquet final labeled dataset fed to the MLP
        run_state.json           stage completion tracking

    models/{run_name}/
        model_a.pt               flat MLP weights
        model_b.pt               two-head MLP weights
        encoders.pkl             LabelEncoders for div + subclass
        hierarchy.pkl            hier_mask tensor (n_sub × n_div)
        metrics.json             train/val accuracy for both models
        report_a.txt             per-class classification report — Model A
        report_b.txt             per-class classification report — Model B
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PIPELINE = Path(__file__).resolve().parent
ROOT     = PIPELINE.parent
sys.path.insert(0, str(PIPELINE))

from config import ENGLISH, MODELS_DIR

RUNS_DIR   = ROOT / "training_runs"
STATE_FILE = "run_state.json"

def _hdbscan_grid(n: int) -> list[tuple[int, int]]:
    """
    HDBSCAN search space: (min_cluster_size, min_samples).

    min_cluster_size is an ABSOLUTE row count, but the cluster count it
    produces depends on the dataset size — e.g. mcs=50 gave 247 clusters at
    n=100k but 600+ at n=300k (same density, 3x more points per cluster).
    A fixed grid that worked at one scale silently produces zero in-range
    configs at another (see PROJECT_DIARY chapter 17).

    So candidates are scaled as fractions of n, keeping the resulting
    cluster count roughly constant (and inside the LCC-aligned target
    window) regardless of sample size. min_samples doesn't need scaling —
    it controls density-estimate conservativeness, not absolute size.
    """
    fracs = [0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006]
    mcs_candidates = sorted(set(max(15, int(n * f)) for f in fracs))
    return [(mcs, ms) for mcs in mcs_candidates for ms in (3, 5, 7)]


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


# ─── State management ─────────────────────────────────────────────────────────

def _load_state(run_dir: Path) -> dict:
    p = run_dir / STATE_FILE
    return json.loads(p.read_text()) if p.exists() else {}

def _save_state(run_dir: Path, stage: str, meta: dict | None = None):
    state = _load_state(run_dir)
    state[stage] = {"done": True, "ts": datetime.now().isoformat(), **(meta or {})}
    (run_dir / STATE_FILE).write_text(json.dumps(state, indent=2))

def _stage_done(run_dir: Path, stage: str) -> bool:
    return _load_state(run_dir).get(stage, {}).get("done", False)


# ─── Atomic file helpers ──────────────────────────────────────────────────────

def _atomic_parquet(df: pd.DataFrame, path: Path, **kwargs):
    """Write parquet via a .tmp file — output is never partial."""
    tmp = path.with_suffix(".tmp.parquet")
    try:
        # Promote null-typed columns to string so all chunks share one schema
        table = pa.Table.from_pandas(df, preserve_index=False)
        fields = [
            pa.field(f.name, pa.string() if f.type == pa.null() else f.type)
            for f in table.schema
        ]
        table = table.cast(pa.schema(fields), safe=False)
        pq.write_table(table, str(tmp), compression="snappy")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".tmp.csv")
    try:
        df.to_csv(str(tmp), index=False, encoding="utf-8-sig")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


# ─── Dynamic module loader ─────────────────────────────────────────────────────

def _load_module(filename: str):
    path = PIPELINE / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — SAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_parquet_chunked(source: Path, k: int, seed: int) -> pd.DataFrame:
    """
    Memory-efficient random sampling from a single parquet file OR a directory
    of shard parquets (output of run_reembed.py).
    Never loads the full dataset into RAM.
    """
    import pyarrow.dataset as _ds

    rng     = np.random.default_rng(seed)
    dataset = _ds.dataset(str(source), format="parquet")
    n_rows  = dataset.count_rows()

    if n_rows <= k:
        return dataset.to_table().to_pandas()

    # Sample row indices (sorted for sequential reads)
    chosen     = np.sort(rng.choice(n_rows, size=k, replace=False))
    chosen_set = set(chosen.tolist())

    rows: list[pd.DataFrame] = []
    global_i = 0
    for batch in dataset.to_batches(batch_size=20_000):
        try:
            chunk = batch.to_pandas()
        except Exception:
            global_i += batch.num_rows
            continue
        n_chunk = len(chunk)
        local   = [i for i in range(n_chunk) if (global_i + i) in chosen_set]
        if local:
            rows.append(chunk.iloc[local].copy())
        global_i += n_chunk
        if global_i > int(chosen[-1]):
            break

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def stage_sample(run_dir: Path, source: Path, sample_size: int,
                 seed: int = 42) -> tuple[int, bool]:
    """
    Returns (n_sampled, has_embeddings).
    If has_embeddings is True, the sample is written to embedded.parquet
    and the embed stage is auto-skipped.

    Source can be:
      - A directory produced by run_reembed.py (sharded, has manifest.json)
      - A single parquet file with an 'embeddings' column
      - A single parquet file without embeddings (text only)
    """
    _log(f"  Source  : {source}")

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # ── Sharded directory from run_reembed.py ─────────────────────────────────
    if source.is_dir():
        from run_reembed import sample_from_shards
        manifest = json.loads((source / "manifest.json").read_text())
        n_total  = manifest["n_docs"]
        k        = min(sample_size, n_total)
        _log(f"  Mode    : sharded directory  ({manifest['n_shards']} shards)")
        _log(f"  Pool    : {n_total:,} documents")
        _log(f"  Sample  : {k:,} documents  (seed={seed})")
        _log("  → embed stage will be skipped (embeddings already computed)")

        sample = sample_from_shards(source, k, seed)

        text_cols = [c for c in sample.columns if c != "embeddings"]
        _atomic_parquet(sample[text_cols], run_dir / "sample.parquet")
        _atomic_parquet(sample,            run_dir / "embedded.parquet")
        _log(f"  ✅ sample.parquet + embedded.parquet → {run_dir}")
        return k, True

    # ── Single parquet file ───────────────────────────────────────────────────
    schema_names   = pq.ParquetFile(source).schema_arrow.names
    has_embeddings = "embeddings" in schema_names
    n_rows         = pq.ParquetFile(source).metadata.num_rows
    k              = min(sample_size, n_rows)

    _log(f"  Mode    : single parquet")
    _log(f"  Pool    : {n_rows:,} documents")
    _log(f"  Sample  : {k:,} documents  (seed={seed})")
    if has_embeddings:
        _log("  → embed stage will be skipped (source has embeddings column)")

    sample = _sample_parquet_chunked(source, k, seed)

    text_cols = [c for c in sample.columns if c != "embeddings"]
    _atomic_parquet(sample[text_cols], run_dir / "sample.parquet")
    _log(f"  ✅ sample.parquet → {run_dir / 'sample.parquet'}")

    if has_embeddings:
        _atomic_parquet(sample, run_dir / "embedded.parquet")
        _log(f"  ✅ embedded.parquet → {run_dir / 'embedded.parquet'}")

    return k, has_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — EMBED
# ═══════════════════════════════════════════════════════════════════════════════

def stage_embed(run_dir: Path, device: str, batch_size: int, model_name: str) -> int:
    in_path  = run_dir / "sample.parquet"
    out_path = run_dir / "embedded.parquet"
    ckpt_dir = run_dir / "embed_checkpoints"

    _log(f"  Input   : {in_path}")
    _log(f"  Model   : {model_name}")
    _log(f"  Device  : {device}  batch={batch_size}")

    _embed = _load_module("03_embed.py")
    _proc  = _load_module("02_process_text.py")

    df = pd.read_parquet(str(in_path))

    if "clean_abstract" not in df.columns:
        _log("  Cleaning text (clean_abstract missing) ...")
        df = _proc.apply_clean_text_to_df(df)

    model = _embed.getModel(device=device, modelName=model_name)
    documents = _embed.get_title_and_abstract(df, title_col="title", abs_col="clean_abstract")

    query = (
        "Given a scientific paper title and abstract, "
        "produce an embedding that captures the research topic."
    )
    embeddings = _embed.getEmbeddings_checkpointed(
        model=model,
        query=query,
        documents=documents,
        device=device,
        batch_size=batch_size,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_name="embed_ckpt",
        save_every_n_batches=10,
    )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)

    _log(f"  Embedding dim : {embeddings.shape[1]}")
    norms = np.linalg.norm(embeddings[:1000], axis=1)
    _log(f"  L2 norms (1k) : mean={norms.mean():.4f}  std={norms.std():.4f}")

    # Write in chunks so the embedding matrix is never fully in RAM as Python objs
    n, chunk = len(df), 2000
    tmp = out_path.with_suffix(".tmp.parquet")
    writer = schema = None
    try:
        for start in range(0, n, chunk):
            end    = min(start + chunk, n)
            c      = df.iloc[start:end].copy().reset_index(drop=True)
            c["embeddings"] = [embeddings[i].tolist() for i in range(start, end)]
            table  = pa.Table.from_pandas(c, preserve_index=False)
            if writer is None:
                fields = [
                    pa.field(f.name, pa.string() if f.type == pa.null() else f.type)
                    for f in table.schema
                ]
                schema = pa.schema(fields)
                writer = pq.ParquetWriter(str(tmp), schema, compression="snappy")
            writer.write_table(table.cast(schema, safe=False))
    except Exception:
        if writer: writer.close()
        tmp.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(tmp, out_path)
    _log(f"  ✅ Saved {n:,} rows → {out_path}")
    return n


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CLUSTER
# ═══════════════════════════════════════════════════════════════════════════════

def _tfidf_keywords(texts: pd.Series, n: int = 12) -> list[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    stop = list(ENGLISH_STOP_WORDS | {
        "abstract", "paper", "study", "result", "results", "method", "methods",
        "using", "used", "also", "data", "analysis", "based", "proposed",
    })
    texts = texts.dropna()
    if len(texts) < 2:
        return []
    vec = TfidfVectorizer(
        max_features=30_000, stop_words=stop,
        ngram_range=(1, 2), min_df=2, sublinear_tf=True,
    )
    try:
        X      = vec.fit_transform(texts)
        scores = np.asarray(X.mean(axis=0)).flatten()
        idx    = scores.argsort()[-n:][::-1]
        return list(vec.get_feature_names_out()[idx])
    except ValueError:
        return []


def _suggest_lcc(df_emb: pd.DataFrame, clusters: np.ndarray,
                 model_dir: Path, device: str) -> dict[int, tuple[str, str, float, float]]:
    """
    Run existing model on embedded docs; for each cluster return the majority-vote
    (lcc_subclass, lcc_division, mean_confidence, agreement_rate).
    Agreement rate = fraction of docs in the cluster that agree with the majority.
    """
    _log("  Loading model artefacts for LCC suggestion bootstrap ...")
    _cls = _load_module("11_classify.py")

    model, le_div, le_sub, hier_mask = _cls.load_artefacts(model_dir, "b", device)

    embs = np.vstack(df_emb["embeddings"].values).astype(np.float32)
    batch = 512
    all_div, all_sub, all_conf = [], [], []
    for i in range(0, len(embs), batch):
        d, s, c = _cls.predict_batch(model, embs[i:i+batch], hier_mask, device, "b")
        all_div.append(d); all_sub.append(s); all_conf.append(c)

    div_idx = np.concatenate(all_div)
    sub_idx = np.concatenate(all_sub)
    conf    = np.concatenate(all_conf)

    pred_sub = le_sub.inverse_transform(sub_idx)
    pred_div = le_div.inverse_transform(div_idx)

    suggestions: dict[int, tuple[str, str, float, float]] = {}
    unique_clusters = np.unique(clusters[clusters != -1])
    for cid in unique_clusters:
        mask = clusters == cid
        subs  = pred_sub[mask]
        divs  = pred_div[mask]
        confs = conf[mask]

        # Majority vote
        maj_sub = pd.Series(subs).mode().iloc[0]
        maj_div = pd.Series(divs[subs == maj_sub]).mode().iloc[0]
        agreement = float((subs == maj_sub).mean())
        mean_conf = float(confs.mean())
        suggestions[int(cid)] = (maj_sub, maj_div, mean_conf, agreement)

    _log(f"  ✅ Bootstrap suggestions generated for {len(suggestions)} clusters")
    return suggestions


def stage_cluster(run_dir: Path,
                  suggest_model_dir: Path | None = None,
                  device: str = "cpu",
                  min_clusters: int = 100,
                  max_clusters: int = 250,
                  umap_components: int = 15) -> int:
    in_path       = run_dir / "embedded.parquet"
    out_clusters  = run_dir / "clusters.parquet"
    out_meta      = run_dir / "cluster_metadata.parquet"
    out_csv       = run_dir / "lcc_mapping.csv"

    _log(f"  Input  : {in_path}")

    try:
        import umap
        import hdbscan as hdbscan_lib
        from sklearn.metrics import silhouette_score
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Install with: pip install umap-learn hdbscan"
        ) from e

    df = pd.read_parquet(str(in_path))
    embs = np.vstack(df["embeddings"].values).astype(np.float32)
    n = len(embs)
    _log(f"  Documents : {n:,}  —  embedding dim: {embs.shape[1]}")

    # ── UMAP reduction ────────────────────────────────────────────────────────
    _log(f"  UMAP reduction to {umap_components} components ...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=umap_components,
        n_neighbors=min(30, n // 10),
        metric="cosine",
        min_dist=0.0,
        random_state=42,
        low_memory=(n > 50_000),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*force_all_finite.*")
        warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
        reduced = reducer.fit_transform(embs)
    _log(f"    done in {time.time()-t0:.1f}s")

    # ── HDBSCAN auto-tune ─────────────────────────────────────────────────────
    _log(f"  Auto-tuning HDBSCAN (target {min_clusters}–{max_clusters} clusters) ...")
    # Score on a random subsample for speed
    sil_sample = min(5_000, n)
    sil_idx    = np.random.default_rng(42).choice(n, sil_sample, replace=False)

    results = []
    for mcs, ms in _hdbscan_grid(n):
        try:
            model = hdbscan_lib.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",
                cluster_selection_method="eom",
                core_dist_n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*force_all_finite.*")
                warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
                labels = model.fit_predict(reduced)
            n_cls  = int((labels >= 0).sum() > 0 and len(set(labels[labels >= 0])))
            n_out  = int((labels == -1).sum())
            out_rate = n_out / n

            if n_cls < min_clusters or n_cls > max_clusters:
                score = -1.0
            else:
                valid_sil = labels[sil_idx] >= 0
                if valid_sil.sum() < 50 or len(set(labels[sil_idx][valid_sil])) < 2:
                    score = -1.0
                else:
                    score = float(silhouette_score(
                        reduced[sil_idx][valid_sil], labels[sil_idx][valid_sil],
                        metric="euclidean", sample_size=min(2000, valid_sil.sum())
                    ))
            results.append(dict(mcs=mcs, ms=ms, n_clusters=n_cls,
                                outlier_rate=round(out_rate, 4), score=round(score, 4)))
            _log(f"    mcs={mcs:2d} ms={ms}  clusters={n_cls:3d}  "
                 f"outliers={out_rate:.1%}  score={score:.4f}")
        except Exception as e:
            _log(f"    mcs={mcs:2d} ms={ms}  ERROR: {e}")

    # Pick best scoring config in valid range
    valid = [r for r in results if r["score"] > 0]
    if not valid:
        raise RuntimeError(
            "No valid HDBSCAN configuration found. "
            "Try adjusting --min-clusters or --max-clusters."
        )
    best = max(valid, key=lambda r: r["score"])
    _log(f"\n  Best config: mcs={best['mcs']} ms={best['ms']}  "
         f"clusters={best['n_clusters']}  outliers={best['outlier_rate']:.1%}  "
         f"score={best['score']:.4f}")

    # Re-fit with best config to get final labels
    final_model = hdbscan_lib.HDBSCAN(
        min_cluster_size=best["mcs"],
        min_samples=best["ms"],
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*force_all_finite.*")
        warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
        cluster_labels = final_model.fit_predict(reduced)

    # Save tuning results alongside state
    (run_dir / "hdbscan_tuning.json").write_text(
        json.dumps({"best": best, "all": results}, indent=2)
    )

    # ── Save clusters.parquet ─────────────────────────────────────────────────
    df_out = df.copy()
    df_out["cluster_id"] = cluster_labels.astype(int)
    _atomic_parquet(df_out, out_clusters)
    _log(f"  ✅ clusters.parquet → {out_clusters}")

    # ── Build cluster metadata ────────────────────────────────────────────────
    _log("  Building cluster metadata (TF-IDF keywords per cluster) ...")
    text_col = next((c for c in ["clean_abstract", "abstract"] if c in df.columns), None)
    rows = []
    cluster_ids = sorted(set(cluster_labels[cluster_labels >= 0]))

    for cid in cluster_ids:
        mask   = cluster_labels == cid
        texts  = df.loc[mask, text_col] if text_col else pd.Series(dtype=str)
        kws    = _tfidf_keywords(texts, n=12)
        label  = " | ".join(kws[:3]) if kws else f"cluster_{cid}"
        rows.append(dict(
            cluster_id=int(cid),
            n_papers=int(mask.sum()),
            label=label,
            keywords=" | ".join(kws),
        ))

    meta_df = pd.DataFrame(rows).sort_values("n_papers", ascending=False)
    _atomic_parquet(meta_df, out_meta)
    _log(f"  ✅ cluster_metadata.parquet → {out_meta}")

    # ── Optional: bootstrap LCC suggestions from existing model ───────────────
    suggestions: dict[int, tuple[str, str, float, float]] = {}
    if suggest_model_dir and suggest_model_dir.exists():
        try:
            suggestions = _suggest_lcc(df_out, cluster_labels, suggest_model_dir, device)
        except Exception as e:
            _log(f"  ⚠  LCC bootstrap failed ({e}) — continuing without suggestions")

    # ── Write lcc_mapping.csv template ───────────────────────────────────────
    csv_rows = []
    for row in rows:
        cid = row["cluster_id"]
        s   = suggestions.get(cid)
        csv_rows.append(dict(
            cluster_id          = cid,
            n_papers            = row["n_papers"],
            label               = row["label"],
            top_keywords        = row["keywords"],
            suggested_lcc_subclass  = s[0] if s else "",
            suggested_lcc_division  = s[1] if s else "",
            suggestion_confidence   = f"{s[2]:.3f}" if s else "",
            suggestion_agreement    = f"{s[3]:.1%}" if s else "",
            lcc_subclass        = "",   # ← USER FILLS THIS
            lcc_division        = "",   # ← USER FILLS THIS
            skip                = "",   # ← write 'yes' to exclude noisy clusters
            notes               = "",
        ))

    csv_df = pd.DataFrame(csv_rows)
    _atomic_csv(csv_df, out_csv)
    _log(f"  ✅ lcc_mapping.csv template → {out_csv}")

    outliers = int((cluster_labels == -1).sum())
    return best["n_clusters"], outliers


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — VALIDATE LCC + TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_lcc_mapping(run_dir: Path) -> pd.DataFrame:
    """
    Read lcc_mapping.csv, validate, and return a clean DataFrame.
    Raises ValueError with a human-readable message if the mapping is incomplete.
    """
    import re
    csv_path = run_dir / "lcc_mapping.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"lcc_mapping.csv not found in {run_dir}.\n"
            "Run the cluster stage first: python pipeline/run_training.py --run-name ... "
            "(without --train)"
        )

    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # Rows marked skip=yes are excluded from training
    df["skip"] = df["skip"].str.strip().str.lower()
    skipped = df[df["skip"] == "yes"]
    active  = df[df["skip"] != "yes"].copy()

    # Resolve: if lcc_subclass is empty, fall back to suggested
    for col, sug in [("lcc_subclass", "suggested_lcc_subclass"),
                     ("lcc_division",  "suggested_lcc_division")]:
        empty = active[col].str.strip() == ""
        active.loc[empty, col] = active.loc[empty, sug].str.strip()

    # Strip whitespace
    active["lcc_subclass"] = active["lcc_subclass"].str.strip().str.upper()
    active["lcc_division"]  = active["lcc_division"].str.strip().str.upper()

    # Find incomplete rows
    missing = active[
        (active["lcc_subclass"] == "") | (active["lcc_division"] == "")
    ]
    if len(missing):
        rows = ", ".join(str(r) for r in missing["cluster_id"].tolist()[:10])
        raise ValueError(
            f"{len(missing)} cluster(s) have no LCC assignment:\n"
            f"  cluster_ids: {rows}\n"
            f"  → Open {csv_path} and fill in lcc_subclass + lcc_division,\n"
            f"    or set skip=yes for noise clusters."
        )

    # Basic format check: lcc_subclass = 1-3 capital letters; lcc_division starts with them
    bad = []
    for _, row in active.iterrows():
        sub = row["lcc_subclass"]
        div = row["lcc_division"]
        if not re.fullmatch(r"[A-Z]{1,3}", sub):
            bad.append(f"cluster {row['cluster_id']}: bad subclass '{sub}'")
        elif not div.startswith(sub):
            bad.append(f"cluster {row['cluster_id']}: division '{div}' "
                       f"doesn't start with subclass '{sub}'")
    if bad:
        raise ValueError("LCC format errors:\n  " + "\n  ".join(bad[:10]))

    _log(f"  Active clusters : {len(active):,}")
    _log(f"  Skipped clusters: {len(skipped):,}  (noise / unassignable)")
    return active[["cluster_id", "lcc_subclass", "lcc_division", "n_papers", "label"]]


def stage_train(run_dir: Path, model_dir: Path,
                epochs: int, batch_size: int, device: str) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

    # ── Validate LCC mapping ──────────────────────────────────────────────────
    _log("  Validating lcc_mapping.csv ...")
    lcc_df = _validate_lcc_mapping(run_dir)

    # Rename for clarity
    lcc_df = lcc_df.rename(columns={
        "lcc_subclass": "lcc",
        "lcc_division": "lcc_division",
    })
    lcc_df["cluster_id"] = lcc_df["cluster_id"].astype(int)

    # ── Build training dataset ────────────────────────────────────────────────
    _log("  Building training dataset ...")
    clusters_path = run_dir / "clusters.parquet"
    df = pd.read_parquet(str(clusters_path))
    df["cluster_id"] = df["cluster_id"].astype(int)

    # Keep only assigned (non-outlier, non-skipped) clusters
    valid_ids = set(lcc_df["cluster_id"].tolist())
    df = df[df["cluster_id"].isin(valid_ids)].copy()
    df = df.merge(lcc_df[["cluster_id", "lcc", "lcc_division"]], on="cluster_id", how="left")
    df = df.dropna(subset=["lcc", "lcc_division"]).reset_index(drop=True)
    _log(f"  Training documents: {len(df):,}")

    # Save training_dataset.parquet
    td_path = run_dir / "training_dataset.parquet"
    _atomic_parquet(df, td_path)
    _log(f"  ✅ training_dataset.parquet → {td_path}")

    # ── Label encoding ────────────────────────────────────────────────────────
    le_div = LabelEncoder().fit(df["lcc_division"])
    le_sub = LabelEncoder().fit(df["lcc"])
    y_div  = le_div.transform(df["lcc_division"])
    y_sub  = le_sub.transform(df["lcc"])
    n_div  = len(le_div.classes_)
    n_sub  = len(le_sub.classes_)
    _log(f"  Divisions: {n_div}   Subclasses: {n_sub}")

    # ── Embeddings ────────────────────────────────────────────────────────────
    X = np.vstack(df["embeddings"].values).astype(np.float32)
    # L2-normalise
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    # ── Hierarchy mask ────────────────────────────────────────────────────────
    import torch
    sub_to_divs: dict[int, list[int]] = {}
    for sub_name in le_sub.classes_:
        s_idx    = int(le_sub.transform([sub_name])[0])
        sub_rows = df[df["lcc"] == sub_name]["lcc_division"].unique()
        d_idxs   = [int(le_div.transform([d])[0])
                    for d in sub_rows if d in le_div.classes_]
        sub_to_divs[s_idx] = d_idxs

    hier_mask = torch.zeros(n_sub, n_div, dtype=torch.bool)
    for s, ds in sub_to_divs.items():
        for d in ds:
            hier_mask[s, d] = True

    # ── Train / val split ─────────────────────────────────────────────────────
    SEED = 42
    X_tr, X_va, yd_tr, yd_va, ys_tr, ys_va = train_test_split(
        X, y_div, y_sub, test_size=0.20, random_state=SEED, stratify=y_div
    )

    def to_t(*a): return [torch.tensor(x) for x in a]

    Xtr, yd_tr_t, ys_tr_t = to_t(X_tr, yd_tr, ys_tr)
    Xva, yd_va_t, ys_va_t = to_t(X_va, yd_va, ys_va)

    train_dl = DataLoader(TensorDataset(Xtr, yd_tr_t, ys_tr_t),
                          batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(Xva, yd_va_t, ys_va_t),
                          batch_size=512, shuffle=False)

    def class_weights(y, n):
        cw = compute_class_weight("balanced", classes=np.arange(n), y=y)
        return torch.tensor(cw, dtype=torch.float32)

    w_div = class_weights(yd_tr, n_div)
    w_sub = class_weights(ys_tr, n_sub)
    _log(f"  Class weight range (div): [{w_div.min():.2f}, {w_div.max():.2f}]")

    # ── Model definitions ─────────────────────────────────────────────────────
    HIDDEN  = [512, 256]
    DROPOUT = 0.3

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], X.shape[1]
            for h in HIDDEN:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
                prev = h
            self.net     = nn.Sequential(*layers)
            self.out_dim = prev
        def forward(self, x): return self.net(x)

    class ModelA(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb   = Backbone()
            self.head = nn.Linear(self.bb.out_dim, n_div)
        def forward(self, x): return self.head(self.bb(x))

    class ModelB(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb       = Backbone()
            d             = self.bb.out_dim
            self.head_sub = nn.Linear(d, n_sub)
            self.head_div = nn.Linear(d, n_div)
        def forward(self, x):
            f = self.bb(x)
            return self.head_sub(f), self.head_div(f)

    # ── Training loop ─────────────────────────────────────────────────────────
    LR         = 1e-3
    LR_PAT     = 8
    LR_FAC     = 0.5
    EARLY_STOP = 15
    ALPHA      = 0.35   # subclass loss weight in Model B

    def _train_loop(model, get_preds_fn, loss_fn, label: str):
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=LR_PAT, factor=LR_FAC)
        best_val, best_state, no_imp = 0.0, None, 0
        history = []

        for ep in range(1, epochs + 1):
            model.train()
            tr_loss = 0.0
            for batch in train_dl:
                opt.zero_grad()
                loss = loss_fn(model, batch)
                loss.backward(); opt.step()
                tr_loss += loss.item() * len(batch[0])
            tr_loss /= len(train_dl.dataset)

            model.eval()
            va_loss = 0.0
            preds, trues = [], []
            with torch.no_grad():
                for batch in val_dl:
                    va_loss += loss_fn(model, batch).item() * len(batch[0])
                    p, t = get_preds_fn(model, batch)
                    preds.append(p); trues.append(t)
            va_loss /= len(val_dl.dataset)
            p_np = torch.cat(preds).numpy(); t_np = torch.cat(trues).numpy()
            val_bal = balanced_accuracy_score(t_np, p_np)
            val_acc = accuracy_score(t_np, p_np)
            sched.step(va_loss)

            history.append(dict(
                epoch=ep, tr_loss=round(tr_loss, 4), va_loss=round(va_loss, 4),
                val_acc=round(val_acc, 4), val_bal=round(val_bal, 4)
            ))
            if ep % 10 == 0 or ep == 1:
                _log(f"    {label} ep{ep:>3}  tr={tr_loss:.4f}  va={va_loss:.4f}"
                     f"  acc={val_acc:.3f}  bal={val_bal:.3f}"
                     f"  lr={opt.param_groups[0]['lr']:.1e}")

            if val_bal > best_val:
                best_val, best_state, no_imp = val_bal, {
                    k: v.clone() for k, v in model.state_dict().items()
                }, 0
            else:
                no_imp += 1
                if no_imp >= EARLY_STOP:
                    _log(f"    {label} early stop at epoch {ep}")
                    break

        model.load_state_dict(best_state)
        return history, best_val

    crit_div = nn.CrossEntropyLoss(weight=w_div)
    crit_sub = nn.CrossEntropyLoss(weight=w_sub)

    _log("\n  Training Model A (flat MLP) ...")
    model_a = ModelA()
    _log(f"    Parameters: {sum(p.numel() for p in model_a.parameters()):,}")
    hist_a, best_a = _train_loop(
        model_a,
        lambda m, b: (m(b[0]).argmax(1), b[1]),
        lambda m, b: crit_div(m(b[0]), b[1]),
        "A",
    )

    _log("\n  Training Model B (two-head MLP) ...")
    model_b = ModelB()
    _log(f"    Parameters: {sum(p.numel() for p in model_b.parameters()):,}")

    def b_preds(m, b):
        ls, ld = m(b[0])
        pred_s = ls.argmax(1)
        ld_m   = ld.masked_fill(~hier_mask[pred_s], float("-inf"))
        return ld_m.argmax(1), b[1]

    def b_loss(m, b):
        ls, ld = m(b[0])
        return ALPHA * crit_sub(ls, b[2]) + (1 - ALPHA) * crit_div(ld, b[1])

    hist_b, best_b = _train_loop(model_b, b_preds, b_loss, "B")

    # ── Final evaluation ──────────────────────────────────────────────────────
    def _eval(model, is_b=False):
        model.eval()
        pd_, td_, ps_, ts_ = [], [], [], []
        with torch.no_grad():
            for xb, yd, ys in val_dl:
                if is_b:
                    ls, ld = model(xb)
                    ps = ls.argmax(1)
                    ld_m = ld.masked_fill(~hier_mask[ps], float("-inf"))
                    pd_.append(ld_m.argmax(1)); ps_.append(ps)
                else:
                    pd_.append(model(xb).argmax(1))
                    # derive sub from div
                    div_to_sub = {}
                    for row in df[["lcc","lcc_division"]].drop_duplicates().itertuples():
                        if row.lcc_division in le_div.classes_ and row.lcc in le_sub.classes_:
                            div_to_sub[int(le_div.transform([row.lcc_division])[0])] = \
                                int(le_sub.transform([row.lcc])[0])
                    ps_.append(torch.tensor([div_to_sub.get(d.item(), -1) for d in pd_[-1]]))
                td_.append(yd); ts_.append(ys)
        pd_ = torch.cat(pd_).numpy(); td_ = torch.cat(td_).numpy()
        ps_ = torch.cat(ps_).numpy(); ts_ = torch.cat(ts_).numpy()
        valid_sub = ps_ >= 0
        return dict(
            val_acc_div = round(accuracy_score(td_, pd_), 4),
            val_bal_div = round(balanced_accuracy_score(td_, pd_), 4),
            val_acc_sub = round(accuracy_score(ts_[valid_sub], ps_[valid_sub]), 4),
            val_bal_sub = round(balanced_accuracy_score(ts_[valid_sub], ps_[valid_sub]), 4),
            report_div  = classification_report(td_, pd_, target_names=le_div.classes_, zero_division=0),
            report_sub  = classification_report(ts_[valid_sub], ps_[valid_sub],
                                                target_names=le_sub.classes_, zero_division=0),
        )

    metrics_a = _eval(model_a, is_b=False)
    metrics_b = _eval(model_b, is_b=True)

    # ── Save artefacts ────────────────────────────────────────────────────────
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model_a.state_dict(), model_dir / "model_a.pt")
    torch.save(model_b.state_dict(), model_dir / "model_b.pt")

    with open(model_dir / "encoders.pkl", "wb") as f:
        pickle.dump({"le_div": le_div, "le_sub": le_sub}, f)

    with open(model_dir / "hierarchy.pkl", "wb") as f:
        pickle.dump({"hier_mask": hier_mask, "sub_to_divs": sub_to_divs}, f)

    metrics_out = dict(
        model_a = dict(
            params=sum(p.numel() for p in model_a.parameters()),
            history=hist_a, best_val_bal=round(best_a, 4),
            **{k: v for k, v in metrics_a.items() if not k.startswith("report")},
        ),
        model_b = dict(
            params=sum(p.numel() for p in model_b.parameters()),
            history=hist_b, best_val_bal=round(best_b, 4),
            **{k: v for k, v in metrics_b.items() if not k.startswith("report")},
        ),
        config = dict(
            n_div=n_div, n_sub=n_sub, hidden=HIDDEN, dropout=DROPOUT,
            divisions=list(le_div.classes_), subclasses=list(le_sub.classes_),
        ),
    )
    (model_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    (model_dir / "report_a.txt").write_text(
        "=== MODEL A — DIVISION ===\n" + metrics_a["report_div"] +
        "\n=== MODEL A — SUBCLASS ===\n" + metrics_a["report_sub"]
    )
    (model_dir / "report_b.txt").write_text(
        "=== MODEL B — DIVISION ===\n" + metrics_b["report_div"] +
        "\n=== MODEL B — SUBCLASS ===\n" + metrics_b["report_sub"]
    )

    _log(f"\n  Saved artefacts → {model_dir}")
    for p in sorted(model_dir.iterdir()):
        _log(f"    {p.name}")

    return metrics_out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Crashproof retraining pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Output layout")[0],
    )
    parser.add_argument("--run-name",    required=True,
                        help="Identifier for this training run (e.g. v3, 50k_v2)")
    parser.add_argument("--source",      default=None, metavar="PATH",
                        help="Source parquet or sharded embedding directory. "
                             "Defaults to the cleaned English parquet from config. "
                             "If the source has an 'embeddings' column the embed "
                             "stage is skipped automatically.")
    parser.add_argument("--sample-size", type=int, default=50_000,
                        help="Documents to sample for clustering (default: 50000)")
    parser.add_argument("--skip",        nargs="*", default=[],
                        choices=["sample", "embed", "cluster"],
                        metavar="STAGE",
                        help="Skip these stages (space-separated)")
    parser.add_argument("--force",       default=None,
                        choices=["sample", "embed", "cluster", "train"],
                        help="Force re-run this stage even if already done")
    parser.add_argument("--train",       action="store_true",
                        help="Run the train stage (after lcc_mapping.csv is filled)")
    parser.add_argument("--suggest-lcc", default=None, metavar="MODEL_DIR",
                        help="Path to an existing model dir — bootstraps LCC suggestions")
    parser.add_argument("--device",      default=os.getenv("DEVICE", "cpu"),
                        help="Compute device (cpu / cuda / mps)")
    parser.add_argument("--batch-size",  type=int, default=32,
                        help="Embedding batch size (default: 32)")
    parser.add_argument("--model-name",  default="Qwen/Qwen3-Embedding-0.6B",
                        help="HuggingFace embedding model")
    parser.add_argument("--epochs",      type=int, default=60,
                        help="Max training epochs (default: 60)")
    parser.add_argument("--train-batch", type=int, default=256,
                        help="MLP training batch size (default: 256)")
    parser.add_argument("--min-clusters", type=int, default=100,
                        help="Lower bound for the HDBSCAN auto-tune target range "
                             "(default: 100). Configs producing fewer clusters than "
                             "this are disqualified regardless of silhouette score. "
                             "LCC has ~200 STEM-relevant subclasses — v2's 169-cluster "
                             "run (26k sample) hit the sweet spot at this granularity, "
                             "so the default window is centred there rather than left "
                             "wide open (silhouette alone tends to favour smaller, "
                             "noisier cores — see PROJECT_DIARY chapter 3 and 15).")
    parser.add_argument("--max-clusters", type=int, default=250,
                        help="Upper bound for the HDBSCAN auto-tune target range "
                             "(default: 250). See --min-clusters for rationale.")
    parser.add_argument("--umap-components", type=int, default=15)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print what would happen without doing it")
    args = parser.parse_args()

    run_dir   = RUNS_DIR / args.run_name
    model_dir = MODELS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    suggest_dir = Path(args.suggest_lcc) if args.suggest_lcc else None
    source      = Path(args.source) if args.source else ENGLISH

    _section(f"TRAINING PIPELINE  run={args.run_name}  {datetime.now():%Y-%m-%d %H:%M}")
    _log(f"  Run dir   : {run_dir}")
    _log(f"  Model dir : {model_dir}")
    _log(f"  Source    : {source}")
    _log(f"  Sample    : {args.sample_size:,}  device={args.device}")
    if suggest_dir:
        _log(f"  LCC hint  : {suggest_dir}")

    def _should_run(stage: str) -> bool:
        # Reload from disk every time — earlier stages in this same run can
        # write state (e.g. stage_sample auto-marking "embed" done when the
        # source already had embeddings), and a stale in-memory snapshot would
        # miss that, causing the stage to run again unnecessarily.
        state = _load_state(run_dir)
        if args.dry_run:
            done = state.get(stage, {}).get("done", False)
            skip = stage in args.skip
            _log(f"  [dry-run] stage={stage}  done={done}  skip={skip}  "
                 f"force={args.force==stage}")
            return False
        if args.force == stage:
            return True
        if stage in args.skip:
            _log(f"  ⏭  {stage} skipped (--skip)")
            return False
        if state.get(stage, {}).get("done"):
            _log(f"  ✅  {stage} already done — skipping")
            return False
        return True

    # ── Stage 1: Sample ───────────────────────────────────────────────────────
    _section("STAGE 1 — SAMPLE")
    if _should_run("sample"):
        k, has_emb = stage_sample(run_dir, source, args.sample_size, args.seed)
        _save_state(run_dir, "sample", {"n_docs": k})
        if has_emb:
            # Source had embeddings — write them as embedded.parquet and skip embed
            _save_state(run_dir, "embed", {"n_docs": k, "skipped": "source had embeddings"})
            _log("  ⏭  embed stage auto-skipped (source already had embeddings)")

    # ── Stage 2: Embed ────────────────────────────────────────────────────────
    _section("STAGE 2 — EMBED")
    if _should_run("embed"):
        k = stage_embed(run_dir, args.device, args.batch_size, args.model_name)
        _save_state(run_dir, "embed", {"n_docs": k})

    # ── Stage 3: Cluster ──────────────────────────────────────────────────────
    _section("STAGE 3 — CLUSTER")
    if _should_run("cluster"):
        n_cls, n_out = stage_cluster(
            run_dir,
            suggest_model_dir=suggest_dir,
            device=args.device,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            umap_components=args.umap_components,
        )
        _save_state(run_dir, "cluster", {"n_clusters": n_cls, "n_outliers": n_out})

    # ── Pause for LCC mapping ─────────────────────────────────────────────────
    if not args.train and not args.dry_run:
        csv_path = run_dir / "lcc_mapping.csv"
        _sep("═")
        _log()
        _log("  NEXT STEP — manual LCC mapping required")
        _sep()
        _log(f"  1. Open: {csv_path}")
        _log("  2. For each cluster, fill in:")
        _log("       lcc_subclass  — e.g. TK, QC, RC")
        _log("       lcc_division  — e.g. TK7874, QC809, RC685")
        if suggest_dir:
            _log("     (suggested_lcc_subclass / suggested_lcc_division are pre-filled")
            _log("      from the existing model — just verify and correct them)")
        _log("  3. Write  skip=yes  for JATS-noise or unclassifiable clusters")
        _log()
        _log("  When done, continue training:")
        _log(f"    python pipeline/run_training.py --run-name {args.run_name} --train")
        _sep("═")
        return

    # ── Stage 4: Train ────────────────────────────────────────────────────────
    _section("STAGE 4 — TRAIN")
    if args.train or args.force == "train":
        metrics = stage_train(run_dir, model_dir, args.epochs, args.train_batch, args.device)
        _save_state(run_dir, "train", {
            "model_a_bal": metrics["model_a"]["best_val_bal"],
            "model_b_bal": metrics["model_b"]["best_val_bal"],
        })

        _section("RESULTS SUMMARY")
        _log(f"  {'Metric':<28} {'Model A':>10} {'Model B':>10}")
        _sep()
        for k in ["val_acc_div", "val_bal_div", "val_acc_sub", "val_bal_sub"]:
            _log(f"  {k:<28} {metrics['model_a'][k]:>10.4f} {metrics['model_b'][k]:>10.4f}")
        _log()
        _log(f"  Model artefacts saved to: {model_dir}")
        _log(f"  To classify documents with the new model, run:")
        _log(f"    python pipeline/11_classify.py \\")
        _log(f"        --input  <embedded.parquet> \\")
        _log(f"        --output <classified.parquet> \\")
        _log(f"        --model-dir {model_dir}")

    _section(f"DONE  {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
