#!/usr/bin/env python3
"""
v2 Clustering Pipeline — CiNii 26k Academic Papers
====================================================
Discovers topics from scratch using:
  1. UMAP (15-dim) — better manifold than the old 5-dim
  2. HDBSCAN parameter search — picks config with best silhouette
  3. Soft outlier assignment — 100% of papers get a label
  4. c-TF-IDF keyword extraction per cluster
  5. Ollama LLM (qwen3.5:0.8b) — names each cluster

Caching: UMAP and HDBSCAN results are cached in clustering_output/v2/_cache/
so re-runs only redo the naming step.

Outputs (all in clustering_output/v2/):
  clustered_26k.parquet    — one row per paper, with cluster_id, cluster_name
  cluster_metadata.parquet — one row per cluster, with keywords, label, description
  best_config.json         — winning HDBSCAN parameters + full search results
  umap_scatter.html        — interactive 2D scatter coloured by cluster
"""

import sys, re, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import numpy as np
import pandas as pd
import requests
import umap
import hdbscan
import plotly.express as px
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import issparse
from config import EMBEDDED_26K, CLUSTER_DIR

OUT_DIR   = CLUSTER_DIR / "v2"
CACHE_DIR = OUT_DIR / "_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:0.8b"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & NORMALIZE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1 — Load & normalise embeddings")
print("=" * 60)

df = pd.read_parquet(str(EMBEDDED_26K))
print(f"Rows: {len(df):,}  Columns: {df.columns.tolist()}")

raw = df["embeddings"].tolist()
embeddings = np.array(raw, dtype=np.float32)
embeddings = normalize(embeddings, norm="l2")   # unit sphere → cosine ≡ euclidean
print(f"Embeddings: {embeddings.shape}  (L2-normalised)")

texts = df["full_text"].fillna(df["title"]).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 2. UMAP — high-dimensional reduction for clustering (cached)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2 — UMAP reduction (n_components=15, for clustering)")
print("=" * 60)

_Z_cache = CACHE_DIR / "Z.npy"
if _Z_cache.exists():
    Z = np.load(str(_Z_cache))
    print(f"  Loaded from cache — shape: {Z.shape}")
else:
    print("  This step takes ~3–6 min on CPU …")
    t0 = time.time()
    reducer_cluster = umap.UMAP(
        n_neighbors=30,       # higher = more global structure captured
        n_components=15,      # 15-dim preserves much more than the old 5-dim
        min_dist=0.0,         # tight packing → better HDBSCAN clusters
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    Z = reducer_cluster.fit_transform(embeddings)
    np.save(str(_Z_cache), Z)
    print(f"  Done in {time.time()-t0:.0f}s — shape: {Z.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. HDBSCAN PARAMETER SEARCH (cached)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3 — HDBSCAN parameter search")
print("=" * 60)

_labels_cache = CACHE_DIR / "best_labels.npy"
_cfg_cache    = CACHE_DIR / "best_config.json"

if _labels_cache.exists() and _cfg_cache.exists():
    best_labels = np.load(str(_labels_cache))
    with open(str(_cfg_cache)) as f:
        _cached = json.load(f)
    best_config    = _cached["best_config"]
    search_results = _cached.get("all_results", [])
    print(f"  Loaded from cache — mcs={best_config['mcs']}, ms={best_config['ms']}, "
          f"{best_config['n_clusters']} clusters, sil={best_config['silhouette']}")

else:
    # Goal: fine-grained sub-topics for LCC mapping — target 150–220 clusters
    # Analysis from find_optimal_k.py: silhouette peaks at k≈170–200 (mcs=20–25)
    HDBSCAN_GRID = [
        # (min_cluster_size, min_samples)
        (15,  3),
        (15,  5),
        (18,  3),
        (18,  5),
        (20,  3),
        (20,  5),
        (22,  5),
        (25,  5),
        (28,  5),
        (30,  5),
    ]

    search_results = []
    best_score     = -999.0
    best_labels    = None
    best_config    = None

    for mcs, ms in tqdm(HDBSCAN_GRID, desc="HDBSCAN search"):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(Z)

        n_clusters   = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers   = int((labels == -1).sum())
        outlier_rate = n_outliers / len(labels)

        if n_clusters < 5 or (labels != -1).sum() < 500:
            search_results.append(dict(mcs=mcs, ms=ms, n_clusters=n_clusters,
                                       outlier_rate=outlier_rate, silhouette=-1.0, score=-999.0))
            continue

        mask = labels != -1
        sil  = silhouette_score(Z[mask], labels[mask], sample_size=6000, random_state=42)

        # Score: reward silhouette + penalise excess outliers + cluster count extremes
        # Target: 150–220 fine-grained sub-topic clusters (optimal per sweep analysis)
        outlier_pen = max(0.0, outlier_rate - 0.50) * 2.0   # looser penalty — high outlier rate is normal here
        lo, hi      = 150, 220
        cluster_pen = (
            max(0.0, (lo - n_clusters) / lo * 0.4) +
            max(0.0, (n_clusters - hi) / hi * 0.4)
        )
        score = sil - outlier_pen - cluster_pen

        search_results.append(dict(mcs=mcs, ms=ms, n_clusters=n_clusters,
                                   outlier_rate=round(outlier_rate, 3),
                                   silhouette=round(sil, 4), score=round(score, 4)))
        print(f"  mcs={mcs:3d}, ms={ms:2d} → {n_clusters:3d} clusters, "
              f"{outlier_rate:.1%} outliers, sil={sil:.3f}, score={score:.3f}")

        if score > best_score:
            best_score  = score
            best_config = dict(mcs=mcs, ms=ms, n_clusters=n_clusters,
                               outlier_rate=outlier_rate, silhouette=round(sil, 4))
            best_labels = labels.copy()

    np.save(str(_labels_cache), best_labels)
    # Convert numpy floats to Python floats for JSON serialisation
    def _to_json(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)):     return int(obj)
        return obj
    def _clean_dict(d): return {k: _to_json(v) for k, v in d.items()}
    with open(str(_cfg_cache), "w") as f:
        json.dump({"best_config": _clean_dict(best_config),
                   "all_results": [_clean_dict(r) for r in search_results]}, f, indent=2)

    print(f"\n  Best: mcs={best_config['mcs']}, ms={best_config['ms']} "
          f"→ {best_config['n_clusters']} clusters, "
          f"{best_config['outlier_rate']:.1%} outliers, "
          f"silhouette={best_config['silhouette']}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SOFT-ASSIGN OUTLIERS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4 — Soft-assign outliers to nearest cluster")
print("=" * 60)

core_mask    = best_labels != -1
outlier_mask = ~core_mask
n_before     = int(outlier_mask.sum())

final_labels = best_labels.copy()

if n_before > 0:
    nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn_model.fit(Z[core_mask])
    _, idx = nn_model.kneighbors(Z[outlier_mask])
    core_positions = np.where(core_mask)[0]
    final_labels[outlier_mask] = best_labels[core_positions[idx.ravel()]]

n_after = int((final_labels == -1).sum())
print(f"  Outliers before: {n_before:,} → after: {n_after}  (all assigned)")
print(f"  Papers covered : {len(final_labels):,} / {len(final_labels):,}  (100%)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. KEYWORD EXTRACTION (c-TF-IDF)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5 — c-TF-IDF keyword extraction per cluster")
print("=" * 60)

cluster_ids = sorted(set(final_labels))

# Build one meta-document per cluster (concat all texts)
meta_docs = [" ".join(texts[i] for i in np.where(final_labels == cid)[0])
             for cid in cluster_ids]

cv = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85,
    max_features=20_000,
)
X_count = cv.fit_transform(meta_docs)
vocab   = np.array(cv.get_feature_names_out())

# IDF over clusters (not individual docs — this is the c-TF-IDF trick)
df_per_term = np.diff(X_count.tocsc().indptr)
idf         = np.log(1 + len(cluster_ids) / (1 + df_per_term))

X_tf = X_count.toarray().astype(np.float32)
row_sums = X_tf.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
X_ctfidf = (X_tf / row_sums) * idf

cluster_keywords = {}
for i, cid in enumerate(cluster_ids):
    top_idx = np.argsort(X_ctfidf[i])[::-1][:15]
    cluster_keywords[cid] = vocab[top_idx].tolist()
    print(f"  [{cid:3d}]  {', '.join(cluster_keywords[cid][:8])}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. LLM CLUSTER NAMING (Ollama via HTTP)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  STEP 6 — Naming clusters with Ollama ({OLLAMA_MODEL})")
print("=" * 60)


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_ollama(prompt: str, timeout: int = 30) -> str:
    # think=False disables qwen3 chain-of-thought → fast, direct responses
    for attempt in range(3):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "think": False,
                      "stream": False,
                      "options": {"temperature": 0}},
                timeout=timeout,
            )
            resp.raise_for_status()
            return _clean(resp.json()["message"]["content"])
        except requests.exceptions.Timeout:
            print(f"    Ollama timeout (attempt {attempt+1}/3), retrying …")
            time.sleep(3)
        except Exception as e:
            print(f"    Ollama error: {e}")
            break
    return "{}"


def name_cluster(cid: int, keywords: list, sample_titles: list) -> dict:
    kw_str = ", ".join(keywords[:10])
    prompt = (
        f"Keywords from an academic paper cluster: {kw_str}\n"
        f'Reply with JSON only: {{"label": "3-5 word topic name", "description": "one sentence"}}'
    )
    raw = ask_ollama(prompt)
    try:
        m    = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        data = {}
    return {
        "label":       data.get("label",       ", ".join(keywords[:3]).title()),
        "description": data.get("description", ""),
    }


metadata_rows = []
for cid in tqdm(cluster_ids, desc="Naming clusters"):
    idx      = np.where(final_labels == cid)[0]
    n_papers = len(idx)
    n_core   = int((best_labels == cid).sum())
    kw       = cluster_keywords[cid]

    # Use core (most representative) titles for naming
    core_idx = np.where(best_labels == cid)[0]
    if len(core_idx) == 0:
        core_idx = idx
    rng          = np.random.default_rng(42)
    sample_idx   = rng.choice(core_idx, min(20, len(core_idx)), replace=False)
    sample_titles = [df["title"].iloc[i] for i in sample_idx]

    info = name_cluster(cid, kw, sample_titles)
    row = {
        "cluster_id":  cid,
        "n_papers":    n_papers,
        "n_core":      n_core,
        "label":       info["label"],
        "description": info["description"],
        "keywords":    kw,
    }
    metadata_rows.append(row)
    print(f"  [{cid:3d}] {row['label']}  ({n_papers} papers, {n_core} core)")

cluster_meta = pd.DataFrame(metadata_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 7. UMAP 2D — visualisation (cached)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 7 — UMAP 2D (visualisation)")
print("=" * 60)

_Z2d_cache = CACHE_DIR / "Z_2d.npy"
if _Z2d_cache.exists():
    Z_2d = np.load(str(_Z2d_cache))
    print(f"  Loaded from cache — shape: {Z_2d.shape}")
else:
    t0 = time.time()
    reducer_viz = umap.UMAP(
        n_neighbors=30,
        n_components=2,
        min_dist=0.1,   # more spread out = better scatter plot
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    Z_2d = reducer_viz.fit_transform(embeddings)
    np.save(str(_Z2d_cache), Z_2d)
    print(f"  Done in {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 8 — Save")
print("=" * 60)

id_to_label = dict(zip(cluster_meta["cluster_id"], cluster_meta["label"]))

df_out = df.copy()
df_out["cluster_id"]   = final_labels.astype(int)
df_out["cluster_name"] = [id_to_label[c] for c in final_labels]
df_out["is_core"]      = core_mask
df_out["umap_x"]       = Z_2d[:, 0]
df_out["umap_y"]       = Z_2d[:, 1]

out_papers = OUT_DIR / "clustered_26k.parquet"
out_meta   = OUT_DIR / "cluster_metadata.parquet"
out_config = OUT_DIR / "best_config.json"

df_out.to_parquet(str(out_papers), index=False)
cluster_meta.to_parquet(str(out_meta), index=False)
def _to_native(obj):
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)):     return int(obj)
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_to_native(v) for v in obj]
    return obj

with open(str(out_config), "w") as f:
    json.dump(_to_native({"best_config": best_config, "all_results": search_results}), f, indent=2)

print(f"  {out_papers}  ({len(df_out):,} rows)")
print(f"  {out_meta}  ({len(cluster_meta)} clusters)")
print(f"  {out_config}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. INTERACTIVE SCATTER
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Building scatter plot …")

df_viz = pd.DataFrame({
    "x":       Z_2d[:, 0],
    "y":       Z_2d[:, 1],
    "cluster": df_out["cluster_id"].astype(str),
    "label":   df_out["cluster_name"],
    "title":   df["title"].str[:120],
    "core":    core_mask,
})
fig = px.scatter(
    df_viz, x="x", y="y",
    color="cluster",
    hover_data={"label": True, "title": True, "core": True,
                "cluster": True, "x": False, "y": False},
    title=(f"CiNii 26k — {best_config['n_clusters']} topic clusters "
           f"(silhouette={best_config['silhouette']})"),
    width=1200, height=800,
    opacity=0.55,
)
fig.update_traces(marker=dict(size=3))
fig.update_layout(legend_title_text="Cluster ID")
out_html = OUT_DIR / "umap_scatter.html"
fig.write_html(str(out_html))
print(f"  {out_html}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TOPIC SUMMARY")
print("=" * 60)
summary = cluster_meta[["cluster_id", "n_papers", "n_core", "label", "description"]].copy()
print(summary.sort_values("n_papers", ascending=False).to_string(index=False))
print()
print(f"  Total clusters      : {len(cluster_meta)}")
print(f"  Total papers        : {len(df_out):,}  (100% labeled)")
print(f"  Silhouette (core pts): {best_config['silhouette']}")
print(f"  HDBSCAN config      : mcs={best_config['mcs']}, ms={best_config['ms']}")
print()
print(f"  Open {out_html} to explore interactively.")
