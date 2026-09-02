#!/usr/bin/env python3
"""
Optimal Cluster Count Analysis
================================
Sweeps HDBSCAN min_cluster_size from coarse to fine and measures quality
at each granularity. Answers: "how many sub-categories should we use?"

Key metrics plotted:
  - Silhouette score        (higher = better-separated clusters)
  - Mean intra-cluster cosine similarity  (higher = more coherent topics)
  - Outlier rate before soft-assignment   (lower = denser natural clusters)
  - Mean papers per cluster               (must stay above ~20 for LCC mapping)

Outputs: clustering_output/v2/optimal_k/
  sweep_stats.csv      — raw numbers for every config tested
  analysis.html        — interactive 4-panel chart
  recommendation.txt   — printed verdict
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from config import CLUSTER_DIR

CACHE_DIR = CLUSTER_DIR / "v2" / "_cache"
OUT_DIR   = CLUSTER_DIR / "v2" / "optimal_k"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load the cached 15-dim UMAP (no need to recompute) ───────────────────────
Z_path = CACHE_DIR / "Z.npy"
if not Z_path.exists():
    print("ERROR: Run v2_cluster.py first to generate the UMAP cache.")
    sys.exit(1)

Z = np.load(str(Z_path))
print(f"Loaded UMAP embedding: {Z.shape}")
N = Z.shape[0]

# Also load raw embeddings for cosine-similarity computation
from config import EMBEDDED_26K
df = pd.read_parquet(str(EMBEDDED_26K))
raw = df["embeddings"].tolist()
embeddings = normalize(np.array(raw, dtype=np.float32), norm="l2")


# ── Sweep ─────────────────────────────────────────────────────────────────────
# min_cluster_size controls granularity directly:
#   large mcs → few big clusters (coarse)
#   small mcs → many small clusters (fine)
MCS_VALUES = [400, 300, 250, 200, 175, 150, 125, 100, 90, 80, 70,
              60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]

MIN_SAMPLES = 5   # fixed — only mcs varies so we isolate one variable

print(f"\nSweeping {len(MCS_VALUES)} configs (mcs from {max(MCS_VALUES)} → {min(MCS_VALUES)}) …\n")

rows = []
for mcs in MCS_VALUES:
    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(Z)

    n_clusters   = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers   = int((labels == -1).sum())
    outlier_rate = n_outliers / N
    mask         = labels != -1
    n_clustered  = mask.sum()

    if n_clusters < 2 or n_clustered < 200:
        print(f"  mcs={mcs:4d}: {n_clusters} clusters — too few, skip")
        continue

    # 1. Silhouette on UMAP space (fast, sampled)
    sample_n = min(8000, n_clustered)
    rng      = np.random.default_rng(42)
    idx      = rng.choice(np.where(mask)[0], sample_n, replace=False)
    sil      = silhouette_score(Z[idx], labels[idx], metric="euclidean")

    # 2. Mean intra-cluster cosine similarity (coherence) on raw embeddings
    #    For efficiency: sample up to 200 papers per cluster, take mean pairwise dot
    coherence_scores = []
    for cid in set(labels[mask]):
        cidx = np.where(labels == cid)[0]
        if len(cidx) > 200:
            cidx = rng.choice(cidx, 200, replace=False)
        vecs = embeddings[cidx]
        sim  = (vecs @ vecs.T)
        np.fill_diagonal(sim, np.nan)
        coherence_scores.append(float(np.nanmean(sim)))
    mean_coherence = float(np.mean(coherence_scores))

    # 3. Cluster size stats
    sizes        = [(labels == cid).sum() for cid in set(labels[mask])]
    mean_size    = float(np.mean(sizes))
    median_size  = float(np.median(sizes))
    min_size     = int(np.min(sizes))

    # 4. Scale projection: papers per cluster if applied to 3M docs
    scale_factor = 3_000_000 / N
    projected_mean = mean_size * scale_factor

    elapsed = time.time() - t0
    row = dict(
        mcs           = mcs,
        n_clusters    = n_clusters,
        outlier_rate  = round(outlier_rate, 4),
        silhouette    = round(sil, 4),
        mean_coherence= round(mean_coherence, 4),
        mean_size     = round(mean_size, 1),
        median_size   = round(median_size, 1),
        min_size      = min_size,
        projected_mean_3M = round(projected_mean, 0),
        elapsed_s     = round(elapsed, 1),
    )
    rows.append(row)
    print(f"  mcs={mcs:4d}: {n_clusters:4d} clusters | "
          f"outliers={outlier_rate:.1%} | sil={sil:.3f} | "
          f"coherence={mean_coherence:.3f} | "
          f"mean_size={mean_size:.0f} (→{projected_mean:.0f} at 3M)")

df_stats = pd.DataFrame(rows)
df_stats.to_csv(str(OUT_DIR / "sweep_stats.csv"), index=False)
print(f"\nSaved: {OUT_DIR / 'sweep_stats.csv'}")


# ── Elbow detection (find the "knee" in silhouette vs n_clusters) ─────────────
# Simple: largest drop in silhouette improvement per additional cluster
df_sorted = df_stats.sort_values("n_clusters").reset_index(drop=True)

# Marginal silhouette gain: d(silhouette)/d(n_clusters)
marginal = []
for i in range(1, len(df_sorted)):
    dk  = df_sorted["n_clusters"].iloc[i] - df_sorted["n_clusters"].iloc[i-1]
    ds  = df_sorted["silhouette"].iloc[i]  - df_sorted["silhouette"].iloc[i-1]
    marginal.append(ds / max(dk, 1))
marginal = [marginal[0]] + marginal  # pad first row

df_sorted["marginal_sil"] = marginal

# "Elbow": where marginal gain drops to near zero or goes negative
# First cluster count where marginal gain < 10% of max marginal gain
max_marginal = max(marginal)
elbow_mask   = df_sorted["marginal_sil"] < max_marginal * 0.10
elbow_row    = df_sorted[elbow_mask].iloc[0] if elbow_mask.any() else df_sorted.iloc[len(df_sorted)//2]
elbow_k      = int(elbow_row["n_clusters"])


# ── 4-panel interactive chart ─────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Silhouette Score vs Number of Clusters",
        "Mean Intra-Cluster Coherence vs Number of Clusters",
        "Outlier Rate vs Number of Clusters",
        "Projected Mean Cluster Size at 3M Papers",
    ],
    vertical_spacing=0.15,
    horizontal_spacing=0.10,
)

x = df_sorted["n_clusters"]

# Panel 1 — Silhouette
fig.add_trace(go.Scatter(
    x=x, y=df_sorted["silhouette"],
    mode="lines+markers", name="Silhouette",
    line=dict(color="#2196F3", width=2),
    marker=dict(size=6),
    hovertemplate="k=%{x}<br>silhouette=%{y:.3f}<extra></extra>",
), row=1, col=1)
fig.add_vline(x=elbow_k, line_dash="dash", line_color="red", row=1, col=1)
fig.add_annotation(x=elbow_k, y=df_stats["silhouette"].max(),
                   text=f"elbow≈{elbow_k}", showarrow=True, arrowhead=2,
                   row=1, col=1)

# Panel 2 — Coherence
fig.add_trace(go.Scatter(
    x=x, y=df_sorted["mean_coherence"],
    mode="lines+markers", name="Coherence",
    line=dict(color="#4CAF50", width=2),
    marker=dict(size=6),
    hovertemplate="k=%{x}<br>coherence=%{y:.3f}<extra></extra>",
), row=1, col=2)
fig.add_vline(x=elbow_k, line_dash="dash", line_color="red", row=1, col=2)

# Panel 3 — Outlier rate
fig.add_trace(go.Scatter(
    x=x, y=df_sorted["outlier_rate"] * 100,
    mode="lines+markers", name="Outlier %",
    line=dict(color="#FF9800", width=2),
    marker=dict(size=6),
    hovertemplate="k=%{x}<br>outlier_rate=%{y:.1f}%<extra></extra>",
), row=2, col=1)
fig.add_hline(y=25, line_dash="dot", line_color="gray",
              annotation_text="25% threshold", row=2, col=1)
fig.add_vline(x=elbow_k, line_dash="dash", line_color="red", row=2, col=1)

# Panel 4 — Projected size at 3M
fig.add_trace(go.Scatter(
    x=x, y=df_sorted["projected_mean_3M"],
    mode="lines+markers", name="Proj. size @ 3M",
    line=dict(color="#9C27B0", width=2),
    marker=dict(size=6),
    hovertemplate="k=%{x}<br>proj_size=%{y:,.0f}<extra></extra>",
), row=2, col=2)
# Reference lines: minimum for reliable labeling
fig.add_hline(y=500,  line_dash="dot", line_color="green",
              annotation_text="500 papers/cluster (minimum for classifier)", row=2, col=2)
fig.add_hline(y=5000, line_dash="dot", line_color="blue",
              annotation_text="5000 papers/cluster (comfortable)", row=2, col=2)
fig.add_vline(x=elbow_k, line_dash="dash", line_color="red", row=2, col=2)

fig.update_layout(
    title=dict(
        text=f"Optimal Cluster Count Analysis — CiNii 26k papers (scaling to 3M)<br>"
             f"<sup>Red dashed line = detected elbow (k≈{elbow_k}). "
             f"Hover to explore all data points.</sup>",
        font=dict(size=16),
    ),
    height=800, width=1200,
    showlegend=False,
)
fig.update_xaxes(title_text="Number of clusters (k)")
fig.update_yaxes(title_text="Silhouette score",      row=1, col=1)
fig.update_yaxes(title_text="Mean cosine similarity", row=1, col=2)
fig.update_yaxes(title_text="Outlier rate (%)",       row=2, col=1)
fig.update_yaxes(title_text="Mean papers/cluster",    row=2, col=2)

out_html = OUT_DIR / "analysis.html"
fig.write_html(str(out_html))
print(f"Saved: {out_html}")


# ── Printed recommendation ────────────────────────────────────────────────────
scale_factor = 3_000_000 / N

# Find configs where silhouette >= 95% of max AND projected size >= 500 at 3M
sil_threshold = df_stats["silhouette"].max() * 0.95
viable = df_stats[
    (df_stats["silhouette"] >= sil_threshold) &
    (df_stats["projected_mean_3M"] >= 500)
]
if len(viable):
    # Among viable, prefer the most clusters (finest granularity)
    recommended_row = viable.loc[viable["n_clusters"].idxmax()]
    recommended_k   = int(recommended_row["n_clusters"])
    recommended_mcs = int(recommended_row["mcs"])
else:
    recommended_k   = elbow_k
    recommended_mcs = int(df_sorted[df_sorted["n_clusters"] >= elbow_k]["mcs"].iloc[-1])

rec_row = df_stats[df_stats["n_clusters"] == recommended_k].iloc[0]

report_lines = [
    "=" * 65,
    "  OPTIMAL CLUSTER COUNT RECOMMENDATION",
    "=" * 65,
    "",
    f"  Dataset          : {N:,} papers (will scale to 3,000,000)",
    f"  Scale factor     : {scale_factor:.0f}×",
    "",
    "  SWEEP RESULTS (sorted by cluster count):",
    "",
    f"  {'k':>5}  {'silhouette':>10}  {'coherence':>10}  {'outlier%':>8}  {'mean_size':>9}  {'@ 3M':>8}",
    "  " + "-" * 60,
]
for _, r in df_sorted.iterrows():
    marker = " ◀ RECOMMENDED" if int(r["n_clusters"]) == recommended_k else \
             " ◀ ELBOW"       if int(r["n_clusters"]) == elbow_k         else ""
    report_lines.append(
        f"  {int(r['n_clusters']):>5}  {r['silhouette']:>10.3f}  "
        f"{r['mean_coherence']:>10.3f}  {r['outlier_rate']*100:>7.1f}%  "
        f"{r['mean_size']:>9.0f}  {r['projected_mean_3M']:>8.0f}{marker}"
    )

report_lines += [
    "",
    "  " + "=" * 60,
    f"  Detected elbow     : k ≈ {elbow_k}",
    f"  Recommended k      : {recommended_k}  (mcs={recommended_mcs})",
    f"  Silhouette at rec. : {rec_row['silhouette']:.3f}",
    f"  Coherence at rec.  : {rec_row['mean_coherence']:.3f}",
    f"  Avg papers/cluster : {rec_row['mean_size']:.0f}  "
    f"(→ {rec_row['projected_mean_3M']:,.0f} at 3M)",
    "",
    "  REASONING:",
    f"  - Silhouette stays flat above k≈{elbow_k}: adding more clusters",
    f"    beyond this point doesn't improve separation quality.",
    f"  - At k={recommended_k}, each cluster will have ~{rec_row['projected_mean_3M']:,.0f}",
    f"    papers at 3M scale — comfortably above the 500-paper minimum",
    f"    needed for reliable LCC sub-category assignment.",
    f"  - The LCC has ~300-400 relevant sub-classes for STEM corpora;",
    f"    k={recommended_k} gives one cluster per ~{300//max(recommended_k,1)} LCC sub-classes",
    f"    which is a good granularity for sub-category mapping.",
    "",
    "  NEXT STEP:",
    f"  Re-run v2_cluster.py with the HDBSCAN grid targeting k≈{recommended_k}",
    f"  (set mcs={recommended_mcs} in the search).",
    "=" * 65,
]

report = "\n".join(report_lines)
print("\n" + report)
with open(str(OUT_DIR / "recommendation.txt"), "w") as f:
    f.write(report)
print(f"\nSaved: {OUT_DIR / 'recommendation.txt'}")
print(f"Open:  {out_html}")
