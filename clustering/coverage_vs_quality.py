#!/usr/bin/env python3
"""
Coverage vs Cluster Quality Tradeoff Analysis
==============================================
Answers: "Is it better to have more papers (lower quality) or
          fewer papers (higher quality) as training data?"

Approach:
  HDBSCAN assigns every non-outlier point a membership probability (0–1).
  Outlier points get probability = 0.
  By varying a probability threshold we get a continuous tradeoff curve:
    threshold=0.0  → 100% coverage  (current approach)
    threshold=1.0  → core-only      (~56% coverage, highest quality)

Metrics at each threshold:
  - Coverage       : % of papers kept
  - Silhouette     : how well-separated the clusters are (higher = better)
  - Mean intra-cluster cosine similarity (coherence)
  - Label noise estimate: % of papers near a cluster boundary

Output: clustering_output/v2/coverage_quality/
  tradeoff.html    — interactive chart
  tradeoff.csv     — raw numbers
  recommendation.txt
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import numpy as np
import pandas as pd
import hdbscan
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from config import CLUSTER_DIR, EMBEDDED_26K

OUT_DIR   = CLUSTER_DIR / "v2" / "coverage_quality"
CACHE_DIR = CLUSTER_DIR / "v2" / "_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load cached UMAP + rerun best HDBSCAN with prediction_data=True ──────────
Z = np.load(str(CACHE_DIR / "Z.npy"))
N = len(Z)
print(f"UMAP embedding: {Z.shape}")

# Load raw embeddings for coherence computation
df_raw   = pd.read_parquet(str(EMBEDDED_26K))
raw_embs = df_raw["embeddings"].tolist()
embeddings = normalize(np.array(raw_embs, dtype=np.float32), norm="l2")

# Best config from previous run: mcs=25, ms=5
print("Running HDBSCAN (mcs=25, ms=5) with prediction_data=True …")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=25,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,      # needed for approximate_predict
)
labels_full = clusterer.fit_predict(Z)
probs_full  = clusterer.probabilities_   # membership probability for each point

n_clusters   = len(set(labels_full)) - (1 if -1 in labels_full else 0)
n_outliers   = (labels_full == -1).sum()
print(f"Clusters: {n_clusters}  |  Outliers: {n_outliers:,} ({n_outliers/N:.1%})")

# For outlier points, assign them to their nearest cluster
# and give them a "soft probability" = 1 - (normalised distance to nearest core)
from sklearn.neighbors import NearestNeighbors

core_mask    = labels_full != -1
outlier_mask = ~core_mask

nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
nn.fit(Z[core_mask])
dists, idxs = nn.kneighbors(Z[outlier_mask])

# Normalise distances: divide by the 95th percentile of core intra-cluster distances
core_positions = np.where(core_mask)[0]
assigned_labels = labels_full[core_positions[idxs.ravel()]]

# Estimate soft probability as exp(-dist / median_core_dist)
# — points close to a cluster core get high probability, distant ones get low
core_nn = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
core_nn.fit(Z[core_mask])
core_dists, _ = core_nn.kneighbors(Z[core_mask])
median_core_dist = float(np.median(core_dists[:, 1]))   # nearest-neighbour dist among core pts

soft_probs = np.exp(-dists.ravel() / (2 * median_core_dist))

# Combine into full arrays
all_labels = labels_full.copy()
all_probs  = probs_full.copy()
all_labels[outlier_mask] = assigned_labels
all_probs[outlier_mask]  = soft_probs

print(f"Median core NN distance: {median_core_dist:.4f}")
print(f"Soft prob stats — mean: {soft_probs.mean():.3f}  "
      f"median: {np.median(soft_probs):.3f}  "
      f">0.5: {(soft_probs > 0.5).mean():.1%}  "
      f">0.3: {(soft_probs > 0.3).mean():.1%}")


# ── Sweep probability thresholds ─────────────────────────────────────────────
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

rng = np.random.default_rng(42)
rows = []

print("\nSweeping thresholds …")
for thresh in THRESHOLDS:
    if thresh == 1.0:
        # Core-only: only original HDBSCAN non-outliers
        mask = core_mask.copy()
    else:
        mask = all_probs >= thresh

    n_kept = mask.sum()
    coverage = n_kept / N

    labs  = all_labels[mask]
    n_cl  = len(set(labs)) - (1 if -1 in labs else 0)

    if n_kept < 500 or n_cl < 5:
        rows.append(dict(threshold=thresh, coverage=coverage, n_kept=n_kept,
                         n_clusters=n_cl, silhouette=np.nan, coherence=np.nan))
        continue

    # Silhouette (sampled for speed)
    Z_sub = Z[mask]
    sample_n = min(8000, n_kept)
    idx      = rng.choice(n_kept, sample_n, replace=False)
    sil = silhouette_score(Z_sub[idx], labs[idx], metric="euclidean")

    # Mean intra-cluster coherence on raw embeddings
    emb_sub = embeddings[mask]
    coh_scores = []
    for cid in set(labs):
        cidx = np.where(labs == cid)[0]
        if len(cidx) < 2: continue
        if len(cidx) > 150:
            cidx = rng.choice(cidx, 150, replace=False)
        vecs = emb_sub[cidx]
        sim  = vecs @ vecs.T
        np.fill_diagonal(sim, np.nan)
        coh_scores.append(float(np.nanmean(sim)))
    mean_coh = float(np.mean(coh_scores))

    row = dict(threshold=thresh, coverage=round(coverage, 4), n_kept=int(n_kept),
               n_clusters=n_cl, silhouette=round(sil, 4), coherence=round(mean_coh, 4))
    rows.append(row)
    print(f"  thresh={thresh:.2f}  coverage={coverage:.1%}  "
          f"n={n_kept:,}  sil={sil:.3f}  coh={mean_coh:.3f}")

df_tr = pd.DataFrame(rows)
df_tr.to_csv(str(OUT_DIR / "tradeoff.csv"), index=False)


# ── Find the recommended threshold ───────────────────────────────────────────
# Criterion: keep the highest threshold where silhouette gain < 0.01
#            AND coverage is still >= 50%
df_valid = df_tr.dropna(subset=["silhouette"])
max_sil   = df_valid["silhouette"].max()

# The "knee": largest drop in silhouette gain relative to coverage cost
# Score = silhouette * coverage^0.3  (heavily reward quality, slightly reward coverage)
df_valid = df_valid.copy()
df_valid["score"] = df_valid["silhouette"] * (df_valid["coverage"] ** 0.3)
best_row  = df_valid.loc[df_valid["score"].idxmax()]
rec_thresh = float(best_row["threshold"])
rec_cov    = float(best_row["coverage"])
rec_sil    = float(best_row["silhouette"])
rec_n      = int(best_row["n_kept"])


# ── Interactive chart ─────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Silhouette Score vs Coverage (quality–quantity tradeoff)",
        "Silhouette & Coherence vs Probability Threshold",
    ],
    horizontal_spacing=0.12,
)

# Panel 1: silhouette vs coverage scatter (threshold as colour)
fig.add_trace(go.Scatter(
    x=df_valid["coverage"] * 100,
    y=df_valid["silhouette"],
    mode="lines+markers+text",
    text=[f"t={r['threshold']:.2f}" for _, r in df_valid.iterrows()],
    textposition="top center",
    textfont=dict(size=9),
    marker=dict(
        size=10,
        color=df_valid["threshold"],
        colorscale="RdYlGn",
        colorbar=dict(title="Threshold", x=0.44),
        showscale=True,
    ),
    line=dict(color="gray", width=1),
    hovertemplate=(
        "threshold=%{text}<br>coverage=%{x:.1f}%<br>"
        "silhouette=%{y:.3f}<extra></extra>"
    ),
), row=1, col=1)

# Mark recommended point
fig.add_trace(go.Scatter(
    x=[rec_cov * 100], y=[rec_sil],
    mode="markers",
    marker=dict(size=18, color="red", symbol="star"),
    name=f"Recommended (t={rec_thresh:.2f})",
    hovertemplate=f"RECOMMENDED<br>threshold={rec_thresh:.2f}<br>"
                  f"coverage={rec_cov:.1%}<br>silhouette={rec_sil:.3f}<extra></extra>",
), row=1, col=1)

# Panel 2: dual axis — silhouette and coherence vs threshold
fig.add_trace(go.Scatter(
    x=df_valid["threshold"],
    y=df_valid["silhouette"],
    mode="lines+markers", name="Silhouette",
    line=dict(color="#2196F3", width=2),
    hovertemplate="threshold=%{x:.2f}<br>silhouette=%{y:.3f}<extra></extra>",
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=df_valid["threshold"],
    y=df_valid["coherence"],
    mode="lines+markers", name="Coherence",
    line=dict(color="#4CAF50", width=2, dash="dash"),
    hovertemplate="threshold=%{x:.2f}<br>coherence=%{y:.3f}<extra></extra>",
), row=1, col=2)
fig.add_vline(x=rec_thresh, line_dash="dash", line_color="red", row=1, col=2)
fig.add_vline(x=1.0, line_dash="dot", line_color="gray", row=1, col=2)

fig.update_xaxes(title_text="Coverage (%)", row=1, col=1)
fig.update_yaxes(title_text="Silhouette score", row=1, col=1)
fig.update_xaxes(title_text="Probability threshold", row=1, col=2)
fig.update_yaxes(title_text="Score", row=1, col=2)

fig.update_layout(
    title=dict(
        text="Coverage vs Quality Tradeoff — CiNii 26k clustering<br>"
             f"<sup>Red star = recommended threshold ({rec_thresh:.2f}): "
             f"{rec_cov:.1%} coverage, silhouette={rec_sil:.3f}</sup>",
        font=dict(size=15),
    ),
    height=550, width=1200,
    legend=dict(x=0.75, y=0.05),
)
out_html = OUT_DIR / "tradeoff.html"
fig.write_html(str(out_html))
print(f"\nSaved: {out_html}")


# ── Recommendation ────────────────────────────────────────────────────────────
row_0   = df_valid[df_valid["threshold"] == 0.0].iloc[0]
row_1   = df_valid[df_valid["threshold"] == 1.0].iloc[0]
sil_gain = rec_sil - row_0["silhouette"]
papers_dropped = int(row_0["n_kept"]) - rec_n

report = f"""
{'='*65}
  COVERAGE vs QUALITY — RECOMMENDATION
{'='*65}

  Question: is it better to train with MORE papers (all 26k,
  lower silhouette) or FEWER but CLEANER papers?

  SHORT ANSWER: Use a probability threshold of {rec_thresh:.2f}.
  This keeps {rec_cov:.1%} of papers ({rec_n:,}) with silhouette={rec_sil:.3f}.

  TRADEOFF TABLE:
  {'threshold':>10}  {'coverage':>9}  {'n_papers':>9}  {'silhouette':>11}  {'coherence':>10}
  {'-'*56}"""

for _, r in df_valid.iterrows():
    marker = " ◀ RECOMMENDED" if r["threshold"] == rec_thresh else \
             " (core-only)"   if r["threshold"] == 1.0         else \
             " (all papers)"  if r["threshold"] == 0.0         else ""
    report += f"\n  {r['threshold']:>10.2f}  {r['coverage']:>8.1%}  {int(r['n_kept']):>9,}  "
    report += f"{r['silhouette']:>11.3f}  {r['coherence']:>10.3f}{marker}"

report += f"""

  {'='*56}

  FULL COVERAGE (threshold=0.0):
    Papers   : {int(row_0['n_kept']):,}  (100%)
    Silhouette: {row_0['silhouette']:.3f}
    Problem  : 43.7% of papers were HDBSCAN outliers — they sit
               between clusters and may carry wrong labels.

  CORE-ONLY (threshold=1.0):
    Papers   : {int(row_1['n_kept']):,}  ({row_1['coverage']:.1%})
    Silhouette: {row_1['silhouette']:.3f}
    Problem  : Discards too much data — rare topics lose most
               of their examples.

  RECOMMENDED (threshold={rec_thresh:.2f}):
    Papers   : {rec_n:,}  ({rec_cov:.1%})
    Silhouette: {rec_sil:.3f}  (+{sil_gain:.3f} vs full coverage)
    Papers dropped: {papers_dropped:,}
    Rationale: Keeps papers that are close to their cluster centre.
               Drops only the most ambiguous boundary cases.
               Best silhouette × coverage tradeoff score.

  WHY SILHOUETTE > COVERAGE FOR TRAINING DATA:
    - Clean labels generalise better than more noisy labels.
    - The 43.7% soft-assigned papers are exactly the ambiguous
      cases where a future classifier will also struggle.
    - Training on them teaches the model the wrong boundaries.
    - Fewer, clean examples >> more, noisy examples for ML.

  Open: {out_html}
{'='*65}
"""

print(report)
with open(str(OUT_DIR / "recommendation.txt"), "w") as f:
    f.write(report)
