"""
Cluster Quality Validator
==========================
Validates clustering label quality using:
  1. Silhouette score (sample-based for speed)
  2. Intra-cluster cosine similarity (cohesion)
  3. Inter-cluster separation
  4. Confusion matrix between similar topics (from classifier)

Usage:
    pip install pandas pyarrow scikit-learn numpy plotly joblib
    python validate_clusters.py \
        --embeddings data/embedded/embedded_data_26k_clean.parquet \
        --labels     clusters/topic_output/papers_with_topics.parquet \
        --output_dir validation/

Optional (adds confusion matrix):
    --classifier classifier/topic_classifier_mlp.joblib
"""

import argparse, os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import normalize

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--embeddings",  required=True)
parser.add_argument("--labels",      required=True)
parser.add_argument("--emb_col",     default="embeddings")
parser.add_argument("--output_dir",  default="clusters/topic_output")
parser.add_argument("--classifier",  default=None, help="Path to .joblib model (optional)")
parser.add_argument("--sample_sil",  type=int, default=8000,
                    help="Samples for silhouette (full 26k is slow, 8k is representative)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── 1. LOAD ────────────────────────────────────────────────────────────────────
print("📂 Loading data...")
df_emb    = pd.read_parquet(args.embeddings)
df_labels = pd.read_parquet(args.labels)

assert len(df_emb) == len(df_labels), "Row count mismatch!"

raw = df_emb[args.emb_col].tolist()
embeddings = normalize(np.array(raw, dtype=np.float32), norm="l2")

topic_ids   = df_labels["topic_id"].values
topic_names = df_labels["topic_name"].values

# Keep only clustered papers (drop outliers)
mask        = topic_ids != -1
X           = embeddings[mask]
y           = topic_ids[mask]
names       = topic_names[mask]
n_topics    = len(np.unique(y))
n_papers    = X.shape[0]

print(f"✅ {n_papers:,} clustered papers | {n_topics} topics")

# ── 2. SILHOUETTE SCORE ───────────────────────────────────────────────────────
print(f"\n🔵 Computing silhouette score (sample={args.sample_sil:,})...")

rng = np.random.default_rng(42)
if n_papers > args.sample_sil:
    idx = rng.choice(n_papers, size=args.sample_sil, replace=False)
    X_s, y_s = X[idx], y[idx]
else:
    X_s, y_s = X, y

sil_global = silhouette_score(X_s, y_s, metric="cosine")
sil_samples = silhouette_samples(X_s, y_s, metric="cosine")

print(f"   Global silhouette score: {sil_global:.4f}  (range -1 to 1, >0.3 is decent, >0.5 is good)")

# Per-topic silhouette (only topics present in sample)
sil_per_topic = {}
for tid in np.unique(y_s):
    mask_t = y_s == tid
    if mask_t.sum() > 1:
        sil_per_topic[tid] = sil_samples[mask_t].mean()

sil_df = pd.DataFrame([
    {"topic_id": tid, "silhouette": score,
     "topic_name": names[y == tid][0],
     "count": (y == tid).sum()}
    for tid, score in sil_per_topic.items()
]).sort_values("silhouette")

# Flag weak topics
weak = sil_df[sil_df["silhouette"] < 0.1]
print(f"   ⚠️  Topics with silhouette < 0.1 (poorly separated): {len(weak)}")
if len(weak):
    print(weak[["topic_id", "topic_name", "silhouette", "count"]].to_string(index=False))

# Plot per-topic silhouette
fig_sil = px.bar(
    sil_df, x="silhouette", y="topic_name",
    orientation="h",
    color="silhouette",
    color_continuous_scale="RdYlGn",
    range_color=[-0.1, 0.8],
    title=f"Per-Topic Silhouette Score (global = {sil_global:.3f})",
    hover_data={"count": True},
    height=max(600, n_topics * 14),
)
fig_sil.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
sil_path = os.path.join(args.output_dir, "01_silhouette_per_topic.html")
fig_sil.write_html(sil_path)
print(f"💾 Silhouette chart → {sil_path}")

# ── 3. INTRA-CLUSTER COHESION ─────────────────────────────────────────────────
print("\n🟡 Computing intra-cluster cosine similarity (cohesion)...")

cohesion = {}
for tid in np.unique(y):
    vecs = X[y == tid]
    if len(vecs) < 2:
        cohesion[tid] = 1.0
        continue
    # Mean cosine similarity = mean dot product (since L2-normalized)
    # Efficient: (V @ V.T) gives all pairwise, take mean off-diagonal
    if len(vecs) > 300:
        vecs = vecs[rng.choice(len(vecs), 300, replace=False)]
    sim_matrix = vecs @ vecs.T
    np.fill_diagonal(sim_matrix, np.nan)
    cohesion[tid] = np.nanmean(sim_matrix)

# ── 4. INTER-CLUSTER SEPARATION ───────────────────────────────────────────────
print("🟠 Computing inter-cluster separation (centroid distances)...")

centroids = {}
for tid in np.unique(y):
    c = X[y == tid].mean(axis=0)
    centroids[tid] = c / np.linalg.norm(c)  # re-normalize centroid

topic_ids_list = list(centroids.keys())
C = np.array([centroids[t] for t in topic_ids_list])
sim_matrix_centroids = C @ C.T  # cosine similarity between all centroids

# Average inter-cluster distance per topic (lower sim = better separated)
separation = {}
for i, tid in enumerate(topic_ids_list):
    others = np.delete(sim_matrix_centroids[i], i)
    separation[tid] = 1 - others.mean()  # convert similarity → distance

# Combined quality df
quality_df = pd.DataFrame({
    "topic_id":   list(cohesion.keys()),
    "cohesion":   list(cohesion.values()),
    "separation": [separation.get(t, np.nan) for t in cohesion.keys()],
    "silhouette": [sil_per_topic.get(t, np.nan) for t in cohesion.keys()],
    "count":      [(y == t).sum() for t in cohesion.keys()],
    "topic_name": [names[y == t][0] for t in cohesion.keys()],
}).sort_values("cohesion", ascending=False)

# Flag problematic topics: low cohesion OR low separation
quality_df["status"] = "✅ good"
quality_df.loc[quality_df["cohesion"] < 0.3, "status"] = "⚠️ low cohesion"
quality_df.loc[quality_df["separation"] < 0.3, "status"] = "⚠️ low separation"
quality_df.loc[
    (quality_df["cohesion"] < 0.3) & (quality_df["separation"] < 0.3), "status"
] = "🔴 review needed"

csv_path = os.path.join(args.output_dir, "cluster_quality_stats.csv")
quality_df.to_csv(csv_path, index=False)
print(f"💾 Quality stats → {csv_path}")

# Scatter: cohesion vs separation (bubble = topic size)
fig_scatter = px.scatter(
    quality_df, x="separation", y="cohesion",
    size="count", color="status",
    hover_data={"topic_name": True, "count": True, "silhouette": ":.3f"},
    title="Topic Quality: Cohesion vs Separation",
    labels={"separation": "Inter-cluster separation (higher=better)",
            "cohesion":   "Intra-cluster cohesion (higher=better)"},
    color_discrete_map={"✅ good": "#2ecc71", "⚠️ low cohesion": "#f39c12",
                        "⚠️ low separation": "#3498db", "🔴 review needed": "#e74c3c"},
    width=900, height=700,
)
fig_scatter.add_hline(y=0.3, line_dash="dot", line_color="gray", annotation_text="cohesion threshold")
fig_scatter.add_vline(x=0.3, line_dash="dot", line_color="gray", annotation_text="separation threshold")
scatter_path = os.path.join(args.output_dir, "02_cohesion_vs_separation.html")
fig_scatter.write_html(scatter_path)
print(f"💾 Cohesion vs separation → {scatter_path}")

# ── 5. CONFUSION MATRIX (requires classifier) ─────────────────────────────────
if args.classifier:
    import joblib
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix

    print("\n🔴 Computing confusion matrix via cross-validation...")
    bundle = joblib.load(args.classifier)
    clf, le = bundle["model"], bundle["label_encoder"]

    y_encoded = le.transform(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true_all, y_pred_all = [], []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_encoded)):
        print(f"   Fold {fold+1}/5...")
        clf_fold = clf.__class__(**clf.get_params()) if hasattr(clf, 'get_params') else type(clf)()
        clf_fold.fit(X[train_idx], y_encoded[train_idx])
        y_true_all.extend(y_encoded[val_idx])
        y_pred_all.extend(clf_fold.predict(X[val_idx]))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Only show the TOP confusions (most confused pairs), not full 178x178 matrix
    n = len(le.classes_)
    confused_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_topic":      le.classes_[i],
                    "predicted_topic": le.classes_[j],
                    "true_name":   names[y == le.classes_[i]][0] if (y == le.classes_[i]).any() else str(le.classes_[i]),
                    "pred_name":   names[y == le.classes_[j]][0] if (y == le.classes_[j]).any() else str(le.classes_[j]),
                    "count":       int(cm[i, j]),
                    "rate":        float(cm_norm[i, j]),
                })

    confused_df = pd.DataFrame(confused_pairs).sort_values("count", ascending=False).head(50)
    confused_path = os.path.join(args.output_dir, "top_confusions.csv")
    confused_df.to_csv(confused_path, index=False)
    print(f"💾 Top confusions → {confused_path}")

    # Heatmap of top-30 most confused topics
    top_topics_idx = list(set(
        confused_df["true_topic"].tolist()[:30] +
        confused_df["predicted_topic"].tolist()[:30]
    ))
    top_topics_idx = sorted(top_topics_idx)[:30]
    idx_map = {t: i for i, t in enumerate(le.classes_) if t in top_topics_idx}
    sub_idx = [list(le.classes_).index(t) for t in top_topics_idx]
    cm_sub = cm_norm[np.ix_(sub_idx, sub_idx)]
    labels_sub = [names[y == t][0][:40] for t in top_topics_idx]

    fig_cm = go.Figure(go.Heatmap(
        z=cm_sub, x=labels_sub, y=labels_sub,
        colorscale="Reds", zmin=0, zmax=0.5,
        hoverongaps=False,
    ))
    fig_cm.update_layout(
        title="Confusion Heatmap — Top 30 Most Confused Topics (normalized)",
        xaxis_tickangle=-45,
        width=1100, height=1000,
    )
    cm_path = os.path.join(args.output_dir, "03_confusion_heatmap.html")
    fig_cm.write_html(cm_path)
    print(f"💾 Confusion heatmap → {cm_path}")

    print("\n🔍 Top 15 most confused topic pairs:")
    print(confused_df[["true_name", "pred_name", "count", "rate"]].head(15).to_string(index=False))
else:
    print("\nℹ️  Skipping confusion matrix — run with --classifier to enable")

# ── 6. SUMMARY REPORT ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  VALIDATION SUMMARY")
print("="*60)
print(f"  Global silhouette score : {sil_global:.4f}")
print(f"  Mean intra cohesion     : {np.nanmean(list(cohesion.values())):.4f}")
print(f"  Mean inter separation   : {np.nanmean(list(separation.values())):.4f}")
print(f"  Topics flagged ⚠️/🔴    : {(quality_df['status'] != '✅ good').sum()} / {n_topics}")
print("="*60)

flagged = quality_df[quality_df["status"] != "✅ good"][["topic_id","topic_name","cohesion","separation","count","status"]]
if len(flagged):
    print("\n🔍 Topics to review:")
    print(flagged.to_string(index=False))

print(f"\n🎉 Done! Open HTML files in: {os.path.abspath(args.output_dir)}")
print("""
Interpreting results:
  Silhouette > 0.5  → excellent separation
  Silhouette 0.3-0.5 → reasonable (acceptable for 178 topics)
  Silhouette < 0.1  → topic overlaps with neighbors → consider merging

  Cohesion > 0.5    → papers in topic are tightly related
  Separation > 0.3  → topic is well separated from others

  Confusion matrix  → pairs with high confusion rate → candidates for merging
""")