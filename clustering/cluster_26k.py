"""
Academic Paper Topic Explorer
==============================
Automatically discovers topics and subtopics from paper embeddings.

Usage:
    pip install pandas pyarrow umap-learn hdbscan bertopic plotly scikit-learn numpy
    python topic_explorer.py --input papers.parquet --col embeddings

Optional args:
    --titles_col   column name with paper titles/abstracts (for hover labels)
    --output_dir   where to save HTML visualizations (default: ./topic_output)
    --min_topic    minimum papers per topic (default: 30)
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import EMBEDDED_26K, VIZ_DIR
import numpy as np
import pandas as pd

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",      default=str(EMBEDDED_26K), help="Path to .parquet file")
parser.add_argument("--col",        default="embeddings", help="Embedding column name")
parser.add_argument("--titles_col", default=None,   help="Column for paper titles/abstracts")
parser.add_argument("--output_dir", default=str(VIZ_DIR / "topic_output"))
parser.add_argument("--min_topic",  type=int, default=30,
                    help="Min papers per topic — lower = more granular")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── 1. LOAD EMBEDDINGS ─────────────────────────────────────────────────────────
print("📂 Loading parquet...")
df = pd.read_parquet(args.input)

raw = df[args.col].tolist()

# Handle various storage formats: list-of-lists, list-of-arrays, or already 2D array
if isinstance(raw[0], (list, np.ndarray)):
    embeddings = np.array(raw, dtype=np.float32)
else:
    # stored as a single numpy array row per cell (e.g. from .to_numpy())
    embeddings = np.vstack(raw).astype(np.float32)

print(f"✅ Loaded {embeddings.shape[0]:,} papers, embedding dim = {embeddings.shape[1]}")

# Optional: use titles/abstracts for BERTopic labels; fall back to empty strings
if args.titles_col and args.titles_col in df.columns:
    docs = df[args.titles_col].fillna("").tolist()
    print(f"📝 Using '{args.titles_col}' column for topic labeling")
else:
    docs = ["" for _ in range(len(embeddings))]
    print("ℹ️  No text column provided — topic labels will use cluster IDs only")

# ── 2. UMAP: reduce to 2D for clustering AND visualization ────────────────────
from umap import UMAP

print("\n🔵 Running UMAP dimensionality reduction...")

# Two UMAP runs:
#   - low min_dist (0.0) for clustering (tight, well-separated clusters)
#   - higher min_dist (0.1) for visualization (more spread, readable scatter)

umap_cluster = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.0,       # tight clusters — better for HDBSCAN
    metric="cosine",
    random_state=42,
    low_memory=True,    # important for 26k+ papers
)
embeddings_2d_cluster = umap_cluster.fit_transform(embeddings)

umap_viz = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,       # spread out — better for visualization
    metric="cosine",
    random_state=42,
    low_memory=True,
)
embeddings_2d_viz = umap_viz.fit_transform(embeddings)

print("✅ UMAP done")

# ── 3. HDBSCAN: auto-discover cluster count ───────────────────────────────────
import hdbscan

print("\n🟡 Running HDBSCAN clustering (auto-finding topic count)...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=args.min_topic,
    min_samples=5,              # lower = more clusters, more noise tolerance
    metric="euclidean",         # on UMAP-reduced space
    cluster_selection_method="eom",   # "eom" finds more clusters; try "leaf" for subtopics
    prediction_data=True,
)
cluster_labels = clusterer.fit_predict(embeddings_2d_cluster)

n_topics    = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_outliers  = (cluster_labels == -1).sum()
print(f"✅ Found {n_topics} topics | {n_outliers:,} outlier papers (label = -1)")

# ── 4. BERTOPIC: auto-label topics using TF-IDF ───────────────────────────────
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

print("\n🟢 Running BERTopic for topic labeling...")

# We pass our pre-computed UMAP + HDBSCAN so BERTopic skips re-running them
topic_model = BERTopic(
    umap_model=umap_cluster,
    hdbscan_model=clusterer,
    vectorizer_model=CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
    ),
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

# ── 5. PRINT TOPIC SUMMARY ────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  TOPIC SUMMARY  ({n_topics} topics discovered)")
print("="*60)
topic_info = topic_model.get_topic_info()
print(topic_info[topic_info["Topic"] != -1]
      [["Topic", "Count", "Name"]].to_string(index=False))

# Save topic info CSV
csv_path = os.path.join(args.output_dir, "topic_summary.csv")
topic_info.to_csv(csv_path, index=False)
print(f"\n💾 Topic summary saved → {csv_path}")

# ── 6. VISUALIZATIONS ─────────────────────────────────────────────────────────
print("\n🎨 Generating visualizations...")

# 6a. Interactive scatter — every paper as a dot, colored by topic
import plotly.express as px

df_viz = pd.DataFrame({
    "x": embeddings_2d_viz[:, 0],
    "y": embeddings_2d_viz[:, 1],
    "topic": [str(t) for t in topics],
    "label": docs if args.titles_col else ["Paper " + str(i) for i in range(len(topics))],
})

fig_scatter = px.scatter(
    df_viz, x="x", y="y",
    color="topic",
    hover_data={"label": True, "topic": True, "x": False, "y": False},
    title=f"Academic Paper Topic Map — {n_topics} topics discovered",
    width=1200, height=800,
    opacity=0.6,
)
fig_scatter.update_traces(marker=dict(size=3))
fig_scatter.update_layout(legend_title_text="Topic ID")

scatter_path = os.path.join(args.output_dir, "01_paper_scatter.html")
fig_scatter.write_html(scatter_path)
print(f"  ✅ Paper scatter map  → {scatter_path}")

# 6b. BERTopic bubble map (topic distance map)
try:
    fig_topics = topic_model.visualize_topics()
    topics_path = os.path.join(args.output_dir, "02_topic_bubbles.html")
    fig_topics.write_html(topics_path)
    print(f"  ✅ Topic bubble map   → {topics_path}")
except Exception as e:
    print(f"  ⚠️  Topic bubble map skipped: {e}")

# 6c. Hierarchy dendrogram (topics + subtopics)
try:
    fig_hier = topic_model.visualize_hierarchy()
    hier_path = os.path.join(args.output_dir, "03_topic_hierarchy.html")
    fig_hier.write_html(hier_path)
    print(f"  ✅ Topic hierarchy    → {hier_path}")
except Exception as e:
    print(f"  ⚠️  Hierarchy skipped: {e}")

# 6d. Heatmap — topic similarity matrix
try:
    fig_heat = topic_model.visualize_heatmap()
    heat_path = os.path.join(args.output_dir, "04_topic_heatmap.html")
    fig_heat.write_html(heat_path)
    print(f"  ✅ Topic heatmap      → {heat_path}")
except Exception as e:
    print(f"  ⚠️  Heatmap skipped: {e}")

# 6e. Document-level scatter with topic labels (BERTopic's own version)
try:
    fig_docs = topic_model.visualize_documents(
        docs,
        embeddings=embeddings_2d_viz,
        hide_annotations=False,
        hide_document_hover=False,
    )
    docs_path = os.path.join(args.output_dir, "05_documents_scatter.html")
    fig_docs.write_html(docs_path)
    print(f"  ✅ Labeled doc scatter → {docs_path}")
except Exception as e:
    print(f"  ⚠️  Doc scatter skipped: {e}")

# ── 7. SAVE ENRICHED PARQUET ──────────────────────────────────────────────────
df["topic_id"]    = topics
df["topic_name"]  = [topic_model.get_topic_info(t)["Name"].values[0]
                     if t != -1 else "outlier"
                     for t in topics]
df["umap_x"]      = embeddings_2d_viz[:, 0]
df["umap_y"]      = embeddings_2d_viz[:, 1]

out_parquet = os.path.join(args.output_dir, "papers_with_topics.parquet")
df.drop(columns=[args.col]).to_parquet(out_parquet, index=False)
print(f"\n💾 Enriched parquet (no embeddings col) saved → {out_parquet}")

print("\n🎉 Done! Open any .html file in your browser to explore.")
print(f"   Output folder: {os.path.abspath(args.output_dir)}")