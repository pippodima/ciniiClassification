import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

from utils import load_df
from config import CLUSTERED_DATA, MERGE_PREFIX


# -------------------------------------------------------------
# 1. ROBUST CLUSTER COHERENCE METRIC
# -------------------------------------------------------------
def cluster_score(dist_matrix, labels):
    """
    A robust metric for evaluating cluster quality:
    - penalizes trivial clusterings (1 cluster)
    - rewards high inter-cluster distance vs low intra-cluster distance
    """
    labels = np.array(labels)
    unique = np.unique(labels)

    if len(unique) < 2:
        return -1.0

    intra, inter = [], []

    for c in unique:
        idx = np.where(labels == c)[0]
        others = np.where(labels != c)[0]

        if len(idx) > 1:
            block = dist_matrix[np.ix_(idx, idx)]
            if not np.isnan(block).any():
                intra.append(block.mean())

        if len(others) > 0:
            block2 = dist_matrix[np.ix_(idx, others)]
            if not np.isnan(block2).any():
                inter.append(block2.mean())

    if not intra or not inter:
        return -1.0

    score = float(np.mean(inter) - np.mean(intra))
    return score if not np.isnan(score) else -1.0


# -------------------------------------------------------------
# 2. COMPUTE CLUSTER CENTROIDS (EXCLUDES -1)
# -------------------------------------------------------------
def compute_centroids(df):
    df = df[df["cluster_id"] != -1]  
    centroids = (
        df.groupby("cluster_id")["embeddings"]
          .apply(lambda x: np.vstack(x.values).mean(axis=0))
          .to_dict()
    )
    return centroids


# -------------------------------------------------------------
# 3. AUTO-TUNING DISTANCE THRESHOLD (BALANCED)
# -------------------------------------------------------------
def evaluate_thresholds(centroid_matrix, thresholds):
    dist_matrix = cosine_distances(centroid_matrix)
    results = {}

    for th in thresholds:
        try:
            model = AgglomerativeClustering(
                metric="precomputed",
                linkage="average",
                distance_threshold=th,
                n_clusters=None,
            )
            labels = model.fit_predict(dist_matrix)
            n_clusters = len(set(labels))
            score = cluster_score(dist_matrix, labels)
            results[th] = {"score": float(score), "n_clusters": int(n_clusters)}
        except Exception:
            results[th] = {"score": -1.0, "n_clusters": 0}

    return results


def choose_best_threshold(
    centroid_matrix,
    target_min_clusters=8,
    target_max_clusters=15
):
    thresholds = np.linspace(0.10, 0.40, 31)
    results = evaluate_thresholds(centroid_matrix, thresholds)

    print("\n=== AUTO-TUNING SUMMARY ===")
    for th, res in results.items():
        print(
            f"Threshold {th:.3f} → "
            f"Score {res['score']:.4f}, "
            f"n_clusters = {res['n_clusters']}"
        )

    # 1) Best score within target cluster count range
    valid = {
        th: r for th, r in results.items()
        if r["score"] > 0
        and target_min_clusters <= r["n_clusters"] <= target_max_clusters
    }

    if valid:
        best_th = max(valid, key=lambda k: valid[k]["score"])
        print(
            f"\nBest threshold (within {target_min_clusters}-{target_max_clusters} clusters): "
            f"{best_th:.3f} → score {valid[best_th]['score']:.4f}, "
            f"n_clusters={valid[best_th]['n_clusters']}"
        )
        return best_th, results

    # 2) Otherwise: pick closest n_clusters to midpoint
    positive = {th: r for th, r in results.items() if r["score"] > 0}
    if positive:
        target_mid = (target_min_clusters + target_max_clusters) / 2
        best_th = min(
            positive.keys(),
            key=lambda th: abs(positive[th]["n_clusters"] - target_mid)
        )
        print(
            f"\nNo thresholds in target range; picked closest to {target_mid}: "
            f"{best_th:.3f} (n_clusters={positive[best_th]['n_clusters']})"
        )
        return best_th, results

    # 3) Fallback
    print("\nNo valid thresholds found; using fallback 0.20")
    return 0.20, results


# -------------------------------------------------------------
# 4. MERGE CLUSTERS
# -------------------------------------------------------------
def merge_clusters(cluster_centroids, distance_threshold):
    cluster_ids = list(cluster_centroids.keys())
    centroid_matrix = np.vstack([cluster_centroids[c] for c in cluster_ids])
    dist_matrix = cosine_distances(centroid_matrix)

    model = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = model.fit_predict(dist_matrix)

    cluster_to_main = {
        cluster_ids[i]: int(labels[i])
        for i in range(len(cluster_ids))
    }

    return cluster_to_main, labels, centroid_matrix, cluster_ids


# -------------------------------------------------------------
# 5. EXTRACT MAIN TOPIC NAMES (NO TF-IDF)
# -------------------------------------------------------------
def extract_topic_keywords(df, cluster_to_main):
    df_valid = df[df["cluster_id"] != -1]

    cluster_names = (
        df_valid.groupby("cluster_id")["name"]
            .apply(lambda x: sorted(set(x.values)))
            .to_dict()
    )

    topic_keywords = {}

    for cid, main_id in cluster_to_main.items():
        topic_keywords.setdefault(main_id, [])
        topic_keywords[main_id].extend(cluster_names.get(cid, []))

    # Deduplicate
    for t in topic_keywords:
        topic_keywords[t] = sorted(set(topic_keywords[t]))

    return topic_keywords


# -------------------------------------------------------------
# 6. TAG DOCUMENTS
# -------------------------------------------------------------
def assign_main_topics(df, cluster_to_main):
    df["main_topic"] = df["cluster_id"].map(cluster_to_main)
    return df


def compute_subtopics(df, cluster_centroids, threshold=0.25):

    def soft_tags(embedding):
        sims = []
        for cid, centroid in cluster_centroids.items():
            if cid == -1:
                continue
            sim = 1 - cosine_distances([embedding], [centroid])[0][0]
            if sim >= threshold:
                sims.append(cid)
        return sims

    df["subtopics"] = df["embeddings"].apply(soft_tags)
    return df


# -------------------------------------------------------------
# 7. DENDROGRAM
# -------------------------------------------------------------
def plot_dendrogram(centroid_matrix, cluster_ids, filename="dendrogram.png"):
    linked = linkage(centroid_matrix, method="average", metric="cosine")

    plt.figure(figsize=(14, 7))
    dendrogram(
        linked,
        labels=[str(cid) for cid in cluster_ids],
        leaf_rotation=90,
        leaf_font_size=7
    )
    plt.title("Cluster Centroid Dendrogram")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Dendrogram saved to {filename}")


# -------------------------------------------------------------
# 8. SAVE OUTPUT
# -------------------------------------------------------------
def save_results(df, topic_keywords, prefix="output"):
    """
    Save:
      - tagged documents as .parquet
      - merged topic keywords as .json
    """
    # Save Parquet (fast, compressed)
    df.to_parquet(f"{prefix}_tagged_documents.parquet", index=False)

    # Save merged cluster keywords/topics
    with open(f"{prefix}_merged_topics.json", "w") as f:
        json.dump(topic_keywords, f, indent=2)

    print(f"Saved Parquet: {prefix}_tagged_documents.parquet")
    print(f"Saved topics JSON: {prefix}_merged_topics.json")


# -------------------------------------------------------------
# 9. MAIN PIPELINE
# -------------------------------------------------------------
def run_pipeline(
    db_path,
    output_prefix="output",
    auto_tune=True,
    manual_threshold=0.18,
    subtopic_similarity=0.25,
    plot=True
):

    df = load_df(db_path)
    print(f"Loaded {len(df)} documents.")

    # Remove outlier cluster -1 NOW
    df = df[df["cluster_id"] != -1].copy()
    print(f"After removing cluster -1: {len(df)} documents")

    cluster_centroids = compute_centroids(df)
    print(f"Computed centroids for {len(cluster_centroids)} clusters.")

    centroid_matrix = np.vstack([cluster_centroids[c] for c in cluster_centroids])

    if auto_tune:
        best_th, _ = choose_best_threshold(centroid_matrix)
        distance_threshold = best_th
    else:
        distance_threshold = manual_threshold

    print(f"\nUsing distance threshold: {distance_threshold:.3f}")

    cluster_to_main, labels, centroid_matrix, cluster_ids = merge_clusters(
        cluster_centroids, distance_threshold
    )
    print(f"Merged into {len(set(labels))} main topics.")

    topic_keywords = extract_topic_keywords(df, cluster_to_main)

    df = assign_main_topics(df, cluster_to_main)
    df = compute_subtopics(df, cluster_centroids, threshold=subtopic_similarity)

    save_results(df, topic_keywords, prefix=output_prefix)
    print("Saved tagged documents and main topic info.")

    if plot:
        plot_dendrogram(centroid_matrix, cluster_ids, filename=f"{output_prefix}_dendrogram.png")


# -------------------------------------------------------------
# RUN SCRIPT
# -------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(
        db_path=str(CLUSTERED_DATA),
        output_prefix=MERGE_PREFIX,
        auto_tune=True,
        manual_threshold=0.18,
        subtopic_similarity=0.25,
        plot=True
    )
