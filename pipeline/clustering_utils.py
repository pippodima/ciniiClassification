import numpy as np
import hdbscan
import umap
from tqdm import tqdm


def run_umap(embeddings, n_neighbors=15, min_dist=0.1, n_components=10, random_state=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
        low_memory=True
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan(embeddings, min_cluster_size=30, min_samples=5, epsilon=0.3):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="leaf"
    )
    return clusterer.fit_predict(embeddings)


def auto_optimize_clustering(embeddings):
    """
    Returns:
        best_umap (np.ndarray)
        best_labels (np.ndarray)
        best_cfg (dict)
    """

    configs = []
    for n in [5, 10, 20]:
        for d in [0.05, 0.1, 0.2]:
            for c in [10, 20, 50]:
                configs.append({"neighbors": n, "min_dist": d, "cluster_size": c})

    best_score, best_cfg = -1, None
    best_labels, best_umap = None, None

    for cfg in tqdm(configs, desc="🔎 Auto-optimizing clustering"):
        reduced = run_umap(
            embeddings,
            n_neighbors=cfg["neighbors"],
            min_dist=cfg["min_dist"]
        )

        labels = run_hdbscan(reduced, min_cluster_size=cfg["cluster_size"])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        outliers = list(labels).count(-1)

        score = n_clusters / (1 + outliers / max(1, len(labels)))

        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_umap = reduced
            best_labels = labels

    return best_umap, best_labels, best_cfg
