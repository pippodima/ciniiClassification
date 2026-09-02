#!/usr/bin/env python3
"""
Auto topic discovery from a parquet DF where embeddings are in column "embeddings".

Install:
  pip install pandas pyarrow numpy umap-learn hdbscan scikit-learn

Run:
  python parquet_auto_topics.py --parquet papers.parquet --emb_col embeddings

Outputs:
  - topic_report.json (best params + all runs)
  - best_labels.npy (topic id per row; -1 = noise)
  - df_with_topics.parquet (original df + topic_id + topic_prob)
"""

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import umap
import hdbscan
from tqdm.auto import tqdm
import time
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score


# ----------------------------
# Utilities: loading embeddings
# ----------------------------

def _parse_embedding_cell(x: Any) -> np.ndarray:
    """
    Parse a single embedding cell into a 1D np.ndarray[float32].
    Supports:
      - list/tuple of floats
      - np.ndarray
      - string representation of list (e.g., "[0.1, 0.2, ...]")
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError("Found null embedding.")

    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, (list, tuple)):
        arr = np.array(x)
    elif isinstance(x, str):
        s = x.strip()
        # Safe-ish parse for list-like strings
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = np.array(json.loads(s))
            except Exception:
                # fallback: split by commas
                inner = s[1:-1].strip()
                if not inner:
                    raise ValueError("Empty embedding string.")
                arr = np.array([float(v) for v in inner.split(",")])
        else:
            raise ValueError(f"Unrecognized embedding string format: {s[:30]}...")
    else:
        # Sometimes parquet stores python objects; try to coerce
        try:
            arr = np.array(x)
        except Exception as e:
            raise ValueError(f"Cannot parse embedding cell of type {type(x)}") from e

    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return arr


def load_embeddings_from_parquet(path: str, emb_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path)
    if emb_col not in df.columns:
        raise KeyError(f"Column '{emb_col}' not found. Columns: {list(df.columns)}")

    embs = []
    bad_rows = []
    for i, v in enumerate(df[emb_col].values):
        try:
            embs.append(_parse_embedding_cell(v))
        except Exception:
            bad_rows.append(i)

    if bad_rows:
        raise ValueError(
            f"Failed to parse embeddings for {len(bad_rows)} rows (e.g., {bad_rows[:10]}). "
            f"Fix/drop these rows or ensure '{emb_col}' contains lists/arrays."
        )

    X = np.vstack(embs)
    # sanity: all same dim
    if len(set([e.shape[0] for e in embs])) != 1:
        dims = pd.Series([e.shape[0] for e in embs]).value_counts().head(10).to_dict()
        raise ValueError(f"Embedding dimensions are not consistent. Top dims: {dims}")

    return df, X


# ----------------------------
# Topic discovery
# ----------------------------

@dataclass
class RunResult:
    n_neighbors: int
    n_components: int
    min_cluster_size: int
    min_samples: int
    seeds: List[int]
    mean_topics: float
    std_topics: float
    mean_noise: float
    std_noise: float
    mean_persistence: float
    mean_membership_prob: float
    mean_ari: float
    score: float


def count_topics(labels: np.ndarray) -> int:
    uniq = set(labels.tolist())
    return len(uniq) - (1 if -1 in uniq else 0)


def run_once(
    X: np.ndarray,
    n_neighbors: int,
    n_components: int,
    min_cluster_size: int,
    min_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
        low_memory=True,
    )
    Z = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(Z)
    probs = clusterer.probabilities_
    return labels, probs, clusterer


def summarize(labels: np.ndarray, probs: np.ndarray, clusterer: hdbscan.HDBSCAN) -> Tuple[int, float, float, float]:
    n_topics = count_topics(labels)
    noise = float(np.mean(labels == -1))
    non_noise = labels != -1
    mean_prob = float(np.mean(probs[non_noise])) if np.any(non_noise) else 0.0
    persistences = getattr(clusterer, "cluster_persistence_", None)
    mean_persist = float(np.mean(persistences)) if persistences is not None and len(persistences) else 0.0
    return n_topics, noise, mean_prob, mean_persist


def score_setting(
    mean_persistence: float,
    mean_membership_prob: float,
    mean_noise: float,
    mean_topics: float,
    std_topics: float,
    mean_ari: float,
    target_topics: Tuple[int, int] = (6, 60),
) -> float:
    lo, hi = target_topics

    if mean_topics <= 0:
        topic_pen = 1.0
    elif mean_topics < lo:
        topic_pen = (lo - mean_topics) / lo
    elif mean_topics > hi:
        topic_pen = (mean_topics - hi) / hi
    else:
        topic_pen = 0.0

    if mean_noise < 0.05:
        noise_pen = (0.05 - mean_noise) / 0.05
    elif mean_noise > 0.25:
        noise_pen = (mean_noise - 0.25) / (1.0 - 0.25)
    else:
        noise_pen = 0.0

    stability_pen = min(1.0, std_topics / max(1.0, mean_topics))

    score = (
        2.8 * mean_persistence +
        1.2 * mean_membership_prob +
        2.0 * mean_ari -
        1.2 * noise_pen -
        1.0 * topic_pen -
        1.5 * stability_pen
    )
    return float(score)


def auto_discover(
    X: np.ndarray,
    neighbors_grid: List[int],
    components_grid: List[int],
    min_cluster_size_grid: List[int],
    min_samples_grid: List[int],
    seeds: List[int],
) -> Tuple[RunResult, Dict[str, Any], List[Dict[str, Any]]]:

    best: Optional[RunResult] = None
    best_artifacts: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []

    total_runs = (
        len(neighbors_grid)
        * len(components_grid)
        * len(min_cluster_size_grid)
        * len(min_samples_grid)
        * len(seeds)
    )

    pbar = tqdm(total=total_runs, desc="Topic discovery", dynamic_ncols=True)

    run_times = []

    for n_neighbors in neighbors_grid:
        for n_components in components_grid:
            for mcs in min_cluster_size_grid:
                for ms in min_samples_grid:

                    per_seed = []
                    labels_list = []
                    probs_list = []

                    for seed in seeds:

                        t0 = time.time()

                        labels, probs, clusterer = run_once(
                            X, n_neighbors, n_components, mcs, ms, seed
                        )

                        dt = time.time() - t0
                        run_times.append(dt)

                        n_topics, noise, mean_prob, mean_persist = summarize(
                            labels, probs, clusterer
                        )

                        per_seed.append((n_topics, noise, mean_prob, mean_persist))
                        labels_list.append(labels)
                        probs_list.append(probs)

                        # ---- progress update ----
                        avg = np.mean(run_times)
                        remaining = avg * (total_runs - pbar.n - 1)

                        pbar.set_postfix({
                            "topics": n_topics,
                            "mcs": mcs,
                            "nn": n_neighbors,
                            "ms": ms,
                            "run_s": f"{dt:.1f}s",
                            "ETA": f"{remaining/60:.1f}m"
                        })
                        pbar.update(1)
                        # -------------------------

                    topics = np.array([x[0] for x in per_seed], dtype=float)
                    noises = np.array([x[1] for x in per_seed], dtype=float)
                    probs_ = np.array([x[2] for x in per_seed], dtype=float)
                    pers_ = np.array([x[3] for x in per_seed], dtype=float)

                    aris = []
                    for i in range(len(labels_list)):
                        for j in range(i + 1, len(labels_list)):
                            aris.append(
                                adjusted_rand_score(labels_list[i], labels_list[j])
                            )
                    mean_ari = float(np.mean(aris)) if aris else 0.0

                    rr = RunResult(
                        n_neighbors=n_neighbors,
                        n_components=n_components,
                        min_cluster_size=mcs,
                        min_samples=ms,
                        seeds=seeds,
                        mean_topics=float(np.mean(topics)),
                        std_topics=float(np.std(topics)),
                        mean_noise=float(np.mean(noises)),
                        std_noise=float(np.std(noises)),
                        mean_persistence=float(np.mean(pers_)),
                        mean_membership_prob=float(np.mean(probs_)),
                        mean_ari=mean_ari,
                        score=0.0,
                    )

                    rr.score = score_setting(
                        mean_persistence=rr.mean_persistence,
                        mean_membership_prob=rr.mean_membership_prob,
                        mean_noise=rr.mean_noise,
                        mean_topics=rr.mean_topics,
                        std_topics=rr.std_topics,
                        mean_ari=rr.mean_ari,
                    )

                    all_rows.append(asdict(rr))

                    if best is None or rr.score > best.score:
                        best = rr
                        idx = int(np.argmin(np.abs(topics - np.mean(topics))))
                        best_artifacts = {
                            "labels": labels_list[idx],
                            "probs": probs_list[idx],
                            "chosen_seed": seeds[idx],
                        }

    pbar.close()

    assert best is not None
    return best, best_artifacts, all_rows


# ----------------------------
# Main
# ----------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import EMBEDDED_26K, CLUSTER_DIR

_out = CLUSTER_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(EMBEDDED_26K), help="Input parquet path.")
    ap.add_argument("--emb_col", default="embeddings", help="Embedding column name.")
    ap.add_argument("--out_report", default=str(_out / "26k_topic_report.json"))
    ap.add_argument("--out_labels", default=str(_out / "26k_best_labels.npy"))
    ap.add_argument("--out_parquet", default=str(_out / "26k_with_embeddings.parquet"))
    ap.add_argument("--no_normalize", action="store_true", help="Disable L2 normalization.")
    args = ap.parse_args()

    df, X = load_embeddings_from_parquet(args.parquet, args.emb_col)

    # L2 normalize: recommended when using cosine geometry
    if not args.no_normalize:
        X = normalize(X, norm="l2")

    n, d = X.shape
    if n < 30:
        raise ValueError(f"Too few rows ({n}). Need at least ~30 for clustering.")
    print(f"Loaded {n} embeddings with dim {d} from {args.parquet}")

    # Tuned grids for ~450 abstracts (adjust if needed)
    neighbors_grid = [10, 15, 20, 25]
    components_grid = [5]
    min_cluster_size_grid = [60, 80, 100, 120, 150, 180]
    min_samples_grid = [1, 3, 5]
    seeds = [22]

    best, artifacts, all_runs = auto_discover(
        X=X,
        neighbors_grid=neighbors_grid,
        components_grid=components_grid,
        min_cluster_size_grid=min_cluster_size_grid,
        min_samples_grid=min_samples_grid,
        seeds=seeds,
    )

    labels = artifacts["labels"]
    probs = artifacts["probs"]

    np.save(args.out_labels, labels)

    df_out = df.copy()
    df_out["topic_id"] = labels.astype(int)
    df_out["topic_prob"] = probs.astype(np.float32)

    df_out.to_parquet(args.out_parquet, index=False)

    report = {
        "best_run": asdict(best),
        "chosen_seed_for_labels": artifacts["chosen_seed"],
        "suggested_integer_topics": int(round(best.mean_topics)),
        "rows": int(n),
        "dim": int(d),
        "outputs": {
            "labels_npy": args.out_labels,
            "df_with_topics_parquet": args.out_parquet,
        },
        "all_runs": all_runs,
        "notes": [
            "topic_id == -1 are outliers/noise.",
            "For fewer topics: increase n_neighbors and/or min_cluster_size.",
            "For more topics: decrease min_cluster_size and/or n_neighbors.",
        ],
    }

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Best run:")
    for k, v in asdict(best).items():
        if k == "seeds":
            continue
        print(f"  {k}: {v}")

    print("\nTopic count estimate:")
    print(f"  mean topics across seeds: {best.mean_topics:.2f} (std {best.std_topics:.2f})")
    print(f"  suggested integer topics: {int(round(best.mean_topics))}")
    print(f"\nWrote:\n  {args.out_report}\n  {args.out_labels}\n  {args.out_parquet}")


if __name__ == "__main__":
    main()
