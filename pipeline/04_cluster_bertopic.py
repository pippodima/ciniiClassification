import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bertopic import BERTopic
from umap import UMAP
import hdbscan
from utils import load_df

import numpy as np
import pandas as pd
import json
import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict
from config import CLUSTERED_DATA, BERT_TOPICS_JSON, BERT_DOCS_PARQUET, BERT_DOCMAP_HTML


# =================================================
# CONFIG
# =================================================
OUTLIER_TOPIC = -1
SIMILARITY_THRESHOLD = 0.85
RANDOM_STATE = 42


# =================================================
# 1. LOAD DATA
# =================================================
df = load_df(str(CLUSTERED_DATA))

documents = df["full_text"].tolist()
doc_embeddings = np.vstack(df["embeddings"].values)


# =================================================
# 2. TRAIN BERTopic
# =================================================
topic_model = BERTopic(
    umap_model=UMAP(
        n_neighbors=25,
        n_components=10,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE
    ),
    hdbscan_model=hdbscan.HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom"
    ),
    verbose=True
)

topics, probs = topic_model.fit_transform(documents, doc_embeddings)
topic_labels = topic_model.generate_topic_labels()


# =================================================
# 3. HIERARCHICAL MERGING (TOPIC LEVEL)
# =================================================
info = topic_model.get_topic_info()
topic_sizes = dict(zip(info["Topic"], info["Count"]))

# valid topics
mask = info["Topic"].astype(int) != OUTLIER_TOPIC
topic_ids = info["Topic"][mask].astype(int).values
topic_embs = topic_model.topic_embeddings_[mask]

# cosine → distance
dist = 1 - cosine_similarity(topic_embs)
condensed_dist = squareform(dist, checks=False)

Z = linkage(condensed_dist, method="average")

cluster_ids = fcluster(
    Z,
    t=1 - SIMILARITY_THRESHOLD,
    criterion="distance"
)

clusters = defaultdict(list)
for tid, cid in zip(topic_ids, cluster_ids):
    clusters[cid].append(tid)

# main topic = largest cluster
subtopics = {}
for members in clusters.values():
    main = max(members, key=lambda t: topic_sizes.get(t, 0))
    subtopics[main] = set(members)

main_topic_map = {
    sub: main
    for main, subs in subtopics.items()
    for sub in subs
}


# =================================================
# 4. REASSIGN DOCUMENT TOPICS
# =================================================
main_topics = [
    OUTLIER_TOPIC if t == OUTLIER_TOPIC else main_topic_map[int(t)]
    for t in topics
]


# =================================================
# 5. EXPORT HIERARCHY (NAMED)
# =================================================
named_merge_map = {
    topic_labels[main]: [topic_labels[s] for s in sorted(subs)]
    for main, subs in subtopics.items()
}

BERT_TOPICS_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(str(BERT_TOPICS_JSON), "w") as f:
    json.dump(named_merge_map, f, indent=2, ensure_ascii=False)


# =================================================
# 6. SAVE DOCUMENT-LEVEL DATA
# =================================================
df_out = pd.DataFrame({
    "document": documents,
    "original_topic": topics,
    "main_topic": main_topics,
    "main_topic_label": [
        topic_labels[t] if t != OUTLIER_TOPIC else "Outlier"
        for t in main_topics
    ],
    "subtopic_label": [
        topic_labels[t] if t != OUTLIER_TOPIC else "Outlier"
        for t in topics
    ]
})

df_out.to_parquet(str(BERT_DOCS_PARQUET))


# =================================================
# 7. MANUAL 2D VISUALIZATION (CORRECT)
# =================================================
umap_2d = UMAP(
    n_neighbors=20,
    n_components=2,
    min_dist=0.1,
    metric="euclidean",
    random_state=RANDOM_STATE
).fit_transform(doc_embeddings)

df_plot = pd.DataFrame({
    "x": umap_2d[:, 0],
    "y": umap_2d[:, 1],
    "main_topic": [
        topic_labels[t] if t != OUTLIER_TOPIC else "Outlier"
        for t in main_topics
    ],
    "subtopic": [
        topic_labels[t] if t != OUTLIER_TOPIC else "Outlier"
        for t in topics
    ]
})

fig = px.scatter(
    df_plot,
    x="x",
    y="y",
    color="main_topic",
    hover_data={
        "main_topic": True,
        "subtopic": True,
        "x": False,
        "y": False
    },
    title="Documents colored by MAIN topic (hierarchical merge)"
)

fig.update_traces(marker=dict(size=4, opacity=0.75))

# add main-topic labels
label_df = (
    df_plot[df_plot["main_topic"] != "Outlier"]
    .groupby("main_topic")[["x", "y"]]
    .mean()
    .reset_index()
)

fig.add_scatter(
    x=label_df["x"],
    y=label_df["y"],
    text=label_df["main_topic"],
    mode="text",
    textposition="middle center",
    showlegend=False,
    hoverinfo="skip"
)

fig.write_html(str(BERT_DOCMAP_HTML))
fig.show()
