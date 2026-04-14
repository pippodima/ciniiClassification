import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from utils import load_df
import numpy as np
import umap

def plot_static(umap_embeddings, labels, path="umap_clusters.png"):
    plt.figure(figsize=(10, 8))
    plt.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=labels,
        s=8,
        cmap="Spectral"
    )
    plt.title("UMAP Clusters (Static)")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"📊 Saved static plot → {path}")


def plot_interactive(df, umap_embeddings, path="umap_clusters.html"):
    df_plot = pd.DataFrame({
        "x": umap_embeddings[:, 0],
        "y": umap_embeddings[:, 1],
        "title": df["title_en"].fillna(""),
        "cluster": df["cluster_id"].astype(str),
        "name": df["name"].fillna("")
    })

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="cluster",
        hover_data=["title", "name"],
        title="UMAP Interactive Clusters"
    )

    fig.write_html(path)
    print(f"🌐 Saved interactive plot → {path}")


def run_umap(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine"):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42
    )
    return reducer.fit_transform(embeddings)

def plot_umap_html(df, html_path="umap_plot.html"):
    """
    df must include columns: x, y, title_en, main_topic_name, subtopic_name
    """
    fig = px.scatter(
        df,
        x="x", y="y",
        hover_name="title_en",
        color="main_topic_name",
        hover_data={
            "main_topic_name": True,
            "subtopic_name": True,
            "x": False,
            "y": False,
        },
        opacity=0.8,
        width=1000,
        height=700
    )
    fig.update_traces(marker=dict(size=6))
    fig.write_html(html_path)
    print(f"Saved interactive plot → {html_path}")
    return fig



def main():
    df = load_df("output/final_data.parquet")
    embeddings = np.vstack(df["embeddings"].values)
    umap_2d = run_umap(embeddings)
    df["x"] = umap_2d[:,0]
    df["y"] = umap_2d[:,1]
    fig = plot_umap_html(df)
    fig.show()



if __name__ == "__main__":
    main()
