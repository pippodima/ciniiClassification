import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_df, save
from config import MERGE_METADATA, MERGE_MAIN_TOPICS, CLUSTERED_DATA, TRAINING_DIR, LABELED_26K
import pandas as pd

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)


# -------------------------------------------------
# MAPPING HELPERS
# -------------------------------------------------

def cluster_id_to_subtopic(df):
    """Build mapping cluster_id → subtopic name."""
    required_cols = {"cluster_id", "subtopic"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    return dict(zip(df["cluster_id"], df["subtopic"]))


def build_subcluster_to_merged_id(df):
    """Build mapping subcluster triplet → merged cluster id."""
    if not {"merged_cluster_id", "subclusters"}.issubset(df.columns):
        raise ValueError("Parquet file must contain 'merged_cluster_id' and 'subclusters'.")

    mapping = {}
    for _, row in df.iterrows():
        merged_id = row["merged_cluster_id"]
        for sub in row["subclusters"]:
            mapping[sub] = merged_id

    return mapping


def build_merged_id_to_main_topic(df):
    """Build mapping merged_id → cleaned main topic name."""
    if not {"merged_cluster_id", "main_topic"}.issubset(df.columns):
        raise ValueError("Parquet file must contain 'merged_cluster_id' and 'main_topic'.")

    df = df.copy()
    df["main_topic_clean"] = df["main_topic"].str.replace(r"\*\*", "", regex=True)

    return dict(zip(df["merged_cluster_id"], df["main_topic_clean"]))


# -------------------------------------------------
# FINAL DATAFRAME BUILDER
# -------------------------------------------------

def build_final_df(
    df,
    subtopic_id_to_name,        # cluster_id → subtopic name
    subcluster_to_merged_id,    # subcluster triplet → merged_id
    merged_id_to_main_name      # merged_id → main topic name
):
    """
    df must contain at least:
        file, title_en, embeddings, cluster_id, name
    """

    df = df.copy()

    # ---------------------------
    # SUBTOPIC FIELDS
    # ---------------------------
    df["subtopic_id"] = df["cluster_id"].astype("Int64")
    df["subtopic_name"] = df["subtopic_id"].map(subtopic_id_to_name)

    # ---------------------------
    # MAIN TOPIC FIELDS
    # ---------------------------
    df["main_topic_id"] = df["name"].map(subcluster_to_merged_id)
    df["main_topic_id"] = df["main_topic_id"].astype("Int64")  # <-- FIXED (no .0 floats)

    df["main_topic_name"] = df["main_topic_id"].map(merged_id_to_main_name)

    # ---------------------------
    # Final column selection
    # ---------------------------
    final_cols = [
        "file",
        "title_en",
        "main_topic_name",
        "main_topic_id",
        "subtopic_name",
        "subtopic_id",
        "embeddings"
    ]

    return df[final_cols]


# -------------------------------------------------
# MAIN TOPIC NAME CLEANING
# -------------------------------------------------

def clean_main_topic_names(df):
    """Clean up markdown artifacts and trailing metadata."""
    df = df.copy()  # avoids SettingWithCopyWarning

    df["main_topic_name"] = (
        df["main_topic_name"]
        .astype(str)
        .str.split("\n").str[0]
        .str.replace(r"\*\(.*?\)\*", "", regex=True)
        .str.strip()
    )
    return df


# -------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------


df_sub = load_df(str(MERGE_METADATA))

df_main = load_df(str(MERGE_MAIN_TOPICS))

data_df = load_df(str(CLUSTERED_DATA))

# Build mappings
sub_id_to_sub_name = cluster_id_to_subtopic(df_sub)
triplets_to_main_id = build_subcluster_to_merged_id(df_main)
main_id_to_main_name = build_merged_id_to_main_topic(df_main)

# Build final dataframe
final_df = build_final_df(
    data_df,
    sub_id_to_sub_name,
    triplets_to_main_id,
    main_id_to_main_name
)

# Clean main topic names
final_df = clean_main_topic_names(final_df)

# Save final parquet
# TRAINING_DIR.mkdir(parents=True, exist_ok=True)
# save(final_df, str(LABELED_26K))
# print("Saved final_df as parquet")

# Display summary
print(final_df.shape)
print(final_df.drop(columns=["embeddings", "file", "title_en"]).head())
print((final_df["main_topic_name"] == "nan").sum())
