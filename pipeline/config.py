"""
Central path configuration for the CiNii classification pipeline.
All paths are absolute, derived from the project root.

Override any path via environment variable before running:
    RAW_DIR=/mnt/storage/cinii/raw  python pipeline/run_pipeline.py --run-name server
"""
import os
from pathlib import Path

# ── Project root (parent of this pipeline/ directory) ────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data: raw ────────────────────────────────────────────────────────────────
# Override with RAW_DIR env var when the raw data lives outside the project tree
# (common on servers where data is on a separate mount/disk).
RAW_DIR = Path(os.getenv("RAW_DIR", str(ROOT / "data" / "raw")))

# ── Data: pipeline stages ────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
CLEANED_DIR   = ROOT / "data" / "cleaned"
EMBEDDED_DIR  = ROOT / "data" / "embedded"

RDF_PARSED  = PROCESSED_DIR / "rdf_parsed.parquet"
ENGLISH     = CLEANED_DIR   / "english.parquet"
OTHER_LANGS = CLEANED_DIR   / "other_languages.parquet"
EMBEDDED_FILE = EMBEDDED_DIR / "embedded.parquet"
EMBEDDED_26K  = EMBEDDED_FILE   # legacy alias

# ── Clustering output ────────────────────────────────────────────────────────
CLUSTER_DIR       = ROOT / "clustering_output"
CLUSTERED_DATA    = CLUSTER_DIR / "clustered_data.parquet"
CLUSTER_METADATA  = CLUSTER_DIR / "cluster_metadata.parquet"
PIPELINE_METADATA = CLUSTER_DIR / "pipeline_metadata.json"

VIZ_DIR   = CLUSTER_DIR / "visualizations"
UMAP_PNG  = VIZ_DIR / "umap_clusters.png"
UMAP_HTML = VIZ_DIR / "umap_clusters.html"

BERT_DIR          = CLUSTER_DIR / "bertopic"
BERT_TOPICS_JSON  = BERT_DIR / "topic_hierarchy_named.json"
BERT_DOCS_PARQUET = BERT_DIR / "documents_with_hierarchical_topics.parquet"
BERT_DOCMAP_HTML  = VIZ_DIR / "document_map_main_topics.html"

MERGE_DIR         = CLUSTER_DIR / "merge"
MERGE_METADATA    = MERGE_DIR / "metadata_clustered_names.parquet"
MERGE_MAIN_TOPICS = MERGE_DIR / "merged_clusters_with_main_topics.parquet"
MERGE_TOPICS_JSON = MERGE_DIR / "cluster_output_merged_topics.json"
MERGE_PREFIX      = str(MERGE_DIR / "cluster_output")

# ── Training data ─────────────────────────────────────────────────────────────
TRAINING_DIR    = ROOT / "training_data"
LABELED_26K     = TRAINING_DIR / "labeled_26k.parquet"
LCC_MAPPING_V2  = CLUSTER_DIR / "v2" / "lcc_mapping_full.parquet"

# ── Training runs (new pipeline) ──────────────────────────────────────────────
RUNS_DIR        = ROOT / "training_runs"

# ── Mappings ──────────────────────────────────────────────────────────────────
MAPPINGS_DIR = ROOT / "mappings"
LCC_MAP      = MAPPINGS_DIR / "lcc_map.json"

# ── Models ────────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
