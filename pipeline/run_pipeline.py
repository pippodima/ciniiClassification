"""
run_pipeline.py — Full pipeline orchestrator: parse → clean → embed.

Usage:
    # New clean run — all outputs get '_clean' suffix, nothing overwritten:
    python pipeline/run_pipeline.py --run-name clean

    # Full re-parse from all 150k raw files:
    python pipeline/run_pipeline.py --run-name clean

    # Skip re-parsing (reuse existing rdf_parsed_clean.parquet):
    python pipeline/run_pipeline.py --run-name clean --skip-parse

    # Skip parse + clean (re-embed only):
    python pipeline/run_pipeline.py --run-name clean --skip-parse --skip-clean

    # Use MPS GPU for embedding:
    DEVICE=mps python pipeline/run_pipeline.py --run-name clean

    # Report-only on existing outputs for a run:
    python pipeline/run_pipeline.py --run-name clean --report-only

Path convention for --run-name NAME:
    data/processed/rdf_parsed_NAME.parquet
    data/cleaned/english_NAME.parquet
    data/cleaned/other_languages_NAME.parquet
    data/embedded/NAME/embedded.parquet
    data/embedded/NAME/checkpoints/

If --run-name is omitted the original default paths are used (may overwrite).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE))
from config import RDF_PARSED, ENGLISH, OTHER_LANGS, EMBEDDED_DIR, PROCESSED_DIR, CLEANED_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Path builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_paths(run_name: str | None) -> dict:
    """Return all stage input/output paths for a given run name."""
    if run_name:
        n = run_name
        return {
            "parsed_output":  str(PROCESSED_DIR / f"rdf_parsed_{n}.parquet"),
            "english_output": str(CLEANED_DIR   / f"english_{n}.parquet"),
            "other_output":   str(CLEANED_DIR   / f"other_languages_{n}.parquet"),
            "embed_dir":      str(EMBEDDED_DIR  / n),
            "embed_file":     str(EMBEDDED_DIR  / n / "embedded.parquet"),
        }
    else:
        return {
            "parsed_output":  str(RDF_PARSED),
            "english_output": str(ENGLISH),
            "other_output":   str(OTHER_LANGS),
            "embed_dir":      str(EMBEDDED_DIR),
            "embed_file":     str(EMBEDDED_DIR / "embedded.parquet"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(label: str, cmd: list[str], env: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"  ▶  {label}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, cwd=PIPELINE, env=env)
    if result.returncode != 0:
        print(f"\n❌  Stage '{label}' FAILED (exit {result.returncode})")
        sys.exit(result.returncode)


def _report_parsed(path: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  REPORT — Stage 01: Parse RDF")
    print(f"{'─'*60}")
    try:
        df = pd.read_parquet(path, columns=["title", "abstract", "error"])
    except Exception as e:
        print(f"  ❌ Cannot read {path}: {e}")
        return
    n = len(df)
    errors       = df["error"].notna().sum()
    has_title    = df["title"].notna().sum()
    has_abstract = df["abstract"].notna().sum()
    jats         = df["abstract"].str.contains("<jats:", na=False).sum()
    print(f"  Total rows            : {n:,}")
    print(f"  Parse errors          : {errors:,}  ({errors/n:.1%})")
    print(f"  Has title             : {has_title:,}  ({has_title/n:.1%})")
    print(f"  Has abstract          : {has_abstract:,}  ({has_abstract/n:.1%})")
    print(f"  JATS in abstract      : {jats}  ← should be 0")
    print(f"  Output                : {path}")


def _report_cleaned(path: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  REPORT — Stage 02: Process text")
    print(f"{'─'*60}")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  ❌ Cannot read {path}: {e}")
        return
    n = len(df)
    jats_abs   = df["abstract"].str.contains("<jats:", na=False).sum() if "abstract" in df.columns else "N/A"
    jats_clean = df["clean_abstract"].str.contains("<jats:", na=False).sum() if "clean_abstract" in df.columns else "N/A"
    lang_col   = "abstract_lang" if "abstract_lang" in df.columns else None
    langs      = df[lang_col].value_counts().head(5).to_dict() if lang_col else {}
    types      = df["type"].value_counts().to_dict() if "type" in df.columns else {}
    print(f"  Rows                  : {n:,}")
    print(f"  Columns               : {list(df.columns)}")
    print(f"  JATS in abstract      : {jats_abs}")
    print(f"  JATS in clean_abstract: {jats_clean}  ← should be 0")
    print(f"  Language distribution : {langs}")
    print(f"  Document types        : {types}")
    print(f"  Output                : {path}")


def _report_embedded(path: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  REPORT — Stage 03: Embeddings")
    print(f"{'─'*60}")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  ❌ Cannot read {path}: {e}")
        return
    n = len(df)
    print(f"  Rows                  : {n:,}")
    print(f"  Columns               : {list(df.columns)}")
    if "embeddings" in df.columns:
        sample = np.vstack(df["embeddings"].values[:min(500, n)])
        norms  = np.linalg.norm(sample, axis=1)
        print(f"  Embedding dim         : {sample.shape[1]}")
        print(f"  L2 norms (sample)     : mean={norms.mean():.4f}  std={norms.std():.4f}  min={norms.min():.4f}")
    if "full_text" in df.columns:
        jats_ft = df["full_text"].str.contains("<jats:", na=False).sum()
        print(f"  JATS in full_text     : {jats_ft}  ← should be 0")
    if "clean_abstract" in df.columns:
        jats_ca = df["clean_abstract"].str.contains("<jats:", na=False).sum()
        print(f"  JATS in clean_abstract: {jats_ca}  ← should be 0")
    print(f"  Output                : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Full parse→clean→embed pipeline")
    parser.add_argument("--run-name",    type=str,   default=None,
                        help="Tag for this run. All output paths get this suffix, "
                             "so existing files are never overwritten. E.g. 'clean'")
    parser.add_argument("--max-docs",    type=int,   default=None,
                        help="Cap on raw documents to parse (default: all)")
    parser.add_argument("--batch-size",  type=int,   default=50_000,
                        help="Parse batch size (default: 50000)")
    parser.add_argument("--skip-parse",  action="store_true",
                        help="Skip stage 01 (use existing parsed parquet)")
    parser.add_argument("--skip-clean",  action="store_true",
                        help="Skip stage 02 (use existing english parquet)")
    parser.add_argument("--skip-embed",  action="store_true",
                        help="Skip stage 03 (use existing embedded parquet)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only print reports on existing outputs; no processing")
    args = parser.parse_args()

    paths = _build_paths(args.run_name)

    # Print resolved paths so the user can verify before anything runs
    print(f"\n{'='*60}")
    print(f"  Pipeline run: {args.run_name or '(default paths)'}")
    print(f"{'='*60}")
    print(f"  [01] parsed  → {paths['parsed_output']}")
    print(f"  [02] english → {paths['english_output']}")
    print(f"       other   → {paths['other_output']}")
    print(f"  [03] embed   → {paths['embed_file']}")
    print(f"{'='*60}\n")

    # Build subprocess env with path overrides
    env = {
        **os.environ,
        "PARSED_OUTPUT":  paths["parsed_output"],
        "PARSED_INPUT":   paths["parsed_output"],
        "ENGLISH_OUTPUT": paths["english_output"],
        "OTHER_OUTPUT":   paths["other_output"],
        "INPUT_PATH":     paths["english_output"],
        "OUTPUT_DIR":     paths["embed_dir"],
    }

    # ── Stage 01: Parse RDF ──────────────────────────────────────────────────
    if not args.skip_parse and not args.report_only:
        cmd = [sys.executable, str(PIPELINE / "01_parse_rdf.py"),
               "--batch-size", str(args.batch_size)]
        if args.max_docs:
            cmd += ["--max-document", str(args.max_docs)]
        _run("01 — Parse RDF", cmd, env)

    _report_parsed(paths["parsed_output"])

    # ── Stage 02: Process text ───────────────────────────────────────────────
    if not args.skip_clean and not args.report_only:
        _run("02 — Process text", [sys.executable, str(PIPELINE / "02_process_text.py")], env)

    _report_cleaned(paths["english_output"])

    # ── Stage 03: Generate embeddings ────────────────────────────────────────
    if not args.skip_embed and not args.report_only:
        _run("03 — Generate embeddings",
             [sys.executable, str(PIPELINE / "run_embedding_clustering.py")], env)

    _report_embedded(paths["embed_file"])

    print(f"\n{'='*60}")
    print(f"  ✅  Pipeline complete.  run-name={args.run_name or 'default'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
