"""
pipeline/run_server.py
======================
Single-command crashproof pipeline: preflight → parse → clean → embed → classify.

Each completed stage is auto-detected; re-running after a crash resumes from where
it stopped without re-doing work.  The embedding step is checkpointed at the batch
level — safe against OOM and disconnections.

Usage:
    RAW_DIR=/mnt/data/cinii DEVICE=cuda BATCH_SIZE=64 SAVE_EVERY_N_BATCHES=50 \\
    python pipeline/run_server.py --run-name server

    # Resume after crash (completed stages are skipped automatically):
    python pipeline/run_server.py --run-name server

    # Force re-run one specific stage:
    python pipeline/run_server.py --run-name server --force-stage embed

    # Dry-run: preflight + path check, no processing:
    python pipeline/run_server.py --run-name server --dry-run

    # Show reports on existing outputs (no processing):
    python pipeline/run_server.py --run-name server --report-only

Output paths (all tagged with --run-name so nothing is overwritten):
    data/processed/rdf_parsed_<name>.parquet
    data/cleaned/english_<name>.parquet
    data/embedded/<name>/embedded.parquet
    classified/classified_<name>.parquet
    logs/run_<name>_<timestamp>.log
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parent
sys.path.insert(0, str(PIPELINE))

from config import (
    RAW_DIR as _CFG_RAW_DIR,
    PROCESSED_DIR, CLEANED_DIR, EMBEDDED_DIR, MODELS_DIR,
)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────────────────
# Tee: orchestrator print() → terminal + log file simultaneously
# (subprocess output goes to terminal only; the log captures stage reports)
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f    = open(path, "a", encoding="utf-8", buffering=1)
        self._real = sys.__stdout__

    def write(self, data: str):
        self._real.write(data)
        self._f.write(data)

    def flush(self):
        self._real.flush()
        self._f.flush()

    def isatty(self) -> bool:
        return False

    def close(self):
        self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# Path builder
# ─────────────────────────────────────────────────────────────────────────────

def _paths(run_name: str) -> dict[str, Path]:
    n = run_name
    return {
        "parse_batches": PROCESSED_DIR / f"{n}_batches",
        "parsed":        PROCESSED_DIR / f"rdf_parsed_{n}.parquet",
        "english":       CLEANED_DIR   / f"english_{n}.parquet",
        "other":         CLEANED_DIR   / f"other_languages_{n}.parquet",
        "embed_dir":     EMBEDDED_DIR  / n,
        "embed":         EMBEDDED_DIR  / n / "embedded.parquet",
        "classified":    ROOT / "classified" / f"classified_{n}.parquet",
        "model_dir":     MODELS_DIR / "release",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks — fail fast with actionable messages
# ─────────────────────────────────────────────────────────────────────────────

def preflight(p: dict, raw_dir: Path, device: str, do_classify: bool,
              skip_parse: bool = False) -> None:
    print(f"\n{'='*62}")
    print("  PRE-FLIGHT CHECKS")
    print(f"{'='*62}")

    errors: list[str] = []

    def chk(label: str, ok: bool, note: str = "") -> None:
        print(f"  {'✓' if ok else '✗'}  {label:<44} {note}")

    # 1. RAW_DIR — only required when the parse stage will actually run
    if skip_parse:
        parsed_exists = p["parsed"].exists()
        chk("RAW_DIR check skipped (parse already done)",
            parsed_exists,
            str(p["parsed"]) if parsed_exists else "parsed parquet missing!")
        if not parsed_exists:
            errors.append(
                f"Parsed parquet not found: {p['parsed']}\n"
                "  Re-run without --force-stage clean (parse stage must run first)"
            )
    else:
        has_rdf = (raw_dir.exists() and
                   next(raw_dir.glob("**/*.rdf"), None) is not None)
        n_subdirs = len([d for d in raw_dir.iterdir() if d.is_dir()]) if raw_dir.exists() else 0
        chk("RAW_DIR exists + .rdf files", raw_dir.exists() and has_rdf,
            f"{n_subdirs:,} subdirs" if has_rdf else "not found or empty")
        if not raw_dir.exists():
            errors.append(
                f"RAW_DIR not found: {raw_dir}\n"
                "  Fix: export RAW_DIR=/path/to/your/rdf/files"
            )
        elif not has_rdf:
            errors.append(f"No .rdf files found under {raw_dir}")

    # 2. Required packages
    required_pkgs = [
        ("torch",                 "torch"),
        ("sentence_transformers", "sentence_transformers"),
        ("pandas",                "pandas"),
        ("pyarrow",               "pyarrow"),
        ("numpy",                 "numpy"),
        ("langdetect",            "langdetect"),
        ("scikit-learn",          "sklearn"),
        ("tqdm",                  "tqdm"),
    ]
    missing_pkgs: list[str] = []
    for display, imp in required_pkgs:
        try:
            __import__(imp)
            chk(f"package: {display}", True, "ok")
        except ImportError:
            chk(f"package: {display}", False, "MISSING")
            missing_pkgs.append(display)
    if missing_pkgs:
        errors.append(
            f"Missing packages: {missing_pkgs}\n"
            "  Fix: pip install -r pipeline/requirements-server.txt"
        )

    # 3. GPU / device
    if device == "cuda":
        try:
            import torch
            ok = torch.cuda.is_available()
            name = torch.cuda.get_device_name(0) if ok else "not available"
            chk("CUDA available", ok, name)
            if not ok:
                errors.append(
                    "DEVICE=cuda but no CUDA GPU detected.\n"
                    "  Fix: check nvidia-smi, or unset DEVICE to fall back to CPU."
                )
        except ImportError:
            pass
    elif device == "mps":
        try:
            import torch
            ok = torch.backends.mps.is_available()
            chk("MPS (Apple Silicon)", ok)
        except ImportError:
            pass
    else:
        chk(f"Device: {device}", True, "no GPU check needed")

    # 4. Model artefacts (only if classify will run)
    if do_classify:
        md = p["model_dir"]
        needed = ["model_b_twohead.pt", "label_encoders.pkl", "hierarchy.pkl"]
        present = [f for f in needed if (md / f).exists()]
        chk("Model artefacts (for classify)", len(present) == len(needed),
            f"{len(present)}/{len(needed)} files in models/release/")
        if len(present) < len(needed):
            missing_m = [f for f in needed if not (md / f).exists()]
            errors.append(
                f"Missing model files: {missing_m}\n"
                f"  Expected in: {md}"
            )

    # 5. Disk space
    try:
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        chk("Disk free ≥ 100 GB (recommended)", free_gb >= 100, f"{free_gb:.0f} GB free")
        if free_gb < 30:
            errors.append(
                f"Only {free_gb:.0f} GB free — not enough for a 3M-doc run.\n"
                "  Free up space before proceeding."
            )
        elif free_gb < 100:
            print(f"  ⚠️  {free_gb:.0f} GB free — monitor disk usage during the run.")
    except Exception:
        pass

    print(f"{'='*62}")

    if errors:
        print("\n  ❌  Pre-flight FAILED — fix before running:\n")
        for err in errors:
            print(f"  • {err}\n")
        sys.exit(1)

    print("  ✅  All checks passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Stage runner — auto-skip if output exists; sys.exit on failure
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(label: str, cmd: list, env: dict, out: Path, force: bool) -> None:
    print(f"\n{'─'*62}")
    print(f"  STAGE [{label.upper()}]")
    print(f"{'─'*62}")

    if out.exists() and not force:
        mb = out.stat().st_size / 1e6
        print(f"  ⏩  Skipping — output already exists ({mb:.0f} MB)")
        print(f"     {out}")
        print(f"  To re-run this stage: add --force-stage {label}")
        return

    if force and out.exists():
        print(f"  ⚡  Force mode — re-running even though output exists.")

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = (time.time() - t0) / 60

    if result.returncode != 0:
        print(f"\n  ❌  Stage [{label.upper()}] FAILED (exit {result.returncode}) "
              f"after {elapsed:.1f} min")
        print("  The pipeline is resumable — fix the issue and re-run the same command.")
        print("  Completed stages will be auto-skipped.")
        sys.exit(result.returncode)

    print(f"  ✅  Stage [{label.upper()}] done in {elapsed:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Per-stage validation reports
# ─────────────────────────────────────────────────────────────────────────────

def _header(label: str, path: Path) -> None:
    print(f"\n  {'─'*55}")
    print(f"  REPORT [{label}]  →  {path.name}")
    print(f"  {'─'*55}")


def report_parsed(path: Path) -> int:
    _header("01 PARSE", path)
    if not path.exists():
        print("  ❌  Output not found!")
        return 0
    df = pd.read_parquet(path)
    n        = len(df)
    errors   = int(df["error"].notna().sum()) if "error" in df.columns else "?"
    has_abs  = int(df["abstract"].notna().sum()) if "abstract" in df.columns else n
    lang_cov = int(df["language"].notna().sum()) if "language" in df.columns else "?"
    jats     = int(df["abstract"].str.contains("<jats:", na=False).sum()) if "abstract" in df.columns else "?"
    print(f"  Total rows      : {n:,}")
    print(f"  Parse errors    : {errors}")
    print(f"  Has abstract    : {has_abs:,}  ({has_abs/max(n,1):.1%})")
    print(f"  dc:language     : {lang_cov} rows with publisher-declared language")
    print(f"  JATS in abstract: {jats}  {'← ✅' if jats == 0 else '← ⚠️  should be 0!'}")
    return n


def report_cleaned(path: Path) -> int:
    _header("02 CLEAN", path)
    if not path.exists():
        print("  ❌  Output not found!")
        return 0
    df    = pd.read_parquet(path)
    n     = len(df)
    jats  = (int(df["clean_abstract"].str.contains("<jats:", na=False).sum())
             if "clean_abstract" in df.columns else "?")
    langs = (df["abstract_lang"].value_counts().head(5).to_dict()
             if "abstract_lang" in df.columns else {})
    print(f"  Rows (EN sci.)  : {n:,}")
    print(f"  JATS in clean   : {jats}  {'← ✅' if jats == 0 else '← ⚠️  should be 0!'}")
    print(f"  Lang dist (top5): {langs}")
    print(f"  Columns         : {list(df.columns)}")
    return n


def report_embedded(path: Path) -> int:
    _header("03 EMBED", path)
    if not path.exists():
        print("  ❌  Output not found!")
        return 0
    n = pq.read_metadata(path).num_rows
    print(f"  Rows embedded   : {n:,}")
    try:
        sample = pd.read_parquet(path, columns=["embeddings"]).head(500)
        embs   = np.vstack(sample["embeddings"].values)
        norms  = np.linalg.norm(embs, axis=1)
        norm_ok = norms.min() > 0.01
        print(f"  Embedding dim   : {embs.shape[1]}")
        print(f"  L2 norms (500)  : mean={norms.mean():.4f}  min={norms.min():.4f}"
              f"  {'← ✅' if norm_ok else '← ⚠️  near-zero norms!'}")
    except Exception as e:
        print(f"  (Norm check skipped: {e})")
    try:
        jats_ft = int(
            pd.read_parquet(path, columns=["full_text"]).head(1000)
            ["full_text"].str.contains("<jats:", na=False).sum()
        )
        print(f"  JATS in text    : {jats_ft}  "
              f"{'← ✅' if jats_ft == 0 else '← ⚠️  contamination!'}")
    except Exception:
        pass
    return n


def report_classified(path: Path) -> int:
    _header("04 CLASSIFY", path)
    if not path.exists():
        print("  ❌  Output not found!")
        return 0
    df = pd.read_parquet(path, columns=["pred_lcc_main", "pred_lcc", "conf_div"])
    n  = len(df)
    print(f"  Classified      : {n:,}")
    print(f"  Top LCC classes :")
    for cls, cnt in df["pred_lcc_main"].value_counts().head(8).items():
        print(f"    {cls:<4}  {cnt:>10,}  ({cnt/n:.1%})")
    m, med = df["conf_div"].mean(), df["conf_div"].median()
    low30  = (df["conf_div"] < 0.30).mean()
    low50  = (df["conf_div"] < 0.50).mean()
    print(f"  conf_div        : mean={m:.3f}  median={med:.3f}")
    print(f"  conf < 0.30     : {low30:.1%}  "
          f"{'← ⚠️  high — consider re-training on server data' if low30 > 0.25 else '← ✅ ok'}")
    print(f"  conf < 0.50     : {low50:.1%}")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crashproof server pipeline: parse → clean → embed → classify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-name",      required=True,
                        help="Tag for this run. All output paths are suffixed with it.")
    parser.add_argument("--force-stage",   choices=["parse", "clean", "embed", "classify"],
                        default=None,
                        help="Force re-run of one stage even if output already exists.")
    parser.add_argument("--skip-embed",    action="store_true",
                        help="Skip the embed stage (run parse+clean only). "
                             "Use when you will embed separately with run_reembed.py.")
    parser.add_argument("--skip-classify", action="store_true",
                        help="Skip the classify step (run parse→clean→embed only).")
    parser.add_argument("--skip-report",  action="store_true",
                        help="Skip the post-classification quality report.")
    parser.add_argument("--max-docs",      type=int, default=None,
                        help="Cap raw docs to parse (for testing; default: all).")
    parser.add_argument("--batch-size",    type=int,
                        default=int(os.getenv("BATCH_SIZE", "64")),
                        help="Embedding batch size (default: $BATCH_SIZE or 64).")
    parser.add_argument("--parse-workers", type=int,
                        default=int(os.getenv("PARSE_WORKERS", "1")),
                        help="Parallel workers for RDF parsing (default: 1). "
                             "Set to CPU core count for large datasets, e.g. --parse-workers 16.")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Preflight + path check only — no processing.")
    parser.add_argument("--report-only",   action="store_true",
                        help="Print reports on existing outputs — no processing.")
    args = parser.parse_args()

    p       = _paths(args.run_name)
    raw_dir = Path(os.getenv("RAW_DIR", str(_CFG_RAW_DIR)))
    device  = os.getenv("DEVICE", "cpu")
    save_n  = int(os.getenv("SAVE_EVERY_N_BATCHES", "50"))
    do_embed  = not args.skip_embed
    do_cls    = not args.skip_classify and do_embed
    do_report = not args.skip_report

    # ── Log file: tees all orchestrator print() calls ─────────────────────────
    log_path = (ROOT / "logs" /
                f"run_{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    tee = _Tee(log_path)
    sys.stdout = tee

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  CiNii Pipeline   run={args.run_name}   device={device}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}")
    print(f"  RAW_DIR       : {raw_dir}")
    print(f"  BATCH_SIZE    : {args.batch_size}  |  SAVE_EVERY_N : {save_n}")
    print(f"  [01] parsed   : {p['parsed']}")
    print(f"  [02] english  : {p['english']}")
    print(f"  [03] embed    : {p['embed']}")
    if do_cls:
        print(f"  [04] classify : {p['classified']}")
        print(f"  model_dir     : {p['model_dir']}")
    if do_cls and do_report:
        report_out = ROOT / "reports" / f"report_{args.run_name}.txt"
        print(f"  [05] report   : {report_out}")
    print(f"  log           : {log_path}")
    print(f"{'='*62}")

    # ── Pre-flight ────────────────────────────────────────────────────────────
    # Parse is effectively skipped when its output already exists and not force-rerun
    parse_will_run = (args.force_stage == "parse") or not p["parsed"].exists()
    preflight(p, raw_dir, device, do_cls, skip_parse=not parse_will_run)

    if args.dry_run:
        print("  Dry-run — stopping here (no data was processed).")
        tee.close()
        return

    # ── Subprocess environment ─────────────────────────────────────────────────
    env = {
        **os.environ,
        "RUN_NAME":             args.run_name,
        "RAW_DIR":              str(raw_dir),
        "PARSED_OUTPUT":        str(p["parsed"]),
        "PARSED_INPUT":         str(p["parsed"]),
        "ENGLISH_OUTPUT":       str(p["english"]),
        "OTHER_OUTPUT":         str(p["other"]),
        "INPUT_PATH":           str(p["english"]),
        "OUTPUT_DIR":           str(p["embed_dir"]),
        "DEVICE":               device,
        "BATCH_SIZE":           str(args.batch_size),
        "SAVE_EVERY_N_BATCHES": str(save_n),
    }

    # ── Stage 01: Parse RDF ────────────────────────────────────────────────────
    if not args.report_only:
        cmd = [sys.executable, str(PIPELINE / "01_parse_rdf.py"),
               "-o", str(p["parse_batches"]),
               "--workers", str(args.parse_workers)]
        if args.max_docs:
            cmd += ["--max-document", str(args.max_docs)]
        run_stage("parse", cmd, env, p["parsed"], force=(args.force_stage == "parse"))
    n1 = report_parsed(p["parsed"])

    # ── Stage 02: Process text ─────────────────────────────────────────────────
    if not args.report_only:
        run_stage("clean",
                  [sys.executable, str(PIPELINE / "02_process_text.py")],
                  env, p["english"], force=(args.force_stage == "clean"))
    n2 = report_cleaned(p["english"])

    # ── Stage 03: Generate embeddings ──────────────────────────────────────────
    if do_embed and not args.report_only:
        run_stage("embed",
                  [sys.executable, str(PIPELINE / "run_embedding_clustering.py")],
                  env, p["embed"], force=(args.force_stage == "embed"))
    n3 = report_embedded(p["embed"]) if p["embed"].exists() else 0

    # ── Stage 04: Classify ─────────────────────────────────────────────────────
    n4 = 0
    if do_cls:
        if not args.report_only:
            p["classified"].parent.mkdir(parents=True, exist_ok=True)
            run_stage("classify",
                      [sys.executable, str(PIPELINE / "11_classify.py"),
                       "--input",     str(p["embed"]),
                       "--output",    str(p["classified"]),
                       "--model",     "b",
                       "--model-dir", str(p["model_dir"]),
                       "--device",    device],
                      env, p["classified"], force=(args.force_stage == "classify"))
        n4 = report_classified(p["classified"])

    # ── Stage 05: Quality report ───────────────────────────────────────────────
    if do_cls and do_report and p["classified"].exists():
        report_out = ROOT / "reports" / f"report_{args.run_name}.txt"
        print(f"\n{'─'*62}")
        print(f"  STAGE [REPORT]")
        print(f"{'─'*62}")
        from config import LCC_MAPPING_V2
        report_cmd = [sys.executable, str(PIPELINE / "12_report_classification.py"),
                      "--classified", str(p["classified"]),
                      "--output",     str(report_out)]
        if LCC_MAPPING_V2.exists():
            report_cmd += ["--lcc-mapping", str(LCC_MAPPING_V2)]
        run_stage("report", report_cmd, env, report_out, force=True)   # always re-generate

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  PIPELINE COMPLETE")
    print(f"  run={args.run_name}   finished {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}")
    print(f"  Stage          Rows")
    print(f"  {'─'*38}")
    print(f"  01 parse     : {n1:>12,}")
    print(f"  02 clean     : {n2:>12,}")
    print(f"  03 embed     : {n3:>12,}")
    if do_cls:
        print(f"  04 classify  : {n4:>12,}")
    print(f"{'='*62}")
    print(f"  Log: {log_path}\n")

    tee.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
