"""
pipeline/11_classify.py
========================
Run trained LCC classifier on any parquet that has an `embeddings` column.
Works on the full 3M-document server dataset — reads and writes in chunks
so the embedding matrix is never fully loaded into RAM.

Usage:
    # Model B (two-head, recommended) on server embeddings
    python pipeline/11_classify.py \
        --input  data/embedded/server/embedded_26k.parquet \
        --output data/classified/classified_server.parquet

    # Model A (flat, faster)
    python pipeline/11_classify.py --model a \
        --input  data/embedded/server/embedded_26k.parquet \
        --output data/classified/classified_server_modela.parquet

    # Adjust chunk size if RAM is tight (default 5000 rows per chunk)
    python pipeline/11_classify.py --chunk-size 2000 ...

Output columns added to the existing parquet:
    pred_lcc_div   – LCC numeric division, e.g. "TJ266"
    pred_lcc       – LCC subclass,          e.g. "TJ"
    pred_lcc_main  – LCC main class,        e.g. "T"
    conf_div       – softmax confidence for pred_lcc_div  (0–1)
    model_used     – "a" or "b"
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR as MODEL_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions (must match 10_train_classifier.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class Backbone(nn.Module):
    def __init__(self, in_dim=1024, hidden=[512, 256], dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class ModelA(nn.Module):
    def __init__(self, n_div, hidden=[512, 256], dropout=0.3):
        super().__init__()
        self.backbone = Backbone(hidden=hidden, dropout=dropout)
        self.head     = nn.Linear(self.backbone.out_dim, n_div)

    def forward(self, x):
        return self.head(self.backbone(x))


class ModelB(nn.Module):
    def __init__(self, n_div, n_sub, hidden=[512, 256], dropout=0.3):
        super().__init__()
        self.backbone = Backbone(hidden=hidden, dropout=dropout)
        d = self.backbone.out_dim
        self.head_sub = nn.Linear(d, n_sub)
        self.head_div = nn.Linear(d, n_div)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_sub(feat), self.head_div(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────────────────────────────────────

def load_artefacts(model_dir: Path, model_choice: str, device: str):
    with open(model_dir / "label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(model_dir / "hierarchy.pkl", "rb") as f:
        hier = pickle.load(f)

    le_div: "LabelEncoder" = encoders["le_div"]
    le_sub: "LabelEncoder" = encoders["le_sub"]
    n_div = len(le_div.classes_)
    n_sub = len(le_sub.classes_)

    # hier_mask: bool tensor (n_sub, n_div) — True where division belongs to subclass
    hier_mask: torch.Tensor = hier["hier_mask"].to(device)   # already bool

    if model_choice == "a":
        model = ModelA(n_div=n_div)
        state = torch.load(model_dir / "model_a_flat.pt", map_location=device,
                           weights_only=True)
    else:
        model = ModelB(n_div=n_div, n_sub=n_sub)
        state = torch.load(model_dir / "model_b_twohead.pt", map_location=device,
                           weights_only=True)

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, le_div, le_sub, hier_mask


# ─────────────────────────────────────────────────────────────────────────────
# Inference on a numpy batch
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(model, embs_np: np.ndarray, hier_mask: torch.Tensor,
                  device: str, model_choice: str):
    """
    Returns:
        div_indices  – int array (N,)
        sub_indices  – int array (N,)  — for model A, derived from div via hier_mask
        conf_div     – float array (N,)  softmax probability of top division
    """
    x = torch.tensor(embs_np, dtype=torch.float32, device=device)

    if model_choice == "a":
        lg_div = model(x)                            # (N, n_div)
        # subclass = whichever subclass contains the winning division
        div_idx = lg_div.argmax(1)
        # for each sample, find which subclass owns the predicted division
        sub_idx = torch.tensor(
            [hier_mask[:, d].nonzero(as_tuple=True)[0][0].item()
             if hier_mask[:, d].any() else 0
             for d in div_idx.cpu().tolist()],
            device=device
        )
        probs_div = F.softmax(lg_div, dim=1)
        conf = probs_div[torch.arange(x.shape[0]), div_idx]

    else:  # model B
        lg_sub, lg_div = model(x)                    # (N, n_sub), (N, n_div)
        pred_sub = lg_sub.argmax(1)

        # Hierarchical masking: zero out divisions not in predicted subclass
        mask = hier_mask[pred_sub]                   # (N, n_div)
        lg_div_masked = lg_div.masked_fill(~mask, float("-inf"))
        div_idx = lg_div_masked.argmax(1)
        sub_idx = pred_sub

        probs_div = F.softmax(lg_div_masked, dim=1)
        conf = probs_div[torch.arange(x.shape[0]), div_idx]

    return (
        div_idx.cpu().numpy(),
        sub_idx.cpu().numpy(),
        conf.cpu().numpy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LCC classifier inference")
    parser.add_argument("--input",      required=True,
                        help="Parquet file with an `embeddings` column")
    parser.add_argument("--output",     required=True,
                        help="Output parquet path (predictions appended)")
    parser.add_argument("--model",      default="b", choices=["a", "b"],
                        help="Model to use: 'a' (flat) or 'b' (two-head). Default: b")
    parser.add_argument("--model-dir",  default=str(MODEL_DIR / "release"),
                        help="Directory containing model artefacts")
    parser.add_argument("--device",     default=os.getenv("DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Inference batch size (default: 512)")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Rows read from parquet per chunk (default: 5000)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Model        : {args.model.upper()}")
    print(f"  Device       : {args.device}")
    print(f"  Model dir    : {model_dir}")
    print(f"  Input        : {args.input}")
    print(f"  Output       : {out_path}")

    # ── Load model + artefacts ───────────────────────────────────────────────
    print("\nLoading model artefacts...")
    model, le_div, le_sub, hier_mask = load_artefacts(
        model_dir, args.model, args.device
    )
    print(f"  LCC divisions : {len(le_div.classes_)}")
    print(f"  LCC subclasses: {len(le_sub.classes_)}")

    # ── Stream input parquet row-group by row-group ──────────────────────────
    pf = pq.ParquetFile(args.input)
    total_rows = pf.metadata.num_rows
    print(f"\nClassifying {total_rows:,} rows in chunks of {args.chunk_size}...")

    writer       = None
    file_schema  = None
    n_classified = 0

    for batch in tqdm(pf.iter_batches(batch_size=args.chunk_size),
                      total=-(total_rows // -args.chunk_size),
                      desc="chunks"):
        chunk = batch.to_pandas()

        # Stack embeddings → (chunk_size, 1024) float32
        embs = np.vstack(chunk["embeddings"].values).astype(np.float32)

        # Run inference in sub-batches to control GPU memory
        all_div, all_sub, all_conf = [], [], []
        for i in range(0, len(embs), args.batch_size):
            b_embs = embs[i : i + args.batch_size]
            d_idx, s_idx, conf = predict_batch(
                model, b_embs, hier_mask, args.device, args.model
            )
            all_div.append(d_idx)
            all_sub.append(s_idx)
            all_conf.append(conf)

        div_idx = np.concatenate(all_div)
        sub_idx = np.concatenate(all_sub)
        conf    = np.concatenate(all_conf)

        # Decode integer indices → string labels
        chunk["pred_lcc_div"]  = le_div.inverse_transform(div_idx)
        chunk["pred_lcc"]      = le_sub.inverse_transform(sub_idx)
        chunk["pred_lcc_main"] = chunk["pred_lcc"].str[0]   # first letter: T, Q, R …
        chunk["conf_div"]      = conf.round(4)
        chunk["model_used"]    = args.model

        # Drop the bulky embeddings column from output (saves disk space)
        chunk = chunk.drop(columns=["embeddings"], errors="ignore")

        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            # Promote null-typed fields → string so all chunks share one stable schema.
            # Columns like language/doi/error can be all-null in some chunks, causing
            # Arrow to infer type null which conflicts with string in other chunks.
            fields = [
                pa.field(f.name, pa.string() if f.type == pa.null() else f.type)
                for f in table.schema
            ]
            file_schema = pa.schema(fields)
            writer = pq.ParquetWriter(str(out_path), file_schema, compression="snappy")

        writer.write_table(table.cast(file_schema, safe=False))
        n_classified += len(chunk)

    if writer:
        writer.close()

    # ── Summary report ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  CLASSIFICATION REPORT")
    print(f"{'='*55}")
    print(f"  Classified     : {n_classified:,} documents")
    print(f"  Output         : {out_path}")

    result = pd.read_parquet(out_path, columns=["pred_lcc_main", "pred_lcc", "pred_lcc_div", "conf_div"])
    print(f"\n  Top LCC main classes:")
    for cls, cnt in result["pred_lcc_main"].value_counts().head(10).items():
        print(f"    {cls:4s}  {cnt:>8,}  ({cnt/n_classified:.1%})")
    print(f"\n  Top LCC subclasses:")
    for cls, cnt in result["pred_lcc"].value_counts().head(10).items():
        print(f"    {cls:6s}  {cnt:>8,}  ({cnt/n_classified:.1%})")
    print(f"\n  Confidence (div):")
    print(f"    mean  = {result['conf_div'].mean():.3f}")
    print(f"    median= {result['conf_div'].median():.3f}")
    print(f"    <0.50 = {(result['conf_div'] < 0.50).sum():,}  ({(result['conf_div'] < 0.50).mean():.1%})")
    print(f"    <0.30 = {(result['conf_div'] < 0.30).sum():,}  ({(result['conf_div'] < 0.30).mean():.1%})")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
