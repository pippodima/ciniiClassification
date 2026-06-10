"""
pipeline/13_eval_models.py
===========================
Honest evaluation + diagnostics for the two MLP classifiers (Model A / Model B).

WHY THIS EXISTS
---------------
The headline val-accuracy (~0.96 on 163 LCC divisions) is suspiciously high for a
163-class problem. This script stress-tests *whether that number means what it
appears to mean*. The labels are not external ground truth — they are produced by
clustering the embeddings (UMAP→HDBSCAN) and then predicted back from the same
embeddings. That makes the task separable almost by construction, and the standard
metrics can be wildly optimistic about real-world classification quality.

This tool quantifies the gap. It produces:

  PLOTS  (→ <out-dir>/*.png)
    1. training_history.png   loss / acc / balanced-acc per epoch, Model A & B
    2. perclass_support.png   per-division support distribution (long tail)
    3. perclass_f1.png        per-division F1 vs support (which classes are weak)
    4. leakage_nn.png         val→train nearest-neighbour cosine-similarity hist
    5. confidence_split.png   model confidence on IN-cluster val vs HELD-OUT outliers
    6. reliability.png        calibration curve on the val set

  DIAGNOSTICS  (→ stdout + <out-dir>/diagnostics.txt)
    A. Baseline comparison: nearest-centroid & 1-NN accuracy on the SAME val split.
       If these match the MLP, the MLP adds ~nothing — the task is trivially
       separable because the labels came from the embedding geometry.
    B. Train/val near-duplicate leakage: fraction of val rows with a train row at
       cosine-sim > {0.999, 0.99, 0.95}. High = inflated accuracy.
    C. Coverage: how many docs are EXCLUDED (outliers / skipped clusters) and thus
       never measured. On the full corpus most docs look like these.
    D. Outlier behaviour: confidence distribution of Model B on the held-out
       outlier points (no ground truth, but over-confidence here is a red flag).

Usage (run on the server where artefacts + parquet live):
    python pipeline/13_eval_models.py \
        --run-dir   training_runs/v3_300k \
        --model-dir models/v3_300k \
        --out-dir   reports/eval_v3_300k

The split is reconstructed with SEED=42 / test_size=0.20 / stratify=y_div, i.e.
IDENTICAL to run_training.py — so the val set here is the very same val set the
reported metrics were computed on.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Must match run_training.py exactly
SEED      = 42
TEST_SIZE = 0.20
HIDDEN    = [512, 256]
DROPOUT   = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions — copied verbatim from run_training.py so state_dicts load.
# ─────────────────────────────────────────────────────────────────────────────
def _build_models(in_dim: int, n_div: int, n_sub: int):
    import torch.nn as nn

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], in_dim
            for h in HIDDEN:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(DROPOUT)]
                prev = h
            self.net = nn.Sequential(*layers)
            self.out_dim = prev
        def forward(self, x): return self.net(x)

    class ModelA(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = Backbone()
            self.head = nn.Linear(self.bb.out_dim, n_div)
        def forward(self, x): return self.head(self.bb(x))

    class ModelB(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = Backbone()
            d = self.bb.out_dim
            self.head_sub = nn.Linear(d, n_sub)
            self.head_div = nn.Linear(d, n_div)
        def forward(self, x):
            f = self.bb(x)
            return self.head_sub(f), self.head_div(f)

    return ModelA(), ModelB()


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _stack_emb(df: pd.DataFrame) -> np.ndarray:
    return _l2(np.vstack(df["embeddings"].values).astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(metrics: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for key, label in [("model_a", "Model A"), ("model_b", "Model B")]:
        hist = metrics.get(key, {}).get("history", [])
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["tr_loss"] for h in hist], label=f"{label} train")
        axes[0].plot(ep, [h["va_loss"] for h in hist], "--", label=f"{label} val")
        axes[1].plot(ep, [h["val_acc"] for h in hist], label=label)
        axes[2].plot(ep, [h["val_bal"] for h in hist], label=label)
    axes[0].set(title="Loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Val accuracy", xlabel="epoch", ylabel="acc")
    axes[2].set(title="Val balanced accuracy", xlabel="epoch", ylabel="bal-acc")
    for a in axes:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "training_history.png", dpi=150)
    plt.close(fig)


def plot_support(y_div: np.ndarray, classes, out: Path):
    counts = np.bincount(y_div, minlength=len(classes))
    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(counts)), counts[order])
    ax.set(title=f"Per-division support (n={len(classes)} divisions, total={counts.sum():,})",
           xlabel="division (rank)", ylabel="# docs", yscale="log")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "perclass_support.png", dpi=150)
    plt.close(fig)
    return counts


def plot_f1_vs_support(report: dict, counts: np.ndarray, classes, out: Path):
    sup, f1 = [], []
    for i, c in enumerate(classes):
        if c in report:
            sup.append(report[c]["support"]); f1.append(report[c]["f1-score"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sup, f1, s=14, alpha=0.6)
    ax.set(title="Per-division F1 vs support (Model B)",
           xlabel="support (val)", ylabel="F1", xscale="log")
    ax.axhline(np.mean(f1), color="r", ls="--", lw=1,
               label=f"macro-F1 ≈ {np.mean(f1):.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "perclass_f1.png", dpi=150)
    plt.close(fig)


def plot_leakage(sims: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=60, color="#d9534f", alpha=0.8)
    for t in (0.95, 0.99, 0.999):
        frac = float((sims >= t).mean())
        ax.axvline(t, ls="--", lw=1, alpha=0.6)
        ax.text(t, ax.get_ylim()[1] * 0.9, f"≥{t}: {frac:.1%}",
                rotation=90, va="top", fontsize=8)
    ax.set(title="Val→train nearest-neighbour cosine similarity\n(high mass near 1.0 ⇒ leakage / near-duplicates)",
           xlabel="max cosine similarity to any train embedding", ylabel="# val docs")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "leakage_nn.png", dpi=150)
    plt.close(fig)


def plot_confidence_split(conf_in: np.ndarray, conf_out: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(conf_in, bins=40, alpha=0.6, density=True, label="in-cluster val (measured)")
    if conf_out is not None and len(conf_out):
        ax.hist(conf_out, bins=40, alpha=0.6, density=True,
                label="held-out outliers (NEVER measured)")
    ax.set(title="Model B confidence: measured vs unmeasured half of the corpus",
           xlabel="softmax confidence of predicted division", ylabel="density")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "confidence_split.png", dpi=150)
    plt.close(fig)


def plot_reliability(conf: np.ndarray, correct: np.ndarray, out: Path):
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(conf, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(xs, ys, "o-", label="Model B")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, y), fontsize=7, alpha=0.7)
    ax.set(title="Reliability curve (val)", xlabel="mean predicted confidence",
           ylabel="empirical accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def nn_leakage(X_tr: np.ndarray, X_va: np.ndarray, batch: int = 2000) -> np.ndarray:
    """Max cosine similarity of each val row to any train row (L2-normed inputs)."""
    sims = np.empty(len(X_va), dtype=np.float32)
    Xtr_T = X_tr.T  # (D, Ntr)
    for i in range(0, len(X_va), batch):
        block = X_va[i:i + batch] @ Xtr_T   # cosine since both L2-normed
        sims[i:i + batch] = block.max(axis=1)
    return sims


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir",   required=True,
                    help="training_runs/<name> (holds clusters.parquet + training_dataset.parquet)")
    ap.add_argument("--model-dir", required=True,
                    help="models/<name> (holds model_a.pt, model_b.pt, encoders.pkl, ...)")
    ap.add_argument("--out-dir",   required=True, help="where plots + diagnostics.txt go")
    ap.add_argument("--max-leakage-train", type=int, default=60000,
                    help="cap train rows used for the O(N*M) NN-leakage scan (default 60000)")
    args = ap.parse_args()

    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  classification_report)

    run_dir   = Path(args.run_dir)
    model_dir = Path(args.model_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg=""):
        print(msg); log_lines.append(msg)

    # ── Load encoders + metrics ──────────────────────────────────────────────
    with open(model_dir / "encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    le_div, le_sub = enc["le_div"], enc["le_sub"]
    n_div, n_sub = len(le_div.classes_), len(le_sub.classes_)

    with open(model_dir / "hierarchy.pkl", "rb") as f:
        hier_mask = pickle.load(f)["hier_mask"]

    metrics = json.loads((model_dir / "metrics.json").read_text())

    # ── Rebuild the exact training dataset + split ───────────────────────────
    td = pd.read_parquet(str(run_dir / "training_dataset.parquet"))
    # column names: lcc (subclass), lcc_division
    X = _stack_emb(td)
    y_div = le_div.transform(td["lcc_division"])
    y_sub = le_sub.transform(td["lcc"])

    X_tr, X_va, yd_tr, yd_va, ys_tr, ys_va, idx_tr, idx_va = train_test_split(
        X, y_div, y_sub, np.arange(len(X)),
        test_size=TEST_SIZE, random_state=SEED, stratify=y_div)

    log("=" * 70)
    log("EVALUATION & LEAKAGE DIAGNOSTICS")
    log("=" * 70)
    log(f"  run-dir   : {run_dir}")
    log(f"  model-dir : {model_dir}")
    log(f"  divisions : {n_div}   subclasses : {n_sub}")
    log(f"  train rows: {len(X_tr):,}   val rows: {len(X_va):,}")
    log("")

    # ── Load Model B and run on val ──────────────────────────────────────────
    _, model_b = _build_models(X.shape[1], n_div, n_sub)
    model_b.load_state_dict(torch.load(model_dir / "model_b.pt", map_location="cpu"))
    model_b.eval()

    Xva_t = torch.tensor(X_va)
    with torch.no_grad():
        ls, ld = model_b(Xva_t)
        pred_sub = ls.argmax(1)
        ld_masked = ld.masked_fill(~hier_mask[pred_sub], float("-inf"))
        prob = torch.softmax(ld_masked, dim=1)
        conf_va, pred_div = prob.max(dim=1)
    pred_div = pred_div.numpy(); conf_va = conf_va.numpy()

    acc = accuracy_score(yd_va, pred_div)
    bal = balanced_accuracy_score(yd_va, pred_div)
    correct = (pred_div == yd_va).astype(int)
    log(f"[Model B] reconstructed val  acc={acc:.4f}  bal-acc={bal:.4f}")
    log(f"          (metrics.json said acc={metrics['model_b'].get('val_acc_div')}, "
        f"bal={metrics['model_b'].get('val_bal_div')})")
    log("")

    # ── A. Baselines: prove separability-by-construction ─────────────────────
    log("-" * 70)
    log("A. BASELINES ON THE SAME VAL SPLIT")
    log("   If a trivial classifier ~matches the MLP, the labels were already")
    log("   encoded in the embedding geometry → the MLP learned almost nothing")
    log("   beyond reproducing the clustering.")
    # nearest centroid (per division)
    cents = np.zeros((n_div, X.shape[1]), dtype=np.float32)
    for c in range(n_div):
        m = yd_tr == c
        if m.any():
            v = X_tr[m].mean(0); cents[c] = v / (np.linalg.norm(v) + 1e-9)
    nc_pred = (X_va @ cents.T).argmax(1)
    nc_acc = accuracy_score(yd_va, nc_pred)
    log(f"   nearest-centroid acc = {nc_acc:.4f}   (MLP acc = {acc:.4f})")
    try:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        knn.fit(X_tr, yd_tr)
        knn_acc = accuracy_score(yd_va, knn.predict(X_va))
        log(f"   1-NN acc             = {knn_acc:.4f}")
    except Exception as e:
        log(f"   1-NN skipped ({e})")
    log("")

    # ── B. Train/val near-duplicate leakage ──────────────────────────────────
    log("-" * 70)
    log("B. TRAIN/VAL NEAR-DUPLICATE LEAKAGE")
    Xtr_scan = X_tr
    if len(X_tr) > args.max_leakage_train:
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(X_tr), args.max_leakage_train, replace=False)
        Xtr_scan = X_tr[sel]
        log(f"   (train subsampled to {len(Xtr_scan):,} rows for the scan)")
    sims = nn_leakage(Xtr_scan, X_va)
    for t in (0.999, 0.99, 0.95):
        log(f"   val rows with a train neighbour ≥ {t}: {(sims >= t).mean():.2%}")
    log(f"   median val→train max-sim = {np.median(sims):.4f}")
    plot_leakage(sims, out_dir)
    log("")

    # ── C. Coverage: how much of the data is never measured ──────────────────
    log("-" * 70)
    log("C. COVERAGE (what the metrics DON'T see)")
    clusters_path = run_dir / "clusters.parquet"
    conf_out = None
    if clusters_path.exists():
        cl = pd.read_parquet(str(clusters_path), columns=None)
        n_total = len(cl)
        n_out = int((cl["cluster_id"] == -1).sum())
        n_used = len(td)
        log(f"   clustered sample size      : {n_total:,}")
        log(f"   HDBSCAN outliers (cid=-1)  : {n_out:,}  ({n_out/n_total:.1%})")
        log(f"   used for train+val         : {n_used:,}  ({n_used/n_total:.1%})")
        log(f"   excluded (outlier+skipped) : {n_total - n_used:,}  "
            f"({(n_total-n_used)/n_total:.1%})")
        log("   → On the full 3.6M corpus, most docs resemble the EXCLUDED set,")
        log("     for which we have NO measured accuracy.")

        # ── D. Outlier behaviour ─────────────────────────────────────────────
        out_df = cl[cl["cluster_id"] == -1]
        if len(out_df) and "embeddings" in out_df.columns:
            sample = out_df.sample(min(20000, len(out_df)), random_state=SEED)
            Xo = _stack_emb(sample)
            with torch.no_grad():
                ls, ld = model_b(torch.tensor(Xo))
                ps = ls.argmax(1)
                ldm = ld.masked_fill(~hier_mask[ps], float("-inf"))
                conf_out = torch.softmax(ldm, 1).max(1).values.numpy()
            log("")
            log("D. OUTLIER OVER-CONFIDENCE (held-out cid=-1, no ground truth)")
            log(f"   mean conf  in-cluster val = {conf_va.mean():.3f}")
            log(f"   mean conf  outliers       = {conf_out.mean():.3f}")
            log(f"   outliers predicted with conf>0.9 = {(conf_out>0.9).mean():.1%}")
            log("   (high confidence on points the clustering called noise is a")
            log("    sign the softmax is not trustworthy off-distribution)")
    else:
        log(f"   clusters.parquet not found at {clusters_path} — skipping coverage/outlier")
    log("")

    # ── Plots that need the report dict ──────────────────────────────────────
    rep = classification_report(yd_va, pred_div, labels=np.arange(n_div),
                                target_names=le_div.classes_,
                                output_dict=True, zero_division=0)
    counts = plot_support(y_div, le_div.classes_, out_dir)
    plot_f1_vs_support(rep, counts, le_div.classes_, out_dir)
    plot_training_history(metrics, out_dir)
    plot_confidence_split(conf_va, conf_out, out_dir)
    plot_reliability(conf_va, correct, out_dir)

    # weakest classes
    weak = sorted(
        [(c, rep[c]["f1-score"], rep[c]["support"]) for c in le_div.classes_
         if c in rep and rep[c]["support"] > 0],
        key=lambda r: r[1])[:15]
    log("-" * 70)
    log("WEAKEST 15 DIVISIONS BY F1 (Model B, val)")
    for c, f1, s in weak:
        log(f"   {c:<14} F1={f1:.3f}  support={int(s)}")
    log("")

    (out_dir / "diagnostics.txt").write_text("\n".join(log_lines))
    log(f"✅ plots + diagnostics.txt written to {out_dir}/")


if __name__ == "__main__":
    main()
