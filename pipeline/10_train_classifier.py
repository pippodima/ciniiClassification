"""
pipeline/10_train_classifier.py
================================
Trains two classifiers on the 14,767-paper training set
(embeddings → LCC division code).

Model A  –  Flat MLP:       embedding → lcc_division  (lcc_subclass derived)
Model B  –  Two-head MLP:   embedding → lcc_subclass + lcc_division (jointly)

Both use:
  • Class-weighted cross-entropy to handle severe imbalance
  • Stratified 80/20 train/val split
  • Adam + ReduceLROnPlateau scheduler
  • L2-normalised 1024-dim input embeddings (already in the parquet)

Model B adds:
  • Hierarchical mask at inference: after predicting subclass, zero-out
    division logits that don't belong to that subclass before argmax

Outputs (saved to models/v2/):
  model_a_flat.pt         – state dict for Model A
  model_b_twohead.pt      – state dict for Model B
  label_encoders.pkl      – LabelEncoders for division and subclass
  hierarchy.pkl           – subclass→division index mapping (for masking)
  results.json            – all evaluation metrics
"""

import json, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, accuracy_score,
                             balanced_accuracy_score)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("clustering_output/v2")
MODEL_DIR = Path("models/v2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
DROP_CLUSTERS  = [123, 127, 153]   # JATS noise / no valid label
HIDDEN_DIMS    = [512, 256]
DROPOUT        = 0.3
EPOCHS         = 60
BATCH_SIZE     = 256
LR             = 1e-3
LR_PATIENCE    = 8                 # epochs before reducing LR
LR_FACTOR      = 0.5
EARLY_STOP     = 15                # epochs without val improvement
ALPHA          = 0.35              # weight of subclass loss in Model B
VAL_SPLIT      = 0.20
SEED           = 42
DEVICE         = "cpu"             # embeddings are small enough for CPU

torch.manual_seed(SEED)
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD AND PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
train_df  = pd.read_parquet(DATA_DIR / "training_data.parquet")
labels_df = pd.read_parquet(DATA_DIR / "lcc_mapping_full.parquet")

df = train_df.merge(
    labels_df[["cluster_id", "lcc", "lcc_division"]],
    on="cluster_id", how="left"
)

# Drop unlabelable clusters
df = df[~df["cluster_id"].isin(DROP_CLUSTERS)].reset_index(drop=True)
print(f"  {len(df):,} papers after dropping clusters {DROP_CLUSTERS}")
print(f"  Unique lcc_division  : {df['lcc_division'].nunique()}")
print(f"  Unique lcc (subclass): {df['lcc'].nunique()}")

# ── Label encoding ────────────────────────────────────────────────────────────
le_div = LabelEncoder().fit(df["lcc_division"])
le_sub = LabelEncoder().fit(df["lcc"])

y_div = le_div.transform(df["lcc_division"])   # int array
y_sub = le_sub.transform(df["lcc"])            # int array

n_div = len(le_div.classes_)
n_sub = len(le_sub.classes_)
print(f"  Classes – division: {n_div}   subclass: {n_sub}")

# ── Embeddings ────────────────────────────────────────────────────────────────
X = np.vstack(df["embeddings"].values).astype(np.float32)
print(f"  Embedding matrix: {X.shape}  norm≈{np.linalg.norm(X, axis=1).mean():.4f}")

# ── Train / val split (stratified on division label) ─────────────────────────
X_tr, X_va, yd_tr, yd_va, ys_tr, ys_va = train_test_split(
    X, y_div, y_sub,
    test_size=VAL_SPLIT, random_state=SEED, stratify=y_div
)
print(f"  Train: {len(X_tr):,}   Val: {len(X_va):,}")

def to_tensor(*arrays):
    return [torch.tensor(a) for a in arrays]

Xtr, yd_tr_t, ys_tr_t = to_tensor(X_tr, yd_tr, ys_tr)
Xva, yd_va_t, ys_va_t = to_tensor(X_va, yd_va, ys_va)

train_ds = TensorDataset(Xtr, yd_tr_t, ys_tr_t)
val_ds   = TensorDataset(Xva, yd_va_t, ys_va_t)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=512,        shuffle=False)

# ── Class weights ─────────────────────────────────────────────────────────────
def class_weights(y, n_classes):
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    return torch.tensor(cw, dtype=torch.float32)

w_div = class_weights(yd_tr, n_div)
w_sub = class_weights(ys_tr, n_sub)
print(f"  Class weight range (div): [{w_div.min():.2f}, {w_div.max():.2f}]")
print(f"  Class weight range (sub): [{w_sub.min():.2f}, {w_sub.max():.2f}]")

# ── Hierarchy: subclass_idx → list of valid division_idx ─────────────────────
sub_to_divs = {}   # int → list[int]
for sub_name in le_sub.classes_:
    sub_idx  = int(le_sub.transform([sub_name])[0])
    # Find all divisions that belong to this subclass
    sub_rows = df[df["lcc"] == sub_name]["lcc_division"].unique()
    div_idxs = [int(le_div.transform([d])[0]) for d in sub_rows if d in le_div.classes_]
    sub_to_divs[sub_idx] = div_idxs

# Build boolean mask tensor [n_sub, n_div]  True = valid
hier_mask = torch.zeros(n_sub, n_div, dtype=torch.bool)
for s, ds in sub_to_divs.items():
    for d in ds:
        hier_mask[s, d] = True

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
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
    """Flat: embedding → lcc_division only."""
    def __init__(self, n_div, hidden=[512, 256], dropout=0.3):
        super().__init__()
        self.backbone = Backbone(hidden=hidden, dropout=dropout)
        self.head     = nn.Linear(self.backbone.out_dim, n_div)

    def forward(self, x):
        return self.head(self.backbone(x))


class ModelB(nn.Module):
    """Two-head: embedding → lcc_subclass + lcc_division jointly."""
    def __init__(self, n_div, n_sub, hidden=[512, 256], dropout=0.3):
        super().__init__()
        self.backbone  = Backbone(hidden=hidden, dropout=dropout)
        d = self.backbone.out_dim
        self.head_sub  = nn.Linear(d, n_sub)
        self.head_div  = nn.Linear(d, n_div)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_sub(feat), self.head_div(feat)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def train_model_a(model, epochs, patience):
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=LR_PATIENCE, factor=LR_FACTOR)
    crit   = nn.CrossEntropyLoss(weight=w_div)

    best_val, best_state, no_imp = 0.0, None, 0
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0
        for xb, yd, _ in train_dl:
            opt.zero_grad()
            loss = crit(model(xb), yd)
            loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_dl.dataset)

        model.eval()
        preds, trues = [], []
        va_loss = 0
        with torch.no_grad():
            for xb, yd, _ in val_dl:
                logits = model(xb)
                va_loss += crit(logits, yd).item() * len(xb)
                preds.append(logits.argmax(1)); trues.append(yd)
        va_loss /= len(val_dl.dataset)
        preds = torch.cat(preds).numpy(); trues = torch.cat(trues).numpy()
        val_acc = accuracy_score(trues, preds)
        val_bal = balanced_accuracy_score(trues, preds)
        sched.step(va_loss)

        history.append(dict(epoch=ep, tr_loss=round(tr_loss,4),
                            va_loss=round(va_loss,4),
                            val_acc=round(val_acc,4),
                            val_bal=round(val_bal,4)))
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:>3}  tr={tr_loss:.4f}  va={va_loss:.4f}"
                  f"  acc={val_acc:.3f}  bal={val_bal:.3f}"
                  f"  lr={opt.param_groups[0]['lr']:.2e}")

        if val_bal > best_val:
            best_val = val_bal
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return history, best_val


def train_model_b(model, epochs, patience):
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=LR_PATIENCE, factor=LR_FACTOR)
    crit_sub = nn.CrossEntropyLoss(weight=w_sub)
    crit_div = nn.CrossEntropyLoss(weight=w_div)

    best_val, best_state, no_imp = 0.0, None, 0
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0
        for xb, yd, ys in train_dl:
            opt.zero_grad()
            lg_sub, lg_div = model(xb)
            loss = ALPHA * crit_sub(lg_sub, ys) + (1 - ALPHA) * crit_div(lg_div, yd)
            loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_dl.dataset)

        model.eval()
        p_div, p_sub, t_div, t_sub = [], [], [], []
        va_loss = 0
        with torch.no_grad():
            for xb, yd, ys in val_dl:
                lg_sub, lg_div = model(xb)
                loss = ALPHA * crit_sub(lg_sub, ys) + (1-ALPHA) * crit_div(lg_div, yd)
                va_loss += loss.item() * len(xb)

                # Hierarchical masking: mask divisions outside predicted subclass
                pred_sub  = lg_sub.argmax(1)
                mask      = hier_mask[pred_sub]           # [B, n_div]
                lg_div_m  = lg_div.masked_fill(~mask, float("-inf"))
                pred_div  = lg_div_m.argmax(1)

                p_div.append(pred_div); t_div.append(yd)
                p_sub.append(pred_sub); t_sub.append(ys)

        va_loss /= len(val_dl.dataset)
        p_div = torch.cat(p_div).numpy(); t_div = torch.cat(t_div).numpy()
        p_sub = torch.cat(p_sub).numpy(); t_sub = torch.cat(t_sub).numpy()
        acc_div = accuracy_score(t_div, p_div)
        bal_div = balanced_accuracy_score(t_div, p_div)
        acc_sub = accuracy_score(t_sub, p_sub)
        sched.step(va_loss)

        history.append(dict(epoch=ep, tr_loss=round(tr_loss,4),
                            va_loss=round(va_loss,4),
                            val_acc_div=round(acc_div,4),
                            val_bal_div=round(bal_div,4),
                            val_acc_sub=round(acc_sub,4)))
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:>3}  tr={tr_loss:.4f}  va={va_loss:.4f}"
                  f"  div_acc={acc_div:.3f}  sub_acc={acc_sub:.3f}"
                  f"  bal={bal_div:.3f}"
                  f"  lr={opt.param_groups[0]['lr']:.2e}")

        if bal_div > best_val:
            best_val = bal_div
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return history, best_val


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def eval_model_a(model):
    model.eval()
    preds, trues_div, trues_sub = [], [], []
    with torch.no_grad():
        for xb, yd, ys in val_dl:
            preds.append(model(xb).argmax(1))
            trues_div.append(yd); trues_sub.append(ys)
    p = torch.cat(preds).numpy()
    td = torch.cat(trues_div).numpy()
    ts = torch.cat(trues_sub).numpy()

    # Derive predicted subclass from predicted division
    # Map: division_idx → subclass_idx
    div_to_sub = {}
    for row in df[["lcc","lcc_division"]].drop_duplicates().itertuples():
        if row.lcc_division in le_div.classes_ and row.lcc in le_sub.classes_:
            d = int(le_div.transform([row.lcc_division])[0])
            s = int(le_sub.transform([row.lcc])[0])
            div_to_sub[d] = s
    p_sub = np.array([div_to_sub.get(d, -1) for d in p])

    return dict(
        val_acc_div  = round(accuracy_score(td, p), 4),
        val_bal_div  = round(balanced_accuracy_score(td, p), 4),
        val_acc_sub  = round(accuracy_score(ts, p_sub), 4),
        val_bal_sub  = round(balanced_accuracy_score(ts, p_sub), 4),
        report_div   = classification_report(
                           td, p,
                           target_names=le_div.classes_, zero_division=0),
        report_sub   = classification_report(
                           ts, p_sub,
                           target_names=le_sub.classes_, zero_division=0),
    )


def eval_model_b(model):
    model.eval()
    p_div, p_sub, t_div, t_sub = [], [], [], []
    with torch.no_grad():
        for xb, yd, ys in val_dl:
            lg_sub, lg_div = model(xb)
            pred_sub = lg_sub.argmax(1)
            mask     = hier_mask[pred_sub]
            lg_div_m = lg_div.masked_fill(~mask, float("-inf"))
            p_div.append(lg_div_m.argmax(1)); t_div.append(yd)
            p_sub.append(pred_sub);           t_sub.append(ys)
    pd_ = torch.cat(p_div).numpy(); td = torch.cat(t_div).numpy()
    ps  = torch.cat(p_sub).numpy(); ts = torch.cat(t_sub).numpy()

    return dict(
        val_acc_div  = round(accuracy_score(td, pd_), 4),
        val_bal_div  = round(balanced_accuracy_score(td, pd_), 4),
        val_acc_sub  = round(accuracy_score(ts, ps), 4),
        val_bal_sub  = round(balanced_accuracy_score(ts, ps), 4),
        report_div   = classification_report(
                           td, pd_,
                           target_names=le_div.classes_, zero_division=0),
        report_sub   = classification_report(
                           ts, ps,
                           target_names=le_sub.classes_, zero_division=0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  RUN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODEL A  –  Flat MLP  (division only)")
print("="*60)
model_a = ModelA(n_div, HIDDEN_DIMS, DROPOUT)
n_params_a = sum(p.numel() for p in model_a.parameters())
print(f"  Parameters: {n_params_a:,}")
t0 = time.time()
hist_a, best_a = train_model_a(model_a, EPOCHS, EARLY_STOP)
time_a = time.time() - t0
print(f"  Training time: {time_a:.1f}s   Best balanced acc: {best_a:.4f}")
metrics_a = eval_model_a(model_a)
print(f"\n  Final eval  –  div acc={metrics_a['val_acc_div']:.4f}"
      f"  div bal={metrics_a['val_bal_div']:.4f}"
      f"  sub acc={metrics_a['val_acc_sub']:.4f}"
      f"  sub bal={metrics_a['val_bal_sub']:.4f}")

print("\n" + "="*60)
print("  MODEL B  –  Two-head MLP  (subclass + division)")
print("="*60)
model_b = ModelB(n_div, n_sub, HIDDEN_DIMS, DROPOUT)
n_params_b = sum(p.numel() for p in model_b.parameters())
print(f"  Parameters: {n_params_b:,}")
t0 = time.time()
hist_b, best_b = train_model_b(model_b, EPOCHS, EARLY_STOP)
time_b = time.time() - t0
print(f"  Training time: {time_b:.1f}s   Best balanced acc: {best_b:.4f}")
metrics_b = eval_model_b(model_b)
print(f"\n  Final eval  –  div acc={metrics_b['val_acc_div']:.4f}"
      f"  div bal={metrics_b['val_bal_div']:.4f}"
      f"  sub acc={metrics_b['val_acc_sub']:.4f}"
      f"  sub bal={metrics_b['val_bal_sub']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  COMPARISON SUMMARY")
print("="*60)
print(f"  {'Metric':<28} {'Model A':>10} {'Model B':>10}")
print("  " + "-"*50)
for k in ['val_acc_div','val_bal_div','val_acc_sub','val_bal_sub']:
    print(f"  {k:<28} {metrics_a[k]:>10.4f} {metrics_b[k]:>10.4f}")

print()
print("  TOP CLASSES — Model A (by F1, division level):")
# Quick per-class from the report
for line in metrics_a['report_div'].split('\n')[2:-4]:
    parts = line.split()
    if len(parts) >= 5 and float(parts[-1]) >= 5:  # support >= 5
        f1 = float(parts[3])
        if f1 < 0.5:
            print(f"    ⚠  {parts[0]:<14} f1={f1:.2f}  support={parts[-1]}")

print()
print("  TOP CLASSES — Model B (by F1, division level):")
for line in metrics_b['report_div'].split('\n')[2:-4]:
    parts = line.split()
    if len(parts) >= 5 and float(parts[-1]) >= 5:
        f1 = float(parts[3])
        if f1 < 0.5:
            print(f"    ⚠  {parts[0]:<14} f1={f1:.2f}  support={parts[-1]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SAVE
# ═══════════════════════════════════════════════════════════════════════════════
torch.save(model_a.state_dict(), MODEL_DIR / "model_a_flat.pt")
torch.save(model_b.state_dict(), MODEL_DIR / "model_b_twohead.pt")

with open(MODEL_DIR / "label_encoders.pkl", "wb") as f:
    pickle.dump({"le_div": le_div, "le_sub": le_sub}, f)

with open(MODEL_DIR / "hierarchy.pkl", "wb") as f:
    pickle.dump({"sub_to_divs": sub_to_divs,
                 "hier_mask":   hier_mask,
                 "div_to_sub":  {d: s for d, divs in sub_to_divs.items()
                                 for s in [d] for d in divs}}, f)

# Save model configs for reloading
model_cfg = dict(n_div=n_div, n_sub=n_sub, hidden=HIDDEN_DIMS, dropout=DROPOUT)
results = dict(
    model_a = dict(params=n_params_a, train_time_s=round(time_a,1),
                   history=hist_a, **{k:v for k,v in metrics_a.items()
                                       if not k.startswith('report')}),
    model_b = dict(params=n_params_b, train_time_s=round(time_b,1),
                   history=hist_b, **{k:v for k,v in metrics_b.items()
                                       if not k.startswith('report')}),
    config  = model_cfg,
    classes = dict(divisions=list(le_div.classes_),
                   subclasses=list(le_sub.classes_)),
)
with open(MODEL_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save full classification reports
with open(MODEL_DIR / "report_a.txt", "w") as f:
    f.write("=== MODEL A — DIVISION REPORT ===\n")
    f.write(metrics_a['report_div'])
    f.write("\n=== MODEL A — SUBCLASS REPORT ===\n")
    f.write(metrics_a['report_sub'])

with open(MODEL_DIR / "report_b.txt", "w") as f:
    f.write("=== MODEL B — DIVISION REPORT ===\n")
    f.write(metrics_b['report_div'])
    f.write("\n=== MODEL B — SUBCLASS REPORT ===\n")
    f.write(metrics_b['report_sub'])

print("\n  Saved:")
for p in sorted(MODEL_DIR.iterdir()):
    print(f"    {p}")
