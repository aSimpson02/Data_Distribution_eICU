#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, accuracy_score, log_loss
)

# paths / constants
DATA = Path("data/splits")
OUT_PREDS = Path("results/preds_uq"); OUT_PREDS.mkdir(parents=True, exist_ok=True)
OUT_TAB = Path("results/tables"); OUT_TAB.mkdir(parents=True, exist_ok=True)
TARGET = "hospital_mortality"
IDCOL = "patientunitstayid"
DROP = {TARGET, IDCOL}

# helpers
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def ece(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m): continue
        e += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(e)

def load_split(split: str):
    tr = pd.read_parquet(DATA / f"{split}_train.parquet")
    va = pd.read_parquet(DATA / f"{split}_val.parquet")
    te = pd.read_parquet(DATA / f"{split}_test.parquet")
    def xy(df):
        cols = [c for c in df.columns if c not in DROP]
        X = df[cols].copy()
        y = df[TARGET].astype(int).values
        return X, y
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    Xte, yte = xy(te)
    id_te = te[IDCOL].values
    return Xtr, ytr, Xva, yva, Xte, yte, id_te

# model
class MLPDropout(nn.Module):
    def __init__(self, in_features: int, hidden=(128, 64), p=0.3):
        super().__init__()
        layers = []
        last = in_features
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p)]
            last = h
        layers += [nn.Linear(last, 1)]  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits

def enable_dropout(model: nn.Module):
    # Keep dropout layers in train mode during MC sampling
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

@torch.no_grad()
def mc_predict(model, X, device, T=50, batch_size=4096):
    model.eval()
    # turn on dropout only
    enable_dropout(model)
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()),
                        batch_size=batch_size, shuffle=False)
    preds = []
    for _ in range(T):
        out_all = []
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            out_all.append(probs)
        preds.append(np.concatenate(out_all, axis=0))
    P = np.vstack(preds)            # [T, N]
    p_mean = P.mean(axis=0)
    p_std = P.std(axis=0, ddof=1)
    # predictive entropy H[p_mean]
    eps = 1e-9
    H = -(p_mean*np.log(p_mean+eps) + (1-p_mean)*np.log(1-p_mean+eps))
    # aleatoric entropy: mean H[p_t]
    H_t = -(P*np.log(P+eps) + (1-P)*np.log(1-P+eps))
    H_alea = H_t.mean(axis=0)
    # epistemic (BALD / mutual information)
    MI = H - H_alea
    return p_mean, p_std, H, MI

# training
def train_one(
    Xtr, ytr, Xva, yva, *,
    hidden=(128,64), dropout=0.3, epochs=30, batch_size=256,
    lr=1e-3, weight_decay=0.0, patience=5, device=None, seed=42
):
    set_seed(seed)
    device = device or device_auto()

    # impute + scale on train only
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = imp.fit_transform(Xtr)
    Xva = imp.transform(Xva)

    Xtr = sc.fit_transform(Xtr)
    Xva = sc.transform(Xva)

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float()
    Xva_t = torch.from_numpy(Xva).float()
    yva_t = torch.from_numpy(yva).float()

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=4096, shuffle=False)

    model = MLPDropout(in_features=Xtr.shape[1], hidden=hidden, p=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state = -np.inf, None
    wait = 0

    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # validate (deterministic)
        model.eval()
        with torch.no_grad():
            logits = []
            ys = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                l = model(xb)
                logits.append(l.cpu())
                ys.append(yb)
            y_val = torch.cat(ys).numpy()
            p_val = torch.sigmoid(torch.cat(logits)).numpy()
            auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val))==2 else np.nan

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, imp, sc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="hospital5src",
                    choices=["random","temporal","hospital","hospital5src"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--hidden", type=str, default="128,64")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--mc-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    device = device_auto()
    print(f"[Info] device={device}, hidden={hidden}, dropout={args.dropout}")

    Xtr, ytr, Xva, yva, Xte, yte, id_te = load_split(args.split)
    model, imp, sc = train_one(
        Xtr, ytr, Xva, yva,
        hidden=hidden, dropout=args.dropout, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, device=device, seed=args.seed
    )

    # transform test with train fitted imputers/scaler
    Xtr_i = sc.transform(imp.transform(Xtr))  # just to get shape for sanity
    Xte_i = sc.transform(imp.transform(Xte))

    # MC sampling
    p_mean, p_std, H, MI = mc_predict(model, Xte_i, device, T=args.mc_samples)

    # metrics for mean probs
    p_clip = np.clip(p_mean, 1e-7, 1-1e-7)
    metrics = {
        "split": args.split,
        "model": "mlp_mc_dropout",
        "auroc": float(roc_auc_score(yte, p_clip)),
        "brier": float(brier_score_loss(yte, p_clip)),
        "accuracy": float(accuracy_score(yte, (p_clip>=0.5).astype(int))),
        "ece": float(ece(yte, p_clip)),
        "logloss": float(log_loss(yte, p_clip)),
        "pos_rate_test": float(np.mean(yte)),
        "mc_samples": int(args.mc_samples),
        "hidden": str(hidden),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs)
    }
    print("[OK] MC Dropout metrics:", metrics)

    # save per-example predictions + uncertainties
    df = pd.DataFrame({
        IDCOL: id_te,
        "y": yte,
        "p_mean": p_mean,
        "p_std": p_std,
        "entropy": H,
        "mutual_info": MI
    })
    fp = OUT_PREDS / f"uq_mcdropout_{args.split}.parquet"
    df.to_parquet(fp, index=False)

    # append summary
    summary = OUT_TAB / "uq_mcdropout_summary.csv"
    pd.DataFrame([metrics]).to_csv(summary, mode="a", index=False, header=not summary.exists())
    print("Saved preds →", fp)
    print("Appended summary →", summary)

if __name__ == "__main__":
    main()
