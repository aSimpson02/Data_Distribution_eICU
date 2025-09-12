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
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    e = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m): 
            continue
        e += (m.sum()/n) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(e)

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    # try both
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    raise FileNotFoundError(path)

def _try_load_split(root: Path, split: str):
    # flat layout
    flat_tr = root / f"{split}_train.parquet"
    flat_va = root / f"{split}_val.parquet"
    flat_te = root / f"{split}_test.parquet"
    if flat_tr.exists() or flat_tr.with_suffix(".csv").exists():
        tr = _read_table(flat_tr)
        va = _read_table(flat_va)
        te = _read_table(flat_te)
        return tr, va, te

    # nested layout
    nested = root / split
    if nested.exists():
        tr = _read_table(nested / "train.parquet")
        va = _read_table(nested / "val.parquet")
        te = _read_table(nested / "test.parquet")
        return tr, va, te

    raise FileNotFoundError(
        f"Could not find split files for '{split}' under {root} "
        f"(tried both flat and nested layouts)."
    )

def load_split(data_root: Path, split: str, label_col: str, id_col: str | None):
    tr, va, te = _try_load_split(data_root, split)

    def xy(df: pd.DataFrame):
        # features: numeric + any engineered columns; drop label and optional ID if present
        drop = {label_col}
        if id_col and id_col in df.columns:
            drop.add(id_col)
        cols = [c for c in df.columns if c not in drop]
        X = df[cols].copy()
        y = df[label_col].astype(int).values
        return X, y

    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    Xte, yte = xy(te)

    # test identifiers (optional; fall back to row indices)
    if id_col and id_col in te.columns:
        id_te = te[id_col].values
    else:
        id_te = np.arange(len(te))
    return Xtr, ytr, Xva, yva, Xte, yte, id_te

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
    enable_dropout(model)  # turn on dropout stochasticity
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


def train_one(
    Xtr, ytr, Xva, yva, *,
    hidden=(128,64), dropout=0.3, epochs=30, batch_size=256,
    lr=1e-3, weight_decay=0.0, patience=5, device=None, seed=42
):
    set_seed(seed)
    device = device or device_auto()

    # numeric-only (robustness if engineered strings slipped in)
    Xtr_num = Xtr.select_dtypes(include=[np.number]).copy()
    Xva_num = Xva.select_dtypes(include=[np.number]).copy()

    # impute + scale on train only
    imp = SimpleImputer(strategy="median").fit(Xtr_num)
    Xtr_i = imp.transform(Xtr_num)
    Xva_i = imp.transform(Xva_num)

    sc = StandardScaler().fit(Xtr_i)
    Xtr_z = sc.transform(Xtr_i)
    Xva_z = sc.transform(Xva_i)

    Xtr_t = torch.from_numpy(Xtr_z).float()
    ytr_t = torch.from_numpy(ytr).float()
    Xva_t = torch.from_numpy(Xva_z).float()
    yva_t = torch.from_numpy(yva).float()

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=4096, shuffle=False)

    model = MLPDropout(in_features=Xtr_t.shape[1], hidden=hidden, p=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, wait = -np.inf, None, 0
    for _epoch in range(1, epochs+1):
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

    if best_state is not None:
        model.load_state_dict(best_state)
    # return model + preprocessors fit on TRAIN ONLY
    return model, imp, sc

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("MC-Dropout MLP on tabular splits")
    ap.add_argument("--data-root", type=Path, default=Path("data/splits"),
                    help="Root folder for splits. Supports flat or nested layout.")
    ap.add_argument("--split", default="hospital", choices=["random","temporal","hospital","hospital5src"])
    ap.add_argument("--label-col", default="hospital_mortality")
    ap.add_argument("--id-col", default="patientunitstayid",
                    help="Optional identifier column; if missing, row indices are used.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--hidden", type=str, default="128,64")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--mc-samples", type=int, default=50)
    ap.add_argument("--ece-bins", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    device = device_auto()
    print(f"[Info] device={device}, hidden={hidden}, dropout={args.dropout}")

    # IO dirs
    OUT_PREDS = Path("results/preds_uq"); OUT_PREDS.mkdir(parents=True, exist_ok=True)
    OUT_TAB   = Path("results/tables");   OUT_TAB.mkdir(parents=True, exist_ok=True)

    # load data
    Xtr, ytr, Xva, yva, Xte, yte, id_te = load_split(args.data_root, args.split, args.label_col, args.id_col)

    # train
    model, imp, sc = train_one(
        Xtr, ytr, Xva, yva,
        hidden=hidden, dropout=args.dropout, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, device=device, seed=args.seed
    )

    # transform TEST using TRAIN-fitted processors
    Xte_num = Xte.select_dtypes(include=[np.number]).copy()
    Xte_i = imp.transform(Xte_num)
    Xte_z = sc.transform(Xte_i)

    # MC sampling
    p_mean, p_std, H, MI = mc_predict(model, Xte_z, device, T=args.mc_samples)

    # metrics for mean probs
    p_clip = np.clip(p_mean, 1e-7, 1-1e-7)
    metrics = {
        "split": args.split,
        "model": "mlp_mc_dropout",
        "auroc": float(roc_auc_score(yte, p_clip)),
        "brier": float(brier_score_loss(yte, p_clip)),
        "accuracy": float(accuracy_score(yte, (p_clip>=0.5).astype(int))),
        "ece": float(ece(yte, p_clip, n_bins=args.ece_bins)),
        "logloss": float(log_loss(yte, p_clip, labels=[0,1])),
        "pos_rate_test": float(np.mean(yte)),
        "mc_samples": int(args.mc_samples),
        "hidden": str(hidden),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs)
    }
    print("[OK] MC-Dropout metrics:", metrics)

    # save per-example predictions + uncertainties
    df = pd.DataFrame({
        (args.id_col if args.id_col else "row_id"): id_te,
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
    # keep CPU thread usage sane on laptops/CI
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    main()
