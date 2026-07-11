"""Train Agora's learned world model on logged episodes (GPU box).

    python train/train_wm.py --data-dir data --out data/wm_weights.json

Shared-trunk MLP, two decision-relevant heads (AAWM-style targets):
  wrong_now       — P(board's current value for a key is incorrect)
  superseded_next — P(an authoritative filing overturns it within lookahead)
plus a tiny logistic stacker mapping (head output, sampled disagreement,
weak flag) -> P(committed answer wrong), fit on the REAL logged calibration
episodes (LLM-built boards) so the sim-to-real gap is measured, not assumed.

Baselines reported honestly: the old hand-set heuristics (doubt blend /
Lomax survival) as single-feature rankers, and HistGradientBoosting on the
same features. Everything exports to one JSON (weights + standardization +
metrics) for dependency-free numpy inference in agora.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = [  # must mirror agora.wmfeat.FEATURES
    "age_days", "exposure_days", "n_assertions", "n_supersessions",
    "n_conflicts", "churn_per_month", "n_distinct_values", "tier_cur",
    "weak_current", "frac_weak_hist", "lomax_p_valid", "doubt_heuristic",
]
I_LOMAX, I_DOUBT = FEATURES.index("lomax_p_valid"), FEATURES.index("doubt_heuristic")


class WMNet(torch.nn.Module):
    def __init__(self, d_in: int, d_h: int = 64):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_h), torch.nn.ReLU(),
            torch.nn.Linear(d_h, d_h), torch.nn.ReLU())
        self.head_wrong = torch.nn.Linear(d_h, 1)
        self.head_sup = torch.nn.Linear(d_h, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head_wrong(h).squeeze(-1), self.head_sup(h).squeeze(-1)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/wm_weights.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} " + (torch.cuda.get_device_name(0) if dev == "cuda" else ""))

    d = np.load(Path(args.data_dir) / "wm_dynamics.npz")
    Xtr, Xva = d["train_X"], d["val_X"]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ztr = torch.tensor((Xtr - mu) / sd, device=dev)
    zva = torch.tensor((Xva - mu) / sd, device=dev)
    ytr = {k: torch.tensor(d[f"train_{k}"], dtype=torch.float32, device=dev)
           for k in ("y_wrong", "y_sup")}
    yva = {k: d[f"val_{k}"].astype(np.float32) for k in ("y_wrong", "y_sup")}

    model = WMNet(Xtr.shape[1]).to(dev)
    pw = torch.tensor((ytr["y_wrong"] == 0).sum() / ytr["y_wrong"].sum(), device=dev)
    bce_w = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    bce_s = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    n, best_auc, best_state, patience = len(ztr), 0.0, None, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 8192):
            idx = perm[i:i + 8192]
            lw, ls = model(ztr[idx])
            loss = bce_w(lw, ytr["y_wrong"][idx]) + bce_s(ls, ytr["y_sup"][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lw, ls = model(zva)
            pw_va = torch.sigmoid(lw).cpu().numpy()
            ps_va = torch.sigmoid(ls).cpu().numpy()
        auc_w = roc_auc_score(yva["y_wrong"], pw_va)
        auc_s = roc_auc_score(yva["y_sup"], ps_va)
        if auc_w > best_auc + 1e-4:
            best_auc, patience = auc_w, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        print(f"epoch {epoch:02d} loss {loss.item():.4f} "
              f"val AUROC wrong={auc_w:.4f} sup={auc_s:.4f}", flush=True)
        if patience >= 8:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lw, ls = model(zva)
        pw_va = torch.sigmoid(lw).cpu().numpy()
        ps_va = torch.sigmoid(ls).cpu().numpy()

    # --- honest baselines on the same val split -------------------------
    metrics = {
        "train_rows": int(len(Xtr)), "val_rows": int(len(Xva)),
        "train_secs": round(time.time() - t0, 1), "device": dev,
        "auroc": {
            "wrong_now/learned": round(float(roc_auc_score(yva["y_wrong"], pw_va)), 4),
            "wrong_now/hand_doubt": round(float(roc_auc_score(yva["y_wrong"], Xva[:, I_DOUBT])), 4),
            "superseded/learned": round(float(roc_auc_score(yva["y_sup"], ps_va)), 4),
            "superseded/hand_lomax": round(float(roc_auc_score(yva["y_sup"], 1 - Xva[:, I_LOMAX])), 4),
        },
        "ece": {"wrong_now/learned": round(ece(pw_va, yva["y_wrong"]), 4),
                "superseded/learned": round(ece(ps_va, yva["y_sup"]), 4)},
    }
    for target, ykey in (("wrong_now", "y_wrong"), ("superseded", "y_sup")):
        gb = HistGradientBoostingClassifier(random_state=0).fit(Xtr, d[f"train_{ykey}"])
        p = gb.predict_proba(Xva)[:, 1]
        metrics["auroc"][f"{target}/gbdt_baseline"] = round(float(roc_auc_score(yva[ykey], p)), 4)

    # --- stacker on REAL logged episodes (sim-to-real) ------------------
    s = np.load(Path(args.data_dir) / "wm_stacker.npz")
    zs = torch.tensor((s["X"] - mu) / sd, dtype=torch.float32, device=dev)
    with torch.no_grad():
        p_head = torch.sigmoid(model(zs)[0]).cpu().numpy()
    Xstk = np.column_stack([p_head, s["extra"]])  # [p_wrong_head, disagreement, weak]
    harm, seeds = s["harm"].astype(int), s["seed"]
    aucs, preds = [], np.zeros_like(harm, dtype=float)
    for hold in np.unique(seeds):  # leave-one-seed-out: honest small-N estimate
        tr, te = seeds != hold, seeds == hold
        lr = LogisticRegression(max_iter=1000).fit(Xstk[tr], harm[tr])
        preds[te] = lr.predict_proba(Xstk[te])[:, 1]
        if len(np.unique(harm[te])) > 1:
            aucs.append(roc_auc_score(harm[te], preds[te]))
    metrics["stacker"] = {
        "n_real_episodes": int(len(harm)), "harm_rate": round(float(harm.mean()), 4),
        "loso_auroc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
        "loso_auroc_overall": round(float(roc_auc_score(harm, preds)), 4),
        "head_alone_auroc": round(float(roc_auc_score(harm, p_head)), 4),
    }
    final_lr = LogisticRegression(max_iter=1000).fit(Xstk, harm)

    # --- export for numpy inference -------------------------------------
    sd_ = {k: v.numpy() for k, v in best_state.items()}
    out = {
        "features": FEATURES, "mu": mu.tolist(), "sd": sd.tolist(),
        "trunk": [[sd_["trunk.0.weight"].tolist(), sd_["trunk.0.bias"].tolist()],
                  [sd_["trunk.2.weight"].tolist(), sd_["trunk.2.bias"].tolist()]],
        "head_wrong": [sd_["head_wrong.weight"].tolist(), sd_["head_wrong.bias"].tolist()],
        "head_sup": [sd_["head_sup.weight"].tolist(), sd_["head_sup.bias"].tolist()],
        "stacker": {"coef": final_lr.coef_[0].tolist(),
                    "intercept": float(final_lr.intercept_[0]),
                    "features": ["p_wrong_head", "disagreement", "weak_current"]},
        "metrics": metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps(metrics, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
