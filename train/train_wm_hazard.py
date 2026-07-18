"""Train the world model's multi-horizon hazard heads — rollout capability.

    python train/train_wm_hazard.py --data-dir data --out data/wm_hazard_weights.json

One shared trunk, one head per horizon k in wmfeat.HAZARD_HORIZONS:
  h_k(x) = P(an authoritative filing overturns this key's current value
             within the next k evidence batches)
The monotone hazard curve h_1 <= h_2 <= h_4 is what lets the model be
ROLLED OUT — board error mass k steps ahead — instead of only queried one
step (train_wm.py's superseded_next == the k=2 point of this curve).
Separate artifact so v1's wm_weights.json / serve path is never touched
(same isolation rule as train_action_wm.py). No class re-weighting: these
probabilities feed planning utilities directly, so calibrated LEVELS matter
(the train_action_wm.py lesson); base rates here (23-41%) don't need it.
Exports JSON for dependency-free numpy inference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from majalis.wmfeat import FEATURES, HAZARD_HORIZONS  # noqa: E402
from train.train_wm import ece  # noqa: E402


class HazardNet(torch.nn.Module):
    def __init__(self, d_in: int, horizons: tuple[int, ...], d_h: int = 64):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_h), torch.nn.ReLU(),
            torch.nn.Linear(d_h, d_h), torch.nn.ReLU())
        self.heads = torch.nn.ModuleDict(
            {str(h): torch.nn.Linear(d_h, 1) for h in horizons})

    def forward(self, x):
        t = self.trunk(x)
        return {int(h): head(t).squeeze(-1) for h, head in self.heads.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/wm_hazard_weights.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(Path(args.data_dir) / "wm_dynamics.npz")
    assert f"train_y_hz{HAZARD_HORIZONS[0]}" in d, (
        "wm_dynamics.npz predates hazard labels — rerun scripts/gen_wm_dataset.py")
    Xtr, Xva = d["train_X"], d["val_X"]
    assert Xtr.shape[1] == len(FEATURES)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ztr = torch.tensor((Xtr - mu) / sd, device=dev)
    zva = torch.tensor((Xva - mu) / sd, device=dev)
    ytr = {h: torch.tensor(d[f"train_y_hz{h}"], dtype=torch.float32, device=dev)
           for h in HAZARD_HORIZONS}
    yva = {h: d[f"val_y_hz{h}"].astype(np.float32) for h in HAZARD_HORIZONS}

    model = HazardNet(Xtr.shape[1], HAZARD_HORIZONS).to(dev)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    n, best, best_state, patience = len(ztr), 0.0, None, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 8192):
            idx = perm[i:i + 8192]
            opt.zero_grad()
            out = model(ztr[idx])
            loss = sum(bce(out[h], ytr[h][idx]) for h in HAZARD_HORIZONS)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = {h: torch.sigmoid(v).cpu().numpy()
                  for h, v in model(zva).items()}
        mean_auc = float(np.mean([roc_auc_score(yva[h], pv[h])
                                  for h in HAZARD_HORIZONS]))
        if mean_auc > best:
            best, best_state, patience = mean_auc, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    model.cpu().eval()
    with torch.no_grad():
        pv = {h: torch.sigmoid(v).numpy() for h, v in model(zva.cpu()).items()}

    metrics = {"train_rows": int(len(Xtr)), "val_rows": int(len(Xva)),
               "epochs_run": epoch + 1, "train_s": round(time.time() - t0, 1)}
    for h in HAZARD_HORIZONS:
        metrics[f"auroc_hz{h}"] = round(float(roc_auc_score(yva[h], pv[h])), 4)
        metrics[f"ece_hz{h}"] = round(ece(pv[h], yva[h]), 4)
        metrics[f"base_rate_hz{h}"] = round(float(yva[h].mean()), 4)
    # Rollout sanity: predicted hazard must be (weakly) monotone in horizon.
    viol = float(np.mean((pv[1] > pv[2] + 0.05) | (pv[2] > pv[4] + 0.05)))
    metrics["monotone_violation_rate_0.05"] = round(viol, 4)

    sd_ = model.state_dict()
    out = {
        "features": FEATURES, "horizons": list(HAZARD_HORIZONS),
        "mu": mu.tolist(), "sd": sd.tolist(),
        "trunk": [[sd_["trunk.0.weight"].tolist(), sd_["trunk.0.bias"].tolist()],
                  [sd_["trunk.2.weight"].tolist(), sd_["trunk.2.bias"].tolist()]],
        "heads": {str(h): [sd_[f"heads.{h}.weight"].tolist(),
                            sd_[f"heads.{h}.bias"].tolist()]
                  for h in HAZARD_HORIZONS},
        "metrics": metrics,
    }
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps(metrics, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
