"""Train the action-conditioned world model's NEW head: p_correct_debate.

    .venv/bin/python train/train_action_wm.py \\
        --train data/action_wm_train.jsonl --heldout data/action_wm_heldout.jsonl \\
        --out data/wm_action_weights.json

Mirrors train_wm.py's WMNet/BCE pattern almost verbatim, for ONE new head
(head_debate) fit on real mined (skip, debate) paired episodes from
scripts/gen_action_wm_dataset.py. Exports to a SEPARATE artifact
(data/wm_action_weights.json) so v1's wm_weights.json / AcceptGate behavior
is never touched or put at risk.

Two honesty numbers are reported (small-N, exactly the culture this repo's
v1 stacker already established):
  auroc_heldout / ece_heldout   — the HEADLINE number: seeds 150-179 are a
                                  genuinely disjoint held-out band (never
                                  seen during training or model selection).
  loso_auroc_mean_train_band    — a robustness check on the TRAIN band
                                  (100-149) via leave-one-seed-out, mirroring
                                  train_wm.py's stacker LOSO convention.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # reuse train_wm.py's ece() — no reimplementation

from majalis.wmfeat_action import FEATURES  # noqa: E402
from train.train_wm import ece  # noqa: E402


class ActionWMNet(torch.nn.Module):
    def __init__(self, d_in: int, d_h: int = 64):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_h), torch.nn.ReLU(),
            torch.nn.Linear(d_h, d_h), torch.nn.ReLU())
        self.head_debate = torch.nn.Linear(d_h, 1)

    def forward(self, x):
        return self.head_debate(self.trunk(x)).squeeze(-1)


def _load_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    if not rows:
        raise ValueError(f"{path} has zero mined rows")
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y = np.array([r["debate_correct"] for r in rows], dtype=np.float32)
    seeds = np.array([r["seed"] for r in rows], dtype=np.int32)
    return X, y, seeds


def _fit(z: torch.Tensor, y: torch.Tensor, epochs: int, seed: int = 0) -> ActionWMNet:
    torch.manual_seed(seed)
    model = ActionWMNet(z.shape[1])
    pos = float(y.sum().item())
    neg = float(len(y)) - pos
    pos_weight = torch.tensor(neg / max(pos, 1.0)) if 0 < pos < len(y) else None
    bce = (torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None
          else torch.nn.BCEWithLogitsLoss())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = bce(model(z), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/action_wm_train.jsonl")
    ap.add_argument("--heldout", default="data/action_wm_heldout.jsonl")
    ap.add_argument("--out", default="data/wm_action_weights.json")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    Xtr, ytr, seeds_tr = _load_rows(Path(args.train))
    Xho, yho, seeds_ho = _load_rows(Path(args.heldout))
    assert Xtr.shape[1] == len(FEATURES), (
        f"expected {len(FEATURES)} features (wmfeat_action.FEATURES), got {Xtr.shape[1]}")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ztr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    zho = torch.tensor((Xho - mu) / sd, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    model = _fit(ztr, ytr_t, args.epochs, seed=args.seed)
    with torch.no_grad():
        logits_ho = model(zho).numpy()
        logits_tr = model(ztr).numpy()

    # Platt-recalibrate on the TRAIN band. _fit's pos_weight rebalancing is
    # right for separability but wrong for probability level: with a ~99.5%
    # positive base rate it down-weights the majority class, and the exported
    # head was measured predicting mean ~0.78 where the true rate is ~1.0.
    # PlannedGate compares p_debate against p_skip in a utility — a level
    # bias there silently turns the gate into never-fire, so the artifact
    # must carry calibrated probabilities, not just a good ranking.
    from sklearn.linear_model import LogisticRegression
    platt = LogisticRegression(C=1e6)
    platt.fit(logits_tr.reshape(-1, 1), ytr)
    pa, pb = float(platt.coef_.ravel()[0]), float(platt.intercept_.ravel()[0])
    p_ho = 1.0 / (1.0 + np.exp(-(pa * logits_ho + pb)))
    p_tr = 1.0 / (1.0 + np.exp(-(pa * logits_tr + pb)))

    metrics: dict = {
        "train_rows": int(len(Xtr)), "heldout_rows": int(len(Xho)),
        "train_debate_correct_rate": round(float(ytr.mean()), 4),
        "heldout_debate_correct_rate": round(float(yho.mean()), 4),
    }
    if len(np.unique(yho)) > 1:
        metrics["auroc_heldout"] = round(float(roc_auc_score(yho, p_ho)), 4)
    else:
        # Falsifier-relevant: a degenerate held-out label column (all one
        # class) means AUROC is undefined, not "good" — report honestly.
        metrics["auroc_heldout"] = None
    metrics["ece_heldout"] = round(ece(p_ho, yho), 4)
    if len(np.unique(ytr)) > 1:
        metrics["auroc_train_resub"] = round(float(roc_auc_score(ytr, p_tr)), 4)

    # Leave-one-seed-out on the TRAIN band — robustness check, not the
    # headline number (the headline is the genuinely held-out 150-179 band).
    aucs = []
    for hold in np.unique(seeds_tr):
        tr_mask, te_mask = seeds_tr != hold, seeds_tr == hold
        if te_mask.sum() == 0 or len(np.unique(ytr[te_mask])) < 2:
            continue
        m = _fit(ztr[tr_mask], ytr_t[tr_mask], args.epochs, seed=args.seed)
        with torch.no_grad():
            p_te = torch.sigmoid(m(ztr[te_mask])).numpy()
        aucs.append(roc_auc_score(ytr[te_mask], p_te))
    metrics["loso_auroc_mean_train_band"] = round(float(np.mean(aucs)), 4) if aucs else None
    metrics["loso_n_folds"] = len(aucs)

    sd_ = model.state_dict()
    out = {
        "features": FEATURES, "mu": mu.tolist(), "sd": sd.tolist(),
        "trunk": [[sd_["trunk.0.weight"].tolist(), sd_["trunk.0.bias"].tolist()],
                  [sd_["trunk.2.weight"].tolist(), sd_["trunk.2.bias"].tolist()]],
        "head_debate": [sd_["head_debate.weight"].tolist(), sd_["head_debate.bias"].tolist()],
        "platt": [pa, pb],
        "metrics": metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps(metrics, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
