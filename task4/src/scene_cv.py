"""Scene-held-out validation for task 4 (2026-11-12).

Why: random 5-fold CV says 0.98 for the fine-tuned ConvNeXt, the LB says ~0.85. The test set holds scenes that are not
in train (e.g. crowds of characters for fiveN, where train is mostly driving), so a fold that shares scenes with its
training part cannot see the drop. Here train images are clustered into scenes inside each class (k-means on frozen
CLIP embeddings) and whole clusters are held out -> an estimate of accuracy on unseen scenes of the same games.

Groups are written once to outputs/scene_groups.npy and reused by every model so scores are comparable.

Usage (GPU venv): python task4/src/scene_cv.py <embedding name> [more names...]   # linear probe per embedding
"""
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import t4data

SEED = 42
K_PER_CLASS = 10
N_FOLDS = 5
GROUP_SRC = "clipL14"  # embedding used to define the scenes


def feats(name: str, split: str) -> np.ndarray:
    """(N, D): mean over views of the L2-normalised embeddings, re-normalised."""
    e = np.load(t4data.OUT / f"emb_{name}_{split}.npy").astype(np.float32)
    if e.ndim == 3:
        e = e.mean(1)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def scene_groups(k: int = K_PER_CLASS) -> np.ndarray:
    f = t4data.OUT / f"scene_groups_k{k}.npy"
    if f.exists():
        return np.load(f)
    X, y = feats(GROUP_SRC, "train"), t4data.train_labels()
    g = np.zeros(len(y), dtype=int)
    for c in range(5):
        idx = np.where(y == c)[0]
        g[idx] = c * 1000 + KMeans(k, n_init=4, random_state=SEED).fit_predict(X[idx])
    np.save(f, g)
    return g


def scene_folds(k: int = K_PER_CLASS):
    y = t4data.train_labels()
    return list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(y, y, scene_groups(k)))


def hard_folds(k: int = 5):
    """Leave-one-region-out: k clusters per class; fold i holds out the i-th largest cluster of every class, so a whole
    contiguous region of each class is unseen (closer to the test set's unseen videos than scene_folds)."""
    y, g = t4data.train_labels(), scene_groups(k)
    rank = np.zeros(len(y), dtype=int)
    for c in range(5):
        ids, cnt = np.unique(g[y == c], return_counts=True)
        order = {gid: r for r, gid in enumerate(ids[np.argsort(-cnt)])}
        rank[y == c] = [order[v] for v in g[y == c]]
    return [(np.where(rank != i)[0], np.where(rank == i)[0]) for i in range(k)]


def random_folds():
    y = t4data.train_labels()
    return list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(y, y, np.arange(len(y))))


def fit_lr(X, y, Xv, l2=1e-4, steps=300, balanced=False):
    """Multinomial logistic regression on GPU (L-BFGS). Returns validation probabilities."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, Xv = torch.tensor(X, device=dev), torch.tensor(Xv, device=dev)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    X, Xv = (X - mu) / sd, (Xv - mu) / sd
    yt = torch.tensor(y, device=dev)
    w = None
    if balanced:
        cnt = torch.bincount(yt, minlength=5).float()
        w = cnt.sum() / (5 * cnt)
    W = torch.zeros(X.shape[1], 5, device=dev, requires_grad=True)
    b = torch.zeros(5, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=steps, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, yt, weight=w) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.softmax(Xv @ W + b, 1).cpu().numpy()


def cv_probe(X, y, folds, **kw):
    oof = np.zeros((len(y), 5), dtype=np.float32)
    for tr, va in folds:
        oof[va] = fit_lr(X[tr], y[tr], X[va], **kw)
    fs = [f1_score(y[va], oof[va].argmax(1), average="macro") for _, va in folds]
    return oof, float(np.mean(fs)), float(np.std(fs)), fs


def report(tag, y, oof, folds):
    fs = [f1_score(y[va], oof[va].argmax(1), average="macro") for _, va in folds]
    rec = [round(float((oof[y == c].argmax(1) == c).mean()), 3) for c in range(5)]
    print(f"{tag:46s} macro F1 {np.mean(fs):.4f} +- {np.std(fs):.4f}  folds {np.round(fs, 3).tolist()}  recall {rec}", flush=True)


if __name__ == "__main__":
    y = t4data.train_labels()
    g = scene_groups()
    print("scene groups:", len(np.unique(g)), "| sizes per class (min/median/max):",
          {c: (int(np.bincount(g[y == c] - c * 1000).min()), int(np.median(np.bincount(g[y == c] - c * 1000))),
               int(np.bincount(g[y == c] - c * 1000).max())) for c in range(5)})
    rf, sf = random_folds(), scene_folds()
    for name in sys.argv[1:]:
        X = feats(name, "train")
        for l2 in (1e-5, 1e-4, 1e-3, 1e-2):
            oof, *_ = cv_probe(X, y, rf, l2=l2)
            report(f"{name} LR l2={l2:g} RANDOM folds", y, oof, rf)
            oof, *_ = cv_probe(X, y, sf, l2=l2)
            report(f"{name} LR l2={l2:g} SCENE folds", y, oof, sf)
        oof, *_ = cv_probe(X, y, sf, l2=1e-3, balanced=True)
        report(f"{name} LR l2=0.001 balanced SCENE folds", y, oof, sf)
        zs = np.load(t4data.OUT / f"zs_{name}_train.npy")
        report(f"{name} zero-shot (4 views), no training", y, zs.mean(1), sf)
