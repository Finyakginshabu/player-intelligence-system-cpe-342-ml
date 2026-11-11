"""Nested-CV ensembling of saved OOF/test prediction pairs (the /ensemble command).

Every blend choice (hill-climbing weights, stacker, threshold) is made on 4 folds' OOF and scored on the 5th, so the
reported score is honest. Supported tasks: 1 (binary F2, tuned threshold) and 2 (macro F1, argmax).
Usage from repo root: python -m common.ensemble 1 lgbm lgbm_sig_5seed cat_sig_5seed ...
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import metrics, utils


def _score_fn(task):
    if task == 1:
        def fit_post(y, p):  # threshold tuned on the given rows
            return metrics.best_fbeta_threshold(y, p, 2, average="binary")[0]

        def apply(y, p, t):
            return metrics.f2(y, (p >= t).astype(int), average="binary")
    elif task == 2:
        def fit_post(y, p):
            return None

        def apply(y, p, t):
            return metrics.macro_f1(y, p.argmax(1))
    else:
        raise ValueError("task must be 1 or 2")
    return fit_post, apply


def hill_climb(P, y, idx, fit_post, apply, n_steps=25):
    """Greedy forward selection with replacement on rows idx. Returns integer counts per model."""
    counts = np.zeros(len(P), int)
    cur = None
    best_score = -np.inf
    for _ in range(n_steps):
        cand = []
        for j, p in enumerate(P):
            q = p[idx] if cur is None else (cur * counts.sum() + p[idx]) / (counts.sum() + 1)
            t = fit_post(y[idx], q)
            cand.append(apply(y[idx], q, t))
        j = int(np.argmax(cand))
        if cand[j] <= best_score + 1e-7:
            break
        best_score = cand[j]
        cur = P[j][idx] if cur is None else (cur * counts.sum() + P[j][idx]) / (counts.sum() + 1)
        counts[j] += 1
    return counts


def run(task, names, folds, y):
    fit_post, apply = _score_fn(task)
    P = [utils.load_preds(task, n)[0] for n in names]
    T = [utils.load_preds(task, n)[1] for n in names]

    def nested(make_pred):
        fs = []
        for k, (_, va) in enumerate(folds):
            inner = np.concatenate([folds[j][1] for j in range(len(folds)) if j != k])
            p_in, p_va = make_pred(inner, va)
            t = fit_post(y[inner], p_in)
            fs.append(apply(y[va], p_va, t))
        return np.mean(fs), np.std(fs), np.round(fs, 4).tolist()

    rows = []
    for n, p in zip(names, P):
        m, s, f = nested(lambda i, v, p=p: (p[i], p[v]))
        rows.append(dict(model=n, nested=round(m, 4), std=round(s, 4)))
    single = pd.DataFrame(rows).sort_values("nested", ascending=False)
    print(single.to_string(index=False))
    flat = [p if p.ndim == 1 else p.reshape(len(p), -1)[:, -1] for p in P]  # task 2: correlate the Whale column
    print("\nOOF correlation" + (" (class-3 probability):" if task == 2 else ":"))
    print(pd.DataFrame(np.corrcoef(flat), index=names, columns=names).round(3).to_string())

    mean_res = nested(lambda i, v: (np.mean([p[i] for p in P], 0), np.mean([p[v] for p in P], 0)))
    print(f"\nsimple mean          nested {mean_res[0]:.4f} ± {mean_res[1]:.4f} {mean_res[2]}")

    chosen = []

    def hc(i, v):
        c = hill_climb(P, y, i, fit_post, apply)
        chosen.append(c.tolist())
        w = c / c.sum()
        return sum(wi * p[i] for wi, p in zip(w, P)), sum(wi * p[v] for wi, p in zip(w, P))

    hc_res = nested(hc)
    print(f"hill climbing        nested {hc_res[0]:.4f} ± {hc_res[1]:.4f} {hc_res[2]} | counts per fold {chosen}")

    def feats(idx):
        if task == 1:
            return np.column_stack([np.log(np.clip(p[idx], 1e-6, 1 - 1e-6) / (1 - np.clip(p[idx], 1e-6, 1 - 1e-6))) for p in P])
        return np.column_stack([np.log(np.clip(p[idx], 1e-9, 1)) for p in P])

    def stack(i, v):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(feats(i), y[i])
        return m.predict_proba(feats(i))[:, 1] if task == 1 else m.predict_proba(feats(i)), \
            m.predict_proba(feats(v))[:, 1] if task == 1 else m.predict_proba(feats(v))

    st_res = nested(stack)
    print(f"LR stacker           nested {st_res[0]:.4f} ± {st_res[1]:.4f} {st_res[2]}")
    best_single = single.iloc[0]
    print(f"\nbest single: {best_single.model} {best_single.nested:.4f} ± {best_single['std']:.4f}")
    for name, r in (("simple mean", mean_res), ("hill climbing", hc_res), ("LR stacker", st_res)):
        gain = r[0] - best_single.nested
        print(f"  {name:14s} gain {gain:+.4f} -> {'KEEP' if gain > best_single['std'] else 'no (<= fold std)'}")
    return single, mean_res, hc_res, st_res


if __name__ == "__main__":
    from sklearn.model_selection import StratifiedKFold

    from common import paths

    task, names = int(sys.argv[1]), sys.argv[2:]
    if task == 1:
        tr = pd.read_csv(paths.task_data(1) / "train.csv")
        y = tr.loc[tr.is_cheater.notna(), "is_cheater"].astype(int).to_numpy()
    else:
        y = pd.read_csv(paths.task_data(2) / "train.csv")["segment"].astype(int).to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=utils.SEED).split(np.zeros(len(y)), y))
    run(task, names, folds, y)
