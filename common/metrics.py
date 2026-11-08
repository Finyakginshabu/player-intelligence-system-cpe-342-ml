"""Exact competition metrics. Use these, not ad-hoc re-implementations, for every CV score.

Per the Kaggle overview page, every classification metric is MACRO-averaged:
Task 1  F2 (Macro): mean of F2 for class 0 and class 1
Task 2  F1 (Macro) over 4 segments
Task 3  NMAE = 1 / (1 + sum|y - yhat| / sum y)   (higher is better)
Task 4  F1 (Macro) over 5 game titles
Task 5  F3 (Macro): mean of F3 for class 0 and class 1
The instruction PDF shows the plain binary formula for F2/F3, so binary scores are kept available
(average="binary") for logging. Optimise the macro version.
Leaderboard = (sum w_i * S_i) / sum w_i. The weights w_i are NOT published.
"""
import numpy as np
from sklearn.metrics import f1_score, fbeta_score


def f2(y_true, y_pred, average: str = "macro") -> float:
    return fbeta_score(y_true, y_pred, beta=2, average=average, zero_division=0)


def f3(y_true, y_pred, average: str = "macro") -> float:
    return fbeta_score(y_true, y_pred, beta=3, average=average, zero_division=0)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def nmae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 1.0 / (1.0 + np.abs(y_true - y_pred).sum() / y_true.sum())


TASK_METRIC = {1: f2, 2: macro_f1, 3: nmae, 4: macro_f1, 5: f3}


def _safe_div(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def fbeta_threshold_curve(y_true, prob, beta: float, average: str = "macro"):
    """F-beta at every distinct threshold (predict positive when prob >= thr).

    Returns (thresholds, scores). Includes a threshold above max(prob) (= all negative).
    average: "macro" (mean of both classes' F-beta, the Kaggle metric) or "binary" (class 1 only).
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    p_s, y_s = p[order], y[order]
    last_of_tie = np.r_[np.diff(p_s) != 0, True]
    tp = np.r_[0, np.cumsum(y_s)[last_of_tie]]
    fp = np.r_[0, np.cumsum(1 - y_s)[last_of_tie]]
    thr = np.r_[np.nextafter(p_s[0], np.inf), p_s[last_of_tie]]
    pos = y.sum()
    fn, tn = pos - tp, (len(y) - pos) - fp
    b2 = beta**2
    f_pos = _safe_div((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp)
    if average == "binary":
        return thr, f_pos
    f_neg = _safe_div((1 + b2) * tn, (1 + b2) * tn + b2 * fp + fn)
    return thr, (f_pos + f_neg) / 2


def best_fbeta_threshold(y_true, prob, beta: float, average: str = "macro") -> tuple[float, float]:
    """Exact F-beta-optimal threshold on OOF probabilities. Predict positive when prob >= thr."""
    thr, score = fbeta_threshold_curve(y_true, prob, beta, average)
    i = int(np.argmax(score))
    return float(thr[i]), float(score[i])
