"""Sanity checks for shared metrics and submission validation. Run: python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score

from common import metrics
from make_submission import N_ROWS, SubmissionError, validate


def test_nmae_perfect_and_known_value():
    y = np.array([0, 10, 30.0])
    assert metrics.nmae(y, y) == 1.0
    # sum|err| = 20, sum y = 40 -> 1 / 1.5
    assert metrics.nmae(y, [0, 0, 20]) == pytest.approx(1 / 1.5)


def test_fbeta_matches_sklearn():
    rng = np.random.default_rng(42)
    y, p = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    assert metrics.f2(y, p) == pytest.approx(fbeta_score(y, p, beta=2, average="macro"))
    assert metrics.f3(y, p) == pytest.approx(fbeta_score(y, p, beta=3, average="macro"))
    assert metrics.f2(y, p, average="binary") == pytest.approx(fbeta_score(y, p, beta=2))


@pytest.mark.parametrize("beta,average", [(2, "macro"), (3, "macro"), (2, "binary")])
def test_threshold_curve_is_exact_and_optimal(beta, average):
    rng = np.random.default_rng(42)
    y = (rng.random(3000) < 0.15).astype(int)  # imbalanced, like task 1
    prob = np.round(np.clip(y * 0.3 + rng.random(3000) * 0.7, 0, 1), 3)  # rounding creates ties
    thr, scores = metrics.fbeta_threshold_curve(y, prob, beta, average)
    for i in rng.choice(len(thr), 25, replace=False):
        sk = fbeta_score(y, (prob >= thr[i]).astype(int), beta=beta, average=average, zero_division=0)
        assert scores[i] == pytest.approx(sk)
    t, best = metrics.best_fbeta_threshold(y, prob, beta, average)
    grid = max(fbeta_score(y, (prob >= g).astype(int), beta=beta, average=average, zero_division=0)
               for g in np.linspace(0, 1, 201))
    assert best >= grid - 1e-12


def _valid_sub():
    ids = [f"ANS{i:05d}" for i in range(1, N_ROWS + 1)]
    sub = pd.DataFrame({"id": ids, "task1": 0, "task2": 1, "task3": 0.0, "task4": 4, "task5": 1})
    return sub, sub.copy()


def test_validate_accepts_valid():
    sub, sample = _valid_sub()
    validate(sub, sample)


@pytest.mark.parametrize("col,val", [("task1", 2), ("task2", 4), ("task3", -1.0),
                                     ("task4", 5), ("task5", np.nan)])
def test_validate_rejects_bad_values(col, val):
    sub, sample = _valid_sub()
    sub.loc[5, col] = val
    with pytest.raises(SubmissionError):
        validate(sub, sample)


def test_build_maps_by_test_row_order(tmp_path, monkeypatch):
    """final_pred rows in any order must land on ANS ids by their position in test.csv."""
    import make_submission as ms

    rng = np.random.default_rng(42)
    sample, _ = _valid_sub()
    for n in range(1, 6):
        pids = [f"P{x:06d}" for x in rng.permutation(10 * N_ROWS)[:N_ROWS]]
        (tmp_path / f"data{n}").mkdir()
        pd.DataFrame({"player_id": pids}).to_csv(tmp_path / f"data{n}" / "test.csv", index=False)
        pred = np.arange(N_ROWS) % 2  # value at position k is k % 2
        if n == 3:
            pred = np.arange(N_ROWS, dtype=float)
        fp = pd.DataFrame({"id": pids, "prediction": pred}).sample(frac=1, random_state=42)
        (tmp_path / f"task{n}" / "outputs").mkdir(parents=True)
        fp.to_csv(tmp_path / f"task{n}" / "outputs" / "final_pred.csv", index=False)
    monkeypatch.setattr(ms, "task_data", lambda n: tmp_path / f"data{n}")
    monkeypatch.setattr(ms, "task_dir", lambda n: tmp_path / f"task{n}")

    sub = ms.build(sample)
    assert (sub["task1"].values == np.arange(N_ROWS) % 2).all()
    assert (sub["task3"].values == np.arange(N_ROWS)).all()


def test_build_from_base_replaces_only_named_task(tmp_path, monkeypatch):
    import make_submission as ms

    sample, base = _valid_sub()
    pids = [f"P{x:06d}" for x in range(N_ROWS)]
    (tmp_path / "data2").mkdir()
    pd.DataFrame({"player_id": pids}).to_csv(tmp_path / "data2" / "test.csv", index=False)
    (tmp_path / "task2" / "outputs").mkdir(parents=True)
    pd.DataFrame({"id": pids, "prediction": np.arange(N_ROWS) % 4}).to_csv(
        tmp_path / "task2" / "outputs" / "final_pred.csv", index=False)
    monkeypatch.setattr(ms, "task_data", lambda n: tmp_path / f"data{n}")
    monkeypatch.setattr(ms, "task_dir", lambda n: tmp_path / f"task{n}")

    sub = ms.build_from_base(sample, base, ["task2"])
    assert (sub["task2"].values == np.arange(N_ROWS) % 4).all()
    for col in ("id", "task1", "task3", "task4", "task5"):
        assert sub[col].equals(base[col])


def test_build_from_base_rejects_invalid_base():
    import make_submission as ms

    sample, base = _valid_sub()
    base.loc[0, "task1"] = 7
    with pytest.raises(SubmissionError):
        ms.build_from_base(sample, base, ["task2"])


def test_validate_rejects_wrong_order():
    sub, sample = _valid_sub()
    sub = sub.iloc[::-1].reset_index(drop=True)
    with pytest.raises(SubmissionError):
        validate(sub, sample)
