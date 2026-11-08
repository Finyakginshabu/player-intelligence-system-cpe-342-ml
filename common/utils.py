"""Seeding and the prediction-file conventions in the project README."""
import os
import random

import numpy as np
import pandas as pd

from common.paths import outputs_dir

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_preds(task: int, name: str, oof: np.ndarray, test: np.ndarray) -> None:
    """Save OOF and test predictions as taskN/outputs/oof_<name>.npy and test_<name>.npy."""
    out = outputs_dir(task)
    np.save(out / f"oof_{name}.npy", oof)
    np.save(out / f"test_{name}.npy", test)


def load_preds(task: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    out = outputs_dir(task)
    return np.load(out / f"oof_{name}.npy"), np.load(out / f"test_{name}.npy")


def write_final_pred(task: int, ids, preds) -> pd.DataFrame:
    """Write taskN/outputs/final_pred.csv keyed by the task's own test ids (player_id / id).

    make_submission.py maps them to ANS##### by test.csv row order.
    """
    df = pd.DataFrame({"id": list(ids), "prediction": list(preds)})
    if df["id"].duplicated().any():
        raise ValueError("duplicate ids in final predictions")
    df.to_csv(outputs_dir(task) / "final_pred.csv", index=False)
    return df
