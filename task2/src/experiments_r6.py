"""Round 6 (2026-11-11): final polish candidates, no player_id.

Task 2: per-class log-offsets on the SmoothNB probabilities (nested CV check, then fit on all OOF, apply to the
fold-averaged test probabilities). Task 3: write the saved 3-seed + CatBoost-gate two-stage test predictions
(round 5, CV 0.7884) as an id,prediction file. Neither file replaces final_pred.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from common import paths, utils

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_r4 as r4  # noqa: E402  (module level only loads data and folds)

oof, tp = utils.load_preds(2, "snb")
base = r4.metrics.macro_f1(r4.y, oof.argmax(1))
nested, b = r4.nested_offsets(oof)
print(f"task 2 SmoothNB argmax {base:.4f} | nested offsets {nested:.4f} | full-OOF offsets {np.round(b, 3).tolist()}")
pred2 = (np.log(np.clip(tp, 1e-12, 1)) + b).argmax(1)
ids2 = r4.test["player_id"].values
pd.DataFrame({"id": ids2, "prediction": pred2}).to_csv(paths.outputs_dir(2) / "pred_snb_offsets.csv", index=False)
final2 = pd.read_csv(paths.outputs_dir(2) / "final_pred.csv")
print(f"  test rows changed vs final_pred.csv: {int((final2.prediction.values != pred2).sum())}; "
      f"class mix {np.round(np.bincount(pred2, minlength=4) / len(pred2), 3).tolist()}")

test3 = pd.read_csv(paths.task_data(3) / "test.csv")
t3 = np.load(paths.outputs_dir(3) / "test_twostage_ms_cat.npy")
pd.DataFrame({"id": test3["player_id"].values, "prediction": np.clip(t3, 0, None)}).to_csv(
    paths.outputs_dir(3) / "pred_twostage_ms_cat.csv", index=False)
final3 = pd.read_csv(paths.outputs_dir(3) / "final_pred.csv")
print(f"task 3: zeros {np.mean(t3 == 0):.3f}, mean |diff| vs final_pred.csv {np.abs(t3 - final3.prediction.values).mean():.1f}, "
      f"rows changed (>1 THB) {int((np.abs(t3 - final3.prediction.values) > 1).sum())}")
