"""LB family test on top of v41 (2026-11-16): v41's six families + extra families, all at equal weight.

Usage (from task4/src): python blend_v41plus.py [--final] <tag> <family spec> [<family spec> ...]
  family spec = comma-separated run names averaged as one family, e.g. f_dinov2l_s42,f_dinov2l_s7
  special spec "seeds2" = use the 2nd seeds inside v41's M38M and DINOv3-B families (no new family)
  --final = also write outputs/final_pred.csv
Writes outputs/pred_<tag>.csv and submissions/sub_<tag>.csv (= sub_v41 with only the task4 column replaced).
Needs only the saved test probabilities (outputs/test_<run>.npy); no data/ folder, no GPU.

FINAL TASK-4 FILE (v43, LB 0.85865):
  python blend_v41plus.py --final v43_t4_seeds_d2l_d3l seeds2 f_dinov2l_s42,f_dinov2l_s7 f_dinov3l_s42
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import t4data
from blend_v41 import FAMILIES, test_ids

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import utils  # noqa: E402

OUT = t4data.OUT
SUBS = t4data.ROOT / "submissions"
args = [a for a in sys.argv[1:] if a != "--final"]
tag, specs = args[0], args[1:]
fams = {k: list(v) for k, v in FAMILIES.items()}
for s in specs:
    if s == "seeds2":
        fams["EVA-02-L M38M"].append("f_eva02lm38_s7")
        fams["DINOv3-B"].append("f_dinov3b_lr2e4_s7")
    else:
        fams[s] = s.split(",")
p = np.mean([np.mean([np.load(OUT / f"test_{n}.npy") for n in names], 0) for names in fams.values()], 0)
pred = p.argmax(1)

v41 = pd.read_csv(OUT / "pred_v41_m38_dino.csv")
print(f"{tag}: {len(fams)} families, rows changed vs v41 {int((pred != v41['prediction'].to_numpy()).sum())}")
pd.DataFrame({"id": v41["id"], "prediction": pred}).to_csv(OUT / f"pred_{tag}.csv", index=False)

sub = pd.read_csv(SUBS / "sub_v41_t4_m38_dino.csv")
assert len(sub) == len(pred) == 25889
sub["task4"] = pred
assert sub.notna().all().all() and sub["task4"].between(0, 4).all()
sub.to_csv(SUBS / f"sub_{tag}.csv", index=False)
print("wrote", SUBS / f"sub_{tag}.csv")
if "--final" in sys.argv:
    np.save(OUT / f"test_blend_{tag}.npy", p)
    utils.write_final_pred(4, test_ids(), pred)
    print("wrote final_pred.csv")
