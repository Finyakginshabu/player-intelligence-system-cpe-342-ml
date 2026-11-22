"""LB family test on top of v41 (2026-11-16): v41's six families + extra families, all at equal weight.

Usage (from task4/src): python blend_v41plus.py [--final] <tag> <family spec> [<family spec> ...]
  family spec = comma-separated run names averaged as one family, e.g. f_dinov2l_s42,f_dinov2l_s7
  special spec "seeds2" = use the 2nd seeds inside v41's M38M and DINOv3-B families (no new family)
  --final = also write outputs/final_pred.csv
  --drop=NAME = remove one of v41's families (e.g. EVA-02-B) before blending, to swap it for a new one
  --prior=A = balanced-prior decision rule: argmax p / pi^A, pi = TRAINING class shares (no test information).
              Our networks are trained with class weights (max/count)^0.5, so they carry a prior of about pi^0.5:
              A = 0.5 removes it completely (equal priors, what macro F1 assumes); A = 0.25 leaves pi^0.25.
Writes outputs/pred_<tag>.csv and submissions/sub_<tag>.csv (= sub_v52, the current tasks 1/2/3/5, with only the task4 column replaced).
Needs only the saved test probabilities (outputs/test_<run>.npy); no data/ folder, no GPU.

FINAL TASK-4 FILE (v47, LB 0.86405) = v43's 8 families + balanced-prior rule A = 2.0:
  python blend_v41plus.py --final --prior=2.0 v47_t4_v43_prior200 seeds2 f_dinov2l_s42,f_dinov2l_s7 f_dinov3l_s42
Previous final (v43, LB 0.85865, no prior):
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
args = [a for a in sys.argv[1:] if a != "--final" and not a.startswith(("--prior=", "--drop=", "--weight=", "--div0="))]
# --div0=K: 7-11k-only rule, divide only the 7-11k probability by K (no boost among the other four classes)
DIV0 = next((float(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--div0=")), 0.0)
DROP = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--drop=")]  # remove a family, e.g. --drop="EVA-02-B"
# --weight=NAME:W gives a family weight W (default 1); NAME = a family name or any run name inside a family spec
WEIGHT = dict(a.split("=", 1)[1].rsplit(":", 1) for a in sys.argv[1:] if a.startswith("--weight="))
tag, specs = args[0], args[1:]
PRIOR_A = next((float(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--prior=")), 0.0)
TRAIN_SHARE = np.array([12429, 4407, 5596, 5926, 3188]) / 31546  # label counts of train.csv (7-11k ... roblock)
fams = {k: list(v) for k, v in FAMILIES.items()}
for s in specs:
    if s == "seeds2":
        fams["EVA-02-L M38M"].append("f_eva02lm38_s7")
        fams["DINOv3-B"].append("f_dinov3b_lr2e4_s7")
    else:
        fams[s] = s.split(",")
for d in DROP:
    fams.pop(d)
w = {k: float(next((v for n, v in WEIGHT.items() if n == k or n in names), 1.0)) for k, names in fams.items()}
unused = [n for n in WEIGHT if not any(n == k or n in names for k, names in fams.items())]
assert not unused, f"--weight names match no family: {unused}"
p = sum(w[k] * np.mean([np.load(OUT / f"test_{n}.npy") for n in names], 0) for k, names in fams.items()) / sum(w.values())
if WEIGHT:
    print("family weights:", {k: v for k, v in w.items() if v != 1.0})
if PRIOR_A:
    p = p / TRAIN_SHARE ** PRIOR_A
    p = p / p.sum(1, keepdims=True)
if DIV0:
    assert not PRIOR_A, "--div0 replaces --prior"
    p = p.copy()
    p[:, 0] /= DIV0
    p = p / p.sum(1, keepdims=True)
pred = p.argmax(1)

v41 = pd.read_csv(OUT / "pred_v41_m38_dino.csv")
print(f"{tag}: {len(fams)} families, rows changed vs v41 {int((pred != v41['prediction'].to_numpy()).sum())}")
pd.DataFrame({"id": v41["id"], "prediction": pred}).to_csv(OUT / f"pred_{tag}.csv", index=False)

sub = pd.read_csv(SUBS / "sub_v52_t3_identity_ratio.csv")  # current tasks 1/2/3/5 (task 3 = identity + ratio model)
assert len(sub) == len(pred) == 25889
sub["task4"] = pred
assert sub.notna().all().all() and sub["task4"].between(0, 4).all()
sub.to_csv(SUBS / f"sub_{tag}.csv", index=False)
print("wrote", SUBS / f"sub_{tag}.csv")
if "--final" in sys.argv:
    np.save(OUT / f"test_blend_{tag}.npy", p)
    utils.write_final_pred(4, test_ids(), pred)
    print("wrote final_pred.csv")
