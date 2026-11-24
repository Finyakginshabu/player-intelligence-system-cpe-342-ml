"""Family test file (2026-11-13): (S x4 + V2-B x3 + NEW) / 3, equal family weights, written as outputs/pred_<tag>.csv.
Usage (GPU venv, from task4/src): python blend_fam.py <tag> <test name> [<test name> ...]   (several names = seed average)"""
import sys

import numpy as np
import pandas as pd

import t4data

OUT = t4data.OUT
tag, names = sys.argv[1], sys.argv[2:]
T = lambda n: np.load(OUT / f"test_{n}.npy")  # noqa: E731
S = np.mean([T(f"f_small_s{s}") for s in (42, 7, 123, 2026)], 0)
B = np.mean([T(f"f_v2base_s{s}") for s in (42, 7, 123)], 0)
new = np.mean([T(n) for n in names], 0)
b = (S + B + new) / 3
v28 = (S + B) / 2
a = lambda p, q: round(float((p.argmax(1) == q.argmax(1)).mean()), 4)  # noqa: E731
print(f"{tag}: new vs S {a(new, S)}, vs V2-B {a(new, B)}, max-prob {new.max(1).mean():.3f}, "
      f"rows changed vs v28 {int((b.argmax(1) != v28.argmax(1)).sum())}")
pd.DataFrame({"id": t4data.load_csv("test")["stem"], "prediction": b.argmax(1)}).to_csv(OUT / f"pred_{tag}.csv", index=False)
