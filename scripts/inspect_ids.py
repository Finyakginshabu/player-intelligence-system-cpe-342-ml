"""Re-check the assumptions behind the positional id mapping. Read-only; trains nothing.

Run: python scripts/inspect_ids.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import SAMPLE_SUBMISSION, task_data  # noqa: E402

sample = pd.read_csv(SAMPLE_SUBMISSION, dtype={"id": str})
print(f"sample_submission: {sample.shape}, ids {sample.id.iloc[0]}..{sample.id.iloc[-1]}, "
      f"sorted={sample.id.is_monotonic_increasing}, non-NaN per task={sample.notna().sum().iloc[1:].to_dict()}")

tests = {}
for n in range(1, 6):
    p = task_data(n) / "test.csv"
    if not p.exists():
        print(f"task{n}: {p} missing")
        continue
    df = pd.read_csv(p, dtype=str)
    col = "player_id" if "player_id" in df.columns else "id"
    ids = df[col]
    tests[n] = set(ids)
    print(f"task{n}: rows={len(df)} id_col={col} unique={ids.nunique()} "
          f"same_len_as_sample={len(df) == len(sample)} ids_in_sample={ids.isin(sample.id).mean():.3f}")

ns = sorted(tests)
for i, a in enumerate(ns):
    for b in ns[i + 1:]:
        print(f"  player overlap task{a} vs task{b}: {len(tests[a] & tests[b])}")
