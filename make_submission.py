"""Merge taskN/outputs/final_pred.csv into one submission and validate.

ID mapping (see README.md): row k of every task's test.csv is submission row ANS{k:05d}.
final_pred.csv is keyed by the task's own test id (player_id, or id for task 4); this script
checks the ids match test.csv exactly and puts them in test.csv order before assigning ANS ids.

Usage:
    python make_submission.py                      # all five tasks required
    python make_submission.py --fill-missing       # unfinished tasks get a constant 0
    python make_submission.py --out submissions/sub_v3.csv
    # One task per submission: start from a scored file, swap in only the named task's final_pred.csv
    python make_submission.py --base submissions/sub_v4_task5_rate15.csv --replace task2 --out submissions/sub_v6_task2.csv
    python make_submission.py --base submissions/sub_v4_task5_rate15.csv \
        --replace task5=task5/outputs/pred_typesfirst_rate15.csv --out submissions/sub_v6_task5_typesfirst15.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from common.paths import ROOT, SAMPLE_SUBMISSION, task_data, task_dir

N_ROWS = 25_889
TASKS = range(1, 6)
VALID = {"task1": {0, 1}, "task2": {0, 1, 2, 3}, "task4": {0, 1, 2, 3, 4}, "task5": {0, 1}}
FILL_VALUE = 0  # valid for every task; only used with --fill-missing


class SubmissionError(Exception):
    pass


def test_ids(n: int) -> pd.Series:
    """The task's test ids in file order. Position k maps to ANS{k+1:05d}."""
    path = task_data(n) / "test.csv"
    if not path.exists():
        raise SubmissionError(f"[task{n}] {path} not found")
    df = pd.read_csv(path, dtype=str)
    if "file_name" in df.columns:
        # Task 4 after the 2026-11-11 data fix: test.csv = id (ANS#####), file_name ({image id}.jpg), label.
        # Predictions are keyed by image id, so map through file_name and check id is the ANS sequence.
        expected = [f"ANS{k + 1:05d}" for k in range(len(df))]
        if df["id"].tolist() != expected:
            raise SubmissionError(f"[task{n}] test.csv id column is not ANS00001..ANS{len(df):05d} in order")
        ids = df["file_name"].str.removesuffix(".jpg")
    else:
        ids = df["player_id" if "player_id" in df.columns else "id"]
    if len(ids) != N_ROWS or ids.duplicated().any():
        raise SubmissionError(f"[task{n}] test.csv must have {N_ROWS} unique ids, got {len(ids)} ({ids.nunique()} unique)")
    return ids.reset_index(drop=True)


def load_task(n: int, path=None) -> pd.Series | None:
    """Predictions of task n in test.csv order. path defaults to taskN/outputs/final_pred.csv."""
    path = task_dir(n) / "outputs" / "final_pred.csv" if path is None else path
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"id": str})
    if list(df.columns) != ["id", "prediction"]:
        raise SubmissionError(f"[task{n}] columns must be ['id','prediction'], got {list(df.columns)}")
    if df["id"].duplicated().any():
        raise SubmissionError(f"[task{n}] duplicate ids")
    ids = test_ids(n)
    missing, extra = set(ids) - set(df["id"]), set(df["id"]) - set(ids)
    if missing or extra:
        raise SubmissionError(
            f"[task{n}] ids differ from test.csv: {len(missing)} missing, {len(extra)} extra "
            f"(e.g. {sorted(missing)[:3]} / {sorted(extra)[:3]})")
    return df.set_index("id")["prediction"].reindex(ids).reset_index(drop=True)


def validate(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    errs = []
    if len(sub) != N_ROWS:
        errs.append(f"expected {N_ROWS} rows, got {len(sub)}")
    if list(sub.columns) != list(sample.columns):
        errs.append(f"columns {list(sub.columns)} != {list(sample.columns)}")
    if not sub["id"].equals(sample["id"]):
        errs.append("id order differs from sample_submission.csv")
    na = sub.isna().sum()
    if na.any():
        errs.append(f"NaN found: {na[na > 0].to_dict()}")
    for col, allowed in VALID.items():
        if not sub[col].isin(allowed).all():
            errs.append(f"{col} must be in {sorted(allowed)}")
    if not (pd.to_numeric(sub["task3"], errors="coerce") >= 0).all():
        errs.append("task3 must be numeric and >= 0")
    if errs:
        raise SubmissionError("VALIDATION FAILED:\n  - " + "\n  - ".join(errs))


def build(sample: pd.DataFrame, fill_missing: bool = False) -> pd.DataFrame:
    sub = pd.DataFrame({"id": sample["id"]})
    for n in TASKS:
        col = f"task{n}"
        pred = load_task(n)
        if pred is None:
            if not fill_missing:
                raise SubmissionError(f"[{col}] missing outputs/final_pred.csv (use --fill-missing for a placeholder)")
            print(f"WARNING: {col} has no final_pred.csv -> filled with {FILL_VALUE}")
            pred = pd.Series(FILL_VALUE, index=sample.index)
        sub[col] = pred.values
    return _finalize(sub, sample)


def _finalize(sub: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    for col in VALID:
        num = pd.to_numeric(sub[col], errors="coerce")
        if num.notna().all() and (num % 1 == 0).all():
            sub[col] = num.astype(int)
    validate(sub, sample)
    return sub


def parse_replace(item: str) -> tuple[str, Path | None]:
    """'task2' -> ('task2', None) = final_pred.csv; 'task5=path.csv' -> ('task5', Path) = a candidate id,prediction file."""
    col, _, path = item.partition("=")
    if col not in {f"task{n}" for n in TASKS}:
        raise SubmissionError(f"--replace expects task1..task5[=path], got {item!r}")
    return col, (Path(path) if path else None)


def build_from_base(sample: pd.DataFrame, base: pd.DataFrame, replace: list[str]) -> pd.DataFrame:
    """Copy an already-scored submission and overwrite only the `replace` columns (taskN or taskN=path)."""
    try:
        validate(base, sample)
    except SubmissionError as e:
        raise SubmissionError(f"base file is invalid: {e}") from e
    sub = base.copy()
    for item in replace:
        col, path = parse_replace(item)
        pred = load_task(int(col[-1]), path)
        if pred is None:
            raise SubmissionError(f"[{col}] prediction file not found ({path or 'outputs/final_pred.csv'})")
        sub[col] = pred.values
    return _finalize(sub, sample)


def diff_summary(sub: pd.DataFrame, base: pd.DataFrame) -> str:
    lines = []
    for col in sub.columns[1:]:
        if col == "task3":
            d = (pd.to_numeric(sub[col]) - pd.to_numeric(base[col])).abs()
            lines.append(f"  {col}: {(d > 0.005).sum():>6} rows changed (mean |diff| {d.mean():.1f})")
        else:
            lines.append(f"  {col}: {(sub[col].values != base[col].values).sum():>6} rows changed")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "submission.csv"))
    ap.add_argument("--fill-missing", action="store_true")
    ap.add_argument("--base", help="already-scored submission to start from (keeps every column not in --replace)")
    ap.add_argument("--replace", nargs="+", metavar="taskN[=path]",
                    help="with --base: columns to overwrite, from taskN/outputs/final_pred.csv or from an explicit "
                         "id,prediction file (e.g. task5=task5/outputs/pred_typesfirst_rate15.csv)")
    args = ap.parse_args()
    if bool(args.base) != bool(args.replace):
        ap.error("--base and --replace must be used together")

    if not SAMPLE_SUBMISSION.exists():
        sys.exit(f"{SAMPLE_SUBMISSION} not found. Put the data in data/ (or set CPE342_DATA).")
    sample = pd.read_csv(SAMPLE_SUBMISSION, dtype={"id": str})
    try:
        if args.base:
            base = pd.read_csv(args.base, dtype={"id": str})
            sub = build_from_base(sample, base, args.replace)
        else:
            sub = build(sample, args.fill_missing)
    except SubmissionError as e:
        sys.exit(str(e))
    sub.to_csv(args.out, index=False)
    if args.base:
        print(f"changes vs {args.base}:\n{diff_summary(sub, base)}")
        if len(args.replace) > 1:
            print("WARNING: more than one task replaced -> the LB delta can't be attributed to one task")
    print(f"OK: wrote {args.out} ({len(sub)} rows). Log it in submissions/log.md after submitting.")


if __name__ == "__main__":
    main()
