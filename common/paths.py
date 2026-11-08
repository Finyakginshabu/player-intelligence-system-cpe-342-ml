"""Single source of truth for file locations.

Actual data layout (Google Drive zip "cpe342-kaggle-main", unpacked into data/):
    data/sample_submission.csv
    data/01_binary_classification/{train,test}.csv, datacard.txt
    data/02_multiclass_classification/...
    data/03_regression/...
    data/04_image_classification/datacard.txt   (+ data/train/*.jpg, data/test/*.jpg, csvs: separate download)
    data/05_anomaly_detection/test.csv, datacard.txt   (no train)
A plain data/taskN/ layout (Kaggle CLI) is also accepted. On Colab/Kaggle kernels set CPE342_DATA
to the folder holding sample_submission.csv.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _find_data_dir() -> Path:
    env = os.environ.get("CPE342_DATA")
    if env:
        return Path(env)
    base = ROOT / "data"
    if (base / "sample_submission.csv").exists():
        return base
    hits = sorted(base.glob("*/sample_submission.csv"))
    return hits[0].parent if hits else base


DATA_DIR = _find_data_dir()
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
SUBMISSIONS_DIR = ROOT / "submissions"


def task_data(task: int) -> Path:
    """Raw data folder for a task: data/0N_<name>/ (or data/taskN/)."""
    hits = sorted(DATA_DIR.glob(f"0{task}_*"))
    return hits[0] if hits else DATA_DIR / f"task{task}"


def datacard(task: int) -> str:
    d = task_data(task)
    for name in ("datacard.txt", "datacard.csv"):
        if (d / name).exists():
            return (d / name).read_text(encoding="utf-8")
    raise FileNotFoundError(f"no datacard in {d}")


def task4_images(split: str) -> Path:
    """Folder of task 4 JPEGs ({id}.jpg) for split 'train' or 'test'."""
    return task_data(4) / "data" / split


def task_dir(task: int) -> Path:
    """Code folder for a task, e.g. task1/."""
    return ROOT / f"task{task}"


def outputs_dir(task: int) -> Path:
    d = task_dir(task) / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d
