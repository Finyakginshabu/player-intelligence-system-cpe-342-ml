"""Build the Kaggle kit for the team's task-4 runs: one self-contained private notebook per team member's own account.

Each notebook embeds the training code, the two csv files and the fold files, downloads the official task-4 images, runs
its job list (one queue per GPU), pauses before the session limit and packs the results into results_account<N>.zip.
Test images are only passed through the trained networks (project rule 2).

Usage (from task4/src):  python build_cloud_kit.py [--out DIR]
Test build:              python build_cloud_kit.py --out DIR --data DIR --outputs DIR --smoke
"""
import argparse
import base64
import hashlib
import io
import tarfile
from pathlib import Path

import nbformat
import pandas as pd
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]
IMAGES_DRIVE_ID = "1C8csylbBo3Oab24ODGKPVom3_SDUJ_Qw"  # official task-4 images (link on the Kaggle data page)
CSV_ORDER_MD5 = {"train": "240607c1609b16eb381fc3bfc5469258", "test": "557644b7fd75cd59bf72a8bb4bd06fcd"}

# Recipes = task4/RECIPE.md section 3; every run also gets COMMON.
COMMON = "--grad-ckpt --ckpt"
LA = "--balance 0 --la-tau 1"
LP = "--lp-epochs 2"
S = "--backbone convnext_small.fb_in22k_ft_in1k --epochs 6 --batch 48 --lr 2e-4 --drop-path 0.2"
V2B = "--backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 1e-4 --drop-path 0.2"
EVAB = ("--backbone eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --epochs 6 --batch 32 --lr 5e-5 "
        "--layer-decay 0.75 --drop-path 0.1")
EVAL = ("--backbone eva02_large_patch14_448.mim_in22k_ft_in22k_in1k --epochs 6 --batch 16 --lr 3e-5 "
        "--layer-decay 0.85 --drop-path 0.2")
M38 = ("--backbone eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --epochs 6 --batch 16 --lr 3e-5 "
       "--layer-decay 0.85 --drop-path 0.2")
D3B = ("--backbone vit_base_patch16_dinov3.lvd1689m --epochs 6 --batch 32 --lr 2e-4 "
       "--layer-decay 0.75 --drop-path 0.1")
D2L = ("--backbone vit_large_patch14_reg4_dinov2.lvd142m --epochs 6 --batch 16 --lr 1e-4 "
       "--layer-decay 0.85 --drop-path 0.2")
D3L = ("--backbone vit_large_patch16_dinov3.lvd1689m --epochs 6 --batch 16 --lr 1e-4 "
       "--layer-decay 0.85 --drop-path 0.2")


def held(fold, recipe, extra=""):
    return f"--cv hard --folds {fold} --no-test --seed 42 {recipe} {extra} {COMMON}".replace("  ", " ")


def full(recipe, extra="", seed=42):
    return f"--cv full --seed {seed} --save-weights {recipe} {extra} {COMMON}".replace("  ", " ")


# Round 2 recipes (2026-11-19): self-supervised ViT-L near the native 120x230 frame size, long schedule, strong HUD dropout.
D3L_N = ("--backbone vit_large_patch16_dinov3.lvd1689m --h 144 --w 272 --epochs 18 --batch 64 --lr 8e-5 "
         "--layer-decay 0.90 --drop-path 0.2 --hud 0.7")
D2L_N = ("--backbone vit_large_patch14_reg4_dinov2.lvd142m --h 126 --w 238 --epochs 15 --batch 64 --lr 3e-5 "
         "--layer-decay 0.85 --drop-path 0.2 --hud 0.7")

# Round 3 recipes (2026-11-21): the round-2 recipe (LB v55 +0.00045) on the EVA-02-L families, the strongest on the LB.
# 126x238 = 9x17 patches of 14 px; CPU check: the 448 checkpoints resample to 154 position tokens, only the head is new.
# LP = 2 head-only epochs first (LP-FT: +0.006 held-out for EVA-02-B).
EVAL_N = ("--backbone eva02_large_patch14_448.mim_in22k_ft_in22k_in1k --h 126 --w 238 --epochs 15 --batch 64 --lr 5e-5 "
          "--layer-decay 0.85 --drop-path 0.2 --hud 0.7")
M38_N = ("--backbone eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --h 126 --w 238 --epochs 15 --batch 64 --lr 5e-5 "
         "--layer-decay 0.85 --drop-path 0.2 --hud 0.7")

# Round 4 recipes (2026-11-22): the near-native DINO recipe (best held-out singles ever, best LB test profile) on the two
# largest DINO backbones that fit a 16 GB T4: only the last 12 blocks are trained (`--freeze-blocks`, CPU-tested), so the
# optimizer state stays small; 12 epochs so a job fits one session (≈ 1,800-1,900 s per epoch estimated on a T4).
D2G_N = ("--backbone vit_giant_patch14_reg4_dinov2.lvd142m --h 126 --w 238 --epochs 12 --batch 64 --lr 3e-5 "
         "--layer-decay 0.90 --drop-path 0.2 --hud 0.7 --freeze-blocks 28")
D3HP_N = ("--backbone vit_huge_plus_patch16_dinov3.lvd1689m --h 144 --w 272 --epochs 12 --batch 64 --lr 6e-5 "
          "--layer-decay 0.90 --drop-path 0.2 --hud 0.7 --freeze-blocks 20")

# start=False: the account waits until the team says go.
ACCOUNTS = {
    1: {"title": "DINOv2-g (1.1B), near-native recipe, all training images (2 seeds)", "start": True, "gpus": [
        {"f_dinov2g_n126_s42": full(D2G_N)}, {"f_dinov2g_n126_s7": full(D2G_N, seed=7)}]},
    2: {"title": "DINOv3 ViT-H+ (841M), near-native recipe, all training images (2 seeds)", "start": True, "gpus": [
        {"f_dinov3hp_n144_s42": full(D3HP_N)}, {"f_dinov3hp_n144_s7": full(D3HP_N, seed=7)}]},
    3: {"title": "Held-out check, hard fold 4: DINOv2-g and DINOv3 ViT-H+", "start": True, "gpus": [
        {"h_dinov2g_n126": held(4, D2G_N)}, {"h_dinov3hp_n144": held(4, D3HP_N)}]},
    4: {"title": "Held-out check, hard fold 3: DINOv2-g and DINOv3 ViT-H+", "start": True, "gpus": [
        {"h3_dinov2g_n126": held(3, D2G_N)}, {"h3_dinov3hp_n144": held(3, D3HP_N)}]},
    5: {"title": "DINOv2-g and DINOv3 ViT-H+, near-native recipe, 3rd seeds", "start": True, "gpus": [
        {"f_dinov2g_n126_s123": full(D2G_N, seed=123)}, {"f_dinov3hp_n144_s123": full(D3HP_N, seed=123)}]},
}

# Round 3 (2026-11-21, finished; results in cloud_kit_t4/run3, LB v58 = 0.86514).
ACCOUNTS_R3 = {
    1: {"title": "EVA-02-L, new recipe, all training images (2 seeds)", "start": True, "gpus": [
        {"f_eva02l_n126_s42": full(EVAL_N, LP)}, {"f_eva02l_n126_s7": full(EVAL_N, LP, 7)}]},
    2: {"title": "EVA-02-L M38M, new recipe, all training images (2 seeds)", "start": True, "gpus": [
        {"f_m38_n126_s42": full(M38_N, LP)}, {"f_m38_n126_s7": full(M38_N, LP, 7)}]},
    3: {"title": "DINOv3-L and DINOv2-L, round-2 recipes, 3rd seeds", "start": True, "gpus": [
        {"f_dinov3l_n144_s123": full(D3L_N, seed=123)}, {"f_dinov2l_n126_s123": full(D2L_N, seed=123)}]},
    4: {"title": "Held-out checks: EVA-02-L new recipe (hard folds 4 and 3)", "start": True, "gpus": [
        {"h_eva02l_n126": held(4, EVAL_N, LP)}, {"h3_eva02l_n126": held(3, EVAL_N, LP)}]},
    5: {"title": "EVA-02-L and M38M, new recipe, 3rd seeds", "start": True, "gpus": [
        {"f_eva02l_n126_s123": full(EVAL_N, LP, 123)}, {"f_m38_n126_s123": full(M38_N, LP, 123)}]},
}

# Round 2 (2026-11-19, finished; results in cloud_kit_t4/run2, LB v55 = 0.86505).
ACCOUNTS_R2 = {
    1: {"title": "Held-out checks: old DINOv3-L recipe, new DINOv2-L recipe (hard fold 4)", "start": True, "gpus": [
        {"h_dinov3l": held(4, D3L)}, {"h_dinov2l_n126": held(4, D2L_N)}]},
    2: {"title": "DINOv3-L, new recipe, all training images (2 seeds)", "start": True, "gpus": [
        {"f_dinov3l_n144_s42": full(D3L_N)}, {"f_dinov3l_n144_s7": full(D3L_N, seed=7)}]},
    3: {"title": "DINOv2-L, new recipe, all training images (2 seeds)", "start": True, "gpus": [
        {"f_dinov2l_n126_s42": full(D2L_N)}, {"f_dinov2l_n126_s7": full(D2L_N, seed=7)}]},
    4: {"title": "Held-out checks: DINOv3-L new recipe (hard folds 4 and 3)", "start": True, "gpus": [
        {"h_dinov3l_n144": held(4, D3L_N)}, {"h3_dinov3l_n144": held(3, D3L_N)}]},
    5: {"title": "LP-FT runs on EVA-02 (round 1; only EVA-02-L is left, resume only if asked)", "start": False, "gpus": [
        {"f_eva02b_lpft_s42": full(EVAB, LP), "f_eva02b_lpft_s7": full(EVAB, LP, 7)},
        {"f_eva02l_lpft_s42": full(EVAL, LP)}]},
}
SMOKE = "--backbone convnext_atto.d2_in1k --epochs 2 --batch 8 --h 64 --w 128 --lr 1e-3 --balance 0 --la-tau 1 --ckpt"
SMOKE_ACCOUNT = {"title": "test build", "start": True, "gpus": [
    {"h3_smoke": f"--cv hard --folds 3 --no-test {SMOKE}", "f_smoke": f"--cv full --save-weights {SMOKE}"}]}

SETUP = r'''
import base64, io, json, os, shlex, shutil, subprocess, sys, tarfile, threading, time, zipfile
from pathlib import Path

T0 = time.time()
assert START, ("This account waits for the stage-1 result. Set START = True in the first code cell only after the "
               "team has confirmed that stage 1 passed.")
WORK = Path(os.environ.get("CLOUD_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("CLOUD_INPUT", "/kaggle/input"))
BASE = Path(os.environ.get("CLOUD_BASE", "/tmp/cpe342"))
MAX_HOURS = float(os.environ.get("CLOUD_MAX_HOURS", MAX_HOURS))
OUT, SRC, DATA = WORK / "outputs", BASE / "task4" / "src", BASE / "data" / "04_image_classification"
OUT.mkdir(parents=True, exist_ok=True)
shutil.rmtree(BASE / "task4", ignore_errors=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(KIT)), mode="r:gz") as tar:
    tar.extractall(BASE)
for f in (BASE / "fold_files").glob("*.npy"):
    shutil.copy(f, OUT / f.name)
os.environ.update(CPE342_DATA=str(BASE / "data"), T4_OUT_DIR=str(OUT), T4_CACHE_DIR=str(BASE / "cache"),
                  PYTHONUNBUFFERED="1")
if not os.environ.get("CLOUD_SKIP_PIP"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm", "gdown"], check=True)

import numpy as np
import pandas as pd
import timm
import torch

n_gpu = torch.cuda.device_count()
print(f"torch {torch.__version__} | timm {timm.__version__} | GPUs: {n_gpu}")
for i in range(n_gpu):
    print(" ", torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 2 ** 30, 1), "GB")
if not n_gpu and not os.environ.get("CLOUD_ALLOW_CPU"):
    raise SystemExit("No GPU. Open Settings > Accelerator and choose GPU T4 x2, then run again.")
if n_gpu < 2:
    QUEUES = [[job for queue in QUEUES for job in queue]]  # one GPU: run everything one after the other
JOBS = {name: args for queue in QUEUES for name, args in queue}
'''

IMAGES = r'''
def find_file(root, name):
    r = subprocess.run(["find", "-L", str(root), "-type", "f", "-name", name, "-print", "-quit"],
                       capture_output=True, text=True)
    return Path(r.stdout.strip()) if r.stdout.strip() else None


train, test = (pd.read_csv(DATA / f"{s}.csv") for s in ("train", "test"))
first = {"train": train.file_name[0], "test": test.file_name[0]}
found = {s: find_file(INPUT, n) for s, n in first.items()} if INPUT.exists() else {"train": None, "test": None}
if not all(found.values()):
    (BASE / "images").mkdir(parents=True, exist_ok=True)
    found = {s: find_file(BASE / "images", n) for s, n in first.items()}
if not all(found.values()):
    archive = BASE / "images.zip"
    print("Downloading the official task-4 images (about 1.3 GB) ...", flush=True)
    r = subprocess.run([sys.executable, "-m", "gdown", IMAGES_DRIVE_ID, "-O", str(archive)])
    if r.returncode or not archive.exists():
        raise SystemExit("Image download failed (Drive may be rate-limited). Try again later, or attach the task-4 "
                         "images as a PRIVATE Kaggle dataset through Add Input and run again.")
    with zipfile.ZipFile(archive) as z:
        z.extractall(BASE / "images")
    archive.unlink()
    found = {s: find_file(BASE / "images", n) for s, n in first.items()}
(DATA / "data").mkdir(parents=True, exist_ok=True)
for split, df in (("train", train), ("test", test)):
    assert found[split] is not None, f"{split} images not found"
    link = DATA / "data" / split
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(found[split].parent, target_is_directory=True)
    have = set(os.listdir(link))
    missing = [f for f in df.file_name if f not in have]
    assert not missing, f"{split}: {len(missing)} images are missing, e.g. {missing[:3]}"
    print(split, len(df), "images OK")
subprocess.run([sys.executable, "-c", "import t4data; t4data.images('train'); t4data.images('test')"], cwd=SRC, check=True)
print(f"image cache ready after {(time.time() - T0) / 60:.1f} min")
'''

RESTORE = r'''
# A paused run continues here: add this notebook's previous output through Add Input.
restored = 0
if INPUT.exists():
    for name in JOBS:
        for pattern in (f"ckpt_{name}_f*.pt", f"ft_{name}.json", f"ft_{name}.log", f"test_{name}.npy", f"oof_{name}.npy",
                        f"model_{name}_f*.pt", f"run_{name}.txt"):
            for f in INPUT.rglob(pattern):
                if not (OUT / f.name).exists():
                    shutil.copy(f, OUT / f.name)
                    restored += 1
print("files restored from earlier sessions:", restored)
'''

RUN = r'''
status, lock = {}, threading.Lock()
MIN_START_HOURS = min(0.4, MAX_HOURS / 2)


def finished(name):
    f = OUT / f"ft_{name}.json"
    return f.exists() and bool(json.loads(f.read_text() or "{}"))


def run_queue(gpu, queue):
    for name, args in queue:
        left = MAX_HOURS - (time.time() - T0) / 3600
        if finished(name):
            result = "done (earlier session)"
        elif left < MIN_START_HOURS:
            result = "not started: no time left in this session"
        else:
            cmd = [sys.executable, "ft.py", "--name", name, *shlex.split(args), "--max-hours", f"{left:.3f}"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            with open(OUT / f"run_{name}.txt", "a") as fh:
                try:
                    code = subprocess.run(cmd, cwd=SRC, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                          timeout=(left + 0.5) * 3600).returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            result = "done" if code == 0 and finished(name) else "paused" if code in (75, "timeout") else f"FAILED ({code})"
        with lock:
            status[name] = result
        print(f"[GPU {gpu}] {name}: {result}", flush=True)


threads = [threading.Thread(target=run_queue, args=(g, q), daemon=True) for g, q in enumerate(QUEUES)]
for t in threads:
    t.start()
while any(t.is_alive() for t in threads):
    for t in threads:
        t.join(timeout=600 / len(threads))
    for name in JOBS:
        f = OUT / f"ft_{name}.log"
        if name not in status and f.exists():
            print(f"{(time.time() - T0) / 3600:.2f} h | {name}: {f.read_text(errors='replace').strip().splitlines()[-1][:110]}", flush=True)
'''

PACK = r'''
n_test, lines, ok = len(test), [], True
for name in JOBS:
    state = status.get(name, "not run")
    if finished(name) and not (OUT / f"test_{name}.npy").exists() and not (OUT / f"oof_{name}.npy").exists():
        state = "FAILED (no prediction file)"
    if (OUT / f"test_{name}.npy").exists():
        p = np.load(OUT / f"test_{name}.npy")
        if p.shape != (n_test, 5) or not np.isfinite(p).all() or abs(p.sum(1) - 1).max() > 1e-3:
            state = f"FAILED (bad test file {p.shape})"
    ok &= state.startswith("done")
    lines.append(f"{name}: {state}")
report = f"account {ACCOUNT} | {'ALL JOBS DONE' if ok else 'NOT FINISHED'}\n" + "\n".join(lines) + "\n"
(WORK / "STATUS.txt").write_text(report)
target = WORK / f"results_account{ACCOUNT}.zip"
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(WORK / "STATUS.txt", "STATUS.txt")
    for name in JOBS:
        if not status.get(name, "").startswith("done"):
            continue
        files = [f"test_{name}.npy", f"ft_{name}.json", f"ft_{name}.log"]
        if name.startswith("h"):
            files.append(f"oof_{name}.npy")  # held-out predictions; all zeros for full-data runs, so not sent
        for f in files:
            if (OUT / f).exists():
                z.write(OUT / f, f)
print(report)
print(f"Download {target.name} from the Output tab and send it to the team." if ok else
      f"{target.name} holds the finished jobs. To continue: Add Input > this notebook's output, then Save Version again.")
'''


def payload(data_dir: Path, outputs_dir: Path) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for rel in ("common/__init__.py", "common/paths.py", "task4/src/ft.py", "task4/src/t4data.py", "task4/src/scene_cv.py"):
            tar.add(ROOT / rel, rel)
        for s in ("train", "test"):
            tar.add(data_dir / f"{s}.csv", f"data/04_image_classification/{s}.csv")
        for f in ("scene_groups_k5.npy", "bright_train.npy"):
            if (outputs_dir / f).exists():
                tar.add(outputs_dir / f, f"fold_files/{f}")
    return base64.b64encode(buf.getvalue()).decode()


def notebook(no: int, acc: dict, kit: str, label: str) -> nbformat.NotebookNode:
    queues = [list(q.items()) for q in acc["gpus"]]
    jobs = "\n".join(f"- GPU {g}: " + ", ".join(f"`{n}`" for n in q) for g, q in enumerate(acc["gpus"]))
    wait = "" if acc["start"] else ("\n**Wait.** Do not run this notebook until the team confirms that stage 1 passed. "
                                    "Then set `START = True` below.\n")
    intro = (f"# CPE342 task 4, {label}: {acc['title']}\n{wait}\n{jobs}\n\n"
             "**Keep this notebook private**: it contains the team's code and the competition csv files.\n\n"
             "Settings: Accelerator = **GPU T4 x2**, Internet = **on**. Then **Save Version > Save & Run All (Commit)**.\n"
             "Do not submit to the competition from here. Test images only pass through the trained networks.")
    config = (f"ACCOUNT = {no}\nSTART = {acc['start']}  # False = wait until the team says go\n"
              "MAX_HOURS = 7.5  # stops cleanly before the session limit; a paused run continues in the next session\n"
              f"IMAGES_DRIVE_ID = {IMAGES_DRIVE_ID!r}\nQUEUES = {queues!r}")
    hidden = new_code_cell(f'KIT = "{kit}"')
    hidden.metadata["jupyter"] = {"source_hidden": True}
    cells = [new_markdown_cell(intro), new_code_cell(config), hidden, new_code_cell(SETUP.strip()),
             new_code_cell(IMAGES.strip()), new_code_cell(RESTORE.strip()), new_code_cell(RUN.strip()),
             new_code_cell(PACK.strip())]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


README = """# Task 4 Kaggle kit (team only)

Use **your own Kaggle account** and only the notebook assigned to you. One person, one account: do not create extra
accounts. Keep every notebook **private**, and do not share this kit outside the team.

| Notebook | Job | When to start |
|---|---|---|
| `account1_dinov2g.ipynb` | DINOv2-g (1.1B) near-native recipe, all images, seeds 42 and 7 (best on a fast GPU) | Now |
| `account2_dinov3hp.ipynb` | DINOv3 ViT-H+ (841M) near-native recipe, all images, seeds 42 and 7 (best on a fast GPU) | Now |
| `account3_big_fold4.ipynb` | Held-out fold 4: DINOv2-g and DINOv3 ViT-H+ | Now |
| `account4_big_fold3.ipynb` | Held-out fold 3: DINOv2-g and DINOv3 ViT-H+ | Now |
| `account5_big_seeds.ipynb` | DINOv2-g and DINOv3 ViT-H+ near-native recipe, seed 123 | Now |

## Steps
1. Kaggle > Settings: verify your phone number (needed for GPU and internet).
2. Kaggle > Code > New Notebook > File > Import Notebook, and upload your `.ipynb`.
3. Right panel > Session options: **Accelerator = GPU T4 x2**, **Internet = On**. Keep it **Private**.
4. All five notebooks start with `START = True`: no waiting this round.
5. **Save Version > Save & Run All (Commit)**. You can close the browser; the run continues.
6. When it ends, open the saved version > **Output**, read `STATUS.txt`, and send `results_account<N>.zip` to the team.

## If STATUS.txt says NOT FINISHED
The notebook stops by itself after 7.5 hours and keeps a checkpoint. Send the zip anyway, then:
Add Input > Notebook Output > **this notebook's latest version** > Save Version > Save & Run All. It continues where it stopped.
A job marked FAILED: send `STATUS.txt` and `outputs/run_<job>.txt`.

If the image download fails, wait and retry, or ask the team for the images as a private Kaggle dataset and attach it with Add Input.
Do not submit to the competition from these notebooks.
"""
FILES = {1: "account1_dinov2g", 2: "account2_dinov3hp", 3: "account3_big_fold4", 4: "account4_big_fold3", 5: "account5_big_seeds"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT.parent / "cloud_kit_t4" / "round4")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "04_image_classification")
    ap.add_argument("--outputs", type=Path, default=ROOT / "task4" / "outputs")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not a.smoke:
        for s, want in CSV_ORDER_MD5.items():
            names = pd.read_csv(a.data / f"{s}.csv")["file_name"]
            assert hashlib.md5("\n".join(names).encode()).hexdigest() == want, f"{s}.csv is not the official row order"
    a.out.mkdir(parents=True, exist_ok=True)
    kit = payload(a.data, a.outputs)
    plan = {0: SMOKE_ACCOUNT} if a.smoke else ACCOUNTS
    for no, acc in plan.items():
        name = "smoke" if a.smoke else FILES[no]
        nbformat.write(notebook(no, acc, kit, "test build" if a.smoke else f"account {no}"), a.out / f"{name}.ipynb")
    if not a.smoke:
        (a.out / "README.md").write_text(README, encoding="utf-8")
    print(f"wrote {len(plan)} notebook(s) to {a.out}; embedded kit {len(kit) / 2 ** 20:.2f} MB")


if __name__ == "__main__":
    main()
