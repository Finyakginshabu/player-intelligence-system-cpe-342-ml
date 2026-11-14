"""Zero-shot CLIP for task 4 (2026-11-14). Nothing is trained: a pretrained CLIP scores each frame against text prompts
for the 5 games. Train labels are used only to SCORE the zero-shot predictions; test images are only predicted.

Why: the test frames come from other videos (broad content shift, 18 % unsure frames). Fine-tuned models only know the
training videos; CLIP's web pretraining has seen screenshots of the real games these names parody (Seven Knights, FiveM /
GTA V, Free Fire, PUBG, Roblox), so its zero-shot scores do not depend on our training videos. Fine-tuning CLIP threw that
away (v27: CLIP fine-tune −0.012 LB t4); mixing zero-shot back in is the WiSE-FT idea (robustness under shift).

Usage (GPU venv, from task4/src): python zeroshot.py <open_clip model> <pretrained tag> <prompt set: card|real|both>
Output: outputs/zs_<tag>_{train,test}.npy (softmax over the 5 classes, temperature 100 = CLIP's logit scale)
"""
import sys

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import t4data

arch, pre, pset = sys.argv[1], sys.argv[2], sys.argv[3]
tag = f"{arch}_{pre}_{pset}".replace("/", "-")
DEV = "cuda"
NAMES = {
    "card": ["7-11 Knights", "FiveN", "Free Fried", "PUBE", "Roblock"],  # datacard names only
    "real": ["Seven Knights", "GTA V FiveM roleplay", "Garena Free Fire", "PUBG Battlegrounds", "Roblox"],
}
TEMPL = ["a screenshot of the video game {}.", "gameplay footage of {}.", "a livestream of someone playing {}.",
         "a low resolution screenshot of {}.", "{} gameplay."]
model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pre, device=DEV)
model.eval().half()
tok = open_clip.get_tokenizer(arch)
sets = ["card", "real"] if pset == "both" else [pset]
with torch.no_grad():
    W = []
    for c in range(5):
        prompts = [t.format(NAMES[s][c]) for s in sets for t in TEMPL]
        e = F.normalize(model.encode_text(tok(prompts).to(DEV)).float(), dim=-1).mean(0)
        W.append(F.normalize(e, dim=0))
    W = torch.stack(W)  # (5, d)
cfg = model.visual.preprocess_cfg if hasattr(model.visual, "preprocess_cfg") else {}
size = model.visual.image_size if isinstance(model.visual.image_size, int) else model.visual.image_size[0]
MEAN = torch.tensor(getattr(model.visual, "image_mean", None) or open_clip.OPENAI_DATASET_MEAN, device=DEV).view(1, 3, 1, 1)
STD = torch.tensor(getattr(model.visual, "image_std", None) or open_clip.OPENAI_DATASET_STD, device=DEV).view(1, 3, 1, 1)


def predict(X, bs=128):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i + bs]).to(DEV).permute(0, 3, 1, 2).float().div_(255)
            # two views: the whole frame squashed to a square, and left + right square halves (aspect kept)
            full = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
            h, w = x.shape[-2:]
            left = F.interpolate(x[..., :, :h], size=(size, size), mode="bicubic", align_corners=False)
            right = F.interpolate(x[..., :, w - h:], size=(size, size), mode="bicubic", align_corners=False)
            logits = 0
            for v in (full, left, right):
                f = F.normalize(model.encode_image(((v - MEAN) / STD).half()).float(), dim=-1)
                logits = logits + 100 * f @ W.T / 3
            out.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(out)


y = t4data.train_labels()
ptr = predict(t4data.images("train"))
np.save(t4data.OUT / f"zs_{tag}_train.npy", ptr)
bt = np.load(t4data.OUT / "bright_train.npy") >= 25
print(f"{tag}: train zero-shot macro F1 {f1_score(y, ptr.argmax(1), average='macro'):.4f} | non-dark "
      f"{f1_score(y[bt], ptr[bt].argmax(1), average='macro'):.4f} | acc {(ptr.argmax(1) == y).mean():.4f}", flush=True)
print("  per-class recall", np.round([(ptr.argmax(1)[y == c] == c).mean() for c in range(5)], 3), flush=True)
pte = predict(t4data.images("test"))
np.save(t4data.OUT / f"zs_{tag}_test.npy", pte)
print("  test class mix", np.round(np.bincount(pte.argmax(1), minlength=5) / len(pte), 3))
