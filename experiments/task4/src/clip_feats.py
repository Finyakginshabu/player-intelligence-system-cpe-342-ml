"""Frozen CLIP-family image embeddings + zero-shot class probabilities for task 4 (pretrained models are allowed here).

Usage (GPU venv):  python task4/src/clip_feats.py <open_clip model> <pretrained tag> [short name] [batch]
Writes task4/outputs/emb_<name>_{train,test}.npy   (N, V, D) float16, one L2-normalised embedding per view
       task4/outputs/zs_<name>_{train,test}.npy    (N, V, 5) float32 zero-shot logits per view (already x logit scale)
Views: 0 = whole frame squashed to a square, 1-3 = left / centre / right square crops (the frame is 230x120).
No training happens here; test images only pass through the frozen network.
"""
import sys
import time

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

import t4data

# The data-card titles are pseudonyms; the prompts name the real games the pretrained model knows.
NAMES = {
    0: ["Seven Knights", "Seven Knights Re:Birth", "Seven Knights 2", "Seven Knights mobile RPG"],
    1: ["FiveM", "GTA V", "Grand Theft Auto V", "GTA 5 roleplay"],
    2: ["Free Fire", "Garena Free Fire"],
    3: ["PUBG", "PUBG Mobile", "PlayerUnknown's Battlegrounds"],
    4: ["Roblox"],
}
TEMPLATES = ["a screenshot of the video game {}.", "gameplay of {}.", "a screenshot of {}.", "{} gameplay screenshot.",
             "a frame from a livestream of {}.", "a distorted, noisy screenshot of the game {}."]


def views(x: torch.Tensor, size: int) -> list[torch.Tensor]:
    """x: float (B, 3, 120, 230) in [0, 1] -> 4 tensors (B, 3, size, size)."""
    crops = [x, x[..., :, 0:120], x[..., :, 55:175], x[..., :, 110:230]]
    return [F.interpolate(c, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1) for c in crops]


def main():
    arch, tag = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else f"{arch}_{tag}".replace("/", "-")
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    dev = "cuda"
    model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=tag, device=dev)
    model.eval().half()
    tok = open_clip.get_tokenizer(arch)
    norm = [t for t in pre.transforms if t.__class__.__name__ == "Normalize"][0]
    mean = torch.tensor(norm.mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(norm.std, device=dev).view(1, 3, 1, 1)
    size = model.visual.image_size if isinstance(model.visual.image_size, int) else model.visual.image_size[0]
    with torch.no_grad():
        txt = []
        for c in range(5):
            prompts = [t.format(n) for n in NAMES[c] for t in TEMPLATES]
            e = F.normalize(model.encode_text(tok(prompts).to(dev)).float(), dim=-1).mean(0)
            txt.append(F.normalize(e, dim=-1))
        txt = torch.stack(txt)  # (5, D)
        scale = float(model.logit_scale.exp()) if hasattr(model, "logit_scale") else 100.0
    print(f"{name}: image size {size}, logit scale {scale:.1f}", flush=True)
    for split in ("train", "test"):
        imgs = t4data.images(split)
        embs, zs, t0 = [], [], time.time()
        for i in range(0, len(imgs), batch):
            x = torch.from_numpy(np.ascontiguousarray(imgs[i:i + batch])).to(dev).permute(0, 3, 1, 2).float() / 255
            with torch.no_grad():
                e = torch.stack([F.normalize(model.encode_image(((v - mean) / std).half()).float(), dim=-1)
                                 for v in views(x, size)], dim=1)  # (B, V, D)
            embs.append(e.cpu().numpy().astype(np.float16))
            zs.append((scale * e @ txt.T).cpu().numpy().astype(np.float32))
            if (i // batch) % 50 == 0:
                print(f"  {split} {i}/{len(imgs)} {time.time() - t0:.0f}s", flush=True)
        embs, zs = np.concatenate(embs), np.concatenate(zs)
        np.save(t4data.OUT / f"emb_{name}_{split}.npy", embs)
        np.save(t4data.OUT / f"zs_{name}_{split}.npy", zs)
        if split == "train":
            from sklearn.metrics import f1_score
            y = t4data.train_labels()
            for label, p in [("squash", zs[:, 0]), ("3 crops", zs[:, 1:].mean(1)), ("all 4 views", zs.mean(1))]:
                pred = p.argmax(1)
                print(f"  zero-shot on TRAIN ({label}): acc {(pred == y).mean():.4f}  macro F1 "
                      f"{f1_score(y, pred, average='macro'):.4f}  per-class recall "
                      f"{[round(float((pred[y == c] == c).mean()), 3) for c in range(5)]}", flush=True)
        else:
            print("  zero-shot test class mix (4 views):", np.round(np.bincount(zs.mean(1).argmax(1), minlength=5) / len(zs), 3))


if __name__ == "__main__":
    main()
