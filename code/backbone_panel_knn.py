"""C7 / : same-kNN backbone-isolation panel across 6 frozen backbones."""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MVTEC_ROOT = "data/MVTec_AD"
VISA_ROOT = "data/VisA/VisA_pytorch/1cls"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "backbone_panel_knn.json")

BACKBONES = {
    "CLIP":       ("vit_large_patch14_clip_224.openai",        "contrastive-language"),
    "SigLIP":     ("vit_large_patch16_siglip_256",             "contrastive-language"),
    "DINOv2":     ("vit_large_patch14_dinov2.lvd142m",         "ssl-spatial"),
    "DINOv2-reg": ("vit_large_patch14_reg4_dinov2.lvd142m",    "ssl-spatial"),
    "MAE":        ("vit_large_patch16_224.mae",                "ssl-spatial"),
    "WRN50":      ("wide_resnet50_2.tv_in1k",                  "supervised"),
}
MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
VISA = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def load_model(name):
    m = timm.create_model(name, pretrained=True, num_classes=0).eval().to(DEVICE)
    cfg = timm.data.resolve_model_data_config(m)
    tf = timm.data.create_transform(**cfg, is_training=False)
    npre = getattr(m, "num_prefix_tokens", 1)
    return m, tf, npre


def list_images(category, dataset, split="test"):
    root = os.path.join(MVTEC_ROOT if dataset == "mvtec" else VISA_ROOT, category, split)
    paths, labels = [], []
    for atype in sorted(os.listdir(root)):
        ad = os.path.join(root, atype)
        if not os.path.isdir(ad):
            continue
        normal = atype in ("good", "ok")
        for fn in sorted(os.listdir(ad)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                paths.append(os.path.join(ad, fn)); labels.append(0 if normal else 1)
    return paths, labels


def ref_path(category, dataset, k=1):
    td = os.path.join(MVTEC_ROOT if dataset == "mvtec" else VISA_ROOT, category, "train", "good")
    files = sorted(os.listdir(td))
    np.random.seed(42)
    idx = np.random.choice(len(files), min(k, len(files)), replace=False)
    return [os.path.join(td, files[i]) for i in idx]


@torch.no_grad()
def patch_feats(model, tf, npre, path):
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    f = model.forward_features(x)
    if f.dim() == 4:                       # CNN [B,C,H,W] -> [N, C]
        B, C, H, W = f.shape
        t = f.reshape(B, C, H * W).permute(0, 2, 1)[0]
    else:                                  # ViT [B,N,D] -> drop prefix tokens
        t = f[0, npre:, :]
    return t.float().cpu()


def knn_auroc(model, tf, npre, category, dataset):
    refs = torch.cat([patch_feats(model, tf, npre, p) for p in ref_path(category, dataset)], dim=0)
    refs = F.normalize(refs, dim=1).to(DEVICE)
    paths, labels = list_images(category, dataset)
    scores = []
    for p in paths:
        fe = F.normalize(patch_feats(model, tf, npre, p), dim=1).to(DEVICE)
        sim = fe @ refs.t()
        dist = 1.0 - sim.max(dim=1).values
        scores.append(dist.max().item())
    if len(set(labels)) < 2:
        return None
    return round(roc_auc_score(labels, scores) * 100, 2)


def main():
    matrix = {}  # backbone -> {category: auroc}
    cats = [("mvtec", c) for c in MVTEC] + [("visa", c) for c in VISA]
    for bname, (model_id, axis) in BACKBONES.items():
        print(f"\n=== {bname} ({model_id}, {axis}) ===", flush=True)
        model, tf, npre = load_model(model_id)
        matrix[bname] = {"_axis": axis}
        for ds, cat in cats:
            try:
                a = knn_auroc(model, tf, npre, cat, ds)
            except Exception as e:
                a = None; print(f"  {cat}: ERROR {e}", flush=True)
            matrix[bname][cat] = a
            print(f"  {cat:<14} {a}", flush=True)
        del model
        torch.cuda.empty_cache()
        with open(OUT, "w") as f:
            json.dump(matrix, f, indent=2)   # checkpoint after each backbone
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
