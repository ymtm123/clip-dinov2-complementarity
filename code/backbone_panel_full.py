"""C7 / : FULL 6-backbone isolation panel — 30 categories, image+pixel, 10 seeds, fusion matrix."""
import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import cv2
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from pixel_fusion import compute_pro

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D9 = "data"
ROOTS = {"mvtec": f"{D9}/MVTec_AD", "visa": f"{D9}/VisA/VisA_pytorch/1cls", "btad": f"{D9}/BTAD/BTech_Dataset_transformed"}
TRAIN_GOOD = {"mvtec": "train/good", "visa": "train/good", "btad": "train/ok"}
OUTDIR = os.path.join(os.path.dirname(BASE), "results")
OUT = os.path.join(OUTDIR, "backbone_panel_full.json")
SCOREDUMP = os.path.join(OUTDIR, "backbone_panel_scores")  # seed-0 per-image scores for the fusion matrix

BACKBONES = {
    "CLIP":       ("vit_large_patch14_clip_224.openai",     "contrastive-language"),
    "SigLIP":     ("vit_large_patch16_siglip_256",          "contrastive-language"),
    "DINOv2":     ("vit_large_patch14_dinov2.lvd142m",      "ssl-spatial"),
    "DINOv2-reg": ("vit_large_patch14_reg4_dinov2.lvd142m", "ssl-spatial"),
    "MAE":        ("vit_large_patch16_224.mae",             "ssl-spatial"),
    "WRN50":      ("wide_resnet50_2.tv_in1k",               "supervised"),
}
MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
         "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
VISA = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
        "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
BTAD = ["01", "02", "03"]
ALL_CATS = [("mvtec", c) for c in MVTEC] + [("visa", c) for c in VISA] + [("btad", c) for c in BTAD]
PIX_SIZE = 256
NEG_CAP = 300_000
RNGPIX = np.random.RandomState(0)


def load_model(name):
    m = timm.create_model(name, pretrained=True, num_classes=0).eval().to(DEVICE)
    cfg = timm.data.resolve_model_data_config(m)
    tf = timm.data.create_transform(**cfg, is_training=False)
    npre = getattr(m, "num_prefix_tokens", 1)
    return m, tf, npre


def find_mask(ds, cat, atype, fn):
    stem = os.path.splitext(fn)[0]
    gtd = os.path.join(ROOTS[ds], cat, "ground_truth", atype)
    for cand in (f"{stem}_mask.png", f"{stem}.png"):
        p = os.path.join(gtd, cand)
        if os.path.exists(p):
            return p
    cands = sorted(glob.glob(os.path.join(gtd, f"{stem}_mask.*")) + glob.glob(os.path.join(gtd, f"{stem}.*")))
    cands = [c for c in cands if not c.lower().endswith(".npy")]
    return cands[0] if cands else None


def list_images(ds, cat):
    root = os.path.join(ROOTS[ds], cat, "test")
    paths, labels, masks = [], [], []
    for atype in sorted(os.listdir(root)):
        ad = os.path.join(root, atype)
        if not os.path.isdir(ad):
            continue
        normal = atype in ("good", "ok")
        for fn in sorted(os.listdir(ad)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                paths.append(os.path.join(ad, fn)); labels.append(0 if normal else 1)
                masks.append(None if normal else find_mask(ds, cat, atype, fn))
    return paths, labels, masks


def ref_paths(ds, cat, seed, k=1):
    td = os.path.join(ROOTS[ds], cat, TRAIN_GOOD[ds])
    files = sorted(os.listdir(td))
    np.random.seed(seed)
    idx = np.random.choice(len(files), min(k, len(files)), replace=False)
    return [os.path.join(td, files[i]) for i in idx]


@torch.no_grad()
def patch_feats(model, tf, npre, path):
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    f = model.forward_features(x)
    if f.dim() == 4:
        B, C, H, W = f.shape
        t = f.reshape(B, C, H * W).permute(0, 2, 1)[0]; grid = (H, W)
    else:
        t = f[0, npre:, :]; n = t.shape[0]; g = int(round(n ** 0.5)); grid = (g, g)
    return F.normalize(t.float(), dim=1).cpu(), grid


def load_mask(mask_path, shape):
    if mask_path is None or not os.path.exists(mask_path):
        return np.zeros(shape, np.uint8)
    m = np.array(Image.open(mask_path).convert("L"))
    return (cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def run_category(model, tf, npre, ds, cat, seeds, want_pixel, dump_scores=False):
    paths, labels, masks = list_images(ds, cat)
    y = np.array(labels)
    if len(np.unique(y)) < 2:
        return None
    # cache test patch features ONCE
    feats, grids = [], []
    for p in paths:
        fe, g = patch_feats(model, tf, npre, p); feats.append(fe.to(DEVICE)); grids.append(g)
    gt = [load_mask(m, (PIX_SIZE, PIX_SIZE)) for m in masks] if want_pixel else None

    img_auroc, pix_auroc, pix_pro = [], [], []
    seed0_scores = None
    for si, seed in enumerate(seeds):
        ref = torch.cat([patch_feats(model, tf, npre, p)[0] for p in ref_paths(ds, cat, seed)], 0).to(DEVICE)
        img_scores = np.empty(len(paths)); pmaps = []
        for i, (fe, g) in enumerate(zip(feats, grids)):
            d = 1.0 - (fe @ ref.t()).max(dim=1).values     # per-patch distance
            img_scores[i] = d.max().item()
            if want_pixel:
                pm = cv2.resize(d.cpu().numpy().reshape(g).astype(np.float32), (PIX_SIZE, PIX_SIZE), interpolation=cv2.INTER_LINEAR)
                pmaps.append(pm)
        img_auroc.append(roc_auc_score(y, img_scores) * 100)
        if si == 0 and dump_scores:
            seed0_scores = {os.path.relpath(p, ROOTS[ds]).replace("\\", "/"): float(s) for p, s in zip(paths, img_scores)}
        if want_pixel and any(m.sum() for m in gt):
            yf = np.concatenate([m.ravel() for m in gt]); pf = np.concatenate([m.ravel() for m in pmaps])
            neg = np.where(yf == 0)[0]
            if len(neg) > NEG_CAP:
                keep = np.concatenate([np.where(yf == 1)[0], RNGPIX.choice(neg, NEG_CAP, replace=False)]); keep.sort()
                yf, pf = yf[keep], pf[keep]
            pix_auroc.append(roc_auc_score(yf, pf) * 100)
            if si == 0:
                anom = [(gt[i], pmaps[i]) for i in range(len(gt)) if gt[i].sum() > 0]
                pix_pro.append(compute_pro([a[0] for a in anom], [a[1] for a in anom]))
    for fe in feats:
        del fe
    torch.cuda.empty_cache()
    return dict(image_auroc_mean=round(float(np.mean(img_auroc)), 2), image_auroc_seeds=[round(a, 2) for a in img_auroc],
                pixel_auroc_mean=(round(float(np.mean(pix_auroc)), 2) if pix_auroc else None),
                pixel_aupro_seed0=(round(float(pix_pro[0]), 2) if pix_pro else None),
                n=len(paths), n_anom=int(y.sum()), seed0_scores=seed0_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--categories", nargs="+", default=None, help="ds:cat tokens for a smoke subset")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--no_pixel", action="store_true")
    ap.add_argument("--out_suffix", type=str, default="")
    args = ap.parse_args()

    cats = ALL_CATS if not args.categories else [tuple(t.split(":")) for t in args.categories]
    out_path = OUT.replace(".json", f"{args.out_suffix}.json")
    os.makedirs(SCOREDUMP, exist_ok=True)
    panel = json.load(open(out_path)) if os.path.exists(out_path) else {}   # resume
    target_cats = {c for _, c in cats}
    print(f"Panel | backbones={args.backbones} | {len(cats)} cats | seeds={args.seeds} | pixel={not args.no_pixel}", flush=True)
    for bname in args.backbones:
        if bname in panel and target_cats.issubset(set(panel[bname]) - {"_axis"}):
            print(f"[skip] {bname} (already complete in {out_path})", flush=True); continue
        model_id, axis = BACKBONES[bname]
        print(f"\n=== {bname} ({model_id}) ===", flush=True)
        model, tf, npre = load_model(model_id)
        panel[bname] = {"_axis": axis}
        dump = {}
        for ds, cat in cats:
            r = run_category(model, tf, npre, ds, cat, args.seeds, want_pixel=not args.no_pixel, dump_scores=True)
            if r is None:
                continue
            if r.get("seed0_scores"):
                dump[f"{ds}:{cat}"] = r.pop("seed0_scores")
            else:
                r.pop("seed0_scores", None)
            panel[bname][cat] = r
            print(f"  {cat:<14} img={r['image_auroc_mean']} pix={r['pixel_auroc_mean']} pro={r['pixel_aupro_seed0']} (n={r['n']})", flush=True)
        del model; torch.cuda.empty_cache()
        json.dump(dump, open(os.path.join(SCOREDUMP, f"{bname}{args.out_suffix}.json"), "w"))
        json.dump(panel, open(out_path, "w"), indent=2)   # checkpoint after each backbone
    print(f"\nSaved: {out_path}\nScore dumps: {SCOREDUMP}/<backbone>{args.out_suffix}.json")


if __name__ == "__main__":
    main()
