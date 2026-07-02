"""
Deep Analysis Experiments: Exploring CLIP-DINOv2 complementarity at a deeper level.

Experiments:
1. Per-defect-type AUROC breakdown (MVTec AD)
2. Per-image score correlation analysis
3. Agreement/disagreement matrix
4. Leave-One-Category-Out (LOCO) VFM prediction validation

These experiments go beyond aggregate category-level metrics to reveal
the internal structure of complementarity.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Environment ──
PYTHON = "python"
MVTEC_ROOT = Path(r"data\MVTec_AD")
VISA_ROOT = Path(r"data\VisA\VisA_pytorch\1cls")

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy import stats

# ── Shared utilities ──

def load_image(path, size=518):
    """Load and preprocess image."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


def load_image_clip(path, preprocess):
    """Load image with CLIP preprocessing."""
    img = Image.open(path).convert('RGB')
    return preprocess(img).unsqueeze(0)


@torch.no_grad()
def extract_clip_patches(clip_model, img_tensor):
    """
    Extract patch-level features from CLIP visual encoder.

    IMPORTANT: Must apply ln_post after transformer for proper normalization.
    Without ln_post, CLIP patch features are in raw transformer output space
    and produce meaningless kNN distances.

    Correct pipeline: conv1 -> reshape -> cls+pos_embed -> ln_pre -> transformer -> ln_post -> patches
    """
    visual = clip_model.visual
    x = visual.conv1(img_tensor)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = visual.transformer(x)
    x = visual.ln_post(x)  # Critical: normalize before extracting patches
    patch_tokens = x[:, 1:, :]  # Remove CLS token
    return patch_tokens


# ═════════════════════════════════════════════════════════════════════
# Experiment 1: Per-defect-type AUROC breakdown
# ═════════════════════════════════════════════════════════════════════

def experiment_1_defect_type_breakdown():
    """
    For each MVTec category, compute AUROC separately for each defect type.
    This reveals whether CLIP and DINOv2 detect DIFFERENT kinds of defects
    even within the same category.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Per-Defect-Type AUROC Breakdown")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load DINOv2
    print("Loading DINOv2 ViT-L/14...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    dinov2 = dinov2.to(device).eval()

    # Load CLIP
    print("Loading CLIP ViT-L/14...")
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    clip_model = clip_model.to(device).eval()

    results = {}

    categories = sorted([d.name for d in MVTEC_ROOT.iterdir()
                        if d.is_dir() and d.name not in ['license.txt', 'meta.json', 'readme.txt']])

    for cat in categories:
        print(f"\n--- {cat} ---")
        cat_dir = MVTEC_ROOT / cat
        test_dir = cat_dir / "test"
        train_dir = cat_dir / "train" / "good"

        # Get defect types
        defect_types = sorted([d.name for d in test_dir.iterdir() if d.is_dir() and d.name != 'good'])
        good_dir = test_dir / "good"

        if not good_dir.exists() or len(defect_types) == 0:
            continue

        # ── Extract reference features (1-shot) ──
        train_images = sorted(list(train_dir.glob("*.png")))
        ref_img_path = train_images[0]

        # DINOv2 reference
        with torch.no_grad():
            ref_tensor = load_image(ref_img_path).to(device)
            dino_ref = dinov2.forward_features(ref_tensor)
            dino_ref_patches = dino_ref['x_norm_patchtokens']  # [1, N, D]

        # CLIP reference (with proper ln_post normalization)
        with torch.no_grad():
            ref_clip = load_image_clip(ref_img_path, clip_preprocess).to(device)
            clip_ref_patches = extract_clip_patches(clip_model, ref_clip)

        # ── Score all test images ──
        cat_results = {"defect_types": {}}

        for defect_type in defect_types:
            defect_dir = test_dir / defect_type
            defect_images = sorted(list(defect_dir.glob("*.png")))

            if len(defect_images) == 0:
                continue

            dino_scores = []
            clip_scores = []

            for img_path in defect_images:
                with torch.no_grad():
                    # DINOv2 score
                    test_tensor = load_image(img_path).to(device)
                    dino_feat = dinov2.forward_features(test_tensor)
                    dino_patches = dino_feat['x_norm_patchtokens']
                    dino_dist = torch.cdist(dino_patches, dino_ref_patches).min(dim=-1).values
                    dino_score = dino_dist.max().item()
                    dino_scores.append(dino_score)

                    # CLIP score (with proper ln_post normalization)
                    test_clip = load_image_clip(img_path, clip_preprocess).to(device)
                    clip_patches = extract_clip_patches(clip_model, test_clip)
                    clip_dist = torch.cdist(
                        clip_patches.float(), clip_ref_patches.float()
                    ).min(dim=-1).values
                    clip_score = clip_dist.max().item()
                    clip_scores.append(clip_score)

            # Also score good images
            good_images = sorted(list(good_dir.glob("*.png")))
            dino_good_scores = []
            clip_good_scores = []

            for img_path in good_images:
                with torch.no_grad():
                    test_tensor = load_image(img_path).to(device)
                    dino_feat = dinov2.forward_features(test_tensor)
                    dino_patches = dino_feat['x_norm_patchtokens']
                    dino_dist = torch.cdist(dino_patches, dino_ref_patches).min(dim=-1).values
                    dino_score = dino_dist.max().item()
                    dino_good_scores.append(dino_score)

                    test_clip = load_image_clip(img_path, clip_preprocess).to(device)
                    clip_patches = extract_clip_patches(clip_model, test_clip)
                    clip_dist = torch.cdist(
                        clip_patches.float(), clip_ref_patches.float()
                    ).min(dim=-1).values
                    clip_score = clip_dist.max().item()
                    clip_good_scores.append(clip_score)

            # Compute AUROC for this defect type (defect vs good)
            n_defect = len(dino_scores)
            n_good = len(dino_good_scores)
            labels = [1]*n_defect + [0]*n_good

            if n_defect >= 2 and n_good >= 2:
                dino_auroc = roc_auc_score(labels, dino_scores + dino_good_scores) * 100
                clip_auroc = roc_auc_score(labels, clip_scores + clip_good_scores) * 100
                diff = clip_auroc - dino_auroc
                winner = "CLIP" if diff > 0 else "DINOv2" if diff < 0 else "tie"

                cat_results["defect_types"][defect_type] = {
                    "n_images": n_defect,
                    "clip_auroc": round(clip_auroc, 2),
                    "dino_auroc": round(dino_auroc, 2),
                    "diff_clip_minus_dino": round(diff, 2),
                    "winner": winner
                }

                print(f"  {defect_type:25s} (n={n_defect:3d}): "
                      f"CLIP={clip_auroc:6.2f}  DINO={dino_auroc:6.2f}  "
                      f"diff={diff:+7.2f}  → {winner}")

        results[cat] = cat_results

    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment 2: Per-image score correlation analysis
# ═════════════════════════════════════════════════════════════════════

def experiment_2_score_correlation(defect_results):
    """
    For each category, compute correlation between CLIP and DINOv2 per-image scores.
    Low correlation = high complementarity = fusion benefit.

    Uses the scores already computed in experiment 1.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Score Correlation Analysis")
    print("="*80)

    # We need to re-run scoring to get per-image scores (not per-defect-type)
    # But actually let's do this more efficiently by computing it fresh
    # with a dedicated function that stores all per-image scores

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    print("Loading models...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    dinov2 = dinov2.to(device).eval()

    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    clip_model = clip_model.to(device).eval()

    results = {}

    categories = sorted([d.name for d in MVTEC_ROOT.iterdir()
                        if d.is_dir() and d.name not in ['license.txt', 'meta.json', 'readme.txt']])

    for cat in categories:
        cat_dir = MVTEC_ROOT / cat
        test_dir = cat_dir / "test"
        train_dir = cat_dir / "train" / "good"

        # Reference
        train_images = sorted(list(train_dir.glob("*.png")))
        ref_img_path = train_images[0]

        with torch.no_grad():
            ref_tensor = load_image(ref_img_path).to(device)
            dino_ref = dinov2.forward_features(ref_tensor)['x_norm_patchtokens']

            ref_clip = load_image_clip(ref_img_path, clip_preprocess).to(device)
            clip_ref = extract_clip_patches(clip_model, ref_clip)

        # Score ALL test images
        all_dino_scores = []
        all_clip_scores = []
        all_labels = []
        all_defect_types = []

        for subdir in sorted(test_dir.iterdir()):
            if not subdir.is_dir():
                continue
            is_good = (subdir.name == "good")
            for img_path in sorted(subdir.glob("*.png")):
                with torch.no_grad():
                    # DINOv2
                    t = load_image(img_path).to(device)
                    dp = dinov2.forward_features(t)['x_norm_patchtokens']
                    ds = torch.cdist(dp, dino_ref).min(dim=-1).values.max().item()
                    all_dino_scores.append(ds)

                    # CLIP (with proper ln_post normalization)
                    tc = load_image_clip(img_path, clip_preprocess).to(device)
                    cp = extract_clip_patches(clip_model, tc)
                    cs = torch.cdist(cp.float(), clip_ref.float()).min(dim=-1).values.max().item()
                    all_clip_scores.append(cs)

                    all_labels.append(0 if is_good else 1)
                    all_defect_types.append("good" if is_good else subdir.name)

        dino_arr = np.array(all_dino_scores)
        clip_arr = np.array(all_clip_scores)
        labels_arr = np.array(all_labels)

        # Correlation (all images)
        pearson_r, pearson_p = stats.pearsonr(clip_arr, dino_arr)
        spearman_r, spearman_p = stats.spearmanr(clip_arr, dino_arr)

        # Correlation (anomaly images only)
        anom_mask = labels_arr == 1
        if anom_mask.sum() >= 3:
            anom_pearson_r, anom_pearson_p = stats.pearsonr(clip_arr[anom_mask], dino_arr[anom_mask])
        else:
            anom_pearson_r, anom_pearson_p = float('nan'), float('nan')

        # AUROC
        dino_auroc = roc_auc_score(labels_arr, dino_arr) * 100
        clip_auroc = roc_auc_score(labels_arr, clip_arr) * 100

        # Fusion AUROC (equal weight, z-score normalized)
        dino_z = (dino_arr - dino_arr.mean()) / (dino_arr.std() + 1e-8)
        clip_z = (clip_arr - clip_arr.mean()) / (clip_arr.std() + 1e-8)
        fusion_scores = 0.5 * clip_z + 0.5 * dino_z
        fusion_auroc = roc_auc_score(labels_arr, fusion_scores) * 100
        fusion_gain = fusion_auroc - max(dino_auroc, clip_auroc)

        results[cat] = {
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 6),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 6),
            "anom_only_pearson_r": round(anom_pearson_r, 4) if not np.isnan(anom_pearson_r) else None,
            "clip_knn_auroc": round(clip_auroc, 2),
            "dino_knn_auroc": round(dino_auroc, 2),
            "fusion_knn_auroc": round(fusion_auroc, 2),
            "fusion_gain_pp": round(fusion_gain, 2),
            "n_total": len(labels_arr),
            "n_anomaly": int(anom_mask.sum()),
        }

        print(f"{cat:15s}: r={pearson_r:+.3f} (p={pearson_p:.4f})  "
              f"anom_r={anom_pearson_r:+.3f}  "
              f"CLIP={clip_auroc:.1f}  DINO={dino_auroc:.1f}  "
              f"Fuse={fusion_auroc:.1f} ({fusion_gain:+.1f}pp)")

    # Summary: correlation vs fusion gain
    print("\n--- Correlation vs Fusion Gain ---")
    rs = [results[c]["pearson_r"] for c in results]
    gains = [results[c]["fusion_gain_pp"] for c in results]
    corr_r, corr_p = stats.pearsonr(rs, gains)
    print(f"Correlation between score_correlation and fusion_gain: r={corr_r:.3f}, p={corr_p:.4f}")
    print("(Negative correlation expected: lower score correlation → higher fusion gain)")

    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment 3: Agreement/Disagreement Analysis
# ═════════════════════════════════════════════════════════════════════

def experiment_3_agreement_analysis(correlation_results=None):
    """
    For each test image, classify into 4 quadrants:
    - Both detect (true positive for both)
    - Both miss (false negative for both)
    - CLIP-only (CLIP detects, DINOv2 misses)
    - DINOv2-only (DINOv2 detects, CLIP misses)

    Uses median anomaly score as threshold per model.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Agreement/Disagreement Analysis")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading models...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    dinov2 = dinov2.to(device).eval()

    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    clip_model = clip_model.to(device).eval()

    results = {}
    all_categories_data = {}

    categories = sorted([d.name for d in MVTEC_ROOT.iterdir()
                        if d.is_dir() and d.name not in ['license.txt', 'meta.json', 'readme.txt']])

    for cat in categories:
        cat_dir = MVTEC_ROOT / cat
        test_dir = cat_dir / "test"
        train_dir = cat_dir / "train" / "good"

        train_images = sorted(list(train_dir.glob("*.png")))
        ref_img_path = train_images[0]

        with torch.no_grad():
            ref_tensor = load_image(ref_img_path).to(device)
            dino_ref = dinov2.forward_features(ref_tensor)['x_norm_patchtokens']

            ref_clip = load_image_clip(ref_img_path, clip_preprocess).to(device)
            clip_ref = extract_clip_patches(clip_model, ref_clip)

        # Collect per-image data
        per_image = []
        for subdir in sorted(test_dir.iterdir()):
            if not subdir.is_dir():
                continue
            is_good = (subdir.name == "good")
            for img_path in sorted(subdir.glob("*.png")):
                with torch.no_grad():
                    t = load_image(img_path).to(device)
                    dp = dinov2.forward_features(t)['x_norm_patchtokens']
                    ds = torch.cdist(dp, dino_ref).min(dim=-1).values.max().item()

                    tc = load_image_clip(img_path, clip_preprocess).to(device)
                    cp = extract_clip_patches(clip_model, tc)
                    cs = torch.cdist(cp.float(), clip_ref.float()).min(dim=-1).values.max().item()

                per_image.append({
                    "path": str(img_path),
                    "label": 0 if is_good else 1,
                    "defect_type": "good" if is_good else subdir.name,
                    "dino_score": ds,
                    "clip_score": cs,
                })

        # Use optimal threshold (Youden's J) for each model
        labels = np.array([d["label"] for d in per_image])
        dino_scores = np.array([d["dino_score"] for d in per_image])
        clip_scores = np.array([d["clip_score"] for d in per_image])

        # Find optimal threshold for each model using good images
        good_dino = dino_scores[labels == 0]
        good_clip = clip_scores[labels == 0]

        # Threshold = mean + 2*std of good scores (simple approach)
        dino_thresh = good_dino.mean() + 2 * good_dino.std()
        clip_thresh = good_clip.mean() + 2 * good_clip.std()

        # Classify anomaly images only
        anom_mask = labels == 1
        anom_dino = dino_scores[anom_mask]
        anom_clip = clip_scores[anom_mask]
        anom_defect_types = [d["defect_type"] for d in per_image if d["label"] == 1]

        dino_detects = anom_dino > dino_thresh
        clip_detects = anom_clip > clip_thresh

        both_detect = np.sum(dino_detects & clip_detects)
        both_miss = np.sum(~dino_detects & ~clip_detects)
        clip_only = np.sum(clip_detects & ~dino_detects)
        dino_only = np.sum(dino_detects & ~clip_detects)
        n_anom = anom_mask.sum()

        # Per defect type breakdown of disagreements
        defect_type_breakdown = defaultdict(lambda: {"clip_only": 0, "dino_only": 0, "both": 0, "neither": 0, "total": 0})
        for i, dt in enumerate(anom_defect_types):
            defect_type_breakdown[dt]["total"] += 1
            if dino_detects[i] and clip_detects[i]:
                defect_type_breakdown[dt]["both"] += 1
            elif clip_detects[i] and not dino_detects[i]:
                defect_type_breakdown[dt]["clip_only"] += 1
            elif dino_detects[i] and not clip_detects[i]:
                defect_type_breakdown[dt]["dino_only"] += 1
            else:
                defect_type_breakdown[dt]["neither"] += 1

        results[cat] = {
            "n_anomaly": int(n_anom),
            "both_detect": int(both_detect),
            "both_miss": int(both_miss),
            "clip_only": int(clip_only),
            "dino_only": int(dino_only),
            "both_detect_pct": round(both_detect / n_anom * 100, 1) if n_anom > 0 else 0,
            "clip_only_pct": round(clip_only / n_anom * 100, 1) if n_anom > 0 else 0,
            "dino_only_pct": round(dino_only / n_anom * 100, 1) if n_anom > 0 else 0,
            "both_miss_pct": round(both_miss / n_anom * 100, 1) if n_anom > 0 else 0,
            "exclusive_detection_rate": round((clip_only + dino_only) / n_anom * 100, 1) if n_anom > 0 else 0,
            "defect_type_breakdown": {k: dict(v) for k, v in defect_type_breakdown.items()},
        }

        print(f"{cat:15s} (n={n_anom:3d}): "
              f"Both={both_detect:3d}({both_detect/n_anom*100:5.1f}%) "
              f"CLIP-only={clip_only:3d}({clip_only/n_anom*100:5.1f}%) "
              f"DINO-only={dino_only:3d}({dino_only/n_anom*100:5.1f}%) "
              f"Neither={both_miss:3d}({both_miss/n_anom*100:5.1f}%)")

    # Global summary
    total_anom = sum(r["n_anomaly"] for r in results.values())
    total_both = sum(r["both_detect"] for r in results.values())
    total_clip_only = sum(r["clip_only"] for r in results.values())
    total_dino_only = sum(r["dino_only"] for r in results.values())
    total_neither = sum(r["both_miss"] for r in results.values())

    print(f"\n--- GLOBAL SUMMARY (N={total_anom}) ---")
    print(f"Both detect:  {total_both:4d} ({total_both/total_anom*100:.1f}%)")
    print(f"CLIP-only:    {total_clip_only:4d} ({total_clip_only/total_anom*100:.1f}%)")
    print(f"DINOv2-only:  {total_dino_only:4d} ({total_dino_only/total_anom*100:.1f}%)")
    print(f"Both miss:    {total_neither:4d} ({total_neither/total_anom*100:.1f}%)")
    print(f"Exclusive:    {total_clip_only+total_dino_only:4d} ({(total_clip_only+total_dino_only)/total_anom*100:.1f}%)")

    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment 4: Leave-One-Category-Out VFM Prediction
# ═════════════════════════════════════════════════════════════════════

def experiment_4_loco_validation():
    """
    Can we predict the optimal VFM for a new category using only product attributes?

    Leave-One-Category-Out: for each category, use the remaining 29 to learn
    the mapping from attributes to optimal alpha, then predict for the held-out.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: Leave-One-Category-Out VFM Prediction")
    print("="*80)

    # Category attributes and oracle alphas from our existing data
    categories = {
        # MVTec
        "bottle":     {"type": "object", "semantic": "shape",  "alpha_img": 0.90},
        "cable":      {"type": "object", "semantic": "shape",  "alpha_img": 0.55},
        "capsule":    {"type": "object", "semantic": "mixed",  "alpha_img": 0.40},
        "carpet":     {"type": "texture","semantic": "color",  "alpha_img": 0.05},
        "grid":       {"type": "texture","semantic": "shape",  "alpha_img": 0.35},
        "hazelnut":   {"type": "object", "semantic": "shape",  "alpha_img": 0.35},
        "leather":    {"type": "texture","semantic": "color",  "alpha_img": 0.50},
        "metal_nut":  {"type": "object", "semantic": "shape",  "alpha_img": 0.90},
        "pill":       {"type": "object", "semantic": "mixed",  "alpha_img": 0.70},
        "screw":      {"type": "object", "semantic": "shape",  "alpha_img": 0.75},
        "tile":       {"type": "texture","semantic": "color",  "alpha_img": 0.00},
        "toothbrush": {"type": "object", "semantic": "shape",  "alpha_img": 0.65},
        "transistor": {"type": "object", "semantic": "mixed",  "alpha_img": 0.55},
        "wood":       {"type": "texture","semantic": "color",  "alpha_img": 0.70},
        "zipper":     {"type": "object", "semantic": "shape",  "alpha_img": 0.65},
        # VisA
        "candle":     {"type": "object", "semantic": "mixed",  "alpha_img": 0.60},
        "capsules_v": {"type": "object", "semantic": "mixed",  "alpha_img": 0.85},
        "cashew":     {"type": "object", "semantic": "shape",  "alpha_img": 0.55},
        "chewinggum": {"type": "object", "semantic": "mixed",  "alpha_img": 0.75},
        "fryum":      {"type": "object", "semantic": "shape",  "alpha_img": 0.65},
        "macaroni1":  {"type": "object", "semantic": "shape",  "alpha_img": 0.50},
        "macaroni2":  {"type": "object", "semantic": "shape",  "alpha_img": 0.65},
        "pcb1":       {"type": "object", "semantic": "shape",  "alpha_img": 0.65},
        "pcb2":       {"type": "object", "semantic": "shape",  "alpha_img": 0.85},
        "pcb3":       {"type": "object", "semantic": "shape",  "alpha_img": 0.90},
        "pcb4":       {"type": "object", "semantic": "shape",  "alpha_img": 0.65},
        "pipe_fryum": {"type": "object", "semantic": "shape",  "alpha_img": 0.45},
        # BTAD
        "btad01":     {"type": "object", "semantic": "shape",  "alpha_img": 0.70},
        "btad02":     {"type": "texture","semantic": "color",  "alpha_img": 0.55},
        "btad03":     {"type": "object", "semantic": "shape",  "alpha_img": 0.75},
    }

    cat_names = list(categories.keys())
    n = len(cat_names)

    # Simple rule-based prediction:
    # For each held-out category, predict alpha based on group means of remaining categories
    correct_direction = 0
    total_error = 0
    wrong_model_penalty = 0

    print(f"\n{'Category':15s} {'True α':>8s} {'Pred α':>8s} {'Error':>8s} {'Direction':>10s}")
    print("-" * 55)

    for i, held_out in enumerate(cat_names):
        held_attrs = categories[held_out]
        true_alpha = held_attrs["alpha_img"]

        # Use remaining categories to compute group means
        remaining = {k: v for k, v in categories.items() if k != held_out}

        # Prediction: use type and semantic attributes
        type_match = [v["alpha_img"] for k, v in remaining.items() if v["type"] == held_attrs["type"]]
        semantic_match = [v["alpha_img"] for k, v in remaining.items() if v["semantic"] == held_attrs["semantic"]]

        # Weighted average of type-based and semantic-based predictions
        type_pred = np.mean(type_match) if type_match else 0.5
        semantic_pred = np.mean(semantic_match) if semantic_match else 0.5
        pred_alpha = 0.5 * type_pred + 0.5 * semantic_pred

        error = abs(pred_alpha - true_alpha)
        total_error += error

        # Direction: does the prediction correctly identify CLIP-preferred (α<0.5) vs DINOv2-preferred (α>0.5)?
        true_model = "CLIP" if true_alpha < 0.5 else "DINOv2" if true_alpha > 0.5 else "neutral"
        pred_model = "CLIP" if pred_alpha < 0.5 else "DINOv2" if pred_alpha > 0.5 else "neutral"
        direction_correct = (true_model == pred_model) or true_model == "neutral"
        if direction_correct:
            correct_direction += 1

        mark = "OK" if direction_correct else "MISS"
        print(f"{held_out:15s} {true_alpha:8.2f} {pred_alpha:8.2f} {error:8.3f} {mark:>10s}")

    accuracy = correct_direction / n * 100
    mae = total_error / n

    print(f"\n--- LOCO Summary ---")
    print(f"Direction accuracy: {correct_direction}/{n} = {accuracy:.1f}%")
    print(f"Mean absolute error: {mae:.3f}")
    print(f"(Random baseline direction accuracy: ~50%)")
    print(f"(Always-DINOv2 baseline: {sum(1 for c in categories.values() if c['alpha_img'] >= 0.5)}/{n} = {sum(1 for c in categories.values() if c['alpha_img'] >= 0.5)/n*100:.1f}%)")

    return {
        "direction_accuracy": round(accuracy, 1),
        "mean_absolute_error": round(mae, 4),
        "n_categories": n,
        "correct_predictions": correct_direction,
    }


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def save_intermediate(results, tag=""):
    """Save intermediate results to avoid losing progress."""
    output_path = Path(__file__).parent / "deep_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [Saved intermediate results ({tag})]")


if __name__ == "__main__":
    output_path = Path(__file__).parent / "deep_analysis_results.json"

    # v2: CLIP feature extraction fixed (ln_post added, patch_dropout removed)
    # Force re-run of experiments 1-3 that use CLIP kNN
    RESULT_VERSION = "v2_ln_post_fixed"

    # Load existing results if available
    if output_path.exists():
        with open(output_path, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded existing results with keys: {list(all_results.keys())}")
        # Check if results need to be re-run due to version change
        if all_results.get("_version") != RESULT_VERSION:
            print(f"[!] Results version mismatch. Clearing experiments 1-3 for re-run.")
            print(f"    Old version: {all_results.get('_version', 'v1_no_ln_post')}")
            print(f"    New version: {RESULT_VERSION}")
            for key in ["experiment_1_defect_types", "experiment_2_correlation", "experiment_3_agreement"]:
                if key in all_results:
                    # Save old results for comparison
                    all_results[f"{key}_old_no_lnpost"] = all_results.pop(key)
            all_results["_version"] = RESULT_VERSION
            save_intermediate(all_results, "version reset")
    else:
        all_results = {"_version": RESULT_VERSION}

    # Experiment 1: Per-defect-type breakdown
    if "experiment_1_defect_types" not in all_results:
        print("\n" + "#"*80)
        print("# Running Experiment 1: Per-Defect-Type AUROC Breakdown")
        print("# (v2: CLIP with ln_post normalization)")
        print("#"*80)
        exp1_results = experiment_1_defect_type_breakdown()
        all_results["experiment_1_defect_types"] = exp1_results
        save_intermediate(all_results, "after exp1 v2")
    else:
        print("Experiment 1 (v2) already completed, skipping.")

    # Experiment 4: LOCO (no GPU needed, uses existing data - no change needed)
    if "experiment_4_loco" not in all_results:
        print("\n" + "#"*80)
        print("# Running Experiment 4: LOCO Validation")
        print("#"*80)
        exp4_results = experiment_4_loco_validation()
        all_results["experiment_4_loco"] = exp4_results
        save_intermediate(all_results, "after exp4")
    else:
        print("Experiment 4 already completed, skipping.")

    # Experiment 2: Score correlation
    if "experiment_2_correlation" not in all_results:
        print("\n" + "#"*80)
        print("# Running Experiment 2: Score Correlation Analysis")
        print("# (v2: CLIP with ln_post normalization)")
        print("#"*80)
        exp2_results = experiment_2_score_correlation(all_results.get("experiment_1_defect_types"))
        all_results["experiment_2_correlation"] = exp2_results
        save_intermediate(all_results, "after exp2 v2")
    else:
        print("Experiment 2 (v2) already completed, skipping.")

    # Experiment 3: Agreement/Disagreement
    if "experiment_3_agreement" not in all_results:
        print("\n" + "#"*80)
        print("# Running Experiment 3: Agreement/Disagreement Analysis")
        print("# (v2: CLIP with ln_post normalization)")
        print("#"*80)
        exp3_results = experiment_3_agreement_analysis()
        all_results["experiment_3_agreement"] = exp3_results
        save_intermediate(all_results, "after exp3 v2")
    else:
        print("Experiment 3 (v2) already completed, skipping.")

    # Final save
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nAll results saved to {output_path}")
    print("\n" + "="*80)
    print("DEEP ANALYSIS COMPLETE")
    print("="*80)
