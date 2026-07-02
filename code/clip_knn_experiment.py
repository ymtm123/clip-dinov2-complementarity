"""
CLIP Visual Encoder kNN Experiment (Backbone Isolation)
=======================================================
Purpose: Run CLIP ViT-L/14 visual encoder in the SAME 1-shot kNN pipeline
as DINOv2 and WRN-50, to directly prove that complementarity patterns
arise from backbone differences, NOT from 0-shot vs 1-shot methodology.

This completes the controlled comparison:
  - CLIP ViT-L/14 (contrastive language-image pre-training) → 1-shot kNN
  - DINOv2 ViT-L/14 (self-supervised distillation) → 1-shot kNN  [already done]
  - WRN-50-2 (supervised ImageNet classification) → 1-shot kNN    [already done]
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import timm

# ========== Configuration ==========
MVTEC_ROOT = "data/MVTec_AD"
VISA_ROOT = "data/VisA/VisA_pytorch/1cls"
OUTPUT_DIR = "."

MVTEC_CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                     'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                     'transistor', 'wood', 'zipper']

VISA_CATEGORIES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                   'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== Model Loading ==========

def get_clip_visual_model():
    """Load CLIP ViT-L/14 visual encoder with patch token access."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        model.eval().to(DEVICE)
        print(f"  Loaded CLIP ViT-L/14 via open_clip")
        return model, preprocess, 'open_clip'
    except ImportError:
        print("  open_clip not available, trying timm...")
        model = timm.create_model('vit_large_patch14_clip_224.openai', pretrained=True)
        model.eval().to(DEVICE)
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
        print(f"  Loaded CLIP ViT-L/14 via timm")
        return model, preprocess, 'timm'


def get_dinov2_model():
    """Load DINOv2 ViT-L/14."""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', verbose=False)
    model.eval().to(DEVICE)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, preprocess


# ========== Feature Extraction ==========

@torch.no_grad()
def extract_patch_features_clip(model, images, preprocess, backend='open_clip'):
    """Extract patch-level features from CLIP visual encoder."""
    features_list = []
    for img in images:
        img_t = preprocess(img).unsqueeze(0).to(DEVICE)

        if backend == 'open_clip':
            # Access the visual encoder's intermediate representations
            visual = model.visual
            x = visual.conv1(img_t)  # patch embedding
            # Reshape: [B, C, H, W] -> [B, HW, C]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            # Add class token
            cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls_token, x], dim=1)
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)
            # Transformer
            x = visual.transformer(x)
            x = visual.ln_post(x)
            # Remove CLS token, keep patch tokens
            patch_tokens = x[:, 1:, :]  # [1, 256, 1024]
            features_list.append(patch_tokens.squeeze(0).cpu())
        else:
            # timm model
            feat = model.forward_features(img_t)
            if isinstance(feat, torch.Tensor) and feat.dim() == 3:
                patch_tokens = feat[:, 1:, :]  # Remove CLS
                features_list.append(patch_tokens.squeeze(0).cpu())
            else:
                # Fallback: use CLS token
                features_list.append(feat.squeeze(0).cpu())

    return features_list


@torch.no_grad()
def extract_patch_features_dinov2(model, images, preprocess):
    """Extract patch-level features from DINOv2."""
    features_list = []
    for img in images:
        img_t = preprocess(img).unsqueeze(0).to(DEVICE)
        feat = model.forward_features(img_t)
        if isinstance(feat, dict):
            patch_tokens = feat['x_norm_patchtokens']
        else:
            patch_tokens = feat[:, 1:, :]
        features_list.append(patch_tokens.squeeze(0).cpu())
    return features_list


# ========== kNN Scoring ==========

def knn_anomaly_score(test_features, ref_features):
    """Compute anomaly scores using kNN distance to reference features."""
    ref_bank = torch.cat(ref_features, dim=0)
    ref_bank = F.normalize(ref_bank, dim=1)

    scores = []
    for feat in test_features:
        feat_norm = F.normalize(feat, dim=1)
        if feat_norm.dim() == 1:
            feat_norm = feat_norm.unsqueeze(0)
        sim = torch.mm(feat_norm, ref_bank.t())
        max_sim, _ = sim.max(dim=1)
        dist = 1.0 - max_sim
        score = dist.max().item()
        scores.append(score)

    return scores


# ========== Data Loading ==========

def load_images(category, dataset='mvtec', split='test'):
    """Load test images for a category."""
    if dataset == 'mvtec':
        root = os.path.join(MVTEC_ROOT, category, split)
    elif dataset == 'visa':
        root = os.path.join(VISA_ROOT, category, split)

    images = []
    labels = []

    for atype in sorted(os.listdir(root)):
        atype_dir = os.path.join(root, atype)
        if not os.path.isdir(atype_dir):
            continue
        is_normal = atype in ('good', 'ok')
        img_files = sorted([f for f in os.listdir(atype_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

        for fname in img_files:
            img = Image.open(os.path.join(atype_dir, fname)).convert('RGB')
            images.append(img)
            labels.append(0 if is_normal else 1)

    return images, labels


def load_reference(category, dataset='mvtec', k=1):
    """Load k normal reference images."""
    if dataset == 'mvtec':
        train_dir = os.path.join(MVTEC_ROOT, category, 'train', 'good')
    elif dataset == 'visa':
        train_dir = os.path.join(VISA_ROOT, category, 'train', 'good')

    img_files = sorted(os.listdir(train_dir))
    np.random.seed(42)
    selected = np.random.choice(len(img_files), min(k, len(img_files)), replace=False)

    images = []
    for idx in selected:
        img = Image.open(os.path.join(train_dir, img_files[idx])).convert('RGB')
        images.append(img)
    return images


# ========== Main Experiment ==========

def run_experiment():
    print("=" * 70)
    print("  BACKBONE ISOLATION: CLIP vs DINOv2 vs WRN (all 1-shot kNN)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load CLIP visual encoder
    print("\n[1/2] Loading CLIP ViT-L/14 visual encoder...")
    clip_model, clip_preprocess, clip_backend = get_clip_visual_model()

    # Load DINOv2
    print("[2/2] Loading DINOv2 ViT-L/14...")
    dinov2_model, dinov2_preprocess = get_dinov2_model()

    results = {'mvtec': {}, 'visa': {}}

    # ===== MVTec AD =====
    print("\n" + "=" * 50)
    print("  MVTec AD (15 categories)")
    print("=" * 50)

    for cat in MVTEC_CATEGORIES:
        print(f"\n  [{cat}]", end=" ", flush=True)

        ref_imgs = load_reference(cat, 'mvtec', k=1)
        test_imgs, labels = load_images(cat, 'mvtec', 'test')

        # CLIP kNN
        ref_feats_clip = extract_patch_features_clip(clip_model, ref_imgs, clip_preprocess, clip_backend)
        test_feats_clip = extract_patch_features_clip(clip_model, test_imgs, clip_preprocess, clip_backend)
        scores_clip = knn_anomaly_score(test_feats_clip, ref_feats_clip)
        auroc_clip = roc_auc_score(labels, scores_clip) * 100

        # DINOv2 kNN
        ref_feats_dino = extract_patch_features_dinov2(dinov2_model, ref_imgs, dinov2_preprocess)
        test_feats_dino = extract_patch_features_dinov2(dinov2_model, test_imgs, dinov2_preprocess)
        scores_dino = knn_anomaly_score(test_feats_dino, ref_feats_dino)
        auroc_dino = roc_auc_score(labels, scores_dino) * 100

        better = "CLIP" if auroc_clip > auroc_dino else "DINOv2"
        print(f"CLIP={auroc_clip:.1f}% DINOv2={auroc_dino:.1f}% → {better}", end="", flush=True)

        results['mvtec'][cat] = {
            'clip_knn': round(auroc_clip, 2),
            'dinov2_knn': round(auroc_dino, 2),
            'better': better
        }

    # ===== VisA =====
    print("\n\n" + "=" * 50)
    print("  VisA (12 categories)")
    print("=" * 50)

    for cat in VISA_CATEGORIES:
        print(f"\n  [{cat}]", end=" ", flush=True)

        ref_imgs = load_reference(cat, 'visa', k=1)
        test_imgs, labels = load_images(cat, 'visa', 'test')

        # CLIP kNN
        ref_feats_clip = extract_patch_features_clip(clip_model, ref_imgs, clip_preprocess, clip_backend)
        test_feats_clip = extract_patch_features_clip(clip_model, test_imgs, clip_preprocess, clip_backend)
        scores_clip = knn_anomaly_score(test_feats_clip, ref_feats_clip)
        auroc_clip = roc_auc_score(labels, scores_clip) * 100

        # DINOv2 kNN
        ref_feats_dino = extract_patch_features_dinov2(dinov2_model, ref_imgs, dinov2_preprocess)
        test_feats_dino = extract_patch_features_dinov2(dinov2_model, test_imgs, dinov2_preprocess)
        scores_dino = knn_anomaly_score(test_feats_dino, ref_feats_dino)
        auroc_dino = roc_auc_score(labels, scores_dino) * 100

        better = "CLIP" if auroc_clip > auroc_dino else "DINOv2"
        print(f"CLIP={auroc_clip:.1f}% DINOv2={auroc_dino:.1f}% → {better}", end="", flush=True)

        results['visa'][cat] = {
            'clip_knn': round(auroc_clip, 2),
            'dinov2_knn': round(auroc_dino, 2),
            'better': better
        }

    # Cleanup
    del clip_model, dinov2_model
    torch.cuda.empty_cache()

    # ===== Summary =====
    print("\n\n" + "=" * 70)
    print("  SUMMARY: Backbone Isolation (all 1-shot kNN)")
    print("=" * 70)

    for dataset in ['mvtec', 'visa']:
        cats = list(results[dataset].keys())
        clip_scores = [results[dataset][c]['clip_knn'] for c in cats]
        dino_scores = [results[dataset][c]['dinov2_knn'] for c in cats]

        clip_wins = sum(1 for c, d in zip(clip_scores, dino_scores) if c > d)
        dino_wins = sum(1 for c, d in zip(clip_scores, dino_scores) if d > c)

        print(f"\n  {dataset.upper()} (N={len(cats)}):")
        print(f"    CLIP kNN:   {np.mean(clip_scores):.2f}% (wins: {clip_wins})")
        print(f"    DINOv2 kNN: {np.mean(dino_scores):.2f}% (wins: {dino_wins})")

        # Show categories where CLIP wins
        clip_winning_cats = [c for c in cats if results[dataset][c]['clip_knn'] > results[dataset][c]['dinov2_knn']]
        if clip_winning_cats:
            print(f"    CLIP wins in: {', '.join(clip_winning_cats)}")

    # Also load WRN results from previous experiment for 3-way comparison
    prev_results_path = os.path.join(OUTPUT_DIR, 'additional_experiment_results.json')
    if os.path.exists(prev_results_path):
        with open(prev_results_path) as f:
            prev = json.load(f)

        print("\n\n  --- 3-BACKBONE COMPARISON (all 1-shot kNN) ---")
        for dataset in ['mvtec', 'visa']:
            cats = list(results[dataset].keys())
            print(f"\n  {dataset.upper()}:")
            print(f"  {'Category':<15} {'CLIP kNN':>10} {'DINOv2 kNN':>12} {'WRN kNN':>10} {'Best':>8}")
            print(f"  {'-'*55}")

            clip_scores = []
            dino_scores = []
            wrn_scores = []
            best_counts = {'CLIP': 0, 'DINOv2': 0, 'WRN': 0}

            for cat in cats:
                c = results[dataset][cat]['clip_knn']
                d = results[dataset][cat]['dinov2_knn']
                w = prev['controlled_setting'][dataset][cat]['wrn_knn']

                clip_scores.append(c)
                dino_scores.append(d)
                wrn_scores.append(w)

                best_val = max(c, d, w)
                if c == best_val:
                    best = 'CLIP'
                elif d == best_val:
                    best = 'DINOv2'
                else:
                    best = 'WRN'
                best_counts[best] += 1

                print(f"  {cat:<15} {c:>10.2f} {d:>12.2f} {w:>10.2f} {best:>8}")

            print(f"  {'Mean':<15} {np.mean(clip_scores):>10.2f} {np.mean(dino_scores):>12.2f} {np.mean(wrn_scores):>10.2f}")
            print(f"  Wins: CLIP={best_counts['CLIP']}, DINOv2={best_counts['DINOv2']}, WRN={best_counts['WRN']}")

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'clip_knn_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    run_experiment()
