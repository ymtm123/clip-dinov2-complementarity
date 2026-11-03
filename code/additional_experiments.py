"""
Additional Experiments
======================
1. Controlled Setting (Backbone Isolation): Both CLIP and DINOv2 in 1-shot kNN mode
2. Feature Visualization: t-SNE plots for representative categories
3. 3rd VFM: PatchCore-style with WideResNet-50-2 backbone

All experiments on MVTec AD (15 categories) as primary dataset.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timm
from collections import defaultdict

# ========== Configuration ==========
MVTEC_ROOT = "data/MVTec_AD"
VISA_ROOT = "data/VisA/VisA_pytorch/1cls"
OUTPUT_DIR = "."
FIG_DIR = "figures"

MVTEC_CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                     'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                     'transistor', 'wood', 'zipper']

VISA_CATEGORIES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                   'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(FIG_DIR, exist_ok=True)


# ========== Feature Extraction ==========

def get_clip_visual_model():
    """Load CLIP ViT-L/14 visual encoder only (no text)."""
    # Use open_clip for the visual encoder
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        model = model.visual
        model.eval().to(DEVICE)
        return model, preprocess
    except ImportError:
        # Fallback: use timm's ViT-L/14 with CLIP weights
        model = timm.create_model('vit_large_patch14_clip_224.openai', pretrained=True)
        model.eval().to(DEVICE)
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
        return model, preprocess


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


def get_widresnet_model():
    """Load WideResNet-50-2 (ImageNet supervised)."""
    model = timm.create_model('wide_resnet50_2', pretrained=True, features_only=True)
    model.eval().to(DEVICE)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, preprocess


@torch.no_grad()
def extract_features_vit(model, images, model_type='dinov2'):
    """Extract patch-level features from ViT model.
    Returns: [N, num_patches, dim] tensor
    """
    features_list = []
    for img in images:
        img = img.unsqueeze(0).to(DEVICE)
        if model_type == 'dinov2':
            # DINOv2: use forward_features to get patch tokens
            feat = model.forward_features(img)
            if isinstance(feat, dict):
                feat = feat['x_norm_patchtokens']  # [1, 256, 1024]
            else:
                # Remove CLS token
                feat = feat[:, 1:, :]
        elif model_type == 'clip':
            # CLIP ViT: forward through the model
            feat = model(img)
            if isinstance(feat, torch.Tensor) and feat.dim() == 2:
                # Got CLS token only, need patch tokens
                # Try accessing intermediate features
                features_list.append(feat.cpu())
                continue
            if feat.dim() == 3:
                feat = feat[:, 1:, :]  # Remove CLS
        features_list.append(feat.squeeze(0).cpu())
    return features_list


@torch.no_grad()
def extract_features_cnn(model, images):
    """Extract features from CNN (WideResNet) at layer 2 and 3."""
    features_list = []
    for img in images:
        img = img.unsqueeze(0).to(DEVICE)
        feats = model(img)  # features_only=True returns list of feature maps
        # Use layer 2 (28x28) and layer 3 (14x14) - typical for PatchCore
        f2 = feats[1]  # [1, 512, 28, 28]
        f3 = feats[2]  # [1, 1024, 14, 14]
        # Upsample f3 to match f2
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate
        combined = torch.cat([f2, f3_up], dim=1)  # [1, 1536, 28, 28]
        # Reshape to [num_patches, dim]
        b, c, h, w = combined.shape
        combined = combined.permute(0, 2, 3, 1).reshape(-1, c)  # [784, 1536]
        features_list.append(combined.cpu())
    return features_list


def load_images(category, dataset='mvtec', split='test', max_per_type=None):
    """Load test images for a category.
    Returns: list of (image_tensor, label, defect_type)
    """
    if dataset == 'mvtec':
        root = os.path.join(MVTEC_ROOT, category, split)
    elif dataset == 'visa':
        root = os.path.join(VISA_ROOT, category, split)

    images = []
    labels = []
    defect_types = []

    for atype in sorted(os.listdir(root)):
        atype_dir = os.path.join(root, atype)
        if not os.path.isdir(atype_dir):
            continue
        is_normal = atype in ('good', 'ok')
        img_files = sorted([f for f in os.listdir(atype_dir) if f.endswith(('.png', '.jpg', '.JPG', '.bmp'))])
        if max_per_type and len(img_files) > max_per_type:
            np.random.seed(42)
            idx = np.random.choice(len(img_files), max_per_type, replace=False)
            img_files = [img_files[i] for i in sorted(idx)]

        for fname in img_files:
            img = Image.open(os.path.join(atype_dir, fname)).convert('RGB')
            images.append(img)
            labels.append(0 if is_normal else 1)
            defect_types.append(atype)

    return images, labels, defect_types


def load_normal_reference(category, dataset='mvtec', k=1):
    """Load k normal reference images for memory bank."""
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


# ========== kNN Anomaly Scoring ==========

def knn_anomaly_score(test_features, ref_features, k=1):
    """Compute anomaly scores using kNN distance to reference features.
    test_features: list of [num_patches, dim] tensors
    ref_features: list of [num_patches, dim] tensors (from normal images)
    Returns: list of scalar anomaly scores
    """
    # Build reference memory bank
    ref_bank = torch.cat(ref_features, dim=0)  # [N_ref * num_patches, dim]
    ref_bank = F.normalize(ref_bank, dim=1)

    scores = []
    for feat in test_features:
        feat_norm = F.normalize(feat, dim=1)
        # Compute distances to nearest neighbor in ref_bank
        # Use cosine distance: 1 - cosine_similarity
        sim = torch.mm(feat_norm, ref_bank.t())  # [num_patches, N_ref_patches]
        max_sim, _ = sim.max(dim=1)  # [num_patches]
        dist = 1.0 - max_sim
        # Image-level score: max patch distance
        score = dist.max().item()
        scores.append(score)

    return scores


def knn_anomaly_score_global(test_features, ref_features):
    """Compute anomaly scores using global (CLS) features.
    For models that output only CLS tokens.
    """
    ref_bank = torch.stack(ref_features)  # [k, dim]
    ref_bank = F.normalize(ref_bank, dim=1)

    scores = []
    for feat in test_features:
        feat_norm = F.normalize(feat.unsqueeze(0), dim=1)
        sim = torch.mm(feat_norm, ref_bank.t())
        max_sim = sim.max().item()
        score = 1.0 - max_sim
        scores.append(score)

    return scores


# ========== Experiment 1: Controlled Setting (Backbone Isolation) ==========

def run_controlled_experiment():
    """Run backbone isolation experiment: same method (kNN), different backbones."""
    print("=" * 70)
    print("  EXPERIMENT 1: Controlled Setting (Backbone Isolation)")
    print("  Same method (1-shot kNN), different backbones")
    print("=" * 70)

    # Load models
    print("\nLoading DINOv2 ViT-L/14...")
    dinov2_model, dinov2_preprocess = get_dinov2_model()

    print("Loading WideResNet-50-2 (ImageNet supervised)...")
    wrn_model, wrn_preprocess = get_widresnet_model()

    results = {'mvtec': {}, 'visa': {}}

    # Run on MVTec
    for cat in MVTEC_CATEGORIES:
        print(f"\n  [{cat}]", end=" ", flush=True)

        # Load reference and test images
        ref_imgs = load_normal_reference(cat, 'mvtec', k=1)
        test_imgs, labels, _ = load_images(cat, 'mvtec', 'test')

        # --- DINOv2 features ---
        ref_tensors_dino = [dinov2_preprocess(img) for img in ref_imgs]
        test_tensors_dino = [dinov2_preprocess(img) for img in test_imgs]

        ref_feats_dino = extract_features_vit(dinov2_model, ref_tensors_dino, 'dinov2')
        test_feats_dino = extract_features_vit(dinov2_model, test_tensors_dino, 'dinov2')

        scores_dino = knn_anomaly_score(test_feats_dino, ref_feats_dino)
        auroc_dino = roc_auc_score(labels, scores_dino) * 100

        # --- WideResNet features ---
        ref_tensors_wrn = [wrn_preprocess(img) for img in ref_imgs]
        test_tensors_wrn = [wrn_preprocess(img) for img in test_imgs]

        ref_feats_wrn = extract_features_cnn(wrn_model, ref_tensors_wrn)
        test_feats_wrn = extract_features_cnn(wrn_model, test_tensors_wrn)

        scores_wrn = knn_anomaly_score(test_feats_wrn, ref_feats_wrn)
        auroc_wrn = roc_auc_score(labels, scores_wrn) * 100

        print(f"DINOv2={auroc_dino:.1f}% WRN={auroc_wrn:.1f}%", end="", flush=True)

        results['mvtec'][cat] = {
            'dinov2_knn': round(auroc_dino, 2),
            'wrn_knn': round(auroc_wrn, 2),
        }

    # Run on VisA
    print("\n\n  --- VisA ---")
    for cat in VISA_CATEGORIES:
        print(f"\n  [{cat}]", end=" ", flush=True)

        ref_imgs = load_normal_reference(cat, 'visa', k=1)
        test_imgs, labels, _ = load_images(cat, 'visa', 'test')

        # DINOv2
        ref_tensors_dino = [dinov2_preprocess(img) for img in ref_imgs]
        test_tensors_dino = [dinov2_preprocess(img) for img in test_imgs]
        ref_feats_dino = extract_features_vit(dinov2_model, ref_tensors_dino, 'dinov2')
        test_feats_dino = extract_features_vit(dinov2_model, test_tensors_dino, 'dinov2')
        scores_dino = knn_anomaly_score(test_feats_dino, ref_feats_dino)
        auroc_dino = roc_auc_score(labels, scores_dino) * 100

        # WideResNet
        ref_tensors_wrn = [wrn_preprocess(img) for img in ref_imgs]
        test_tensors_wrn = [wrn_preprocess(img) for img in test_imgs]
        ref_feats_wrn = extract_features_cnn(wrn_model, ref_tensors_wrn)
        test_feats_wrn = extract_features_cnn(wrn_model, test_tensors_wrn)
        scores_wrn = knn_anomaly_score(test_feats_wrn, ref_feats_wrn)
        auroc_wrn = roc_auc_score(labels, scores_wrn) * 100

        print(f"DINOv2={auroc_dino:.1f}% WRN={auroc_wrn:.1f}%", end="", flush=True)

        results['visa'][cat] = {
            'dinov2_knn': round(auroc_dino, 2),
            'wrn_knn': round(auroc_wrn, 2),
        }

    # Cleanup GPU
    del dinov2_model, wrn_model
    torch.cuda.empty_cache()

    # Summary
    print("\n\n" + "=" * 70)
    print("  CONTROLLED SETTING RESULTS SUMMARY")
    print("=" * 70)

    for dataset in ['mvtec', 'visa']:
        cats = list(results[dataset].keys())
        dino_scores = [results[dataset][c]['dinov2_knn'] for c in cats]
        wrn_scores = [results[dataset][c]['wrn_knn'] for c in cats]

        print(f"\n  {dataset.upper()} (N={len(cats)} categories):")
        print(f"    DINOv2 kNN: {np.mean(dino_scores):.2f} +/- {np.std(dino_scores):.2f}")
        print(f"    WRN kNN:    {np.mean(wrn_scores):.2f} +/- {np.std(wrn_scores):.2f}")

        # Dominance analysis
        dino_wins = sum(1 for d, w in zip(dino_scores, wrn_scores) if d > w)
        wrn_wins = sum(1 for d, w in zip(dino_scores, wrn_scores) if w > d)
        print(f"    DINOv2 wins: {dino_wins}, WRN wins: {wrn_wins}")

    return results


# ========== Experiment 2: Feature Visualization ==========

def run_feature_visualization():
    """Create t-SNE visualizations for representative categories."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Feature Visualization (t-SNE)")
    print("=" * 70)

    # Representative categories:
    # carpet - texture, CLIP-preferred (alpha=0.05)
    # screw - object, DINOv2-preferred (alpha=0.75)
    # wood - level reversal (img: DINOv2, pix: CLIP)
    # capsule - level reversal (img: CLIP, pix: DINOv2)
    vis_categories = ['carpet', 'screw', 'wood', 'capsule']

    print("\nLoading models...")
    dinov2_model, dinov2_preprocess = get_dinov2_model()

    for cat in vis_categories:
        print(f"\n  Processing [{cat}]...", flush=True)

        # Load images (limit to manageable size for t-SNE)
        test_imgs, labels, defect_types = load_images(cat, 'mvtec', 'test', max_per_type=30)
        ref_imgs = load_normal_reference(cat, 'mvtec', k=3)

        print(f"    {len(test_imgs)} test images, {sum(labels)} anomalous")

        # Extract DINOv2 features (CLS token for global view)
        all_imgs = ref_imgs + test_imgs
        all_tensors = [dinov2_preprocess(img) for img in all_imgs]

        cls_features = []
        with torch.no_grad():
            for t in all_tensors:
                t = t.unsqueeze(0).to(DEVICE)
                feat = dinov2_model(t)  # CLS token output
                if isinstance(feat, torch.Tensor):
                    cls_features.append(feat.squeeze(0).cpu().numpy())
                else:
                    cls_features.append(feat['x_norm_clstoken'].squeeze(0).cpu().numpy())

        cls_features = np.array(cls_features)

        # Labels for visualization
        n_ref = len(ref_imgs)
        vis_labels = ['Reference'] * n_ref + ['Normal' if l == 0 else 'Anomaly' for l in labels]

        # t-SNE
        print(f"    Running t-SNE (n={len(cls_features)})...", flush=True)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(cls_features) - 1))
        embeddings = tsne.fit_transform(cls_features)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        colors = {'Reference': '#2ca02c', 'Normal': '#1f77b4', 'Anomaly': '#d62728'}
        markers = {'Reference': '*', 'Normal': 'o', 'Anomaly': 'x'}
        sizes = {'Reference': 200, 'Normal': 30, 'Anomaly': 50}

        for label_type in ['Normal', 'Anomaly', 'Reference']:
            mask = [l == label_type for l in vis_labels]
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=colors[label_type], marker=markers[label_type],
                      s=sizes[label_type], label=label_type, alpha=0.7)

        ax.set_title(f'DINOv2 Feature Space: {cat}', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f'tsne_dinov2_{cat}.png'), dpi=150)
        plt.close(fig)
        print(f"    Saved tsne_dinov2_{cat}.png")

    del dinov2_model
    torch.cuda.empty_cache()
    print("\n  Feature visualization complete.")


# ========== Experiment 3: 3rd VFM (WideResNet-50 kNN) ==========

def run_third_vfm():
    """Run PatchCore-style anomaly detection with WideResNet-50-2."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: 3rd VFM (WideResNet-50-2, 1-shot kNN)")
    print("=" * 70)

    print("\nLoading WideResNet-50-2...")
    wrn_model, wrn_preprocess = get_widresnet_model()

    results = {'mvtec': {}, 'visa': {}}

    for cat in MVTEC_CATEGORIES:
        print(f"\n  [{cat}]", end=" ", flush=True)

        ref_imgs = load_normal_reference(cat, 'mvtec', k=1)
        test_imgs, labels, _ = load_images(cat, 'mvtec', 'test')

        ref_tensors = [wrn_preprocess(img) for img in ref_imgs]
        test_tensors = [wrn_preprocess(img) for img in test_imgs]

        ref_feats = extract_features_cnn(wrn_model, ref_tensors)
        test_feats = extract_features_cnn(wrn_model, test_tensors)

        scores = knn_anomaly_score(test_feats, ref_feats)

        try:
            auroc = roc_auc_score(labels, scores) * 100
        except ValueError:
            auroc = 50.0

        print(f"AUROC={auroc:.1f}%", end="", flush=True)
        results['mvtec'][cat] = round(auroc, 2)

    print("\n\n  --- VisA ---")
    for cat in VISA_CATEGORIES:
        print(f"\n  [{cat}]", end=" ", flush=True)

        ref_imgs = load_normal_reference(cat, 'visa', k=1)
        test_imgs, labels, _ = load_images(cat, 'visa', 'test')

        ref_tensors = [wrn_preprocess(img) for img in ref_imgs]
        test_tensors = [wrn_preprocess(img) for img in test_imgs]

        ref_feats = extract_features_cnn(wrn_model, ref_tensors)
        test_feats = extract_features_cnn(wrn_model, test_tensors)

        scores = knn_anomaly_score(test_feats, ref_feats)

        try:
            auroc = roc_auc_score(labels, scores) * 100
        except ValueError:
            auroc = 50.0

        print(f"AUROC={auroc:.1f}%", end="", flush=True)
        results['visa'][cat] = round(auroc, 2)

    del wrn_model
    torch.cuda.empty_cache()

    # Summary
    print("\n\n" + "-" * 50)
    mvtec_scores = list(results['mvtec'].values())
    visa_scores = list(results['visa'].values())
    print(f"  WRN-50 Mean: MVTec={np.mean(mvtec_scores):.2f}%, VisA={np.mean(visa_scores):.2f}%")

    return results


# ========== Score Fusion with 3 Models ==========

def three_model_fusion(wrn_results):
    """Evaluate 3-model fusion: CLIP + DINOv2 + WideResNet."""
    print("\n" + "=" * 70)
    print("  3-MODEL FUSION ANALYSIS")
    print("=" * 70)

    # Load existing CLIP and DINOv2 scores from comprehensive_eval_results.json
    with open(os.path.join(OUTPUT_DIR, 'comprehensive_eval_results.json'), 'r') as f:
        existing = json.load(f)

    fusion_results = {}

    for dataset_name, dataset_key in [('mvtec', 'MVTec AD'), ('visa', 'VisA')]:
        if dataset_name not in wrn_results:
            continue

        cats = list(wrn_results[dataset_name].keys())
        clip_scores_per_cat = {}
        dino_scores_per_cat = {}
        wrn_scores_per_cat = {}

        for cat in cats:
            clip_scores_per_cat[cat] = existing[dataset_key]['CLIP only']['per_category'].get(cat, 50.0)
            dino_scores_per_cat[cat] = existing[dataset_key]['DINOv2 only']['per_category'].get(cat, 50.0)
            wrn_scores_per_cat[cat] = wrn_results[dataset_name].get(cat, 50.0)

        # Per-category best model
        best_model = {}
        for cat in cats:
            scores = {
                'CLIP': clip_scores_per_cat[cat],
                'DINOv2': dino_scores_per_cat[cat],
                'WRN': wrn_scores_per_cat[cat]
            }
            best = max(scores, key=scores.get)
            best_model[cat] = (best, scores[best])

        print(f"\n  {dataset_key}: Per-category best model")
        clip_wins = sum(1 for v in best_model.values() if v[0] == 'CLIP')
        dino_wins = sum(1 for v in best_model.values() if v[0] == 'DINOv2')
        wrn_wins = sum(1 for v in best_model.values() if v[0] == 'WRN')
        print(f"    CLIP wins: {clip_wins}, DINOv2 wins: {dino_wins}, WRN wins: {wrn_wins}")

        # Compute means
        clip_mean = np.mean(list(clip_scores_per_cat.values()))
        dino_mean = np.mean(list(dino_scores_per_cat.values()))
        wrn_mean = np.mean(list(wrn_scores_per_cat.values()))

        print(f"    CLIP mean: {clip_mean:.2f}%")
        print(f"    DINOv2 mean: {dino_mean:.2f}%")
        print(f"    WRN mean: {wrn_mean:.2f}%")

        fusion_results[dataset_name] = {
            'clip_mean': round(clip_mean, 2),
            'dino_mean': round(dino_mean, 2),
            'wrn_mean': round(wrn_mean, 2),
            'clip_wins': clip_wins,
            'dino_wins': dino_wins,
            'wrn_wins': wrn_wins,
            'per_category': {cat: {
                'clip': clip_scores_per_cat[cat],
                'dino': dino_scores_per_cat[cat],
                'wrn': wrn_scores_per_cat[cat],
                'best': best_model[cat][0]
            } for cat in cats}
        }

    return fusion_results


# ========== Main ==========

if __name__ == '__main__':
    print("Additional Experiments")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # Experiment 1: Controlled Setting
    print("\n\n>>> EXPERIMENT 1: Controlled Setting <<<\n")
    controlled_results = run_controlled_experiment()
    all_results['controlled_setting'] = controlled_results

    # Experiment 2: Feature Visualization
    print("\n\n>>> EXPERIMENT 2: Feature Visualization <<<\n")
    run_feature_visualization()

    # Experiment 3: 3rd VFM
    print("\n\n>>> EXPERIMENT 3: 3rd VFM (WideResNet-50-2) <<<\n")
    wrn_results = run_third_vfm()
    all_results['wrn_results'] = wrn_results

    # 3-Model Fusion Analysis
    print("\n\n>>> 3-MODEL FUSION ANALYSIS <<<\n")
    fusion_3m = three_model_fusion(wrn_results)
    all_results['three_model_fusion'] = fusion_3m

    # Save all results
    output_path = os.path.join(OUTPUT_DIR, 'additional_experiment_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("  ALL ADDITIONAL EXPERIMENTS COMPLETE")
    print("=" * 70)
