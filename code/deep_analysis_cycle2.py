"""
Deep Analysis Cycle 2: Follow-up experiments based on v2 findings.

Experiments:
5. kNN fusion weight optimization (find optimal alpha per category)
6. CLS token vs patch token comparison for CLIP
7. Defect-type-level taxonomy: statistical test of "color→CLIP, structure→DINOv2"

Based on discoveries from cycle 1 (v2):
- CLIP wins 34.2% of defect types (color/texture anomalies)
- Fusion is effective for 12/15 categories at equal weight
- CLIP-only detections (17.8%) ≈ DINOv2-only detections (15.3%)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

MVTEC_ROOT = Path(r"data\MVTec_AD")

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy import stats


# ── Shared utilities ──

def load_image(path, size=518):
    """Load and preprocess image for DINOv2."""
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
    """Extract CLIP patch features WITH ln_post (corrected)."""
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
    x = visual.ln_post(x)
    patch_tokens = x[:, 1:, :]
    return patch_tokens


@torch.no_grad()
def extract_clip_cls(clip_model, img_tensor):
    """Extract CLIP CLS token (global image representation)."""
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
    x = visual.ln_post(x)
    cls_out = x[:, 0, :]  # CLS token only
    # Apply projection if available
    if visual.proj is not None:
        cls_out = cls_out @ visual.proj
    return cls_out


@torch.no_grad()
def extract_dinov2_cls(dinov2, img_tensor):
    """Extract DINOv2 CLS token."""
    feat = dinov2.forward_features(img_tensor)
    return feat['x_norm_clstoken']  # [1, D]


def get_categories():
    """Get sorted list of MVTec categories."""
    return sorted([d.name for d in MVTEC_ROOT.iterdir()
                  if d.is_dir() and d.name not in ['license.txt', 'meta.json', 'readme.txt']])


def collect_scores(device, dinov2, clip_model, clip_preprocess, cat,
                   use_cls_clip=False, use_cls_dino=False):
    """
    Collect per-image scores for a category.
    Returns: dict with dino_scores, clip_scores, labels, defect_types (arrays)
    """
    cat_dir = MVTEC_ROOT / cat
    test_dir = cat_dir / "test"
    train_dir = cat_dir / "train" / "good"

    train_images = sorted(list(train_dir.glob("*.png")))
    ref_img_path = train_images[0]

    # Reference features
    with torch.no_grad():
        ref_tensor = load_image(ref_img_path).to(device)
        if use_cls_dino:
            dino_ref = extract_dinov2_cls(dinov2, ref_tensor)  # [1, D]
        else:
            dino_ref = dinov2.forward_features(ref_tensor)['x_norm_patchtokens']

        ref_clip = load_image_clip(ref_img_path, clip_preprocess).to(device)
        if use_cls_clip:
            clip_ref = extract_clip_cls(clip_model, ref_clip)  # [1, D]
        else:
            clip_ref = extract_clip_patches(clip_model, ref_clip)

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
                if use_cls_dino:
                    dp = extract_dinov2_cls(dinov2, t)
                    ds = torch.cdist(dp, dino_ref).item()
                else:
                    dp = dinov2.forward_features(t)['x_norm_patchtokens']
                    ds = torch.cdist(dp, dino_ref).min(dim=-1).values.max().item()
                all_dino_scores.append(ds)

                # CLIP
                tc = load_image_clip(img_path, clip_preprocess).to(device)
                if use_cls_clip:
                    cp = extract_clip_cls(clip_model, tc)
                    cs = torch.cdist(cp.float(), clip_ref.float()).item()
                else:
                    cp = extract_clip_patches(clip_model, tc)
                    cs = torch.cdist(cp.float(), clip_ref.float()).min(dim=-1).values.max().item()
                all_clip_scores.append(cs)

                all_labels.append(0 if is_good else 1)
                all_defect_types.append("good" if is_good else subdir.name)

    return {
        'dino_scores': np.array(all_dino_scores),
        'clip_scores': np.array(all_clip_scores),
        'labels': np.array(all_labels),
        'defect_types': all_defect_types,
    }


# ═════════════════════════════════════════════════════════════════════
# Experiment 5: kNN Fusion Weight Optimization
# ═════════════════════════════════════════════════════════════════════

def experiment_5_fusion_weight_optimization():
    """
    For each category, sweep alpha from 0 to 1 and find the optimal fusion weight.
    Compare with:
    - Equal fusion (alpha=0.5)
    - Best single model
    - Oracle from native mode
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: kNN Fusion Weight Optimization")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading models...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    dinov2 = dinov2.to(device).eval()

    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    clip_model = clip_model.to(device).eval()

    # Oracle alphas from native mode (AnomalyCLIP + AnomalyDINO)
    native_oracle = {
        "bottle": 0.90, "cable": 0.55, "capsule": 0.40, "carpet": 0.05,
        "grid": 0.35, "hazelnut": 0.35, "leather": 0.50, "metal_nut": 0.90,
        "pill": 0.70, "screw": 0.75, "tile": 0.00, "toothbrush": 0.65,
        "transistor": 0.55, "wood": 0.70, "zipper": 0.65,
    }

    results = {}
    alphas = np.arange(0, 1.05, 0.05)  # 0.00, 0.05, ..., 1.00

    for cat in get_categories():
        print(f"\n--- {cat} ---")
        data = collect_scores(device, dinov2, clip_model, clip_preprocess, cat)

        dino_z = (data['dino_scores'] - data['dino_scores'].mean()) / (data['dino_scores'].std() + 1e-8)
        clip_z = (data['clip_scores'] - data['clip_scores'].mean()) / (data['clip_scores'].std() + 1e-8)

        clip_auroc = roc_auc_score(data['labels'], data['clip_scores']) * 100
        dino_auroc = roc_auc_score(data['labels'], data['dino_scores']) * 100

        # Sweep alpha
        alpha_aurocs = {}
        for alpha in alphas:
            alpha_r = round(alpha, 2)
            fused = alpha_r * dino_z + (1 - alpha_r) * clip_z
            auroc = roc_auc_score(data['labels'], fused) * 100
            alpha_aurocs[str(alpha_r)] = round(auroc, 2)

        # Find optimal
        best_alpha = max(alpha_aurocs, key=alpha_aurocs.get)
        best_auroc = alpha_aurocs[best_alpha]
        equal_auroc = alpha_aurocs["0.5"]

        native_alpha = native_oracle.get(cat, 0.5)
        native_alpha_key = str(round(native_alpha, 2))
        native_alpha_auroc = alpha_aurocs.get(native_alpha_key, None)

        results[cat] = {
            "clip_auroc": round(clip_auroc, 2),
            "dino_auroc": round(dino_auroc, 2),
            "best_single": round(max(clip_auroc, dino_auroc), 2),
            "equal_fusion_auroc": round(equal_auroc, 2),
            "oracle_knn_alpha": float(best_alpha),
            "oracle_knn_auroc": round(best_auroc, 2),
            "native_oracle_alpha": native_alpha,
            "native_alpha_knn_auroc": round(native_alpha_auroc, 2) if native_alpha_auroc else None,
            "equal_gain_over_best_single": round(equal_auroc - max(clip_auroc, dino_auroc), 2),
            "oracle_gain_over_equal": round(best_auroc - equal_auroc, 2),
            "alpha_sweep": alpha_aurocs,
        }

        winner_single = "CLIP" if clip_auroc > dino_auroc else "DINOv2"
        print(f"  CLIP={clip_auroc:.1f} DINO={dino_auroc:.1f} ({winner_single})")
        print(f"  Equal(0.5)={equal_auroc:.1f} Oracle(a={best_alpha})={best_auroc:.1f} "
              f"Native(a={native_alpha})={native_alpha_auroc:.1f}" if native_alpha_auroc else "")
        print(f"  kNN oracle alpha={best_alpha} vs native oracle alpha={native_alpha}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: kNN Oracle Alpha vs Native Mode Oracle Alpha")
    print("=" * 80)
    print(f"{'Category':15s} {'kNN a':>7s} {'Native a':>8s} {'Diff':>7s} {'kNN AUROC':>9s} {'Equal':>6s} {'Gain':>6s}")
    print("-" * 65)

    alpha_diffs = []
    for cat in sorted(results.keys()):
        r = results[cat]
        knn_a = r['oracle_knn_alpha']
        nat_a = r['native_oracle_alpha']
        diff = knn_a - nat_a
        alpha_diffs.append(diff)
        gain = r['oracle_gain_over_equal']
        print(f"{cat:15s} {knn_a:7.2f} {nat_a:8.2f} {diff:+7.2f} {r['oracle_knn_auroc']:9.2f} "
              f"{r['equal_fusion_auroc']:6.2f} {gain:+6.2f}")

    corr, p = stats.pearsonr(
        [results[c]['oracle_knn_alpha'] for c in sorted(results.keys())],
        [results[c]['native_oracle_alpha'] for c in sorted(results.keys())]
    )
    print(f"\nkNN Oracle alpha vs Native Oracle alpha correlation: r={corr:.3f}, p={p:.4f}")
    print(f"Mean absolute alpha difference: {np.mean(np.abs(alpha_diffs)):.3f}")

    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment 6: CLS Token vs Patch Token Comparison
# ═════════════════════════════════════════════════════════════════════

def experiment_6_cls_vs_patch():
    """
    Compare CLIP CLS token kNN vs CLIP patch token kNN.
    Also compare DINOv2 CLS vs patch.

    Hypothesis: CLS token captures global semantics while patch tokens
    capture local anomalies. The relative performance may differ by
    anomaly type.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: CLS Token vs Patch Token Comparison")
    print("=" * 80)

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

    for cat in get_categories():
        print(f"\n--- {cat} ---")

        # Collect scores with patch tokens (default)
        patch_data = collect_scores(device, dinov2, clip_model, clip_preprocess, cat,
                                   use_cls_clip=False, use_cls_dino=False)

        # Collect scores with CLS tokens
        cls_data = collect_scores(device, dinov2, clip_model, clip_preprocess, cat,
                                 use_cls_clip=True, use_cls_dino=True)

        labels = patch_data['labels']

        # AUROC for each combination
        clip_patch = roc_auc_score(labels, patch_data['clip_scores']) * 100
        clip_cls = roc_auc_score(labels, cls_data['clip_scores']) * 100
        dino_patch = roc_auc_score(labels, patch_data['dino_scores']) * 100
        dino_cls = roc_auc_score(labels, cls_data['dino_scores']) * 100

        # Cross-fusion: CLIP CLS + DINOv2 patch (semantic + spatial)
        clip_cls_z = (cls_data['clip_scores'] - cls_data['clip_scores'].mean()) / (cls_data['clip_scores'].std() + 1e-8)
        dino_patch_z = (patch_data['dino_scores'] - patch_data['dino_scores'].mean()) / (patch_data['dino_scores'].std() + 1e-8)
        clip_patch_z = (patch_data['clip_scores'] - patch_data['clip_scores'].mean()) / (patch_data['clip_scores'].std() + 1e-8)
        dino_cls_z = (cls_data['dino_scores'] - cls_data['dino_scores'].mean()) / (cls_data['dino_scores'].std() + 1e-8)

        # Fusion: CLIP_CLS + DINOv2_patch (semantic global + spatial local)
        cross_fused = 0.5 * clip_cls_z + 0.5 * dino_patch_z
        cross_auroc = roc_auc_score(labels, cross_fused) * 100

        # Standard fusion: CLIP_patch + DINOv2_patch
        standard_fused = 0.5 * clip_patch_z + 0.5 * dino_patch_z
        standard_auroc = roc_auc_score(labels, standard_fused) * 100

        # 3-way fusion: CLIP_CLS + CLIP_patch + DINOv2_patch
        triple_fused = (1/3) * clip_cls_z + (1/3) * clip_patch_z + (1/3) * dino_patch_z
        triple_auroc = roc_auc_score(labels, triple_fused) * 100

        results[cat] = {
            "clip_patch_auroc": round(clip_patch, 2),
            "clip_cls_auroc": round(clip_cls, 2),
            "dino_patch_auroc": round(dino_patch, 2),
            "dino_cls_auroc": round(dino_cls, 2),
            "clip_cls_advantage": round(clip_cls - clip_patch, 2),
            "dino_cls_advantage": round(dino_cls - dino_patch, 2),
            "standard_fusion_auroc": round(standard_auroc, 2),
            "cross_fusion_auroc": round(cross_auroc, 2),
            "triple_fusion_auroc": round(triple_auroc, 2),
            "cross_vs_standard": round(cross_auroc - standard_auroc, 2),
            "triple_vs_standard": round(triple_auroc - standard_auroc, 2),
        }

        print(f"  CLIP: patch={clip_patch:.1f} CLS={clip_cls:.1f} (diff={clip_cls-clip_patch:+.1f})")
        print(f"  DINO: patch={dino_patch:.1f} CLS={dino_cls:.1f} (diff={dino_cls-dino_patch:+.1f})")
        print(f"  Fusion: standard={standard_auroc:.1f} cross(CLIP_CLS+DINO_patch)={cross_auroc:.1f} "
              f"triple={triple_auroc:.1f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    clip_cls_advantages = [results[c]['clip_cls_advantage'] for c in sorted(results.keys())]
    dino_cls_advantages = [results[c]['dino_cls_advantage'] for c in sorted(results.keys())]
    cross_vs_std = [results[c]['cross_vs_standard'] for c in sorted(results.keys())]

    print(f"CLIP CLS advantage over patch: mean={np.mean(clip_cls_advantages):+.2f} "
          f"(positive {sum(1 for x in clip_cls_advantages if x > 0)}/15)")
    print(f"DINOv2 CLS advantage over patch: mean={np.mean(dino_cls_advantages):+.2f} "
          f"(positive {sum(1 for x in dino_cls_advantages if x > 0)}/15)")
    print(f"Cross fusion (CLIP_CLS+DINO_patch) vs standard: mean={np.mean(cross_vs_std):+.2f} "
          f"(positive {sum(1 for x in cross_vs_std if x > 0)}/15)")

    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment 7: Defect-Type Taxonomy Statistical Test
# ═════════════════════════════════════════════════════════════════════

def experiment_7_defect_taxonomy():
    """
    Classify each defect type by its anomaly nature and test whether
    the "color→CLIP, structure→DINOv2" pattern holds statistically.

    Uses results from experiment 1 v2 (already computed).
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: Defect-Type Level Taxonomy")
    print("=" * 80)

    # Manual classification of MVTec defect types by anomaly nature
    # Categories: color, texture_surface, structural, geometric, foreign_object, semantic
    defect_taxonomy = {
        # bottle
        ("bottle", "broken_large"): "structural",
        ("bottle", "broken_small"): "structural",
        ("bottle", "contamination"): "foreign_object",
        # cable
        ("cable", "bent_wire"): "geometric",
        ("cable", "cable_swap"): "semantic",
        ("cable", "combined"): "mixed",
        ("cable", "cut_inner_insulation"): "geometric",
        ("cable", "cut_outer_insulation"): "geometric",
        ("cable", "missing_cable"): "semantic",
        ("cable", "missing_wire"): "semantic",
        ("cable", "poke_insulation"): "structural",
        # capsule
        ("capsule", "crack"): "structural",
        ("capsule", "faulty_imprint"): "color",
        ("capsule", "poke"): "structural",
        ("capsule", "scratch"): "texture_surface",
        ("capsule", "squeeze"): "geometric",
        # carpet
        ("carpet", "color"): "color",
        ("carpet", "cut"): "structural",
        ("carpet", "hole"): "structural",
        ("carpet", "metal_contamination"): "foreign_object",
        ("carpet", "thread"): "texture_surface",
        # grid
        ("grid", "bent"): "geometric",
        ("grid", "broken"): "structural",
        ("grid", "glue"): "foreign_object",
        ("grid", "metal_contamination"): "foreign_object",
        ("grid", "thread"): "texture_surface",
        # hazelnut
        ("hazelnut", "crack"): "structural",
        ("hazelnut", "cut"): "structural",
        ("hazelnut", "hole"): "structural",
        ("hazelnut", "print"): "color",
        # leather
        ("leather", "color"): "color",
        ("leather", "cut"): "structural",
        ("leather", "fold"): "geometric",
        ("leather", "glue"): "foreign_object",
        ("leather", "poke"): "structural",
        # metal_nut
        ("metal_nut", "bent"): "geometric",
        ("metal_nut", "color"): "color",
        ("metal_nut", "flip"): "semantic",
        ("metal_nut", "scratch"): "texture_surface",
        # pill
        ("pill", "color"): "color",
        ("pill", "combined"): "mixed",
        ("pill", "contamination"): "foreign_object",
        ("pill", "crack"): "structural",
        ("pill", "faulty_imprint"): "color",
        ("pill", "pill_type"): "semantic",
        ("pill", "scratch"): "texture_surface",
        # screw
        ("screw", "manipulated_front"): "geometric",
        ("screw", "scratch_head"): "texture_surface",
        ("screw", "scratch_neck"): "texture_surface",
        ("screw", "thread_side"): "texture_surface",
        ("screw", "thread_top"): "texture_surface",
        # tile
        ("tile", "crack"): "structural",
        ("tile", "glue_strip"): "foreign_object",
        ("tile", "gray_stroke"): "color",
        ("tile", "oil"): "foreign_object",
        ("tile", "rough"): "texture_surface",
        # toothbrush
        ("toothbrush", "defective"): "structural",
        # transistor
        ("transistor", "bent_lead"): "geometric",
        ("transistor", "cut_lead"): "geometric",
        ("transistor", "damaged_case"): "structural",
        ("transistor", "misplaced"): "semantic",
        # wood
        ("wood", "color"): "color",
        ("wood", "combined"): "mixed",
        ("wood", "hole"): "structural",
        ("wood", "liquid"): "foreign_object",
        ("wood", "scratch"): "texture_surface",
        # zipper
        ("zipper", "broken_teeth"): "structural",
        ("zipper", "combined"): "mixed",
        ("zipper", "fabric_border"): "texture_surface",
        ("zipper", "fabric_interior"): "texture_surface",
        ("zipper", "rough"): "texture_surface",
        ("zipper", "split_teeth"): "structural",
        ("zipper", "squeezed_teeth"): "geometric",
    }

    # Load experiment 1 v2 results
    results_path = Path(__file__).parent / "deep_analysis_results.json"
    with open(results_path, 'r') as f:
        all_data = json.load(f)

    exp1 = all_data['experiment_1_defect_types']

    # Collect CLIP-DINOv2 advantage per taxonomy category
    taxonomy_stats = defaultdict(list)

    for cat in exp1:
        for dt, vals in exp1[cat]['defect_types'].items():
            key = (cat, dt)
            if key in defect_taxonomy:
                nature = defect_taxonomy[key]
                diff = vals['diff_clip_minus_dino']
                taxonomy_stats[nature].append({
                    'category': cat,
                    'defect_type': dt,
                    'clip_auroc': vals['clip_auroc'],
                    'dino_auroc': vals['dino_auroc'],
                    'diff': diff,
                })

    # Statistical analysis
    print("\n--- CLIP-DINOv2 AUROC Difference by Anomaly Nature ---")
    print(f"{'Nature':20s} {'N':>4s} {'Mean diff':>10s} {'Median':>8s} {'Std':>8s} {'CLIP wins':>10s}")
    print("-" * 70)

    nature_summary = {}
    for nature in sorted(taxonomy_stats.keys()):
        items = taxonomy_stats[nature]
        diffs = [x['diff'] for x in items]
        n = len(diffs)
        mean_d = np.mean(diffs)
        median_d = np.median(diffs)
        std_d = np.std(diffs)
        clip_wins = sum(1 for d in diffs if d > 0)
        print(f"{nature:20s} {n:4d} {mean_d:+10.2f} {median_d:+8.2f} {std_d:8.2f} {clip_wins}/{n}")

        # One-sample t-test: is mean diff significantly different from 0?
        if n >= 3:
            t_stat, p_val = stats.ttest_1samp(diffs, 0)
        else:
            t_stat, p_val = float('nan'), float('nan')

        nature_summary[nature] = {
            'n': n,
            'mean_diff': round(mean_d, 2),
            'median_diff': round(median_d, 2),
            'std': round(std_d, 2),
            'clip_win_rate': round(clip_wins / n * 100, 1),
            't_stat': round(t_stat, 3) if not np.isnan(t_stat) else None,
            'p_value': round(p_val, 4) if not np.isnan(p_val) else None,
            'defect_types': items,
        }

    # Test: "color" vs "structural" vs "geometric"
    print("\n--- Key Comparisons ---")

    color_diffs = [x['diff'] for x in taxonomy_stats.get('color', [])]
    structural_diffs = [x['diff'] for x in taxonomy_stats.get('structural', [])]
    geometric_diffs = [x['diff'] for x in taxonomy_stats.get('geometric', [])]
    surface_diffs = [x['diff'] for x in taxonomy_stats.get('texture_surface', [])]

    if color_diffs and structural_diffs:
        t, p = stats.ttest_ind(color_diffs, structural_diffs)
        print(f"Color vs Structural: t={t:.3f}, p={p:.4f}")
        print(f"  Color mean:      {np.mean(color_diffs):+.2f} (n={len(color_diffs)})")
        print(f"  Structural mean: {np.mean(structural_diffs):+.2f} (n={len(structural_diffs)})")

    if color_diffs and geometric_diffs:
        t, p = stats.ttest_ind(color_diffs, geometric_diffs)
        print(f"Color vs Geometric: t={t:.3f}, p={p:.4f}")
        print(f"  Color mean:     {np.mean(color_diffs):+.2f} (n={len(color_diffs)})")
        print(f"  Geometric mean: {np.mean(geometric_diffs):+.2f} (n={len(geometric_diffs)})")

    if surface_diffs and geometric_diffs:
        t, p = stats.ttest_ind(surface_diffs, geometric_diffs)
        print(f"Texture_surface vs Geometric: t={t:.3f}, p={p:.4f}")

    # ANOVA across all groups
    all_groups = [v for v in taxonomy_stats.values() if len(v) >= 3]
    all_diffs_groups = [[x['diff'] for x in g] for g in all_groups if len(g) >= 3]
    if len(all_diffs_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*all_diffs_groups)
        print(f"\nOne-way ANOVA across all anomaly natures: F={f_stat:.3f}, p={p_val:.4f}")

    return nature_summary


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def save_results(results, tag=""):
    output_path = Path(__file__).parent / "deep_analysis_cycle2_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [Saved results ({tag})]")


if __name__ == "__main__":
    output_path = Path(__file__).parent / "deep_analysis_cycle2_results.json"

    if output_path.exists():
        with open(output_path, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")
    else:
        all_results = {}

    # Experiment 7 first (no GPU needed, uses existing data)
    if "experiment_7_defect_taxonomy" not in all_results:
        print("\n" + "#" * 80)
        print("# Running Experiment 7: Defect-Type Taxonomy")
        print("#" * 80)
        exp7 = experiment_7_defect_taxonomy()
        all_results["experiment_7_defect_taxonomy"] = exp7
        save_results(all_results, "after exp7")
    else:
        print("Experiment 7 already completed, skipping.")

    # Experiment 5: Fusion weight optimization (needs GPU)
    if "experiment_5_fusion_weights" not in all_results:
        print("\n" + "#" * 80)
        print("# Running Experiment 5: kNN Fusion Weight Optimization")
        print("#" * 80)
        exp5 = experiment_5_fusion_weight_optimization()
        all_results["experiment_5_fusion_weights"] = exp5
        save_results(all_results, "after exp5")
    else:
        print("Experiment 5 already completed, skipping.")

    # Experiment 6: CLS vs patch (needs GPU, runs 2x per category)
    if "experiment_6_cls_vs_patch" not in all_results:
        print("\n" + "#" * 80)
        print("# Running Experiment 6: CLS vs Patch Token")
        print("#" * 80)
        exp6 = experiment_6_cls_vs_patch()
        all_results["experiment_6_cls_vs_patch"] = exp6
        save_results(all_results, "after exp6")
    else:
        print("Experiment 6 already completed, skipping.")

    # Final save
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nAll results saved to {output_path}")
    print("=" * 80)
    print("CYCLE 2 COMPLETE")
    print("=" * 80)
