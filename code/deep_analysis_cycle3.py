"""
Deep Analysis Cycle 3: k-shot experiments and difficulty analysis.

Experiments:
8. k-shot comparison (k=1,2,4,8): How does reference count affect VFM complementarity?
9. Anomaly difficulty analysis: What makes an anomaly "hard" for each model?

Based on Cycle 2 discoveries:
- Optimal alpha is method-dependent (r=0.128 between kNN and native)
- Color→CLIP, geometric→DINOv2 is statistically significant (ANOVA p=0.04)
- Equal fusion is nearly optimal (oracle only +0.80pp better)
"""

import os
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


def load_image(path, size=518):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


def load_image_clip(path, preprocess):
    img = Image.open(path).convert('RGB')
    return preprocess(img).unsqueeze(0)


@torch.no_grad()
def extract_clip_patches(clip_model, img_tensor):
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
    return x[:, 1:, :]


def get_categories():
    return sorted([d.name for d in MVTEC_ROOT.iterdir()
                  if d.is_dir() and d.name not in ['license.txt', 'meta.json', 'readme.txt']])


# ═════════════════════════════════════════════════════════════════════
# Experiment 8: k-shot comparison
# ═════════════════════════════════════════════════════════════════════

def experiment_8_kshot():
    """
    Compare CLIP kNN and DINOv2 kNN with k=1,2,4,8 reference images.

    Key questions:
    - Does CLIP benefit more from additional references than DINOv2?
    - Does the VFM winner change with k?
    - Does complementarity (fusion gain) change with k?
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 8: k-shot Comparison (k=1,2,4,8)")
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

    k_values = [1, 2, 4, 8]
    results = {}

    for cat in get_categories():
        print(f"\n--- {cat} ---")
        cat_dir = MVTEC_ROOT / cat
        test_dir = cat_dir / "test"
        train_dir = cat_dir / "train" / "good"
        train_images = sorted(list(train_dir.glob("*.png")))

        cat_results = {}

        for k in k_values:
            if k > len(train_images):
                print(f"  k={k}: Not enough training images ({len(train_images)} available)")
                continue

            # Extract reference features for k images
            ref_imgs = train_images[:k]

            with torch.no_grad():
                # DINOv2 references
                dino_refs = []
                for ref_path in ref_imgs:
                    ref_tensor = load_image(ref_path).to(device)
                    feat = dinov2.forward_features(ref_tensor)['x_norm_patchtokens']
                    dino_refs.append(feat)
                dino_ref_all = torch.cat(dino_refs, dim=1)  # [1, k*N, D]

                # CLIP references
                clip_refs = []
                for ref_path in ref_imgs:
                    ref_clip = load_image_clip(ref_path, clip_preprocess).to(device)
                    patches = extract_clip_patches(clip_model, ref_clip)
                    clip_refs.append(patches)
                clip_ref_all = torch.cat(clip_refs, dim=1)  # [1, k*N, D]

            # Score test images
            all_dino_scores = []
            all_clip_scores = []
            all_labels = []

            for subdir in sorted(test_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                is_good = (subdir.name == "good")
                for img_path in sorted(subdir.glob("*.png")):
                    with torch.no_grad():
                        # DINOv2
                        t = load_image(img_path).to(device)
                        dp = dinov2.forward_features(t)['x_norm_patchtokens']
                        ds = torch.cdist(dp, dino_ref_all).min(dim=-1).values.max().item()
                        all_dino_scores.append(ds)

                        # CLIP
                        tc = load_image_clip(img_path, clip_preprocess).to(device)
                        cp = extract_clip_patches(clip_model, tc)
                        cs = torch.cdist(cp.float(), clip_ref_all.float()).min(dim=-1).values.max().item()
                        all_clip_scores.append(cs)

                        all_labels.append(0 if is_good else 1)

            labels = np.array(all_labels)
            dino_arr = np.array(all_dino_scores)
            clip_arr = np.array(all_clip_scores)

            dino_auroc = roc_auc_score(labels, dino_arr) * 100
            clip_auroc = roc_auc_score(labels, clip_arr) * 100

            # Fusion
            dino_z = (dino_arr - dino_arr.mean()) / (dino_arr.std() + 1e-8)
            clip_z = (clip_arr - clip_arr.mean()) / (clip_arr.std() + 1e-8)
            fusion = 0.5 * dino_z + 0.5 * clip_z
            fusion_auroc = roc_auc_score(labels, fusion) * 100
            gain = fusion_auroc - max(dino_auroc, clip_auroc)

            cat_results[str(k)] = {
                "clip_auroc": round(clip_auroc, 2),
                "dino_auroc": round(dino_auroc, 2),
                "fusion_auroc": round(fusion_auroc, 2),
                "fusion_gain": round(gain, 2),
                "winner": "CLIP" if clip_auroc > dino_auroc else "DINOv2",
                "gap": round(abs(clip_auroc - dino_auroc), 2),
            }

            winner = "CLIP" if clip_auroc > dino_auroc else "DINOv2"
            print(f"  k={k}: CLIP={clip_auroc:.1f} DINO={dino_auroc:.1f} Fusion={fusion_auroc:.1f} "
                  f"({winner}, gap={abs(clip_auroc-dino_auroc):.1f}, gain={gain:+.1f})")

        results[cat] = cat_results

    # Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY: How k affects VFM complementarity")
    print("=" * 80)

    for k in k_values:
        k_str = str(k)
        clip_wins = sum(1 for c in results.values() if k_str in c and c[k_str]['winner'] == 'CLIP')
        total = sum(1 for c in results.values() if k_str in c)
        avg_gap = np.mean([c[k_str]['gap'] for c in results.values() if k_str in c])
        avg_gain = np.mean([c[k_str]['fusion_gain'] for c in results.values() if k_str in c])
        avg_clip = np.mean([c[k_str]['clip_auroc'] for c in results.values() if k_str in c])
        avg_dino = np.mean([c[k_str]['dino_auroc'] for c in results.values() if k_str in c])
        avg_fuse = np.mean([c[k_str]['fusion_auroc'] for c in results.values() if k_str in c])
        pos_gain = sum(1 for c in results.values() if k_str in c and c[k_str]['fusion_gain'] > 0)

        print(f"k={k}: CLIP wins {clip_wins}/{total} | "
              f"Avg CLIP={avg_clip:.1f} DINO={avg_dino:.1f} Fusion={avg_fuse:.1f} | "
              f"Avg gap={avg_gap:.1f} Avg gain={avg_gain:+.1f} | "
              f"Positive gain: {pos_gain}/{total}")

    # Check winner changes
    print("\n--- Categories where winner changes with k ---")
    for cat in sorted(results.keys()):
        winners = {k: results[cat][str(k)]['winner'] for k in k_values if str(k) in results[cat]}
        if len(set(winners.values())) > 1:
            print(f"  {cat}: " + " -> ".join(f"k={k}:{w}" for k, w in sorted(winners.items())))

    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment 9: Anomaly Difficulty Analysis
# ═════════════════════════════════════════════════════════════════════

def experiment_9_difficulty_analysis():
    """
    Analyze what makes anomalies "hard" or "easy" for each model.

    - Categorize images by detection status (both detect, CLIP-only, DINOv2-only, both miss)
    - Analyze properties: anomaly size, edge contrast, defect type
    - Compare CLIP-only detections vs DINOv2-only detections
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 9: Anomaly Difficulty Analysis")
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

    # For mask analysis
    mask_stats = defaultdict(list)
    all_image_data = []

    for cat in get_categories():
        print(f"\n--- {cat} ---")
        cat_dir = MVTEC_ROOT / cat
        test_dir = cat_dir / "test"
        train_dir = cat_dir / "train" / "good"
        mask_dir = cat_dir / "ground_truth"

        train_images = sorted(list(train_dir.glob("*.png")))
        ref_img_path = train_images[0]

        with torch.no_grad():
            ref_tensor = load_image(ref_img_path).to(device)
            dino_ref = dinov2.forward_features(ref_tensor)['x_norm_patchtokens']

            ref_clip = load_image_clip(ref_img_path, clip_preprocess).to(device)
            clip_ref = extract_clip_patches(clip_model, ref_clip)

        # Collect scores and masks
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

                # Get mask info for anomalous images
                mask_area_ratio = 0.0
                if not is_good:
                    mask_subdir = mask_dir / subdir.name
                    mask_name = img_path.stem + "_mask.png"
                    mask_path = mask_subdir / mask_name
                    if mask_path.exists():
                        mask = np.array(Image.open(mask_path).convert('L'))
                        mask_area_ratio = (mask > 128).sum() / mask.size
                    else:
                        # Try without _mask suffix
                        mask_path2 = mask_subdir / img_path.name
                        if mask_path2.exists():
                            mask = np.array(Image.open(mask_path2).convert('L'))
                            mask_area_ratio = (mask > 128).sum() / mask.size

                per_image.append({
                    "path": str(img_path),
                    "category": cat,
                    "label": 0 if is_good else 1,
                    "defect_type": "good" if is_good else subdir.name,
                    "dino_score": ds,
                    "clip_score": cs,
                    "mask_area_ratio": mask_area_ratio,
                })

        # Threshold and classify
        labels = np.array([d["label"] for d in per_image])
        dino_scores = np.array([d["dino_score"] for d in per_image])
        clip_scores = np.array([d["clip_score"] for d in per_image])

        good_dino = dino_scores[labels == 0]
        good_clip = clip_scores[labels == 0]
        dino_thresh = good_dino.mean() + 2 * good_dino.std()
        clip_thresh = good_clip.mean() + 2 * good_clip.std()

        anom_mask = labels == 1
        for i, d in enumerate(per_image):
            if d["label"] == 1:
                d["dino_detects"] = dino_scores[i] > dino_thresh
                d["clip_detects"] = clip_scores[i] > clip_thresh
                if d["dino_detects"] and d["clip_detects"]:
                    d["detection_class"] = "both"
                elif d["clip_detects"]:
                    d["detection_class"] = "clip_only"
                elif d["dino_detects"]:
                    d["detection_class"] = "dino_only"
                else:
                    d["detection_class"] = "neither"
                all_image_data.append(d)

    # Analyze mask area by detection class
    print("\n" + "=" * 80)
    print("Anomaly Size (mask area ratio) by Detection Class")
    print("=" * 80)

    detection_classes = ["both", "clip_only", "dino_only", "neither"]
    class_areas = {}
    for dc in detection_classes:
        areas = [d["mask_area_ratio"] for d in all_image_data
                if d["detection_class"] == dc and d["mask_area_ratio"] > 0]
        if areas:
            class_areas[dc] = areas
            print(f"{dc:12s} (n={len(areas):4d}): "
                  f"mean={np.mean(areas):.4f} median={np.median(areas):.4f} "
                  f"std={np.std(areas):.4f}")

    # Statistical tests: are CLIP-only and DINOv2-only anomalies different sizes?
    if "clip_only" in class_areas and "dino_only" in class_areas:
        t, p = stats.ttest_ind(class_areas["clip_only"], class_areas["dino_only"])
        u, up = stats.mannwhitneyu(class_areas["clip_only"], class_areas["dino_only"],
                                   alternative='two-sided')
        print(f"\nCLIP-only vs DINOv2-only mask area:")
        print(f"  t-test: t={t:.3f}, p={p:.4f}")
        print(f"  Mann-Whitney U: U={u:.0f}, p={up:.4f}")

    if "both" in class_areas and "neither" in class_areas:
        t, p = stats.ttest_ind(class_areas["both"], class_areas["neither"])
        print(f"\nBoth-detect vs Neither mask area:")
        print(f"  t-test: t={t:.3f}, p={p:.4f}")

    # Analyze by defect type
    print("\n" + "=" * 80)
    print("Detection Class Distribution by Defect Type (top disagreement types)")
    print("=" * 80)

    defect_breakdown = defaultdict(lambda: defaultdict(int))
    for d in all_image_data:
        key = f"{d['category']}/{d['defect_type']}"
        defect_breakdown[key][d["detection_class"]] += 1
        defect_breakdown[key]["total"] += 1

    # Sort by exclusive detection rate
    exclusive_rates = []
    for dt_key, counts in defect_breakdown.items():
        total = counts["total"]
        clip_only = counts.get("clip_only", 0)
        dino_only = counts.get("dino_only", 0)
        exclusive_rate = (clip_only + dino_only) / total if total > 0 else 0
        clip_only_rate = clip_only / total if total > 0 else 0
        dino_only_rate = dino_only / total if total > 0 else 0
        exclusive_rates.append({
            "defect": dt_key,
            "total": total,
            "both": counts.get("both", 0),
            "clip_only": clip_only,
            "dino_only": dino_only,
            "neither": counts.get("neither", 0),
            "exclusive_rate": exclusive_rate,
            "clip_only_rate": clip_only_rate,
            "dino_only_rate": dino_only_rate,
        })

    exclusive_rates.sort(key=lambda x: x["exclusive_rate"], reverse=True)

    print(f"{'Defect Type':35s} {'Total':>5s} {'Both':>5s} {'CLIP':>5s} {'DINO':>5s} {'Neither':>7s} {'Excl%':>6s}")
    print("-" * 75)
    for item in exclusive_rates[:25]:
        print(f"{item['defect']:35s} {item['total']:5d} {item['both']:5d} "
              f"{item['clip_only']:5d} {item['dino_only']:5d} {item['neither']:7d} "
              f"{item['exclusive_rate']*100:5.1f}%")

    # Compile results
    results = {
        "mask_area_by_class": {
            dc: {
                "n": len(areas),
                "mean": round(np.mean(areas), 6),
                "median": round(np.median(areas), 6),
                "std": round(np.std(areas), 6),
            }
            for dc, areas in class_areas.items()
        },
        "defect_type_breakdown": exclusive_rates[:30],
    }

    return results


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def save_results(results, tag=""):
    output_path = Path(__file__).parent / "deep_analysis_cycle3_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [Saved results ({tag})]")


if __name__ == "__main__":
    output_path = Path(__file__).parent / "deep_analysis_cycle3_results.json"

    if output_path.exists():
        with open(output_path, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")
    else:
        all_results = {}

    # Experiment 8: k-shot comparison
    if "experiment_8_kshot" not in all_results:
        print("\n" + "#" * 80)
        print("# Running Experiment 8: k-shot Comparison")
        print("#" * 80)
        exp8 = experiment_8_kshot()
        all_results["experiment_8_kshot"] = exp8
        save_results(all_results, "after exp8")
    else:
        print("Experiment 8 already completed, skipping.")

    # Experiment 9: Difficulty analysis
    if "experiment_9_difficulty" not in all_results:
        print("\n" + "#" * 80)
        print("# Running Experiment 9: Difficulty Analysis")
        print("#" * 80)
        exp9 = experiment_9_difficulty_analysis()
        all_results["experiment_9_difficulty"] = exp9
        save_results(all_results, "after exp9")
    else:
        print("Experiment 9 already completed, skipping.")

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll results saved to {output_path}")
    print("=" * 80)
    print("CYCLE 3 COMPLETE")
    print("=" * 80)
