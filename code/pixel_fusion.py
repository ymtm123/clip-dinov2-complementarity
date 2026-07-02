"""
Pixel-Level Fusion of AnomalyCLIP and AnomalyDINO Anomaly Maps
================================================================

Extends score-level fusion to pixel-level anomaly segmentation.
Loads anomaly maps from both models, resizes to common resolution,
applies fusion strategies, and evaluates pixel-level AUROC/AUPRO.

AnomalyCLIP maps: [H, W] at 518x518 resolution
AnomalyDINO maps: [32, 32] patch grid (needs upsampling)

Both are resized to a common target resolution for fair comparison.
"""

import os
import re
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from PIL import Image
import cv2


# ============================================================================
# Map Loading
# ============================================================================

def load_anomalyclip_maps(maps_dir, dataset_root, dataset='mvtec'):
    """
    Load AnomalyCLIP anomaly maps and corresponding ground truth masks.

    Directory structure:
      maps_dir/anomaly_maps/{object}/test/{anomaly_type}/{image_id}.npy

    Returns: {object: [{map: ndarray, gt_mask: ndarray, anomaly_type: str, img_id: str}, ...]}
    """
    data = defaultdict(list)
    am_dir = os.path.join(maps_dir, "anomaly_maps")

    if not os.path.exists(am_dir):
        print(f"  [WARNING] AnomalyCLIP maps directory not found: {am_dir}")
        return dict(data)

    for obj in sorted(os.listdir(am_dir)):
        test_dir = os.path.join(am_dir, obj, "test")
        if not os.path.isdir(test_dir):
            continue

        for atype in sorted(os.listdir(test_dir)):
            atype_dir = os.path.join(test_dir, atype)
            if not os.path.isdir(atype_dir):
                continue

            for npy_file in sorted(os.listdir(atype_dir)):
                if not npy_file.endswith('.npy'):
                    continue

                img_id = os.path.splitext(npy_file)[0]
                amap = np.load(os.path.join(atype_dir, npy_file))

                # Load ground truth mask
                # Support both 'good' (MVTec/VisA) and 'ok' (BTAD) naming
                is_normal = (atype in ('good', 'ok'))
                if is_normal:
                    gt_mask = np.zeros_like(amap, dtype=np.float32)
                else:
                    # Find GT mask — extension-agnostic. Naming differs by dataset:
                    # MVTec '<id>_mask.png', VisA/BTAD-01/02 '<id>.png', BTAD-03 '<id>.bmp'.
                    # Try the named patterns first, then glob ANY extension so e.g. BTAD-03's
                    # .bmp masks are loaded (the previous .png-only lookup silently fell through
                    # to an all-ones mask, fabricating pixel ground truth -- never do that).
                    import glob as _glob
                    gt_dir = os.path.join(dataset_root, obj, "ground_truth", atype)
                    gt_mask_path = None
                    for cand in (f"{img_id}_mask.png", f"{img_id}.png"):
                        p = os.path.join(gt_dir, cand)
                        if os.path.exists(p):
                            gt_mask_path = p
                            break
                    if gt_mask_path is None:
                        cands = sorted(_glob.glob(os.path.join(gt_dir, f"{img_id}_mask.*")) +
                                       _glob.glob(os.path.join(gt_dir, f"{img_id}.*")))
                        cands = [c for c in cands if not c.lower().endswith(".npy")]
                        gt_mask_path = cands[0] if cands else None
                    if gt_mask_path is not None:
                        gt_img = Image.open(gt_mask_path).convert('L')
                        gt_mask = np.array(gt_img, dtype=np.float32) / 255.0
                    else:
                        raise FileNotFoundError(
                            f"No ground-truth mask for anomalous {obj}/{atype}/{img_id} in {gt_dir} "
                            f"(refusing to fabricate an all-ones mask)")

                data[obj].append({
                    'map': amap.astype(np.float32),
                    'gt_mask': gt_mask,
                    'anomaly_type': atype,
                    'img_id': img_id,
                    'is_normal': is_normal,
                })

    return dict(data)


def load_anomalydino_maps(results_dir):
    """
    Load AnomalyDINO anomaly maps (.npy patch grids).

    Directory structure:
      results_dir/anomaly_maps/seed=0/{object}/test/{anomaly_type}/{image_id}.npy

    Returns: {object: [{map: ndarray, anomaly_type: str, img_id: str}, ...]}
    """
    data = defaultdict(list)
    am_dir = os.path.join(results_dir, "anomaly_maps", "seed=0")

    if not os.path.exists(am_dir):
        print(f"  [WARNING] AnomalyDINO maps directory not found: {am_dir}")
        return dict(data)

    for obj in sorted(os.listdir(am_dir)):
        test_dir = os.path.join(am_dir, obj, "test")
        if not os.path.isdir(test_dir):
            continue

        for atype in sorted(os.listdir(test_dir)):
            atype_dir = os.path.join(test_dir, atype)
            if not os.path.isdir(atype_dir):
                continue

            for npy_file in sorted(os.listdir(atype_dir)):
                if not npy_file.endswith('.npy'):
                    continue

                img_id = os.path.splitext(npy_file)[0]
                amap = np.load(os.path.join(atype_dir, npy_file))

                data[obj].append({
                    'map': amap.astype(np.float32),
                    'anomaly_type': atype,
                    'img_id': img_id,
                    'is_normal': (atype in ('good', 'ok')),
                })

    return dict(data)


# ============================================================================
# Map Alignment and Resizing
# ============================================================================

def resize_map(amap, target_size):
    """Resize anomaly map to target_size using bilinear interpolation."""
    if amap.shape[0] == target_size[0] and amap.shape[1] == target_size[1]:
        return amap
    resized = cv2.resize(amap, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return resized


def normalize_map(amap, method='minmax'):
    """Normalize anomaly map to [0, 1]."""
    if method == 'minmax':
        amin, amax = amap.min(), amap.max()
        if amax - amin < 1e-10:
            return np.zeros_like(amap)
        return (amap - amin) / (amax - amin)
    return amap


def align_maps(clip_data, dino_data, target_size=(256, 256), dino_sigma=4):
    """
    Align AnomalyCLIP and AnomalyDINO maps for the same test images.

    Matches by anomaly_type/img_id key.
    Resizes all maps to target_size.
    """
    aligned = {}
    common_objs = set(clip_data.keys()) & set(dino_data.keys())

    for obj in sorted(common_objs):
        # Build index by (anomaly_type, img_id) key
        clip_index = {}
        for item in clip_data[obj]:
            key = f"{item['anomaly_type']}/{item['img_id']}"
            clip_index[key] = item

        dino_index = {}
        for item in dino_data[obj]:
            key = f"{item['anomaly_type']}/{item['img_id']}"
            dino_index[key] = item

        common_keys = set(clip_index.keys()) & set(dino_index.keys())

        if len(common_keys) == 0:
            print(f"  {obj}: no common samples (CLIP: {len(clip_index)}, DINO: {len(dino_index)})")
            # Debug: show sample keys
            if clip_index:
                print(f"    CLIP sample keys: {list(clip_index.keys())[:3]}")
            if dino_index:
                print(f"    DINO sample keys: {list(dino_index.keys())[:3]}")
            continue

        aligned_items = []
        for key in sorted(common_keys):
            ci = clip_index[key]
            di = dino_index[key]

            # Resize CLIP map to target
            clip_map = resize_map(ci['map'], target_size)
            # Resize GT mask to target
            gt_mask = resize_map(ci['gt_mask'], target_size)
            gt_mask = (gt_mask > 0.5).astype(np.float32)

            # Resize DINO map (from patch grid to full resolution) + gaussian smooth
            dino_map = resize_map(di['map'], target_size)
            dino_map = gaussian_filter(dino_map, sigma=dino_sigma)

            aligned_items.append({
                'clip_map': clip_map,
                'dino_map': dino_map,
                'gt_mask': gt_mask,
                'anomaly_type': ci['anomaly_type'],
                'img_id': ci['img_id'],
                'is_normal': ci['is_normal'],
            })

        n_total_clip = len(clip_data[obj])
        n_total_dino = len(dino_data[obj])
        n_aligned = len(aligned_items)
        if n_aligned == max(n_total_clip, n_total_dino):
            print(f"  {obj}: perfectly aligned {n_aligned} samples")
        else:
            print(f"  {obj}: aligned {n_aligned}/{n_total_clip} (CLIP) and {n_aligned}/{n_total_dino} (DINO)")

        aligned[obj] = aligned_items

    return aligned


# ============================================================================
# Pixel-Level Evaluation
# ============================================================================

def compute_pixel_auroc(gt_masks, pred_maps):
    """Compute pixel-level AUROC."""
    gt_flat = np.concatenate([m.ravel() for m in gt_masks])
    pred_flat = np.concatenate([m.ravel() for m in pred_maps])

    if len(np.unique(gt_flat)) < 2:
        return float('nan')

    return roc_auc_score(gt_flat, pred_flat) * 100


def compute_pro(gt_masks, pred_maps, num_thresholds=200, max_fpr=0.3):
    """Compute per-region overlap (PRO) metric. Simplified version."""
    from scipy.ndimage import label as connected_components

    gt_flat = np.concatenate([m.ravel() for m in gt_masks])
    pred_flat = np.concatenate([m.ravel() for m in pred_maps])

    if gt_flat.sum() == 0:
        return float('nan')

    thresholds = np.linspace(pred_flat.max(), pred_flat.min(), num_thresholds)

    fprs = []
    pros = []

    for threshold in thresholds:
        binary_pred = (pred_flat >= threshold).astype(np.float32)

        # FPR
        fp = binary_pred[gt_flat == 0].sum()
        tn = (1 - binary_pred[gt_flat == 0]).sum()
        fpr = fp / (fp + tn + 1e-10)

        # PRO: per-region overlap
        region_overlaps = []
        for gt_m, pred_m in zip(gt_masks, pred_maps):
            if gt_m.sum() == 0:
                continue
            binary_p = (pred_m >= threshold).astype(np.float32)
            labeled, n_regions = connected_components(gt_m > 0.5)
            for r in range(1, n_regions + 1):
                region_mask = (labeled == r).astype(np.float32)
                overlap = (binary_p * region_mask).sum() / (region_mask.sum() + 1e-10)
                region_overlaps.append(overlap)

        pro = np.mean(region_overlaps) if region_overlaps else 0
        fprs.append(fpr)
        pros.append(pro)

        if fpr > max_fpr:
            break

    # Compute AUPRO (area under PRO curve up to max_fpr)
    fprs = np.array(fprs)
    pros = np.array(pros)
    sorted_idx = np.argsort(fprs)
    fprs = fprs[sorted_idx]
    pros = pros[sorted_idx]

    # Clip to max_fpr
    valid = fprs <= max_fpr
    if valid.sum() < 2:
        return 0.0

    aupro = np.trapz(pros[valid], fprs[valid]) / max_fpr
    return aupro * 100


# ============================================================================
# Fusion Strategies (Pixel-Level)
# ============================================================================

def pixel_fusion(clip_maps, dino_maps, gt_masks, alpha=0.5, normalize_method='minmax'):
    """
    Fuse pixel-level anomaly maps.
    Returns fused maps and evaluation metrics.
    """
    fused_maps = []

    for cm, dm in zip(clip_maps, dino_maps):
        if normalize_method:
            cm_norm = normalize_map(cm, normalize_method)
            dm_norm = normalize_map(dm, normalize_method)
        else:
            cm_norm = cm
            dm_norm = dm

        fused = (1 - alpha) * cm_norm + alpha * dm_norm
        fused_maps.append(fused)

    return fused_maps


def evaluate_pixel_fusion(aligned_data, dataset_name=""):
    """Evaluate multiple pixel-level fusion strategies."""

    print(f"\n{'='*90}")
    print(f"  PIXEL-LEVEL FUSION RESULTS -- {dataset_name}")
    print(f"{'='*90}")

    strategies = ['CLIP only', 'DINOv2 only', 'Equal (a=0.5)', 'CLIP-heavy (a=0.3)',
                  'DINO-heavy (a=0.7)', 'Max fusion']

    # Header
    print(f"\n{'Category':<15} {'CLIP':>8} {'DINO':>8} {'Eq(0.5)':>8} {'C(0.3)':>8} "
          f"{'D(0.7)':>8} {'Max':>8} {'Oracle':>8}")
    print("-" * 85)

    all_results = {s: {} for s in strategies}
    all_results['Oracle'] = {}
    oracle_alphas = {}

    for obj in sorted(aligned_data.keys()):
        items = aligned_data[obj]
        if not items:
            continue

        clip_maps = [item['clip_map'] for item in items]
        dino_maps = [item['dino_map'] for item in items]
        gt_masks = [item['gt_mask'] for item in items]

        # Check if any anomaly pixels exist
        total_anomaly_pixels = sum(m.sum() for m in gt_masks)
        total_normal_pixels = sum((1-m).sum() for m in gt_masks)
        if total_anomaly_pixels == 0 or total_normal_pixels == 0:
            print(f"{obj:<15} [skipped: no anomaly pixels in GT]")
            continue

        # Normalize all maps per-category
        all_clip = np.concatenate([m.ravel() for m in clip_maps])
        all_dino = np.concatenate([m.ravel() for m in dino_maps])
        clip_min, clip_max = all_clip.min(), all_clip.max()
        dino_min, dino_max = all_dino.min(), all_dino.max()

        clip_norm_maps = [(m - clip_min) / (clip_max - clip_min + 1e-10) for m in clip_maps]
        dino_norm_maps = [(m - dino_min) / (dino_max - dino_min + 1e-10) for m in dino_maps]

        # Strategy evaluations
        clip_auroc = compute_pixel_auroc(gt_masks, clip_norm_maps)
        dino_auroc = compute_pixel_auroc(gt_masks, dino_norm_maps)

        fused_eq = [(1-0.5)*c + 0.5*d for c, d in zip(clip_norm_maps, dino_norm_maps)]
        eq_auroc = compute_pixel_auroc(gt_masks, fused_eq)

        fused_ch = [(1-0.3)*c + 0.3*d for c, d in zip(clip_norm_maps, dino_norm_maps)]
        ch_auroc = compute_pixel_auroc(gt_masks, fused_ch)

        fused_dh = [(1-0.7)*c + 0.7*d for c, d in zip(clip_norm_maps, dino_norm_maps)]
        dh_auroc = compute_pixel_auroc(gt_masks, fused_dh)

        fused_max = [np.maximum(c, d) for c, d in zip(clip_norm_maps, dino_norm_maps)]
        max_auroc = compute_pixel_auroc(gt_masks, fused_max)

        # Oracle: sweep alpha
        best_auroc = 0
        best_a = 0.5
        for a in [i / 20.0 for i in range(21)]:
            fused = [(1-a)*c + a*d for c, d in zip(clip_norm_maps, dino_norm_maps)]
            auroc = compute_pixel_auroc(gt_masks, fused)
            if auroc > best_auroc:
                best_auroc = auroc
                best_a = a

        all_results['CLIP only'][obj] = clip_auroc
        all_results['DINOv2 only'][obj] = dino_auroc
        all_results['Equal (a=0.5)'][obj] = eq_auroc
        all_results['CLIP-heavy (a=0.3)'][obj] = ch_auroc
        all_results['DINO-heavy (a=0.7)'][obj] = dh_auroc
        all_results['Max fusion'][obj] = max_auroc
        all_results['Oracle'][obj] = best_auroc
        oracle_alphas[obj] = best_a

        print(f"{obj:<15} {clip_auroc:>7.1f}% {dino_auroc:>7.1f}% {eq_auroc:>7.1f}% "
              f"{ch_auroc:>7.1f}% {dh_auroc:>7.1f}% {max_auroc:>7.1f}% {best_auroc:>7.1f}%")

    # Means
    print("-" * 85)
    means = {}
    for s in list(all_results.keys()):
        vals = [v for v in all_results[s].values() if not np.isnan(v)]
        means[s] = np.mean(vals) if vals else 0

    print(f"{'MEAN':<15} {means.get('CLIP only',0):>7.1f}% {means.get('DINOv2 only',0):>7.1f}% "
          f"{means.get('Equal (a=0.5)',0):>7.1f}% {means.get('CLIP-heavy (a=0.3)',0):>7.1f}% "
          f"{means.get('DINO-heavy (a=0.7)',0):>7.1f}% {means.get('Max fusion',0):>7.1f}% "
          f"{means.get('Oracle',0):>7.1f}%")

    # Oracle alpha distribution
    print(f"\n--- Oracle Alpha Distribution (Pixel-Level) ---")
    for obj in sorted(oracle_alphas.keys()):
        oa = oracle_alphas[obj]
        winner = "DINOv2" if oa > 0.5 else ("CLIP" if oa < 0.5 else "equal")
        print(f"  {obj:<15} a={oa:.2f} ({winner})")

    # Summary comparison with image-level
    print(f"\n--- Pixel-Level Fusion Summary ({dataset_name}) ---")
    for s in ['CLIP only', 'DINOv2 only', 'Equal (a=0.5)', 'DINO-heavy (a=0.7)', 'Max fusion', 'Oracle']:
        print(f"  {s:<20}: {means.get(s, 0):.1f}%")

    return all_results, oracle_alphas


# ============================================================================
# Main
# ============================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = "data/MVTec_AD"

    # =====================================================
    # MVTec AD Pixel-Level Fusion
    # =====================================================
    clip_maps_dir = os.path.join(base_dir, "AnomalyCLIP", "results", "maps_mvtec")
    dino_maps_weak = os.path.join(base_dir, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                   "1-shot_preprocess=agnostic_weak_cats")
    dino_maps_strong = os.path.join(base_dir, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                     "1-shot_preprocess=agnostic_strong_cats")

    print("=" * 90)
    print("  PIXEL-LEVEL FUSION: AnomalyCLIP + AnomalyDINO")
    print("=" * 90)

    # Load CLIP maps
    print("\n--- Loading AnomalyCLIP maps ---")
    clip_data = load_anomalyclip_maps(clip_maps_dir, data_root, 'mvtec')
    if clip_data:
        print(f"  Loaded {sum(len(v) for v in clip_data.values())} maps across {len(clip_data)} categories")
        for obj, items in clip_data.items():
            map_shape = items[0]['map'].shape if items else 'N/A'
            print(f"    {obj}: {len(items)} maps, shape={map_shape}")
    else:
        print("  [ERROR] No AnomalyCLIP maps found. Run test_save_maps.py first.")
        return

    # Load DINO maps (merge weak and strong)
    print("\n--- Loading AnomalyDINO maps ---")
    dino_data = {}
    for dino_dir in [dino_maps_weak, dino_maps_strong]:
        if os.path.exists(dino_dir):
            d = load_anomalydino_maps(dino_dir)
            dino_data.update(d)
    if dino_data:
        print(f"  Loaded {sum(len(v) for v in dino_data.values())} maps across {len(dino_data)} categories")
        for obj, items in dino_data.items():
            map_shape = items[0]['map'].shape if items else 'N/A'
            print(f"    {obj}: {len(items)} maps, shape={map_shape}")
    else:
        print("  [ERROR] No AnomalyDINO maps found.")
        return

    # Align and evaluate
    print("\n--- Aligning pixel-level maps ---")
    aligned = align_maps(clip_data, dino_data, target_size=(256, 256), dino_sigma=4)

    results, oracle_alphas = evaluate_pixel_fusion(aligned, "MVTec AD")

    # =====================================================
    # VisA Pixel-Level Fusion
    # =====================================================
    visa_data_root = "data/VisA/VisA_pytorch/1cls"
    visa_clip_maps_dir = os.path.join(base_dir, "AnomalyCLIP", "results", "maps_visa")
    visa_dino_maps_dir = os.path.join(base_dir, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                                       "1-shot_preprocess=agnostic_all")

    print("\n\n" + "=" * 90)
    print("  PIXEL-LEVEL FUSION: AnomalyCLIP + AnomalyDINO (VisA)")
    print("=" * 90)

    # Load CLIP maps for VisA
    print("\n--- Loading AnomalyCLIP maps (VisA) ---")
    visa_clip_data = load_anomalyclip_maps(visa_clip_maps_dir, visa_data_root, 'visa')
    if visa_clip_data:
        print(f"  Loaded {sum(len(v) for v in visa_clip_data.values())} maps across {len(visa_clip_data)} categories")
        for obj, items in visa_clip_data.items():
            map_shape = items[0]['map'].shape if items else 'N/A'
            print(f"    {obj}: {len(items)} maps, shape={map_shape}")
    else:
        print("  [ERROR] No AnomalyCLIP VisA maps found.")

    # Load DINO maps for VisA
    print("\n--- Loading AnomalyDINO maps (VisA) ---")
    visa_dino_data = {}
    if os.path.exists(visa_dino_maps_dir):
        visa_dino_data = load_anomalydino_maps(visa_dino_maps_dir)
    if visa_dino_data:
        print(f"  Loaded {sum(len(v) for v in visa_dino_data.values())} maps across {len(visa_dino_data)} categories")
        for obj, items in visa_dino_data.items():
            map_shape = items[0]['map'].shape if items else 'N/A'
            print(f"    {obj}: {len(items)} maps, shape={map_shape}")
    else:
        print("  [ERROR] No AnomalyDINO VisA maps found.")

    # Align and evaluate VisA
    visa_results = {}
    visa_oracle_alphas = {}
    if visa_clip_data and visa_dino_data:
        print("\n--- Aligning pixel-level maps (VisA) ---")
        visa_aligned = align_maps(visa_clip_data, visa_dino_data, target_size=(256, 256), dino_sigma=4)
        visa_results, visa_oracle_alphas = evaluate_pixel_fusion(visa_aligned, "VisA")

    # =====================================================
    # Cross-Dataset Comparison
    # =====================================================
    if visa_results and results:
        print("\n\n" + "=" * 90)
        print("  CROSS-DATASET COMPARISON: MVTec AD vs VisA (Pixel-Level)")
        print("=" * 90)

        for strategy in ['CLIP only', 'DINOv2 only', 'Equal (a=0.5)', 'DINO-heavy (a=0.7)', 'Oracle']:
            mvtec_vals = [v for v in results[strategy].values() if not np.isnan(v)]
            visa_vals = [v for v in visa_results[strategy].values() if not np.isnan(v)]
            mvtec_mean = np.mean(mvtec_vals) if mvtec_vals else 0
            visa_mean = np.mean(visa_vals) if visa_vals else 0
            print(f"  {strategy:<20}: MVTec={mvtec_mean:.1f}%  VisA={visa_mean:.1f}%")

        # Oracle alpha comparison
        print(f"\n--- Oracle Alpha Distribution Comparison ---")
        print(f"  {'Category':<15} {'MVTec alpha':>12} {'VisA alpha':>12}")
        print(f"  {'-'*40}")
        for obj in sorted(oracle_alphas.keys()):
            oa = oracle_alphas[obj]
            winner = "DINOv2" if oa > 0.5 else ("CLIP" if oa < 0.5 else "equal")
            print(f"  {obj:<15} {oa:.2f} ({winner})")
        print(f"  {'--- VisA ---':<15}")
        for obj in sorted(visa_oracle_alphas.keys()):
            oa = visa_oracle_alphas[obj]
            winner = "DINOv2" if oa > 0.5 else ("CLIP" if oa < 0.5 else "equal")
            print(f"  {obj:<15} {oa:.2f} ({winner})")

        # Complementarity pattern analysis
        mvtec_clip_dominant = sum(1 for a in oracle_alphas.values() if a < 0.5)
        mvtec_dino_dominant = sum(1 for a in oracle_alphas.values() if a > 0.5)
        mvtec_equal = sum(1 for a in oracle_alphas.values() if a == 0.5)
        visa_clip_dominant = sum(1 for a in visa_oracle_alphas.values() if a < 0.5)
        visa_dino_dominant = sum(1 for a in visa_oracle_alphas.values() if a > 0.5)
        visa_equal = sum(1 for a in visa_oracle_alphas.values() if a == 0.5)

        print(f"\n--- Complementarity Pattern Summary (Pixel-Level) ---")
        print(f"  MVTec AD: CLIP-dominant={mvtec_clip_dominant}, DINOv2-dominant={mvtec_dino_dominant}, Equal={mvtec_equal} / {len(oracle_alphas)} categories")
        print(f"  VisA:     CLIP-dominant={visa_clip_dominant}, DINOv2-dominant={visa_dino_dominant}, Equal={visa_equal} / {len(visa_oracle_alphas)} categories")

    # =====================================================
    # BTAD Pixel-Level Fusion
    # =====================================================
    btad_data_root = "data/BTAD/BTech_Dataset_transformed"
    btad_clip_maps_dir = os.path.join(base_dir, "AnomalyCLIP", "results", "maps_btad")
    btad_dino_maps_dir = os.path.join(base_dir, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                                       "1-shot_preprocess=agnostic_btad_all")

    print("\n\n" + "=" * 90)
    print("  PIXEL-LEVEL FUSION: AnomalyCLIP + AnomalyDINO (BTAD)")
    print("=" * 90)

    # Load CLIP maps for BTAD
    print("\n--- Loading AnomalyCLIP maps (BTAD) ---")
    btad_clip_data = load_anomalyclip_maps(btad_clip_maps_dir, btad_data_root, 'btad')
    if btad_clip_data:
        print(f"  Loaded {sum(len(v) for v in btad_clip_data.values())} maps across {len(btad_clip_data)} categories")
        for obj, items in btad_clip_data.items():
            map_shape = items[0]['map'].shape if items else 'N/A'
            print(f"    {obj}: {len(items)} maps, shape={map_shape}")
    else:
        print("  [ERROR] No AnomalyCLIP BTAD maps found.")

    # Load DINO maps for BTAD
    print("\n--- Loading AnomalyDINO maps (BTAD) ---")
    btad_dino_data = {}
    if os.path.exists(btad_dino_maps_dir):
        btad_dino_data = load_anomalydino_maps(btad_dino_maps_dir)
    if btad_dino_data:
        print(f"  Loaded {sum(len(v) for v in btad_dino_data.values())} maps across {len(btad_dino_data)} categories")
        for obj, items in btad_dino_data.items():
            map_shape = items[0]['map'].shape if items else 'N/A'
            print(f"    {obj}: {len(items)} maps, shape={map_shape}")
    else:
        print("  [ERROR] No AnomalyDINO BTAD maps found.")

    # Align and evaluate BTAD
    btad_results = {}
    btad_oracle_alphas = {}
    if btad_clip_data and btad_dino_data:
        print("\n--- Aligning pixel-level maps (BTAD) ---")
        btad_aligned = align_maps(btad_clip_data, btad_dino_data, target_size=(256, 256), dino_sigma=4)
        btad_results, btad_oracle_alphas = evaluate_pixel_fusion(btad_aligned, "BTAD")

    # Save results
    output_path = os.path.join(base_dir, "pixel_fusion_results.json")
    output = {
        'mvtec': {
            'strategies': {k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in results.items()},
            'oracle_alphas': oracle_alphas,
        }
    }
    if visa_results:
        output['visa'] = {
            'strategies': {k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in visa_results.items()},
            'oracle_alphas': visa_oracle_alphas,
        }
    if btad_results:
        output['btad'] = {
            'strategies': {k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in btad_results.items()},
            'oracle_alphas': btad_oracle_alphas,
        }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
