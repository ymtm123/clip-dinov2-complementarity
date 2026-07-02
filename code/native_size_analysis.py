"""
Experiment 11: Anomaly Size x Method Interaction in Native Mode
=================================================================

Goal: Test whether the kNN finding "CLIP detects smaller anomalies"
holds in native mode (AnomalyCLIP 0-shot + AnomalyDINO 1-shot).

kNN findings (Experiment 9):
  - CLIP-only detected anomalies: median mask area 1.01%
  - DINOv2-only detected anomalies: median mask area 2.89%
  - Both: median 2.47%
  - Neither: median 0.74%

This experiment replicates the analysis using native mode scores.
"""

import csv
import json
import os
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats
from PIL import Image


# ============================================================
# Data loading (reuse from native_defect_type_analysis.py)
# ============================================================

def load_anomalyclip_per_image(csv_path, data_root):
    """Load AnomalyCLIP scores with per-image paths for mask lookup."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            obj = row[0]
            img_path = row[1].replace('\\', '/')
            score = float(row[2])
            gt = int(row[3])

            match = re.search(r'/test/([^/]+)/', img_path)
            if match:
                defect_type = match.group(1)
            else:
                defect_type = 'good' if gt == 0 else 'unknown'

            results.append({
                'category': obj,
                'defect_type': defect_type,
                'score': score,
                'gt': gt,
                'img_path': img_path,
            })
    return results


def load_anomalydino_per_image(csv_path):
    """Load AnomalyDINO per-image scores."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            obj = row[0]
            sample = row[1]  # e.g., "contamination/000.png"
            score = float(row[2])

            defect_type = sample.split('/')[0]
            img_name = sample.split('/')[-1]
            is_good = (defect_type == 'good')

            results.append({
                'category': obj,
                'defect_type': defect_type,
                'score': score,
                'gt': 0 if is_good else 1,
                'sample': sample,
                'img_name': img_name,
            })
    return results


def get_mask_area_ratio(data_root, category, defect_type, img_name):
    """Compute anomaly mask area ratio for a given image."""
    if defect_type == 'good':
        return 0.0

    # Try standard mask path
    mask_dir = data_root / category / 'ground_truth' / defect_type
    mask_name = os.path.splitext(img_name)[0] + '_mask.png'
    mask_path = mask_dir / mask_name

    if mask_path.exists():
        mask = np.array(Image.open(mask_path).convert('L'))
        return float((mask > 128).sum()) / mask.size

    # Try without _mask suffix
    mask_path2 = mask_dir / img_name
    if mask_path2.exists():
        mask = np.array(Image.open(mask_path2).convert('L'))
        return float((mask > 128).sum()) / mask.size

    return -1.0  # not found


def main():
    src_dir = Path(__file__).parent
    data_root = Path(r"data\MVTec_AD")

    # Load scores
    clip_path = src_dir / 'AnomalyCLIP' / 'results' / 'scores_mvtec' / 'anomalyclip_scores.csv'
    dino_path_strong = src_dir / 'AnomalyDINO' / 'results_MVTec' / 'dinov2_vitl14_448' / \
                       '1-shot_preprocess=agnostic_strong_cats' / 'measurements_seed=0.csv'
    dino_path_weak = src_dir / 'AnomalyDINO' / 'results_MVTec' / 'dinov2_vitl14_448' / \
                     '1-shot_preprocess=agnostic_weak_cats' / 'measurements_seed=0.csv'

    print("Loading AnomalyCLIP scores...")
    clip_data = load_anomalyclip_per_image(clip_path, data_root)
    print(f"  Loaded {len(clip_data)} entries")

    print("Loading AnomalyDINO scores...")
    dino_data = []
    for p in [dino_path_strong, dino_path_weak]:
        dino_data.extend(load_anomalydino_per_image(p))
    print(f"  Loaded {len(dino_data)} entries")

    # Build per-category normal score distributions for thresholding
    # For each model, compute optimal threshold per category (Youden's J)
    from sklearn.metrics import roc_curve

    # Group CLIP scores by category
    clip_by_cat = defaultdict(lambda: {'scores': [], 'gts': []})
    for d in clip_data:
        clip_by_cat[d['category']]['scores'].append(d['score'])
        clip_by_cat[d['category']]['gts'].append(d['gt'])

    # Group DINO scores by category
    dino_by_cat = defaultdict(lambda: {'scores': [], 'gts': []})
    for d in dino_data:
        dino_by_cat[d['category']]['scores'].append(d['score'])
        dino_by_cat[d['category']]['gts'].append(d['gt'])

    # Compute thresholds (Youden's J method)
    clip_thresholds = {}
    for cat, info in clip_by_cat.items():
        scores = np.array(info['scores'])
        gts = np.array(info['gts'])
        if len(set(gts)) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(gts, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        clip_thresholds[cat] = thresholds[best_idx]

    dino_thresholds = {}
    for cat, info in dino_by_cat.items():
        scores = np.array(info['scores'])
        gts = np.array(info['gts'])
        if len(set(gts)) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(gts, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        dino_thresholds[cat] = thresholds[best_idx]

    common_cats = sorted(set(clip_thresholds.keys()) & set(dino_thresholds.keys()))
    print(f"\nCommon categories with thresholds: {len(common_cats)}")

    # ============================================================
    # Build per-image detection classification with mask areas
    # ============================================================
    print("\n" + "=" * 80)
    print("Building per-image detection classes with mask area ratios...")
    print("=" * 80)

    # Index CLIP scores by (category, defect_type, img_identifier)
    clip_score_map = {}
    for d in clip_data:
        # Extract image identifier from path
        match = re.search(r'/test/[^/]+/([^/]+)$', d['img_path'])
        if match:
            img_id = match.group(1)
        else:
            img_id = d['img_path'].split('/')[-1]
        key = (d['category'], d['defect_type'], img_id)
        clip_score_map[key] = d['score']

    # Index DINO scores by (category, defect_type, img_name)
    dino_score_map = {}
    for d in dino_data:
        key = (d['category'], d['defect_type'], d['img_name'])
        dino_score_map[key] = d['score']

    # Find matching images
    all_images = []
    mask_not_found = 0

    for cat in common_cats:
        # Get all defect types for this category
        clip_defects = set()
        dino_defects = set()
        for d in clip_data:
            if d['category'] == cat and d['defect_type'] != 'good':
                clip_defects.add(d['defect_type'])
        for d in dino_data:
            if d['category'] == cat and d['defect_type'] != 'good':
                dino_defects.add(d['defect_type'])

        common_defects = clip_defects & dino_defects

        for dt in sorted(common_defects):
            # Get matching images
            clip_imgs = {}
            for d in clip_data:
                if d['category'] == cat and d['defect_type'] == dt:
                    match = re.search(r'/test/[^/]+/([^/]+)$', d['img_path'])
                    if match:
                        img_id = match.group(1)
                        clip_imgs[img_id] = d['score']

            dino_imgs = {}
            for d in dino_data:
                if d['category'] == cat and d['defect_type'] == dt:
                    dino_imgs[d['img_name']] = d['score']

            common_imgs = set(clip_imgs.keys()) & set(dino_imgs.keys())

            for img_name in sorted(common_imgs):
                clip_score = clip_imgs[img_name]
                dino_score = dino_imgs[img_name]

                clip_detected = clip_score >= clip_thresholds[cat]
                dino_detected = dino_score >= dino_thresholds[cat]

                if clip_detected and dino_detected:
                    detection_class = 'both'
                elif clip_detected:
                    detection_class = 'clip_only'
                elif dino_detected:
                    detection_class = 'dino_only'
                else:
                    detection_class = 'neither'

                # Get mask area
                mask_area = get_mask_area_ratio(data_root, cat, dt, img_name)
                if mask_area < 0:
                    mask_not_found += 1
                    continue

                all_images.append({
                    'category': cat,
                    'defect_type': dt,
                    'img_name': img_name,
                    'clip_score': clip_score,
                    'dino_score': dino_score,
                    'clip_detected': clip_detected,
                    'dino_detected': dino_detected,
                    'detection_class': detection_class,
                    'mask_area': mask_area,
                })

    print(f"\nTotal matched anomalous images: {len(all_images)}")
    print(f"Masks not found (skipped): {mask_not_found}")

    # ============================================================
    # Analysis 1: Mask area by detection class (native mode)
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 1: Mask Area by Detection Class (Native Mode)")
    print("=" * 80)

    detection_classes = ['both', 'clip_only', 'dino_only', 'neither']
    class_stats = {}
    for dc in detection_classes:
        areas = [d['mask_area'] for d in all_images
                 if d['detection_class'] == dc and d['mask_area'] > 0]
        if areas:
            class_stats[dc] = {
                'n': len(areas),
                'mean': np.mean(areas),
                'median': np.median(areas),
                'std': np.std(areas),
                'q25': np.percentile(areas, 25),
                'q75': np.percentile(areas, 75),
            }

    print(f"\n{'Class':12s} {'N':>5s} {'Mean%':>8s} {'Median%':>8s} {'Q25%':>8s} {'Q75%':>8s}")
    print("-" * 55)
    for dc in detection_classes:
        if dc in class_stats:
            s = class_stats[dc]
            print(f"{dc:12s} {s['n']:5d} {s['mean']*100:7.2f}% {s['median']*100:7.2f}% "
                  f"{s['q25']*100:7.2f}% {s['q75']*100:7.2f}%")

    # Compare with kNN results
    print("\n--- Comparison with kNN backbone isolation (Experiment 9) ---")
    knn_results = {
        'both': {'n': 433, 'mean': 0.059546, 'median': 0.024662},
        'clip_only': {'n': 224, 'mean': 0.025587, 'median': 0.010066},
        'dino_only': {'n': 193, 'mean': 0.074017, 'median': 0.028911},
        'neither': {'n': 408, 'mean': 0.023051, 'median': 0.007385},
    }

    print(f"\n{'Class':12s} {'kNN median%':>12s} {'Native median%':>14s} {'Change':>8s}")
    print("-" * 55)
    for dc in detection_classes:
        if dc in class_stats and dc in knn_results:
            knn_med = knn_results[dc]['median'] * 100
            nat_med = class_stats[dc]['median'] * 100
            print(f"{dc:12s} {knn_med:11.2f}% {nat_med:13.2f}% {nat_med-knn_med:+7.2f}%")

    # ============================================================
    # Analysis 2: Statistical tests
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 2: Statistical Tests on Mask Area Differences")
    print("=" * 80)

    # Mann-Whitney U test: clip_only vs dino_only mask areas
    if 'clip_only' in class_stats and 'dino_only' in class_stats:
        clip_areas = [d['mask_area'] for d in all_images
                      if d['detection_class'] == 'clip_only' and d['mask_area'] > 0]
        dino_areas = [d['mask_area'] for d in all_images
                      if d['detection_class'] == 'dino_only' and d['mask_area'] > 0]

        U, p_mw = stats.mannwhitneyu(clip_areas, dino_areas, alternative='less')
        print(f"\nMann-Whitney U test (CLIP-only < DINOv2-only mask area):")
        print(f"  CLIP-only: n={len(clip_areas)}, median={np.median(clip_areas)*100:.2f}%")
        print(f"  DINO-only: n={len(dino_areas)}, median={np.median(dino_areas)*100:.2f}%")
        print(f"  U={U:.0f}, p={p_mw:.4f} {'*' if p_mw < 0.05 else 'n.s.'}")
        print(f"  Effect size (rank-biserial r): {1 - 2*U/(len(clip_areas)*len(dino_areas)):.3f}")

    # ============================================================
    # Analysis 3: Size bins with detection rates
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 3: Detection Rate by Anomaly Size Bins")
    print("=" * 80)

    # Define size bins
    size_bins = [
        ('tiny (<0.5%)', 0, 0.005),
        ('small (0.5-2%)', 0.005, 0.02),
        ('medium (2-5%)', 0.02, 0.05),
        ('large (5-15%)', 0.05, 0.15),
        ('very_large (>15%)', 0.15, 1.0),
    ]

    print(f"\n{'Size bin':22s} {'N':>4s} {'Both%':>6s} {'CLIP%':>6s} {'DINO%':>6s} {'Neither%':>8s} "
          f"{'CLIP TPR':>8s} {'DINO TPR':>8s}")
    print("-" * 80)

    bin_results = []
    for bin_name, lo, hi in size_bins:
        bin_images = [d for d in all_images if lo <= d['mask_area'] < hi]
        if len(bin_images) == 0:
            continue

        n = len(bin_images)
        n_both = sum(1 for d in bin_images if d['detection_class'] == 'both')
        n_clip = sum(1 for d in bin_images if d['detection_class'] == 'clip_only')
        n_dino = sum(1 for d in bin_images if d['detection_class'] == 'dino_only')
        n_neither = sum(1 for d in bin_images if d['detection_class'] == 'neither')

        clip_tpr = (n_both + n_clip) / n * 100
        dino_tpr = (n_both + n_dino) / n * 100

        print(f"{bin_name:22s} {n:4d} {n_both/n*100:5.1f}% {n_clip/n*100:5.1f}% "
              f"{n_dino/n*100:5.1f}% {n_neither/n*100:7.1f}% "
              f"{clip_tpr:7.1f}% {dino_tpr:7.1f}%")

        bin_results.append({
            'bin': bin_name,
            'lo': lo, 'hi': hi,
            'n': n,
            'both': n_both, 'clip_only': n_clip, 'dino_only': n_dino, 'neither': n_neither,
            'clip_tpr': clip_tpr, 'dino_tpr': dino_tpr,
        })

    # ============================================================
    # Analysis 4: Per-category size analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 4: Per-Category Median Anomaly Size vs VFM Preference")
    print("=" * 80)

    cat_size_data = {}
    for d in all_images:
        cat = d['category']
        if cat not in cat_size_data:
            cat_size_data[cat] = {
                'areas': [],
                'clip_detected': 0,
                'dino_detected': 0,
                'total': 0,
            }
        cat_size_data[cat]['areas'].append(d['mask_area'])
        cat_size_data[cat]['total'] += 1
        if d['clip_detected']:
            cat_size_data[cat]['clip_detected'] += 1
        if d['dino_detected']:
            cat_size_data[cat]['dino_detected'] += 1

    print(f"\n{'Category':15s} {'Med Size%':>10s} {'CLIP TPR%':>9s} {'DINO TPR%':>9s} {'Diff':>6s} {'Better':>8s}")
    print("-" * 65)

    cat_medians = []
    cat_diffs = []
    for cat in sorted(cat_size_data.keys()):
        s = cat_size_data[cat]
        med_area = np.median(s['areas']) * 100
        clip_tpr = s['clip_detected'] / s['total'] * 100
        dino_tpr = s['dino_detected'] / s['total'] * 100
        diff = clip_tpr - dino_tpr
        better = 'CLIP' if diff > 0 else 'DINOv2'

        print(f"{cat:15s} {med_area:9.2f}% {clip_tpr:8.1f}% {dino_tpr:8.1f}% {diff:+5.1f} {better:>8s}")
        cat_medians.append(med_area)
        cat_diffs.append(diff)

    # Correlation: median anomaly size vs CLIP-DINOv2 TPR difference
    r, p_corr = stats.pearsonr(cat_medians, cat_diffs)
    print(f"\nCorrelation (median size vs CLIP-DINOv2 TPR diff): r={r:.3f}, p={p_corr:.4f}")
    if p_corr < 0.05:
        print(f"  SIGNIFICANT: {'larger anomalies favor DINOv2' if r < 0 else 'larger anomalies favor CLIP'}")
    else:
        print(f"  Not significant: anomaly size does not predict VFM advantage in native mode")

    # Spearman rank correlation
    rho, p_spear = stats.spearmanr(cat_medians, cat_diffs)
    print(f"Spearman rank correlation: rho={rho:.3f}, p={p_spear:.4f}")

    # ============================================================
    # Analysis 5: Anomaly size continuum
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 5: CLIP Advantage Ratio by Anomaly Size Percentile")
    print("=" * 80)

    # Sort all images by mask area and compute rolling CLIP advantage
    sorted_imgs = sorted([d for d in all_images if d['mask_area'] > 0],
                         key=lambda x: x['mask_area'])

    # Use 5 equal-size bins
    n_per_bin = len(sorted_imgs) // 5
    print(f"\n{'Percentile':12s} {'N':>4s} {'Size range%':>15s} {'CLIP TPR%':>9s} {'DINO TPR%':>9s} {'CLIP adv':>8s}")
    print("-" * 65)

    percentile_results = []
    for i in range(5):
        start = i * n_per_bin
        end = start + n_per_bin if i < 4 else len(sorted_imgs)
        bin_imgs = sorted_imgs[start:end]

        min_area = bin_imgs[0]['mask_area'] * 100
        max_area = bin_imgs[-1]['mask_area'] * 100
        n = len(bin_imgs)
        clip_tpr = sum(1 for d in bin_imgs if d['clip_detected']) / n * 100
        dino_tpr = sum(1 for d in bin_imgs if d['dino_detected']) / n * 100
        clip_adv = clip_tpr - dino_tpr

        label = f"Q{i+1} ({(i)*20}-{(i+1)*20}%)"
        print(f"{label:12s} {n:4d} {min_area:5.2f}-{max_area:5.2f}% {clip_tpr:8.1f}% {dino_tpr:8.1f}% {clip_adv:+7.1f}")

        percentile_results.append({
            'quintile': i+1,
            'n': n,
            'min_area': min_area,
            'max_area': max_area,
            'clip_tpr': clip_tpr,
            'dino_tpr': dino_tpr,
            'clip_advantage': clip_adv,
        })

    # ============================================================
    # Save results
    # ============================================================
    output = {
        'experiment_11_native_size': {
            'total_images': len(all_images),
            'mask_area_by_class': {
                dc: {
                    'n': class_stats[dc]['n'],
                    'mean': round(class_stats[dc]['mean'], 6),
                    'median': round(class_stats[dc]['median'], 6),
                    'std': round(class_stats[dc]['std'], 6),
                } for dc in detection_classes if dc in class_stats
            },
            'size_bin_detection_rates': bin_results,
            'percentile_analysis': percentile_results,
            'correlation': {
                'pearson_r': round(r, 4),
                'pearson_p': round(p_corr, 4),
                'spearman_rho': round(rho, 4),
                'spearman_p': round(p_spear, 4),
            },
        }
    }

    output_path = src_dir / 'native_size_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
