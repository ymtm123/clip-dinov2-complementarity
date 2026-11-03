"""
BTAD Score-Level Fusion: AnomalyCLIP + AnomalyDINO
===================================================
Combines scores from both models on BTAD (3 categories: 01, 02, 03)
and evaluates score-level fusion strategies.
"""

import os
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

def load_anomalyclip_scores_btad(csv_path):
    """Load AnomalyCLIP BTAD scores from CSV."""
    scores = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row['Object']
            score = float(row['AnomalyScore'])
            gt = int(row['GroundTruth'])
            img_path = row['ImagePath']
            scores[obj].append({
                'score': score,
                'gt': gt,
                'img_path': img_path,
            })
    return dict(scores)


def load_anomalydino_scores_btad(csv_path):
    """Load AnomalyDINO BTAD scores from measurements CSV."""
    scores = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row['Object']
            sample = row['Sample']  # e.g., "ko/0000.bmp" or "ok/0000.bmp"
            score = float(row['Anomaly_Score'])

            # Determine ground truth from folder name
            anomaly_type = sample.split('/')[0]
            gt = 0 if anomaly_type == 'ok' else 1

            scores[obj].append({
                'score': score,
                'gt': gt,
                'img_path': sample,
                'anomaly_type': anomaly_type,
            })
    return dict(scores)


def normalize_scores(scores, method='minmax'):
    """Min-max normalize scores to [0, 1]."""
    arr = np.array(scores)
    if method == 'minmax':
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    return arr


def align_scores(clip_data, dino_data):
    """Align AnomalyCLIP and AnomalyDINO scores by category."""
    aligned = {}
    common_cats = set(clip_data.keys()) & set(dino_data.keys())

    for cat in sorted(common_cats):
        clip_items = clip_data[cat]
        dino_items = dino_data[cat]

        # Sort both by image path for alignment
        # AnomalyCLIP img_path format varies; AnomalyDINO uses "ko/0000.bmp"
        # Need to extract the key parts for matching

        # Build index by (type, id) for CLIP
        clip_index = {}
        for item in clip_items:
            # AnomalyCLIP path might be like "01/test/ko/0000.bmp"
            parts = item['img_path'].replace('\\', '/').split('/')
            # Find the anomaly type and image id
            if 'test' in parts:
                idx = parts.index('test')
                atype = parts[idx+1] if idx+1 < len(parts) else ''
                img_name = parts[idx+2] if idx+2 < len(parts) else ''
            else:
                atype = parts[-2] if len(parts) >= 2 else ''
                img_name = parts[-1]
            img_id = os.path.splitext(img_name)[0]
            key = f"{atype}/{img_id}"
            clip_index[key] = item

        # Build index by (type, id) for DINO
        dino_index = {}
        for item in dino_items:
            parts = item['img_path'].replace('\\', '/').split('/')
            atype = parts[0]
            img_name = parts[1] if len(parts) > 1 else parts[0]
            img_id = os.path.splitext(img_name)[0]
            key = f"{atype}/{img_id}"
            dino_index[key] = item

        # Find common keys
        common_keys = set(clip_index.keys()) & set(dino_index.keys())

        if len(common_keys) == 0:
            print(f"  {cat}: no common samples!")
            print(f"    CLIP keys sample: {list(clip_index.keys())[:3]}")
            print(f"    DINO keys sample: {list(dino_index.keys())[:3]}")
            continue

        aligned_items = []
        for key in sorted(common_keys):
            ci = clip_index[key]
            di = dino_index[key]
            aligned_items.append({
                'clip_score': ci['score'],
                'dino_score': di['score'],
                'gt': ci['gt'],
                'key': key,
            })

        print(f"  {cat}: aligned {len(aligned_items)} / CLIP:{len(clip_items)} / DINO:{len(dino_items)}")
        aligned[cat] = aligned_items

    return aligned


def evaluate_fusion(aligned_data):
    """Evaluate score-level fusion on BTAD."""
    print(f"\n{'='*80}")
    print(f"  BTAD SCORE-LEVEL FUSION RESULTS")
    print(f"{'='*80}")

    strategies = ['CLIP only', 'DINOv2 only', 'Equal (a=0.5)', 'CLIP-heavy (a=0.3)',
                  'DINO-heavy (a=0.7)', 'Max fusion']

    print(f"\n{'Category':<12} {'CLIP':>8} {'DINO':>8} {'Eq(0.5)':>8} {'C(0.3)':>8} "
          f"{'D(0.7)':>8} {'Max':>8} {'Oracle':>8}")
    print("-" * 80)

    all_results = {s: {} for s in strategies}
    all_results['Oracle'] = {}
    oracle_alphas = {}

    for cat in sorted(aligned_data.keys()):
        items = aligned_data[cat]
        if not items:
            continue

        clip_scores = np.array([item['clip_score'] for item in items])
        dino_scores = np.array([item['dino_score'] for item in items])
        gt_labels = np.array([item['gt'] for item in items])

        # Check if there are both normal and anomaly samples
        if len(np.unique(gt_labels)) < 2:
            print(f"{cat:<12} [skipped: only one class]")
            continue

        # Normalize scores per-category
        clip_norm = normalize_scores(clip_scores)
        dino_norm = normalize_scores(dino_scores)

        # Evaluate strategies
        clip_auroc = roc_auc_score(gt_labels, clip_norm) * 100
        dino_auroc = roc_auc_score(gt_labels, dino_norm) * 100

        eq_scores = 0.5 * clip_norm + 0.5 * dino_norm
        eq_auroc = roc_auc_score(gt_labels, eq_scores) * 100

        ch_scores = 0.7 * clip_norm + 0.3 * dino_norm
        ch_auroc = roc_auc_score(gt_labels, ch_scores) * 100

        dh_scores = 0.3 * clip_norm + 0.7 * dino_norm
        dh_auroc = roc_auc_score(gt_labels, dh_scores) * 100

        max_scores = np.maximum(clip_norm, dino_norm)
        max_auroc = roc_auc_score(gt_labels, max_scores) * 100

        # Oracle: sweep alpha
        best_auroc = 0
        best_a = 0.5
        for a in [i / 20.0 for i in range(21)]:
            fused = (1 - a) * clip_norm + a * dino_norm
            auroc = roc_auc_score(gt_labels, fused) * 100
            if auroc > best_auroc:
                best_auroc = auroc
                best_a = a

        all_results['CLIP only'][cat] = clip_auroc
        all_results['DINOv2 only'][cat] = dino_auroc
        all_results['Equal (a=0.5)'][cat] = eq_auroc
        all_results['CLIP-heavy (a=0.3)'][cat] = ch_auroc
        all_results['DINO-heavy (a=0.7)'][cat] = dh_auroc
        all_results['Max fusion'][cat] = max_auroc
        all_results['Oracle'][cat] = best_auroc
        oracle_alphas[cat] = best_a

        print(f"{cat:<12} {clip_auroc:>7.1f}% {dino_auroc:>7.1f}% {eq_auroc:>7.1f}% "
              f"{ch_auroc:>7.1f}% {dh_auroc:>7.1f}% {max_auroc:>7.1f}% {best_auroc:>7.1f}%")

    # Means
    print("-" * 80)
    means = {}
    for s in list(all_results.keys()):
        vals = [v for v in all_results[s].values()]
        means[s] = np.mean(vals) if vals else 0

    print(f"{'MEAN':<12} {means.get('CLIP only',0):>7.1f}% {means.get('DINOv2 only',0):>7.1f}% "
          f"{means.get('Equal (a=0.5)',0):>7.1f}% {means.get('CLIP-heavy (a=0.3)',0):>7.1f}% "
          f"{means.get('DINO-heavy (a=0.7)',0):>7.1f}% {means.get('Max fusion',0):>7.1f}% "
          f"{means.get('Oracle',0):>7.1f}%")

    # Oracle alpha distribution
    print(f"\n--- Oracle Alpha Distribution (BTAD Image-Level) ---")
    for cat in sorted(oracle_alphas.keys()):
        oa = oracle_alphas[cat]
        winner = "DINOv2" if oa > 0.5 else ("CLIP" if oa < 0.5 else "equal")
        print(f"  {cat:<12} a={oa:.2f} ({winner})")

    return all_results, oracle_alphas


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load AnomalyCLIP BTAD scores
    clip_csv = os.path.join(base_dir, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv")
    print("--- Loading AnomalyCLIP BTAD scores ---")
    clip_data = load_anomalyclip_scores_btad(clip_csv)
    for cat, items in clip_data.items():
        n_anomaly = sum(1 for i in items if i['gt'] == 1)
        n_normal = sum(1 for i in items if i['gt'] == 0)
        print(f"  {cat}: {len(items)} samples (normal={n_normal}, anomaly={n_anomaly})")

    # Load AnomalyDINO BTAD scores
    dino_csv = os.path.join(base_dir, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                             "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv")
    print("\n--- Loading AnomalyDINO BTAD scores ---")
    dino_data = load_anomalydino_scores_btad(dino_csv)
    for cat, items in dino_data.items():
        n_anomaly = sum(1 for i in items if i['gt'] == 1)
        n_normal = sum(1 for i in items if i['gt'] == 0)
        print(f"  {cat}: {len(items)} samples (normal={n_normal}, anomaly={n_anomaly})")

    # Align scores
    print("\n--- Aligning scores ---")
    aligned = align_scores(clip_data, dino_data)

    # Evaluate fusion
    results, oracle_alphas = evaluate_fusion(aligned)

    # Save results
    import json
    output = {
        'btad_image_level': {
            'strategies': {k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in results.items()},
            'oracle_alphas': oracle_alphas,
        }
    }
    output_path = os.path.join(base_dir, "btad_fusion_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
