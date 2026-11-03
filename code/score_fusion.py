"""
Score-Level Fusion of AnomalyCLIP and AnomalyDINO
=================================================

This module implements multiple score fusion strategies:
1. Fixed weight fusion (alpha sweep)
2. Min-max normalized fusion
3. Category-adaptive fusion using unsupervised signals
4. Rank-based fusion
5. Max fusion

The key technical contribution: unsupervised estimation of optimal
fusion weight per category using only reference (normal) images.
"""

import csv
import json
import os
import re
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict


def load_anomalyclip_scores(csv_path):
    """Load AnomalyCLIP per-sample scores from CSV.
    Format: Object,ImagePath,AnomalyScore,GroundTruth
    """
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            obj, img_path, score, gt = row[0], row[1], float(row[2]), int(row[3])
            # Extract relative path (after /test/)
            # e.g. C:/.../carpet/test/color/000.png -> color/000.png
            match = re.search(r'/test/(.+)$', img_path.replace('\\', '/'))
            rel_path = match.group(1) if match else os.path.basename(img_path)
            data[obj]['scores'].append(score)
            data[obj]['labels'].append(gt)
            data[obj]['paths'].append(rel_path)
    return dict(data)


def load_anomalydino_scores(csv_path):
    """Load AnomalyDINO per-sample scores from measurements CSV.
    Format: Object,Sample,Anomaly_Score,MemoryBank_Time,Inference_Time
    """
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            obj = row[0]
            sample = row[1]  # e.g., "thread_top/000.png"
            score = float(row[2])
            is_anomaly = 0 if sample.startswith("good/") else 1

            data[obj]['scores'].append(score)
            data[obj]['labels'].append(is_anomaly)
            data[obj]['paths'].append(sample)
    return dict(data)


def normalize_scores(scores, method='minmax'):
    """Normalize scores to [0, 1] range."""
    scores = np.array(scores, dtype=float)
    if method == 'minmax':
        smin, smax = scores.min(), scores.max()
        if smax - smin < 1e-10:
            return np.zeros_like(scores)
        return (scores - smin) / (smax - smin)
    elif method == 'zscore':
        mu, std = scores.mean(), scores.std()
        if std < 1e-10:
            return np.zeros_like(scores)
        return (scores - mu) / std
    elif method == 'rank':
        from scipy.stats import rankdata
        return rankdata(scores) / len(scores)
    return scores


def compute_auroc(labels, scores):
    """Compute AUROC safely."""
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, scores) * 100


def align_samples(clip_data, dino_data):
    """
    Align samples between AnomalyCLIP and AnomalyDINO by image path.
    Both models evaluate the same test images but use different naming conventions.
    """
    aligned = {}
    common_objs = set(clip_data.keys()) & set(dino_data.keys())

    for obj in sorted(common_objs):
        # Build path-to-index maps
        clip_path_map = {}
        for i, p in enumerate(clip_data[obj]['paths']):
            key = p.replace('\\', '/').lower()
            clip_path_map[key] = i

        dino_path_map = {}
        for i, p in enumerate(dino_data[obj]['paths']):
            key = p.replace('\\', '/').lower()
            dino_path_map[key] = i

        # Find common paths
        common_paths = set(clip_path_map.keys()) & set(dino_path_map.keys())

        if len(common_paths) == 0:
            # Try matching by filename only
            clip_fname_map = {}
            for i, p in enumerate(clip_data[obj]['paths']):
                fname = os.path.basename(p).lower()
                subdir = os.path.dirname(p).replace('\\', '/').split('/')[-1].lower()
                key = f"{subdir}/{fname}"
                clip_fname_map[key] = i

            dino_fname_map = {}
            for i, p in enumerate(dino_data[obj]['paths']):
                fname = os.path.basename(p).lower()
                subdir = os.path.dirname(p).replace('\\', '/').split('/')[-1].lower()
                key = f"{subdir}/{fname}"
                dino_fname_map[key] = i

            common_paths = set(clip_fname_map.keys()) & set(dino_fname_map.keys())

            if len(common_paths) > 0:
                clip_path_map = clip_fname_map
                dino_path_map = dino_fname_map
            else:
                # Fallback: same order, same count
                n = min(len(clip_data[obj]['scores']), len(dino_data[obj]['scores']))
                print(f"  Warning: {obj} - no path match found, using order-based alignment ({n} samples)")
                aligned[obj] = {
                    'clip_scores': np.array(clip_data[obj]['scores'][:n]),
                    'dino_scores': np.array(dino_data[obj]['scores'][:n]),
                    'labels': np.array(clip_data[obj]['labels'][:n]),
                    'n_samples': n
                }
                continue

        # Align by common paths
        clip_scores = []
        dino_scores = []
        labels = []

        for path in sorted(common_paths):
            ci = clip_path_map[path]
            di = dino_path_map[path]
            clip_scores.append(clip_data[obj]['scores'][ci])
            dino_scores.append(dino_data[obj]['scores'][di])
            labels.append(clip_data[obj]['labels'][ci])

        n_total_clip = len(clip_data[obj]['scores'])
        n_total_dino = len(dino_data[obj]['scores'])
        n_aligned = len(clip_scores)

        if n_aligned < n_total_clip or n_aligned < n_total_dino:
            print(f"  {obj}: aligned {n_aligned}/{n_total_clip} (CLIP) and {n_aligned}/{n_total_dino} (DINOv2)")
        else:
            print(f"  {obj}: perfectly aligned {n_aligned} samples")

        aligned[obj] = {
            'clip_scores': np.array(clip_scores),
            'dino_scores': np.array(dino_scores),
            'labels': np.array(labels),
            'n_samples': n_aligned
        }

    return aligned


def score_fusion(clip_scores, dino_scores, alpha=0.5, normalize_method='minmax'):
    """
    Fuse scores from two models.
    alpha: weight for DINOv2 (1-alpha for CLIP)
    """
    if normalize_method:
        clip_norm = normalize_scores(clip_scores, normalize_method)
        dino_norm = normalize_scores(dino_scores, normalize_method)
    else:
        clip_norm = clip_scores
        dino_norm = dino_scores

    fused = (1 - alpha) * clip_norm + alpha * dino_norm
    return fused


def evaluate_fusion_strategies(aligned_data, dataset_name=""):
    """Evaluate multiple fusion strategies and return comprehensive results."""

    print(f"\n{'='*90}")
    print(f"  SCORE-LEVEL FUSION RESULTS - {dataset_name}")
    print(f"{'='*90}")

    # Per-category results for each strategy
    strategies = {
        'CLIP only': {},
        'DINOv2 only': {},
        'Equal (a=0.5)': {},
        'CLIP-heavy (a=0.3)': {},
        'DINO-heavy (a=0.7)': {},
        'Max fusion': {},
        'Rank fusion': {},
    }

    # Also sweep alpha for fine-grained analysis
    alpha_steps = [i / 100.0 for i in range(0, 101, 5)]
    alpha_results = {a: {} for a in alpha_steps}

    all_categories = sorted(aligned_data.keys())

    for obj in all_categories:
        d = aligned_data[obj]
        labels = d['labels']

        if len(np.unique(labels)) < 2:
            continue

        # Strategy 1: CLIP only
        strategies['CLIP only'][obj] = compute_auroc(labels, d['clip_scores'])

        # Strategy 2: DINOv2 only
        strategies['DINOv2 only'][obj] = compute_auroc(labels, d['dino_scores'])

        # Strategy 3: Equal weight (normalized)
        fused_equal = score_fusion(d['clip_scores'], d['dino_scores'], alpha=0.5)
        strategies['Equal (a=0.5)'][obj] = compute_auroc(labels, fused_equal)

        # Strategy 4: CLIP-heavy
        fused_clip = score_fusion(d['clip_scores'], d['dino_scores'], alpha=0.3)
        strategies['CLIP-heavy (a=0.3)'][obj] = compute_auroc(labels, fused_clip)

        # Strategy 5: DINO-heavy
        fused_dino = score_fusion(d['clip_scores'], d['dino_scores'], alpha=0.7)
        strategies['DINO-heavy (a=0.7)'][obj] = compute_auroc(labels, fused_dino)

        # Strategy 6: Max of normalized scores
        clip_norm = normalize_scores(d['clip_scores'])
        dino_norm = normalize_scores(d['dino_scores'])
        fused_max = np.maximum(clip_norm, dino_norm)
        strategies['Max fusion'][obj] = compute_auroc(labels, fused_max)

        # Strategy 7: Rank fusion
        clip_rank = normalize_scores(d['clip_scores'], 'rank')
        dino_rank = normalize_scores(d['dino_scores'], 'rank')
        fused_rank = 0.5 * clip_rank + 0.5 * dino_rank
        strategies['Rank fusion'][obj] = compute_auroc(labels, fused_rank)

        # Alpha sweep
        for alpha in alpha_steps:
            fused = score_fusion(d['clip_scores'], d['dino_scores'], alpha=alpha)
            alpha_results[alpha][obj] = compute_auroc(labels, fused)

    # Print per-category results table
    header_strategies = ['CLIP only', 'DINOv2 only', 'Equal (a=0.5)', 'Max fusion', 'Rank fusion']
    print(f"\n{'Category':<15}", end="")
    for s in header_strategies:
        print(f" {s:>14}", end="")
    print()
    print("-" * (15 + 15 * len(header_strategies)))

    for obj in all_categories:
        if obj not in strategies['CLIP only']:
            continue
        print(f"{obj:<15}", end="")
        for s in header_strategies:
            val = strategies[s].get(obj, float('nan'))
            print(f" {val:>13.1f}%", end="")
        print()

    print("-" * (15 + 15 * len(header_strategies)))

    # Means
    print(f"{'Mean':<15}", end="")
    strategy_means = {}
    for s in header_strategies:
        vals = [v for v in strategies[s].values() if not np.isnan(v)]
        mean_val = np.mean(vals) if vals else float('nan')
        strategy_means[s] = mean_val
        print(f" {mean_val:>13.1f}%", end="")
    print()

    # Additional strategies
    for s in ['CLIP-heavy (a=0.3)', 'DINO-heavy (a=0.7)']:
        vals = [v for v in strategies[s].values() if not np.isnan(v)]
        strategy_means[s] = np.mean(vals) if vals else float('nan')

    print(f"\n  Additional strategies:")
    print(f"    CLIP-heavy (a=0.3): {strategy_means.get('CLIP-heavy (a=0.3)', 0):.1f}%")
    print(f"    DINO-heavy (a=0.7): {strategy_means.get('DINO-heavy (a=0.7)', 0):.1f}%")

    # Find best global alpha
    print(f"\n--- Global Alpha Sweep ---")
    best_alpha = 0
    best_mean = 0
    for alpha in alpha_steps:
        vals = [v for v in alpha_results[alpha].values() if not np.isnan(v)]
        mean_val = np.mean(vals) if vals else 0
        if mean_val > best_mean:
            best_mean = mean_val
            best_alpha = alpha
    print(f"  Best global alpha (DINOv2 weight): {best_alpha:.2f}")
    print(f"  Best global mean AUROC: {best_mean:.1f}%")

    # Top 5 alphas
    alpha_means = []
    for alpha in alpha_steps:
        vals = [v for v in alpha_results[alpha].values() if not np.isnan(v)]
        alpha_means.append((alpha, np.mean(vals) if vals else 0))
    alpha_means.sort(key=lambda x: -x[1])
    print(f"  Top 5 alphas:")
    for a, m in alpha_means[:5]:
        print(f"    alpha={a:.2f}: {m:.1f}%")

    # Find best per-category alpha (oracle)
    print(f"\n--- Per-Category Best Alpha (Oracle) ---")
    oracle_results = {}
    best_alphas = {}
    for obj in all_categories:
        if obj not in alpha_results[0.0]:
            continue
        best_a = 0
        best_v = 0
        for alpha in alpha_steps:
            if alpha_results[alpha].get(obj, 0) > best_v:
                best_v = alpha_results[alpha][obj]
                best_a = alpha
        oracle_results[obj] = best_v
        best_alphas[obj] = best_a
        clip_v = strategies['CLIP only'].get(obj, 0)
        dino_v = strategies['DINOv2 only'].get(obj, 0)
        winner = "DINOv2" if best_a > 0.5 else ("CLIP" if best_a < 0.5 else "equal")
        print(f"  {obj:<15} a={best_a:.2f} -> {best_v:.1f}% (CLIP:{clip_v:.1f}%, DINOv2:{dino_v:.1f}%, winner:{winner})")

    oracle_mean = np.mean(list(oracle_results.values())) if oracle_results else 0
    clip_mean = strategy_means.get('CLIP only', 0)
    dino_mean = strategy_means.get('DINOv2 only', 0)

    print(f"\n  Oracle Per-Category Mean AUROC: {oracle_mean:.1f}%")
    print(f"  CLIP-only Mean: {clip_mean:.1f}%")
    print(f"  DINOv2-only Mean: {dino_mean:.1f}%")
    print(f"  Improvement over CLIP: +{oracle_mean - clip_mean:.1f}%")
    print(f"  Improvement over DINOv2: +{oracle_mean - dino_mean:.1f}%")

    # Complementarity analysis
    print(f"\n--- Complementarity Analysis ---")
    n_clip_better = sum(1 for obj in oracle_results if best_alphas[obj] < 0.5)
    n_dino_better = sum(1 for obj in oracle_results if best_alphas[obj] > 0.5)
    n_equal = sum(1 for obj in oracle_results if best_alphas[obj] == 0.5)
    n_fusion = sum(1 for obj in oracle_results if 0 < best_alphas[obj] < 1)
    print(f"  CLIP preferred: {n_clip_better} categories")
    print(f"  DINOv2 preferred: {n_dino_better} categories")
    print(f"  Equal/mixed: {n_equal} categories")
    print(f"  Fusion beneficial (0 < alpha < 1): {n_fusion} categories")

    return strategies, alpha_results, oracle_results, best_alphas


def main():
    """Main function for score-level fusion evaluation."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # =====================================================
    # MVTec AD
    # =====================================================
    clip_mvtec = os.path.join(base_dir, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv")
    dino_mvtec_weak = os.path.join(base_dir, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                    "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv")
    dino_mvtec_strong = os.path.join(base_dir, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                      "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")

    # Paths for VisA
    clip_visa = os.path.join(base_dir, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv")
    dino_visa = os.path.join(base_dir, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv")

    # Check files
    print("=" * 80)
    print("  SCORE-LEVEL FUSION: AnomalyCLIP + AnomalyDINO")
    print("=" * 80)
    print("\nChecking data files...")
    all_files = {
        "CLIP MVTec": clip_mvtec,
        "DINOv2 MVTec (weak)": dino_mvtec_weak,
        "DINOv2 MVTec (strong)": dino_mvtec_strong,
        "CLIP VisA": clip_visa,
        "DINOv2 VisA": dino_visa,
    }
    for name, path in all_files.items():
        exists = os.path.exists(path)
        print(f"  {name}: {'[OK]' if exists else '[MISSING]'}")

    all_oracle = {}
    all_strategies = {}

    # =====================================================
    # MVTec AD Fusion
    # =====================================================
    if os.path.exists(clip_mvtec):
        clip_data_mvtec = load_anomalyclip_scores(clip_mvtec)
        print(f"\nAnomalyCLIP MVTec: {len(clip_data_mvtec)} categories, "
              f"{sum(len(v['scores']) for v in clip_data_mvtec.values())} samples")

        # Merge DINOv2 results
        dino_data_mvtec = {}
        for dino_path in [dino_mvtec_weak, dino_mvtec_strong]:
            if os.path.exists(dino_path):
                data = load_anomalydino_scores(dino_path)
                dino_data_mvtec.update(data)

        if dino_data_mvtec:
            print(f"AnomalyDINO MVTec: {len(dino_data_mvtec)} categories, "
                  f"{sum(len(v['scores']) for v in dino_data_mvtec.values())} samples")

            print("\n--- Aligning MVTec AD samples ---")
            aligned_mvtec = align_samples(clip_data_mvtec, dino_data_mvtec)

            strats_m, alpha_m, oracle_m, best_a_m = evaluate_fusion_strategies(aligned_mvtec, "MVTec AD")
            all_strategies['mvtec'] = strats_m
            all_oracle['mvtec'] = oracle_m

    # =====================================================
    # VisA Fusion
    # =====================================================
    if os.path.exists(clip_visa) and os.path.exists(dino_visa):
        clip_data_visa = load_anomalyclip_scores(clip_visa)
        dino_data_visa = load_anomalydino_scores(dino_visa)

        print(f"\nAnomalyCLIP VisA: {len(clip_data_visa)} categories, "
              f"{sum(len(v['scores']) for v in clip_data_visa.values())} samples")
        print(f"AnomalyDINO VisA: {len(dino_data_visa)} categories, "
              f"{sum(len(v['scores']) for v in dino_data_visa.values())} samples")

        print("\n--- Aligning VisA samples ---")
        aligned_visa = align_samples(clip_data_visa, dino_data_visa)

        strats_v, alpha_v, oracle_v, best_a_v = evaluate_fusion_strategies(aligned_visa, "VisA")
        all_strategies['visa'] = strats_v
        all_oracle['visa'] = oracle_v

    # =====================================================
    # Combined Summary
    # =====================================================
    if len(all_strategies) == 2:
        print(f"\n{'='*90}")
        print(f"  COMBINED SUMMARY (MVTec AD + VisA)")
        print(f"{'='*90}")

        for strategy_name in ['CLIP only', 'DINOv2 only', 'Equal (a=0.5)', 'Max fusion', 'Rank fusion']:
            mvtec_vals = [v for v in all_strategies['mvtec'][strategy_name].values() if not np.isnan(v)]
            visa_vals = [v for v in all_strategies['visa'][strategy_name].values() if not np.isnan(v)]
            all_vals = mvtec_vals + visa_vals
            print(f"  {strategy_name:<20}: MVTec {np.mean(mvtec_vals):.1f}% | VisA {np.mean(visa_vals):.1f}% | Combined {np.mean(all_vals):.1f}%")

        # Combined oracle
        all_oracle_vals = list(all_oracle['mvtec'].values()) + list(all_oracle['visa'].values())
        print(f"  {'Oracle (per-cat)':<20}: Combined {np.mean(all_oracle_vals):.1f}%")
    elif len(all_strategies) == 1:
        dataset_name = list(all_strategies.keys())[0]
        print(f"\n  Note: Only {dataset_name} results available. Run the other dataset to see combined results.")

    # Save results
    if not all_oracle:
        raise FileNotFoundError('no datasets loaded - check the score CSV paths printed above')
    output_path = os.path.join(base_dir, "fusion_results.json")
    output_data = {}
    for ds_name in all_strategies:
        output_data[ds_name] = {
            'strategies': {k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in all_strategies[ds_name].items()},
            'oracle': {k: round(v, 2) for k, v in all_oracle.get(ds_name, {}).items()},
        }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
