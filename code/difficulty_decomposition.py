"""
Experiment 28: Category Difficulty Decomposition & Fusion Gain Mechanism

Decomposes why fusion helps more for difficult categories:
A. What makes a category "difficult"? (score overlap, margin, sample size)
B. Exclusive detection analysis: per-image gain decomposition
C. Score agreement patterns: normal vs anomaly conditional agreement
D. Fusion gain stability: per-image bootstrap within categories
"""

import csv
import json
import os
import re
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from collections import defaultdict

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "difficulty_decomposition_results.json")


# === Data Loading (shared with other scripts) ===

def load_anomalyclip_scores(csv_path):
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            obj, img_path, score, gt = row[0], row[1], float(row[2]), int(row[3])
            match = re.search(r'/test/(.+)$', img_path.replace('\\', '/'))
            rel_path = match.group(1) if match else os.path.basename(img_path)
            data[obj]['scores'].append(score)
            data[obj]['labels'].append(gt)
            data[obj]['paths'].append(rel_path)
    return dict(data)


def load_anomalydino_scores(csv_path):
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            obj = row[0]
            sample = row[1]
            score = float(row[2])
            is_anomaly = 0 if sample.startswith("good/") else 1
            data[obj]['scores'].append(score)
            data[obj]['labels'].append(is_anomaly)
            data[obj]['paths'].append(sample)
    return dict(data)


def align_samples(clip_data, dino_data):
    aligned = {}
    common_objs = set(clip_data.keys()) & set(dino_data.keys())
    for obj in sorted(common_objs):
        clip_path_map = {}
        for i, p in enumerate(clip_data[obj]['paths']):
            key = p.replace('\\', '/').lower()
            clip_path_map[key] = i
        dino_path_map = {}
        for i, p in enumerate(dino_data[obj]['paths']):
            key = p.replace('\\', '/').lower()
            dino_path_map[key] = i
        common_paths = set(clip_path_map.keys()) & set(dino_path_map.keys())
        if len(common_paths) == 0:
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
                n = min(len(clip_data[obj]['scores']), len(dino_data[obj]['scores']))
                aligned[obj] = {
                    'clip_scores': np.array(clip_data[obj]['scores'][:n]),
                    'dino_scores': np.array(dino_data[obj]['scores'][:n]),
                    'labels': np.array(clip_data[obj]['labels'][:n]),
                    'n_samples': n
                }
                continue
        clip_scores, dino_scores, labels = [], [], []
        for path in sorted(common_paths):
            ci = clip_path_map[path]
            di = dino_path_map[path]
            clip_scores.append(clip_data[obj]['scores'][ci])
            dino_scores.append(dino_data[obj]['scores'][di])
            labels.append(clip_data[obj]['labels'][ci])
        aligned[obj] = {
            'clip_scores': np.array(clip_scores),
            'dino_scores': np.array(dino_scores),
            'labels': np.array(labels),
            'n_samples': len(labels)
        }
    return aligned


def load_all_datasets():
    base = os.path.dirname(__file__)
    datasets = {}

    clip_mvtec = load_anomalyclip_scores(
        os.path.join(base, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv"))
    dino_mvtec = load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv"))
    dino_mvtec.update(load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")))
    datasets["MVTec AD"] = align_samples(clip_mvtec, dino_mvtec)

    clip_visa = load_anomalyclip_scores(
        os.path.join(base, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv"))
    dino_visa = load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv"))
    datasets["VisA"] = align_samples(clip_visa, dino_visa)

    clip_btad = load_anomalyclip_scores(
        os.path.join(base, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv"))
    dino_btad = load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv"))
    datasets["BTAD"] = align_samples(clip_btad, dino_btad)

    return datasets


def normalize_minmax(scores):
    s = np.array(scores, dtype=float)
    smin, smax = s.min(), s.max()
    if smax - smin < 1e-10:
        return np.zeros_like(s)
    return (s - smin) / (smax - smin)


def compute_auroc(labels, scores):
    labels = np.array(labels)
    if len(np.unique(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, scores) * 100


def part_a_difficulty_factors(datasets):
    """Part A: What makes a category difficult? Decompose difficulty."""
    print("=" * 70)
    print("PART A: Category Difficulty Decomposition")
    print("=" * 70)

    cat_data = []

    for ds_name, ds_data in datasets.items():
        for cat, d in ds_data.items():
            labels = d['labels']
            clip_z = normalize_minmax(d['clip_scores'])
            dino_z = normalize_minmax(d['dino_scores'])
            fused = 0.65 * dino_z + 0.35 * clip_z

            normal_mask = labels == 0
            anomaly_mask = labels == 1

            if sum(anomaly_mask) == 0 or sum(normal_mask) == 0:
                continue

            # Difficulty metrics
            clip_auroc = compute_auroc(labels, clip_z)
            dino_auroc = compute_auroc(labels, dino_z)
            fused_auroc = compute_auroc(labels, fused)
            best_single = max(clip_auroc, dino_auroc)
            fusion_gain = fused_auroc - best_single

            # Score overlap: fraction of normal scores above min anomaly score
            for model_name, scores in [('clip', clip_z), ('dino', dino_z), ('fused', fused)]:
                normal_scores = scores[normal_mask]
                anomaly_scores = scores[anomaly_mask]

                # Overlap: fraction of normal scores >= median anomaly score
                overlap = np.mean(normal_scores >= np.median(anomaly_scores))

                # Cohen's d separation
                pooled_std = np.sqrt((np.var(normal_scores) + np.var(anomaly_scores)) / 2 + 1e-10)
                cohens_d = (np.mean(anomaly_scores) - np.mean(normal_scores)) / pooled_std

                if model_name == 'dino':
                    dino_overlap = overlap
                    dino_separation = cohens_d
                elif model_name == 'clip':
                    clip_overlap = overlap
                    clip_separation = cohens_d

            # Score disagreement on anomalies
            anomaly_clip = clip_z[anomaly_mask]
            anomaly_dino = dino_z[anomaly_mask]
            anomaly_corr = stats.spearmanr(anomaly_clip, anomaly_dino)[0] if len(anomaly_clip) > 2 else np.nan

            # Normal agreement
            normal_clip = clip_z[normal_mask]
            normal_dino = dino_z[normal_mask]
            normal_corr = stats.spearmanr(normal_clip, normal_dino)[0] if len(normal_clip) > 2 else np.nan

            cat_data.append({
                'dataset': ds_name,
                'category': cat,
                'clip_auroc': clip_auroc,
                'dino_auroc': dino_auroc,
                'best_single': best_single,
                'fused_auroc': fused_auroc,
                'fusion_gain': fusion_gain,
                'n_normal': int(sum(normal_mask)),
                'n_anomaly': int(sum(anomaly_mask)),
                'n_total': len(labels),
                'anomaly_ratio': float(sum(anomaly_mask) / len(labels)),
                'dino_overlap': float(dino_overlap),
                'clip_overlap': float(clip_overlap),
                'dino_separation': float(dino_separation),
                'clip_separation': float(clip_separation),
                'anomaly_score_corr': float(anomaly_corr) if not np.isnan(anomaly_corr) else None,
                'normal_score_corr': float(normal_corr) if not np.isnan(normal_corr) else None,
            })

    # Correlations: what predicts difficulty?
    best_singles = [c['best_single'] for c in cat_data]
    fusion_gains = [c['fusion_gain'] for c in cat_data]

    print(f"\n  N = {len(cat_data)} categories")
    print(f"\n  What predicts fusion gain?")
    predictors = [
        ('best_single', 'Best single AUROC (difficulty)'),
        ('dino_separation', 'DINOv2 score separation (Cohen\'s d)'),
        ('clip_separation', 'CLIP score separation (Cohen\'s d)'),
        ('n_anomaly', 'Number of anomaly images'),
        ('anomaly_ratio', 'Anomaly ratio'),
    ]

    predictor_corrs = {}
    for pred_key, pred_name in predictors:
        vals = [c[pred_key] for c in cat_data]
        if all(v is not None for v in vals):
            r, p = stats.spearmanr(vals, fusion_gains)
            predictor_corrs[pred_key] = {'rho': float(r), 'p': float(p)}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"    {pred_name:>45s}: rho={r:+.3f}, p={p:.4f} {sig}")

    # Anomaly score correlation: normal vs anomaly
    anom_corrs = [c['anomaly_score_corr'] for c in cat_data if c['anomaly_score_corr'] is not None]
    norm_corrs = [c['normal_score_corr'] for c in cat_data if c['normal_score_corr'] is not None]
    print(f"\n  Score correlation (CLIP vs DINOv2):")
    print(f"    On normal images:  mean rho = {np.mean(norm_corrs):.3f} (sd={np.std(norm_corrs):.3f})")
    print(f"    On anomaly images: mean rho = {np.mean(anom_corrs):.3f} (sd={np.std(anom_corrs):.3f})")
    if len(anom_corrs) > 2:
        t, p_ttest = stats.ttest_rel(norm_corrs[:len(anom_corrs)], anom_corrs)
        print(f"    Paired t-test (normal vs anomaly): t={t:.3f}, p={p_ttest:.4f}")

    # Does low anomaly correlation predict high fusion gain?
    anom_corr_data = [(c['anomaly_score_corr'], c['fusion_gain'])
                      for c in cat_data if c['anomaly_score_corr'] is not None]
    if len(anom_corr_data) > 3:
        ac, fg = zip(*anom_corr_data)
        r, p = stats.spearmanr(ac, fg)
        print(f"\n  Anomaly score disagreement -> fusion gain:")
        print(f"    rho(anomaly_corr, fusion_gain) = {r:+.3f}, p={p:.4f}")
        predictor_corrs['anomaly_score_corr'] = {'rho': float(r), 'p': float(p)}

    return {
        'n_categories': len(cat_data),
        'predictor_correlations': predictor_corrs,
        'mean_normal_corr': float(np.mean(norm_corrs)),
        'mean_anomaly_corr': float(np.mean(anom_corrs)),
    }


def part_b_exclusive_detection(datasets):
    """Part B: Per-image exclusive detection analysis."""
    print("\n" + "=" * 70)
    print("PART B: Per-Image Exclusive Detection Analysis")
    print("=" * 70)

    # Use the median of normal scores as threshold for each model per category
    total_anomalies = 0
    clip_exclusive = 0
    dino_exclusive = 0
    both_detect = 0
    neither_detect = 0
    fusion_rescues = 0  # fusion detects but best single doesn't

    cat_results = []

    for ds_name, ds_data in datasets.items():
        for cat, d in ds_data.items():
            labels = d['labels']
            clip_z = normalize_minmax(d['clip_scores'])
            dino_z = normalize_minmax(d['dino_scores'])
            fused = 0.65 * dino_z + 0.35 * clip_z

            normal_mask = labels == 0
            anomaly_mask = labels == 1

            if sum(anomaly_mask) == 0 or sum(normal_mask) == 0:
                continue

            # Threshold: 95th percentile of normal scores (5% FPR operating point)
            clip_thresh = np.percentile(clip_z[normal_mask], 95)
            dino_thresh = np.percentile(dino_z[normal_mask], 95)
            fused_thresh = np.percentile(fused[normal_mask], 95)

            # Per anomaly image: who detects it?
            n_anom = sum(anomaly_mask)
            anom_clip = clip_z[anomaly_mask] >= clip_thresh
            anom_dino = dino_z[anomaly_mask] >= dino_thresh
            anom_fused = fused[anomaly_mask] >= fused_thresh

            cat_clip_exc = int(np.sum(anom_clip & ~anom_dino))
            cat_dino_exc = int(np.sum(anom_dino & ~anom_clip))
            cat_both = int(np.sum(anom_clip & anom_dino))
            cat_neither = int(np.sum(~anom_clip & ~anom_dino))

            # Best single: use whichever model has higher AUROC
            clip_auroc = compute_auroc(labels, clip_z)
            dino_auroc = compute_auroc(labels, dino_z)
            if clip_auroc >= dino_auroc:
                best_single_detects = anom_clip
            else:
                best_single_detects = anom_dino

            cat_fusion_rescues = int(np.sum(anom_fused & ~best_single_detects))

            total_anomalies += n_anom
            clip_exclusive += cat_clip_exc
            dino_exclusive += cat_dino_exc
            both_detect += cat_both
            neither_detect += cat_neither
            fusion_rescues += cat_fusion_rescues

            cat_results.append({
                'dataset': ds_name,
                'category': cat,
                'n_anomaly': n_anom,
                'clip_exclusive': cat_clip_exc,
                'dino_exclusive': cat_dino_exc,
                'both_detect': cat_both,
                'neither_detect': cat_neither,
                'fusion_rescues': cat_fusion_rescues,
                'clip_exclusive_pct': cat_clip_exc / n_anom * 100 if n_anom > 0 else 0,
                'dino_exclusive_pct': cat_dino_exc / n_anom * 100 if n_anom > 0 else 0,
            })

    print(f"\n  At 5% FPR operating point:")
    print(f"  Total anomaly images: {total_anomalies}")
    print(f"  Both models detect:       {both_detect:4d} ({both_detect/total_anomalies*100:.1f}%)")
    print(f"  CLIP exclusively detects: {clip_exclusive:4d} ({clip_exclusive/total_anomalies*100:.1f}%)")
    print(f"  DINOv2 exclusively detects: {dino_exclusive:4d} ({dino_exclusive/total_anomalies*100:.1f}%)")
    print(f"  Neither model detects:    {neither_detect:4d} ({neither_detect/total_anomalies*100:.1f}%)")
    print(f"  Fusion rescues (detects when best single doesn't): {fusion_rescues:4d} ({fusion_rescues/total_anomalies*100:.1f}%)")

    # Top categories by exclusive detection
    print(f"\n  Top categories by CLIP-exclusive detection (at 5% FPR):")
    sorted_by_clip = sorted(cat_results, key=lambda x: x['clip_exclusive_pct'], reverse=True)
    for c in sorted_by_clip[:5]:
        print(f"    {c['dataset']}/{c['category']:15s}: {c['clip_exclusive']}/{c['n_anomaly']} "
              f"({c['clip_exclusive_pct']:.0f}%)")

    print(f"\n  Top categories by DINOv2-exclusive detection (at 5% FPR):")
    sorted_by_dino = sorted(cat_results, key=lambda x: x['dino_exclusive_pct'], reverse=True)
    for c in sorted_by_dino[:5]:
        print(f"    {c['dataset']}/{c['category']:15s}: {c['dino_exclusive']}/{c['n_anomaly']} "
              f"({c['dino_exclusive_pct']:.0f}%)")

    return {
        'total_anomalies': total_anomalies,
        'both_detect_pct': float(both_detect / total_anomalies * 100),
        'clip_exclusive_pct': float(clip_exclusive / total_anomalies * 100),
        'dino_exclusive_pct': float(dino_exclusive / total_anomalies * 100),
        'neither_detect_pct': float(neither_detect / total_anomalies * 100),
        'fusion_rescue_pct': float(fusion_rescues / total_anomalies * 100),
    }


def part_c_agreement_asymmetry(datasets):
    """Part C: Normal vs anomaly agreement asymmetry (why fusion works mechanistically)."""
    print("\n" + "=" * 70)
    print("PART C: Score Agreement Asymmetry (Normal vs Anomaly)")
    print("=" * 70)

    # For each category, compute rank correlation of CLIP and DINOv2 scores
    # separately for normal and anomaly images
    normal_corrs = []
    anomaly_corrs = []
    agreement_gaps = []  # normal_corr - anomaly_corr

    for ds_name, ds_data in datasets.items():
        for cat, d in ds_data.items():
            labels = d['labels']
            clip_z = normalize_minmax(d['clip_scores'])
            dino_z = normalize_minmax(d['dino_scores'])

            normal_mask = labels == 0
            anomaly_mask = labels == 1

            if sum(anomaly_mask) < 3 or sum(normal_mask) < 3:
                continue

            r_norm, _ = stats.spearmanr(clip_z[normal_mask], dino_z[normal_mask])
            r_anom, _ = stats.spearmanr(clip_z[anomaly_mask], dino_z[anomaly_mask])

            normal_corrs.append(r_norm)
            anomaly_corrs.append(r_anom)
            agreement_gaps.append(r_norm - r_anom)

    print(f"\n  N = {len(normal_corrs)} categories")
    print(f"\n  Score correlation (Spearman rho, CLIP vs DINOv2):")
    print(f"    On normal images:  mean = {np.mean(normal_corrs):.3f} (sd = {np.std(normal_corrs):.3f})")
    print(f"    On anomaly images: mean = {np.mean(anomaly_corrs):.3f} (sd = {np.std(anomaly_corrs):.3f})")
    print(f"    Agreement gap:     mean = {np.mean(agreement_gaps):+.3f}")

    # Test: is the gap significant?
    t, p = stats.ttest_rel(normal_corrs, anomaly_corrs)
    print(f"    Paired t-test: t={t:.3f}, p={p:.6f}")

    # How many categories show the pattern (higher agreement on normals)?
    n_higher_normal = sum(1 for g in agreement_gaps if g > 0)
    print(f"    Categories with higher normal agreement: {n_higher_normal}/{len(agreement_gaps)} "
          f"({n_higher_normal/len(agreement_gaps)*100:.0f}%)")

    # Does the gap predict fusion gain?
    fusion_gains = []
    for ds_name, ds_data in datasets.items():
        for cat, d in ds_data.items():
            labels = d['labels']
            clip_z = normalize_minmax(d['clip_scores'])
            dino_z = normalize_minmax(d['dino_scores'])

            normal_mask = labels == 0
            anomaly_mask = labels == 1
            if sum(anomaly_mask) < 3 or sum(normal_mask) < 3:
                continue

            best_single = max(compute_auroc(labels, clip_z), compute_auroc(labels, dino_z))
            fused = 0.65 * dino_z + 0.35 * clip_z
            fused_auroc = compute_auroc(labels, fused)
            fusion_gains.append(fused_auroc - best_single)

    r_gap, p_gap = stats.spearmanr(agreement_gaps, fusion_gains)
    print(f"\n  Agreement gap -> fusion gain: rho={r_gap:+.3f}, p={p_gap:.4f}")

    return {
        'mean_normal_corr': float(np.mean(normal_corrs)),
        'mean_anomaly_corr': float(np.mean(anomaly_corrs)),
        'mean_gap': float(np.mean(agreement_gaps)),
        'ttest_p': float(p),
        'higher_normal_agreement_pct': float(n_higher_normal / len(agreement_gaps) * 100),
        'gap_predicts_gain': {'rho': float(r_gap), 'p': float(p_gap)},
    }


def part_d_per_image_bootstrap(datasets):
    """Part D: Per-image bootstrap to estimate fusion gain stability."""
    print("\n" + "=" * 70)
    print("PART D: Per-Image Bootstrap Stability")
    print("=" * 70)

    np.random.seed(42)
    n_boot = 1000

    cat_stability = []

    for ds_name, ds_data in datasets.items():
        for cat, d in ds_data.items():
            labels = d['labels']
            clip_z = normalize_minmax(d['clip_scores'])
            dino_z = normalize_minmax(d['dino_scores'])
            fused = 0.65 * dino_z + 0.35 * clip_z

            if len(np.unique(labels)) < 2:
                continue

            n = len(labels)
            boot_gains = []
            for _ in range(n_boot):
                idx = np.random.choice(n, n, replace=True)
                b_labels = labels[idx]
                if len(np.unique(b_labels)) < 2:
                    continue
                b_clip = compute_auroc(b_labels, clip_z[idx])
                b_dino = compute_auroc(b_labels, dino_z[idx])
                b_fused = compute_auroc(b_labels, fused[idx])
                boot_gains.append(b_fused - max(b_clip, b_dino))

            if len(boot_gains) < 100:
                continue

            lo, hi = np.percentile(boot_gains, [2.5, 97.5])
            mean_gain = np.mean(boot_gains)
            sig = lo > 0  # CI excludes zero

            cat_stability.append({
                'dataset': ds_name,
                'category': cat,
                'mean_gain': float(mean_gain),
                'ci_lo': float(lo),
                'ci_hi': float(hi),
                'significant': bool(sig),
                'n_images': n,
            })

    n_sig = sum(1 for c in cat_stability if c['significant'])
    n_total = len(cat_stability)
    n_positive = sum(1 for c in cat_stability if c['mean_gain'] > 0)

    print(f"\n  Per-category image-level bootstrap (N={n_boot}):")
    print(f"  Total categories analyzed: {n_total}")
    print(f"  Positive mean fusion gain: {n_positive}/{n_total} ({n_positive/n_total*100:.0f}%)")
    print(f"  Significantly positive (CI > 0): {n_sig}/{n_total} ({n_sig/n_total*100:.0f}%)")

    # Show categories where fusion is negative
    negatives = [c for c in cat_stability if c['mean_gain'] < 0]
    if negatives:
        print(f"\n  Categories where fusion hurts (mean gain < 0):")
        for c in sorted(negatives, key=lambda x: x['mean_gain']):
            print(f"    {c['dataset']}/{c['category']:15s}: {c['mean_gain']:+.2f}pp "
                  f"[{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]")

    # Show strongest fusion beneficiaries
    print(f"\n  Top 5 fusion beneficiaries:")
    for c in sorted(cat_stability, key=lambda x: x['mean_gain'], reverse=True)[:5]:
        sig_mark = "***" if c['significant'] else ""
        print(f"    {c['dataset']}/{c['category']:15s}: {c['mean_gain']:+.2f}pp "
              f"[{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}] {sig_mark}")

    return {
        'n_categories': n_total,
        'positive_rate': float(n_positive / n_total * 100),
        'significant_rate': float(n_sig / n_total * 100),
        'n_negative': len(negatives),
    }


def main():
    print("Experiment 28: Category Difficulty Decomposition & Fusion Gain Mechanism")
    print("=" * 70)

    datasets = load_all_datasets()
    total_cats = sum(len(d) for d in datasets.values())
    for ds_name, ds_data in datasets.items():
        print(f"  {ds_name}: {len(ds_data)} categories loaded")
    print(f"  Total: {total_cats} categories")

    results = {}
    results["part_a"] = part_a_difficulty_factors(datasets)
    results["part_b"] = part_b_exclusive_detection(datasets)
    results["part_c"] = part_c_agreement_asymmetry(datasets)
    results["part_d"] = part_d_per_image_bootstrap(datasets)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"experiment_28_difficulty_decomposition": results}, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
