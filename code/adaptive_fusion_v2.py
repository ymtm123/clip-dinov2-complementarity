"""
Adaptive Score-Level Fusion v2 (AdaFuse-v2)
============================================

Improvements over v1:
1. Parameter-free methods (no grid search needed)
2. Image-level adaptive weighting (per-image alpha)
3. Multi-signal ensemble with automatic selection
4. Cross-dataset evaluation

Key insight: Instead of estimating a single alpha per category,
estimate per-IMAGE alpha based on each model's confidence for that
specific test image.
"""

import csv
import json
import os
import re
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from collections import defaultdict

from score_fusion import (
    load_anomalyclip_scores,
    load_anomalydino_scores,
    align_samples,
    normalize_scores,
    compute_auroc,
)


# ============================================================================
# Method 1: Parameter-Free Category-Level Adaptive Fusion
# ============================================================================

def parameterfree_adaptive(aligned_data, signal_fns, base_alpha=0.65):
    """
    Parameter-free adaptive fusion.
    Alpha = base_alpha + delta, where delta is derived from signal ratios.
    No grid search or hyperparameter tuning.

    signal_fns: dict of {name: function} to compute confidence signals
    """
    results = {}

    for obj in sorted(aligned_data.keys()):
        d = aligned_data[obj]
        labels = d['labels']
        if len(np.unique(labels)) < 2:
            continue

        clip_norm = normalize_scores(d['clip_scores'], 'minmax')
        dino_norm = normalize_scores(d['dino_scores'], 'minmax')

        # Compute all signals
        ratios = []
        for name, fn in signal_fns.items():
            sc = fn(clip_norm)
            sd = fn(dino_norm)
            if sc + sd > 1e-10:
                ratio = sd / (sc + sd)
            else:
                ratio = 0.5
            ratios.append(ratio)

        # Average signal ratio
        mean_ratio = np.mean(ratios)  # [0, 1], >0.5 means DINOv2 more confident

        # Map to alpha: ratio 0.5 -> base_alpha, deviations scale linearly
        alpha = base_alpha + 0.5 * (mean_ratio - 0.5)
        alpha = np.clip(alpha, 0.1, 0.9)

        fused = (1 - alpha) * clip_norm + alpha * dino_norm
        auroc = compute_auroc(labels, fused)

        results[obj] = {
            'alpha': round(alpha, 3),
            'auroc': round(auroc, 2),
            'mean_signal_ratio': round(mean_ratio, 3),
        }

    return results


# ============================================================================
# Method 2: Image-Level Adaptive Fusion
# ============================================================================

def image_level_adaptive(aligned_data, method='extremity'):
    """
    Per-image adaptive fusion: each test image gets its own alpha.

    Methods:
    - 'extremity': trust the model whose normalized score is more extreme
      (further from 0.5 = ambiguous zone). Inspired by confidence weighting.
    - 'rank_extremity': same but on rank-normalized scores
    - 'softmax': softmax of score magnitudes to determine weights
    """
    results = {}

    for obj in sorted(aligned_data.keys()):
        d = aligned_data[obj]
        labels = d['labels']
        if len(np.unique(labels)) < 2:
            continue

        clip_norm = normalize_scores(d['clip_scores'], 'minmax')
        dino_norm = normalize_scores(d['dino_scores'], 'minmax')

        n = len(labels)
        fused = np.zeros(n)

        if method == 'extremity':
            # For each image, weight based on how extreme (far from 0.5) each score is
            clip_conf = np.abs(clip_norm - 0.5)  # Confidence: distance from ambiguity
            dino_conf = np.abs(dino_norm - 0.5)
            total_conf = clip_conf + dino_conf + 1e-10
            alpha_per_image = dino_conf / total_conf  # per-image alpha

            fused = (1 - alpha_per_image) * clip_norm + alpha_per_image * dino_norm

        elif method == 'rank_extremity':
            clip_rank = normalize_scores(d['clip_scores'], 'rank')
            dino_rank = normalize_scores(d['dino_scores'], 'rank')
            clip_conf = np.abs(clip_rank - 0.5)
            dino_conf = np.abs(dino_rank - 0.5)
            total_conf = clip_conf + dino_conf + 1e-10
            alpha_per_image = dino_conf / total_conf
            fused = (1 - alpha_per_image) * clip_norm + alpha_per_image * dino_norm

        elif method == 'softmax':
            # Softmax over score magnitudes with temperature
            temperature = 2.0
            clip_logit = clip_norm / temperature
            dino_logit = dino_norm / temperature
            max_logit = np.maximum(clip_logit, dino_logit)
            clip_exp = np.exp(clip_logit - max_logit)
            dino_exp = np.exp(dino_logit - max_logit)
            alpha_per_image = dino_exp / (clip_exp + dino_exp)
            fused = (1 - alpha_per_image) * clip_norm + alpha_per_image * dino_norm

        elif method == 'agreement_weighted':
            # When models agree (both high or both low), trust the mean
            # When they disagree, trust the one with higher absolute score
            agreement = 1 - np.abs(clip_norm - dino_norm)
            # In agreement: equal weight; in disagreement: trust higher score
            base_alpha = 0.65  # Slight DINOv2 preference
            # Disagreement signal: shift alpha toward the more confident model
            clip_conf = clip_norm  # Higher raw score = more anomalous
            dino_conf = dino_norm
            conf_ratio = dino_conf / (clip_conf + dino_conf + 1e-10)
            # Blend: in agreement use base_alpha, in disagreement use conf_ratio
            alpha_per_image = agreement * base_alpha + (1 - agreement) * conf_ratio
            fused = (1 - alpha_per_image) * clip_norm + alpha_per_image * dino_norm

        elif method == 'hybrid_cat_img':
            # Hybrid: category-level base + image-level adjustment
            # First compute category-level signal (variance-based)
            sig_clip = np.std(clip_norm)
            sig_dino = np.std(dino_norm)
            total = sig_clip + sig_dino + 1e-10
            cat_alpha = sig_dino / total
            cat_alpha = 0.65 + 0.5 * (cat_alpha - 0.5)  # Center at 0.65

            # Image-level adjustment: shift toward more extreme model
            clip_conf = np.abs(clip_norm - 0.5)
            dino_conf = np.abs(dino_norm - 0.5)
            total_conf = clip_conf + dino_conf + 1e-10
            img_alpha = dino_conf / total_conf

            # Blend: 70% category, 30% image
            alpha_per_image = 0.7 * cat_alpha + 0.3 * img_alpha
            alpha_per_image = np.clip(alpha_per_image, 0.1, 0.9)
            fused = (1 - alpha_per_image) * clip_norm + alpha_per_image * dino_norm

        auroc = compute_auroc(labels, fused)
        results[obj] = {'auroc': round(auroc, 2)}

    return results


# ============================================================================
# Method 3: Unsupervised Model Selection via Score Distribution
# ============================================================================

def distribution_based_selection(aligned_data, method='gmm_bic'):
    """
    Use score distribution properties to select between models or determine weights.

    Methods:
    - 'gmm_bic': Fit 1-comp and 2-comp GMM, use BIC difference as signal.
      If 2-comp GMM is much better -> model separates normal/anomaly well.
    - 'ks_test': Use KS statistic between top-half and bottom-half scores.
    """
    results = {}

    for obj in sorted(aligned_data.keys()):
        d = aligned_data[obj]
        labels = d['labels']
        if len(np.unique(labels)) < 2:
            continue

        clip_norm = normalize_scores(d['clip_scores'], 'minmax')
        dino_norm = normalize_scores(d['dino_scores'], 'minmax')

        if method == 'gmm_bic':
            # Fit GMMs to each model's score distribution
            def gmm_bic_diff(scores):
                X = scores.reshape(-1, 1)
                try:
                    gmm1 = GaussianMixture(1, random_state=42).fit(X)
                    gmm2 = GaussianMixture(2, random_state=42).fit(X)
                    return gmm1.bic(X) - gmm2.bic(X)  # Positive = 2-comp preferred
                except:
                    return 0.0

            clip_bic = gmm_bic_diff(clip_norm)
            dino_bic = gmm_bic_diff(dino_norm)

            # Higher BIC difference -> model separates better
            total = abs(clip_bic) + abs(dino_bic) + 1e-10
            # If both positive, weight proportionally
            # If only one positive, prefer that one
            clip_w = max(clip_bic, 0)
            dino_w = max(dino_bic, 0)
            if clip_w + dino_w > 0:
                alpha = 0.65 + 0.3 * (dino_w / (clip_w + dino_w) - 0.5)
            else:
                alpha = 0.65  # Default
            alpha = np.clip(alpha, 0.1, 0.9)

        elif method == 'ks_test':
            # Split scores at median and test if top/bottom halves differ
            def ks_separation(scores):
                median = np.median(scores)
                low = scores[scores <= median]
                high = scores[scores > median]
                if len(low) < 2 or len(high) < 2:
                    return 0.0
                ks_stat, _ = scipy_stats.ks_2samp(low, high)
                return ks_stat

            clip_ks = ks_separation(clip_norm)
            dino_ks = ks_separation(dino_norm)
            total = clip_ks + dino_ks + 1e-10
            alpha = 0.65 + 0.3 * (dino_ks / total - 0.5)
            alpha = np.clip(alpha, 0.1, 0.9)

        fused = (1 - alpha) * clip_norm + alpha * dino_norm
        auroc = compute_auroc(labels, fused)
        results[obj] = {
            'alpha': round(alpha, 3),
            'auroc': round(auroc, 2),
        }

    return results


# ============================================================================
# Method 4: Cross-dataset LOOCV
# ============================================================================

def cross_dataset_eval(aligned_mvtec, aligned_visa, signal_name, signal_fn):
    """
    Train hyperparameters on one dataset, evaluate on the other.
    This is the strongest evaluation: completely unseen categories.
    """
    from adaptive_fusion import grid_search_hyperparams, adaptive_fusion_single_signal

    results = {}

    # Train on MVTec, test on VisA
    (ba_m, sens_m), _, _ = grid_search_hyperparams(aligned_mvtec, signal_name)
    visa_results = adaptive_fusion_single_signal(aligned_visa, signal_name, ba_m, sens_m)
    visa_aurocs = [r['auroc'] for r in visa_results.values()]
    results['train_mvtec_test_visa'] = {
        'params': (ba_m, sens_m),
        'mean_auroc': np.mean(visa_aurocs) if visa_aurocs else 0,
        'per_category': {obj: r['auroc'] for obj, r in visa_results.items()},
    }

    # Train on VisA, test on MVTec
    (ba_v, sens_v), _, _ = grid_search_hyperparams(aligned_visa, signal_name)
    mvtec_results = adaptive_fusion_single_signal(aligned_mvtec, signal_name, ba_v, sens_v)
    mvtec_aurocs = [r['auroc'] for r in mvtec_results.values()]
    results['train_visa_test_mvtec'] = {
        'params': (ba_v, sens_v),
        'mean_auroc': np.mean(mvtec_aurocs) if mvtec_aurocs else 0,
        'per_category': {obj: r['auroc'] for obj, r in mvtec_results.items()},
    }

    return results


# ============================================================================
# Signal Functions (simplified copies to avoid circular import)
# ============================================================================

def sig_variance(scores):
    return np.std(scores)

def sig_iqr(scores):
    q75, q25 = np.percentile(scores, [75, 25])
    return q75 - q25

def sig_skewness(scores):
    return abs(scipy_stats.skew(scores))

def sig_topk_gap(scores, k_ratio=0.2):
    sorted_scores = np.sort(scores)[::-1]
    k = max(1, int(len(scores) * k_ratio))
    return sorted_scores[:k].mean() - sorted_scores[k:].mean()

def sig_bimodality(scores):
    n = len(scores)
    if n < 4:
        return 0.0
    skew = scipy_stats.skew(scores)
    kurt = scipy_stats.kurtosis(scores, fisher=False)
    if kurt < 1e-10:
        return 0.0
    return (skew ** 2 + 1) / kurt

def sig_gmm_sep(scores):
    if len(scores) < 10:
        return 0.0
    X = scores.reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
        gmm.fit(X)
        mu1, mu2 = gmm.means_.flatten()
        s1, s2 = np.sqrt(gmm.covariances_.flatten())
        if s1 ** 2 + s2 ** 2 < 1e-10:
            return 0.0
        return np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(s1 ** 2 + s2 ** 2)
    except:
        return 0.0

ALL_SIGNALS = {
    'variance': sig_variance,
    'iqr': sig_iqr,
    'skewness': sig_skewness,
    'topk_gap': sig_topk_gap,
    'bimodality': sig_bimodality,
    'gmm_sep': sig_gmm_sep,
}


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Data paths
    clip_mvtec = os.path.join(base_dir, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv")
    dino_mvtec_weak = os.path.join(base_dir, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                    "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv")
    dino_mvtec_strong = os.path.join(base_dir, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                      "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")
    clip_visa = os.path.join(base_dir, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv")
    dino_visa = os.path.join(base_dir, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv")

    # Load data
    clip_data_mvtec = load_anomalyclip_scores(clip_mvtec)
    dino_data_mvtec = {}
    for p in [dino_mvtec_weak, dino_mvtec_strong]:
        if os.path.exists(p):
            dino_data_mvtec.update(load_anomalydino_scores(p))

    clip_data_visa = load_anomalyclip_scores(clip_visa)
    dino_data_visa = load_anomalydino_scores(dino_visa)

    print("--- Aligning MVTec AD samples ---")
    aligned_mvtec = align_samples(clip_data_mvtec, dino_data_mvtec)
    print("\n--- Aligning VisA samples ---")
    aligned_visa = align_samples(clip_data_visa, dino_data_visa)

    all_results = {}

    for ds_name, aligned in [("MVTec AD", aligned_mvtec), ("VisA", aligned_visa)]:
        print(f"\n{'='*90}")
        print(f"  ADAPTIVE FUSION v2 -- {ds_name}")
        print(f"{'='*90}")

        ds_results = {}

        # ----- Baselines -----
        baselines = {}

        # CLIP only
        clip_mean = np.mean([compute_auroc(d['labels'], d['clip_scores'])
                           for d in aligned.values() if len(np.unique(d['labels'])) >= 2])
        baselines['CLIP only'] = clip_mean

        # DINOv2 only
        dino_mean = np.mean([compute_auroc(d['labels'], d['dino_scores'])
                           for d in aligned.values() if len(np.unique(d['labels'])) >= 2])
        baselines['DINOv2 only'] = dino_mean

        # Equal (a=0.5)
        equal_aurocs = []
        for obj, d in aligned.items():
            if len(np.unique(d['labels'])) < 2: continue
            cn = normalize_scores(d['clip_scores'], 'minmax')
            dn = normalize_scores(d['dino_scores'], 'minmax')
            equal_aurocs.append(compute_auroc(d['labels'], 0.5*cn + 0.5*dn))
        baselines['Equal (a=0.5)'] = np.mean(equal_aurocs)

        # Fixed best (a=0.65)
        fixed_aurocs = []
        for obj, d in aligned.items():
            if len(np.unique(d['labels'])) < 2: continue
            cn = normalize_scores(d['clip_scores'], 'minmax')
            dn = normalize_scores(d['dino_scores'], 'minmax')
            fixed_aurocs.append(compute_auroc(d['labels'], 0.35*cn + 0.65*dn))
        baselines['Fixed (a=0.65)'] = np.mean(fixed_aurocs)

        # Oracle
        oracle_aurocs = []
        for obj, d in aligned.items():
            if len(np.unique(d['labels'])) < 2: continue
            cn = normalize_scores(d['clip_scores'], 'minmax')
            dn = normalize_scores(d['dino_scores'], 'minmax')
            best = max(compute_auroc(d['labels'], (1-a)*cn + a*dn)
                      for a in [i/20.0 for i in range(21)])
            oracle_aurocs.append(best)
        baselines['Oracle'] = np.mean(oracle_aurocs)

        ds_results['baselines'] = baselines

        # ----- Method 1: Parameter-free category-level -----
        pf_result = parameterfree_adaptive(aligned, ALL_SIGNALS, base_alpha=0.65)
        pf_mean = np.mean([r['auroc'] for r in pf_result.values()])
        ds_results['ParamFree-AllSignals'] = pf_mean

        # Try with subset of best signals
        best_signals = {k: v for k, v in ALL_SIGNALS.items() if k in ['variance', 'bimodality', 'gmm_sep']}
        pf_result2 = parameterfree_adaptive(aligned, best_signals, base_alpha=0.65)
        pf_mean2 = np.mean([r['auroc'] for r in pf_result2.values()])
        ds_results['ParamFree-BestSignals'] = pf_mean2

        # ----- Method 2: Image-level adaptive -----
        img_methods = ['extremity', 'rank_extremity', 'softmax', 'agreement_weighted', 'hybrid_cat_img']
        for method in img_methods:
            il_result = image_level_adaptive(aligned, method=method)
            il_mean = np.mean([r['auroc'] for r in il_result.values()])
            ds_results[f'ImageLevel-{method}'] = il_mean

        # ----- Method 3: Distribution-based -----
        for method in ['gmm_bic', 'ks_test']:
            db_result = distribution_based_selection(aligned, method=method)
            db_mean = np.mean([r['auroc'] for r in db_result.values()])
            ds_results[f'DistBased-{method}'] = db_mean

        # ----- Print Summary -----
        print(f"\n{'Method':<30} {'Mean AUROC':>12} {'vs Equal':>10} {'vs Fixed':>10} {'vs Oracle':>10}")
        print("-" * 75)

        equal_val = baselines['Equal (a=0.5)']
        fixed_val = baselines['Fixed (a=0.65)']
        oracle_val = baselines['Oracle']

        all_methods = {**{k: v for k, v in baselines.items()},
                       **{k: v for k, v in ds_results.items() if k != 'baselines'}}

        for name, val in sorted(all_methods.items(), key=lambda x: -x[1]):
            is_new = name not in baselines
            prefix = ">> " if is_new else "   "
            print(f"{prefix}{name:<27} {val:>11.2f}% {val-equal_val:>+9.2f}% "
                  f"{val-fixed_val:>+9.2f}% {val-oracle_val:>+9.2f}%")

        all_results[ds_name] = ds_results

    # ----- Cross-Dataset Evaluation -----
    print(f"\n{'='*90}")
    print(f"  CROSS-DATASET EVALUATION")
    print(f"{'='*90}")

    for sig_name in ['variance', 'gmm_sep', 'bimodality']:
        cross_results = cross_dataset_eval(
            aligned_mvtec, aligned_visa, sig_name, ALL_SIGNALS[sig_name]
        )
        print(f"\n  Signal: {sig_name}")
        for setting, result in cross_results.items():
            print(f"    {setting}: {result['mean_auroc']:.2f}% (params: ba={result['params'][0]}, sens={result['params'][1]})")

    # ----- Per-category detail for best image-level method -----
    print(f"\n{'='*90}")
    print(f"  PER-CATEGORY DETAIL: Image-Level Methods")
    print(f"{'='*90}")

    for ds_name, aligned in [("MVTec AD", aligned_mvtec), ("VisA", aligned_visa)]:
        print(f"\n--- {ds_name} ---")
        print(f"{'Category':<15} {'CLIP':>8} {'DINOv2':>8} {'Equal':>8} {'Fixed65':>8} "
              f"{'Extrem':>8} {'Agree':>8} {'Hybrid':>8} {'Oracle':>8}")
        print("-" * 85)

        methods_per_cat = {}
        for method in ['extremity', 'agreement_weighted', 'hybrid_cat_img']:
            il = image_level_adaptive(aligned, method=method)
            methods_per_cat[method] = il

        for obj in sorted(aligned.keys()):
            d = aligned[obj]
            if len(np.unique(d['labels'])) < 2: continue
            cn = normalize_scores(d['clip_scores'], 'minmax')
            dn = normalize_scores(d['dino_scores'], 'minmax')

            clip_v = compute_auroc(d['labels'], d['clip_scores'])
            dino_v = compute_auroc(d['labels'], d['dino_scores'])
            equal_v = compute_auroc(d['labels'], 0.5*cn + 0.5*dn)
            fixed_v = compute_auroc(d['labels'], 0.35*cn + 0.65*dn)
            oracle_v = max(compute_auroc(d['labels'], (1-a)*cn + a*dn)
                          for a in [i/20.0 for i in range(21)])

            ext_v = methods_per_cat['extremity'].get(obj, {}).get('auroc', 0)
            agr_v = methods_per_cat['agreement_weighted'].get(obj, {}).get('auroc', 0)
            hyb_v = methods_per_cat['hybrid_cat_img'].get(obj, {}).get('auroc', 0)

            print(f"{obj:<15} {clip_v:>7.1f}% {dino_v:>7.1f}% {equal_v:>7.1f}% {fixed_v:>7.1f}% "
                  f"{ext_v:>7.1f}% {agr_v:>7.1f}% {hyb_v:>7.1f}% {oracle_v:>7.1f}%")

        # Means
        print("-" * 85)
        cats = [obj for obj in aligned.keys() if len(np.unique(aligned[obj]['labels'])) >= 2]
        means = []
        for method in ['extremity', 'agreement_weighted', 'hybrid_cat_img']:
            vals = [methods_per_cat[method].get(obj, {}).get('auroc', 0) for obj in cats]
            means.append(np.mean(vals))

        clip_m = np.mean([compute_auroc(aligned[o]['labels'], aligned[o]['clip_scores']) for o in cats])
        dino_m = np.mean([compute_auroc(aligned[o]['labels'], aligned[o]['dino_scores']) for o in cats])
        equal_m_list = []
        fixed_m_list = []
        oracle_m_list = []
        for o in cats:
            cn = normalize_scores(aligned[o]['clip_scores'], 'minmax')
            dn = normalize_scores(aligned[o]['dino_scores'], 'minmax')
            equal_m_list.append(compute_auroc(aligned[o]['labels'], 0.5*cn+0.5*dn))
            fixed_m_list.append(compute_auroc(aligned[o]['labels'], 0.35*cn+0.65*dn))
            oracle_m_list.append(max(compute_auroc(aligned[o]['labels'], (1-a)*cn+a*dn)
                                    for a in [i/20.0 for i in range(21)]))

        print(f"{'MEAN':<15} {clip_m:>7.1f}% {dino_m:>7.1f}% {np.mean(equal_m_list):>7.1f}% "
              f"{np.mean(fixed_m_list):>7.1f}% {means[0]:>7.1f}% {means[1]:>7.1f}% {means[2]:>7.1f}% "
              f"{np.mean(oracle_m_list):>7.1f}%")

    # Save all results
    output_path = os.path.join(base_dir, "adaptive_fusion_v2_results.json")
    # Convert numpy types for JSON
    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=to_serializable)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
