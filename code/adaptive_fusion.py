"""
Adaptive Score-Level Fusion (AdaFuse)
=====================================

Core technical contribution: unsupervised estimation of per-category
fusion weights using test-time score distribution analysis.

Key idea: The optimal weight alpha for combining CLIP and DINOv2 scores
varies per category. We estimate this weight WITHOUT anomaly labels by
analyzing the discriminability of each model's score distribution.

Fusion formula:
  score_fused(x) = (1 - alpha*) * score_CLIP_norm(x) + alpha* * score_DINOv2_norm(x)

where alpha* is estimated per category using unsupervised signals.

Methods implemented:
1. Score Variance - weight proportional to score spread
2. Score Range (IQR) - weight proportional to interquartile range
3. Bimodality Coefficient - weight proportional to distribution bimodality
4. Top-k Gap - weight proportional to separation of top anomaly candidates
5. GMM Separability - weight proportional to Gaussian mixture separation
6. Combined (Meta-estimator) - weighted combination of signals
7. Entropy-based - weight inversely proportional to score entropy
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

# Re-use data loading from score_fusion.py
from score_fusion import (
    load_anomalyclip_scores,
    load_anomalydino_scores,
    align_samples,
    normalize_scores,
    compute_auroc,
)


# ============================================================================
# Unsupervised Confidence Signals
# ============================================================================

def signal_variance(scores):
    """Score variance as confidence signal.
    Higher variance -> more discriminative -> higher confidence."""
    return np.std(scores)


def signal_iqr(scores):
    """Interquartile range as confidence signal.
    Robust to outliers, measures central spread."""
    q75, q25 = np.percentile(scores, [75, 25])
    return q75 - q25


def signal_skewness(scores):
    """Absolute skewness as confidence signal.
    Highly skewed distribution suggests clear anomaly candidates
    (a few extreme scores pulling the distribution)."""
    return abs(scipy_stats.skew(scores))


def signal_topk_gap(scores, k_ratio=0.2):
    """Gap between top-k% and remaining scores.
    Large gap -> model confidently flags some images as anomalous."""
    sorted_scores = np.sort(scores)[::-1]
    k = max(1, int(len(scores) * k_ratio))
    top_mean = sorted_scores[:k].mean()
    rest_mean = sorted_scores[k:].mean()
    return top_mean - rest_mean


def signal_bimodality_coefficient(scores):
    """Bimodality coefficient: BC = (skew^2 + 1) / kurtosis.
    BC > 5/9 ~ 0.555 suggests bimodal distribution -> good separation."""
    n = len(scores)
    if n < 4:
        return 0.0
    skew = scipy_stats.skew(scores)
    kurt = scipy_stats.kurtosis(scores, fisher=False)  # excess=False -> Pearson kurtosis
    if kurt < 1e-10:
        return 0.0
    bc = (skew ** 2 + 1) / kurt
    return bc


def signal_gmm_separability(scores, n_components=2):
    """Fit 2-component GMM and compute Ashman's D.
    D = sqrt(2) * |mu1 - mu2| / sqrt(sigma1^2 + sigma2^2)
    D > 2 suggests clear bimodal separation."""
    if len(scores) < 10:
        return 0.0
    X = scores.reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=200)
        gmm.fit(X)
        mu1, mu2 = gmm.means_.flatten()
        s1, s2 = np.sqrt(gmm.covariances_.flatten())
        if s1 ** 2 + s2 ** 2 < 1e-10:
            return 0.0
        D = np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(s1 ** 2 + s2 ** 2)
        return D
    except Exception:
        return 0.0


def signal_entropy(scores, n_bins=20):
    """Score entropy (discretized). Lower entropy -> more peaked -> more confident.
    We return NEGATIVE entropy so higher = more confident (consistent with other signals)."""
    hist, _ = np.histogram(scores, bins=n_bins, density=True)
    hist = hist[hist > 0]
    bin_width = (scores.max() - scores.min()) / n_bins if scores.max() > scores.min() else 1.0
    probs = hist * bin_width
    probs = probs / probs.sum()  # Ensure normalization
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    # Invert: more confident = lower entropy = higher signal
    return max(0, np.log(n_bins) - entropy)  # Relative negentropy


def signal_tail_weight(scores, threshold_pct=90):
    """Weight of distribution in the upper tail.
    More tail weight -> model detects more potential anomalies -> more informative."""
    threshold = np.percentile(scores, threshold_pct)
    tail_scores = scores[scores >= threshold]
    if len(tail_scores) == 0:
        return 0.0
    return (tail_scores.mean() - np.median(scores)) / (np.std(scores) + 1e-10)


# ============================================================================
# Alpha Estimation Methods
# ============================================================================

def estimate_alpha_from_signal(clip_signal, dino_signal, base_alpha=0.5, sensitivity=1.0):
    """
    Estimate fusion weight alpha from confidence signals.

    alpha = base_alpha + sensitivity * (dino_signal - clip_signal) / (dino_signal + clip_signal + eps)

    Clamped to [0.05, 0.95] to avoid degeneracy.
    """
    total = clip_signal + dino_signal + 1e-10
    ratio = dino_signal / total  # [0, 1], higher = DINOv2 more confident
    # Smooth interpolation centered at base_alpha
    alpha = base_alpha + sensitivity * (ratio - 0.5)
    return np.clip(alpha, 0.05, 0.95)


def estimate_alpha_softmax(clip_signal, dino_signal, temperature=1.0):
    """
    Softmax-based alpha estimation.
    alpha = exp(dino_signal/T) / (exp(clip_signal/T) + exp(dino_signal/T))
    """
    # Normalize signals to prevent overflow
    max_val = max(clip_signal, dino_signal)
    clip_exp = np.exp((clip_signal - max_val) / temperature)
    dino_exp = np.exp((dino_signal - max_val) / temperature)
    alpha = dino_exp / (clip_exp + dino_exp)
    return np.clip(alpha, 0.05, 0.95)


# ============================================================================
# Adaptive Fusion Methods
# ============================================================================

SIGNAL_FUNCTIONS = {
    'variance': signal_variance,
    'iqr': signal_iqr,
    'skewness': signal_skewness,
    'topk_gap': signal_topk_gap,
    'bimodality': signal_bimodality_coefficient,
    'gmm_sep': signal_gmm_separability,
    'entropy': signal_entropy,
    'tail_weight': signal_tail_weight,
}


def adaptive_fusion_single_signal(aligned_data, signal_name, base_alpha=0.5, sensitivity=1.0):
    """
    Perform adaptive fusion using a single confidence signal.

    Returns per-category results: {category: {'alpha': a, 'auroc': AUROC, 'signal_clip': ..., 'signal_dino': ...}}
    """
    signal_fn = SIGNAL_FUNCTIONS[signal_name]
    results = {}

    for obj in sorted(aligned_data.keys()):
        d = aligned_data[obj]
        labels = d['labels']
        if len(np.unique(labels)) < 2:
            continue

        clip_scores = d['clip_scores']
        dino_scores = d['dino_scores']

        # Normalize scores
        clip_norm = normalize_scores(clip_scores, 'minmax')
        dino_norm = normalize_scores(dino_scores, 'minmax')

        # Compute confidence signals on normalized scores
        sig_clip = signal_fn(clip_norm)
        sig_dino = signal_fn(dino_norm)

        # Estimate alpha
        alpha = estimate_alpha_from_signal(sig_clip, sig_dino, base_alpha, sensitivity)

        # Compute fused scores
        fused = (1 - alpha) * clip_norm + alpha * dino_norm
        auroc = compute_auroc(labels, fused)

        results[obj] = {
            'alpha': round(alpha, 3),
            'auroc': round(auroc, 2),
            'signal_clip': round(float(sig_clip), 4),
            'signal_dino': round(float(sig_dino), 4),
        }

    return results


def adaptive_fusion_combined(aligned_data, signal_weights=None, base_alpha=0.5):
    """
    Combined meta-estimator: use multiple signals with learned weights.

    If signal_weights is None, use equal weights for all signals.
    """
    if signal_weights is None:
        signal_weights = {name: 1.0 for name in SIGNAL_FUNCTIONS}

    results = {}
    for obj in sorted(aligned_data.keys()):
        d = aligned_data[obj]
        labels = d['labels']
        if len(np.unique(labels)) < 2:
            continue

        clip_scores = d['clip_scores']
        dino_scores = d['dino_scores']
        clip_norm = normalize_scores(clip_scores, 'minmax')
        dino_norm = normalize_scores(dino_scores, 'minmax')

        # Aggregate confidence signals
        total_clip = 0.0
        total_dino = 0.0
        total_weight = 0.0

        for sig_name, weight in signal_weights.items():
            if weight == 0:
                continue
            sig_fn = SIGNAL_FUNCTIONS[sig_name]
            sc = sig_fn(clip_norm)
            sd = sig_fn(dino_norm)
            total_clip += weight * sc
            total_dino += weight * sd
            total_weight += weight

        if total_weight > 0:
            total_clip /= total_weight
            total_dino /= total_weight

        alpha = estimate_alpha_from_signal(total_clip, total_dino, base_alpha, sensitivity=1.0)
        fused = (1 - alpha) * clip_norm + alpha * dino_norm
        auroc = compute_auroc(labels, fused)

        results[obj] = {
            'alpha': round(alpha, 3),
            'auroc': round(auroc, 2),
        }

    return results


def adaptive_fusion_rank_weighted(aligned_data, signal_name='gmm_sep'):
    """
    Rank-weighted fusion: use rank-normalized scores with adaptive weights.
    Combines the robustness of rank fusion with adaptive weighting.
    """
    signal_fn = SIGNAL_FUNCTIONS[signal_name]
    results = {}

    for obj in sorted(aligned_data.keys()):
        d = aligned_data[obj]
        labels = d['labels']
        if len(np.unique(labels)) < 2:
            continue

        clip_rank = normalize_scores(d['clip_scores'], 'rank')
        dino_rank = normalize_scores(d['dino_scores'], 'rank')

        # Use min-max normalized scores for signal computation
        clip_norm = normalize_scores(d['clip_scores'], 'minmax')
        dino_norm = normalize_scores(d['dino_scores'], 'minmax')

        sig_clip = signal_fn(clip_norm)
        sig_dino = signal_fn(dino_norm)
        alpha = estimate_alpha_softmax(sig_clip, sig_dino, temperature=1.0)

        fused = (1 - alpha) * clip_rank + alpha * dino_rank
        auroc = compute_auroc(labels, fused)

        results[obj] = {
            'alpha': round(alpha, 3),
            'auroc': round(auroc, 2),
        }

    return results


# ============================================================================
# Hyperparameter Search for Sensitivity/Base-Alpha
# ============================================================================

def grid_search_hyperparams(aligned_data, signal_name, dataset_name=""):
    """
    Grid search over base_alpha and sensitivity to find best hyperparameters.
    This uses oracle knowledge (labels) -- useful for understanding the signal.
    """
    base_alphas = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    sensitivities = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    best_mean = 0
    best_params = (0.5, 1.0)
    all_results = []

    for ba in base_alphas:
        for sens in sensitivities:
            results = adaptive_fusion_single_signal(aligned_data, signal_name, ba, sens)
            aurocs = [r['auroc'] for r in results.values()]
            mean_auroc = np.mean(aurocs) if aurocs else 0
            all_results.append((ba, sens, mean_auroc))
            if mean_auroc > best_mean:
                best_mean = mean_auroc
                best_params = (ba, sens)

    return best_params, best_mean, all_results


# ============================================================================
# Evaluation & Comparison
# ============================================================================

def evaluate_all_methods(aligned_data, dataset_name=""):
    """
    Evaluate all adaptive fusion methods and compare with baselines.
    """
    print(f"\n{'='*90}")
    print(f"  ADAPTIVE FUSION EVALUATION -- {dataset_name}")
    print(f"{'='*90}")

    # ----- Baselines -----
    baseline_results = {}

    # CLIP only
    clip_aurocs = {}
    for obj, d in aligned_data.items():
        if len(np.unique(d['labels'])) < 2:
            continue
        clip_aurocs[obj] = compute_auroc(d['labels'], d['clip_scores'])
    baseline_results['CLIP only'] = clip_aurocs

    # DINOv2 only
    dino_aurocs = {}
    for obj, d in aligned_data.items():
        if len(np.unique(d['labels'])) < 2:
            continue
        dino_aurocs[obj] = compute_auroc(d['labels'], d['dino_scores'])
    baseline_results['DINOv2 only'] = dino_aurocs

    # Fixed alpha = 0.5
    equal_aurocs = {}
    for obj, d in aligned_data.items():
        if len(np.unique(d['labels'])) < 2:
            continue
        clip_norm = normalize_scores(d['clip_scores'], 'minmax')
        dino_norm = normalize_scores(d['dino_scores'], 'minmax')
        fused = 0.5 * clip_norm + 0.5 * dino_norm
        equal_aurocs[obj] = compute_auroc(d['labels'], fused)
    baseline_results['Equal (a=0.5)'] = equal_aurocs

    # Fixed alpha = 0.65 (best global from previous experiments)
    fixed65_aurocs = {}
    for obj, d in aligned_data.items():
        if len(np.unique(d['labels'])) < 2:
            continue
        clip_norm = normalize_scores(d['clip_scores'], 'minmax')
        dino_norm = normalize_scores(d['dino_scores'], 'minmax')
        fused = 0.35 * clip_norm + 0.65 * dino_norm
        fixed65_aurocs[obj] = compute_auroc(d['labels'], fused)
    baseline_results['Fixed (a=0.65)'] = fixed65_aurocs

    # Rank fusion
    rank_aurocs = {}
    for obj, d in aligned_data.items():
        if len(np.unique(d['labels'])) < 2:
            continue
        clip_rank = normalize_scores(d['clip_scores'], 'rank')
        dino_rank = normalize_scores(d['dino_scores'], 'rank')
        fused = 0.5 * clip_rank + 0.5 * dino_rank
        rank_aurocs[obj] = compute_auroc(d['labels'], fused)
    baseline_results['Rank fusion'] = rank_aurocs

    # Oracle per-category
    oracle_aurocs = {}
    oracle_alphas = {}
    for obj, d in aligned_data.items():
        if len(np.unique(d['labels'])) < 2:
            continue
        best_auroc = 0
        best_a = 0.5
        clip_norm = normalize_scores(d['clip_scores'], 'minmax')
        dino_norm = normalize_scores(d['dino_scores'], 'minmax')
        for a in [i / 100.0 for i in range(0, 101, 5)]:
            fused = (1 - a) * clip_norm + a * dino_norm
            auroc = compute_auroc(d['labels'], fused)
            if auroc > best_auroc:
                best_auroc = auroc
                best_a = a
        oracle_aurocs[obj] = best_auroc
        oracle_alphas[obj] = best_a
    baseline_results['Oracle'] = oracle_aurocs

    # ----- Adaptive Methods -----
    adaptive_results = {}
    adaptive_alphas = {}

    for signal_name in SIGNAL_FUNCTIONS:
        # Grid search for best hyperparameters
        (best_ba, best_sens), best_mean, _ = grid_search_hyperparams(
            aligned_data, signal_name, dataset_name
        )
        results = adaptive_fusion_single_signal(
            aligned_data, signal_name, best_ba, best_sens
        )
        aurocs = {obj: r['auroc'] for obj, r in results.items()}
        alphas = {obj: r['alpha'] for obj, r in results.items()}
        adaptive_results[f"Adapt-{signal_name}"] = aurocs
        adaptive_alphas[f"Adapt-{signal_name}"] = alphas
        adaptive_results[f"Adapt-{signal_name}|params"] = {'base_alpha': best_ba, 'sensitivity': best_sens}

    # Combined method
    combined = adaptive_fusion_combined(aligned_data, base_alpha=0.6)
    adaptive_results['Adapt-combined'] = {obj: r['auroc'] for obj, r in combined.items()}
    adaptive_alphas['Adapt-combined'] = {obj: r['alpha'] for obj, r in combined.items()}

    # Rank-weighted
    for sig in ['gmm_sep', 'variance', 'topk_gap']:
        rw = adaptive_fusion_rank_weighted(aligned_data, signal_name=sig)
        adaptive_results[f'RankAdapt-{sig}'] = {obj: r['auroc'] for obj, r in rw.items()}
        adaptive_alphas[f'RankAdapt-{sig}'] = {obj: r['alpha'] for obj, r in rw.items()}

    # ----- Print Results Table -----
    all_methods = {}
    all_methods.update(baseline_results)
    all_methods.update({k: v for k, v in adaptive_results.items() if '|' not in k})

    categories = sorted(set().union(*(m.keys() for m in all_methods.values())))

    # Summary table
    print(f"\n{'Method':<25} {'Mean AUROC':>12} {'vs Equal':>10} {'vs Oracle':>10}")
    print("-" * 60)

    equal_mean = np.mean(list(equal_aurocs.values()))
    oracle_mean = np.mean(list(oracle_aurocs.values()))
    method_means = {}

    for method_name, aurocs in sorted(all_methods.items(), key=lambda x: -np.mean(list(x[1].values()))):
        vals = list(aurocs.values())
        if not vals:
            continue
        mean_val = np.mean(vals)
        method_means[method_name] = mean_val
        diff_equal = mean_val - equal_mean
        diff_oracle = mean_val - oracle_mean
        prefix = "*" if method_name.startswith("Adapt") or method_name.startswith("Rank") else " "
        print(f"{prefix}{method_name:<24} {mean_val:>11.2f}% {diff_equal:>+9.2f}% {diff_oracle:>+9.2f}%")

    # ----- Per-Category Alpha Comparison -----
    print(f"\n--- Per-Category Alpha Comparison (Top Methods) ---")

    # Find best adaptive method
    adaptive_only = {k: v for k, v in method_means.items()
                     if k.startswith("Adapt") or k.startswith("Rank")}
    if adaptive_only:
        best_adapt_name = max(adaptive_only, key=adaptive_only.get)
        best_adapt_alphas = adaptive_alphas.get(best_adapt_name, {})

        print(f"\nBest adaptive method: {best_adapt_name} (Mean: {adaptive_only[best_adapt_name]:.2f}%)")
        print(f"\n{'Category':<15} {'Oracle a':>10} {'Adapt a':>10} {'|Diff|':>8} {'Oracle':>10} {'Adapt':>10} {'Equal':>10}")
        print("-" * 75)

        alpha_errors = []
        for obj in categories:
            oa = oracle_alphas.get(obj, 0.5)
            aa = best_adapt_alphas.get(obj, 0.5)
            diff = abs(oa - aa)
            alpha_errors.append(diff)
            o_auroc = oracle_aurocs.get(obj, 0)
            a_auroc = all_methods.get(best_adapt_name, {}).get(obj, 0)
            e_auroc = equal_aurocs.get(obj, 0)
            print(f"{obj:<15} {oa:>10.2f} {aa:>10.3f} {diff:>8.3f} {o_auroc:>9.1f}% {a_auroc:>9.1f}% {e_auroc:>9.1f}%")

        print("-" * 75)
        print(f"{'Mean a error':<15} {'':>10} {'':>10} {np.mean(alpha_errors):>8.3f}")
        print(f"{'Correlation':<15} {'':>10} {'':>10} ", end="")
        oracle_a_list = [oracle_alphas.get(obj, 0.5) for obj in categories]
        adapt_a_list = [best_adapt_alphas.get(obj, 0.5) for obj in categories]
        if len(oracle_a_list) > 2:
            corr, pval = scipy_stats.pearsonr(oracle_a_list, adapt_a_list)
            print(f"r={corr:.3f} (p={pval:.3f})")
        else:
            print("N/A")

    # ----- Signal Correlation Analysis -----
    print(f"\n--- Signal Correlation with Oracle Alpha ---")
    print(f"{'Signal':<20} {'r(alpha)':>10} {'p-value':>10} {'Direction':>12}")
    print("-" * 55)

    for signal_name, signal_fn in SIGNAL_FUNCTIONS.items():
        clip_signals = []
        dino_signals = []
        oracle_a_list = []

        for obj in categories:
            if obj not in aligned_data:
                continue
            d = aligned_data[obj]
            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')

            sc = signal_fn(clip_norm)
            sd = signal_fn(dino_norm)

            ratio = sd / (sc + sd + 1e-10)  # DINOv2 relative confidence
            clip_signals.append(sc)
            dino_signals.append(sd)
            oracle_a_list.append(oracle_alphas.get(obj, 0.5))

        if len(oracle_a_list) > 2:
            # Correlation between DINOv2/(CLIP+DINOv2) signal ratio and oracle alpha
            signal_ratios = [dino_signals[i] / (clip_signals[i] + dino_signals[i] + 1e-10)
                           for i in range(len(clip_signals))]
            corr, pval = scipy_stats.pearsonr(signal_ratios, oracle_a_list)
            direction = "OK correct" if corr > 0.1 else ("NG wrong" if corr < -0.1 else "~ neutral")
            print(f"{signal_name:<20} {corr:>10.3f} {pval:>10.3f} {direction:>12}")

    return baseline_results, adaptive_results, adaptive_alphas, oracle_alphas


# ============================================================================
# Leave-One-Out Cross-Validation for Fair Evaluation
# ============================================================================

def loocv_evaluation(aligned_data, signal_name, dataset_name=""):
    """
    Leave-one-category-out cross-validation.
    For each category:
    1. Use remaining categories to learn optimal base_alpha/sensitivity
    2. Apply learned parameters to estimate alpha for held-out category
    3. Evaluate on held-out category

    This simulates a truly unsupervised setting where we learn hyperparameters
    from auxiliary categories.
    """
    categories = sorted([obj for obj in aligned_data.keys()
                        if len(np.unique(aligned_data[obj]['labels'])) >= 2])

    loocv_results = {}

    for held_out in categories:
        # Training: all other categories
        train_cats = [c for c in categories if c != held_out]

        # Grid search on training categories
        train_aligned = {c: aligned_data[c] for c in train_cats}
        (best_ba, best_sens), _, _ = grid_search_hyperparams(
            train_aligned, signal_name
        )

        # Apply to held-out category
        result = adaptive_fusion_single_signal(
            {held_out: aligned_data[held_out]}, signal_name, best_ba, best_sens
        )

        if held_out in result:
            loocv_results[held_out] = result[held_out]

    return loocv_results


# ============================================================================
# Main
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

    all_output = {}

    # =====================================================
    # MVTec AD
    # =====================================================
    if os.path.exists(clip_mvtec):
        clip_data = load_anomalyclip_scores(clip_mvtec)
        dino_data = {}
        for p in [dino_mvtec_weak, dino_mvtec_strong]:
            if os.path.exists(p):
                dino_data.update(load_anomalydino_scores(p))

        if dino_data:
            print("\n--- Aligning MVTec AD samples ---")
            aligned_mvtec = align_samples(clip_data, dino_data)
            baselines_m, adaptive_m, alphas_m, oracle_a_m = evaluate_all_methods(
                aligned_mvtec, "MVTec AD"
            )

            # LOOCV for top signals
            print(f"\n{'='*90}")
            print(f"  LEAVE-ONE-OUT CROSS-VALIDATION -- MVTec AD")
            print(f"{'='*90}")

            loocv_all = {}
            for sig_name in ['variance', 'topk_gap', 'gmm_sep', 'bimodality', 'entropy']:
                loocv = loocv_evaluation(aligned_mvtec, sig_name, "MVTec AD")
                aurocs = [r['auroc'] for r in loocv.values()]
                mean_auroc = np.mean(aurocs) if aurocs else 0
                loocv_all[sig_name] = {'mean': mean_auroc, 'results': loocv}
                print(f"  LOOCV {sig_name:<15}: {mean_auroc:.2f}%")

            all_output['mvtec'] = {
                'adaptive_methods': {k: v for k, v in adaptive_m.items() if '|' not in k},
                'oracle_alphas': oracle_a_m,
                'loocv': {k: v['mean'] for k, v in loocv_all.items()},
            }

    # =====================================================
    # VisA
    # =====================================================
    if os.path.exists(clip_visa) and os.path.exists(dino_visa):
        clip_data = load_anomalyclip_scores(clip_visa)
        dino_data = load_anomalydino_scores(dino_visa)

        print("\n--- Aligning VisA samples ---")
        aligned_visa = align_samples(clip_data, dino_data)
        baselines_v, adaptive_v, alphas_v, oracle_a_v = evaluate_all_methods(
            aligned_visa, "VisA"
        )

        # LOOCV
        print(f"\n{'='*90}")
        print(f"  LEAVE-ONE-OUT CROSS-VALIDATION -- VisA")
        print(f"{'='*90}")

        loocv_all_v = {}
        for sig_name in ['variance', 'topk_gap', 'gmm_sep', 'bimodality', 'entropy']:
            loocv = loocv_evaluation(aligned_visa, sig_name, "VisA")
            aurocs = [r['auroc'] for r in loocv.values()]
            mean_auroc = np.mean(aurocs) if aurocs else 0
            loocv_all_v[sig_name] = {'mean': mean_auroc, 'results': loocv}
            print(f"  LOOCV {sig_name:<15}: {mean_auroc:.2f}%")

        all_output['visa'] = {
            'adaptive_methods': {k: v for k, v in adaptive_v.items() if '|' not in k},
            'oracle_alphas': oracle_a_v,
            'loocv': {k: v['mean'] for k, v in loocv_all_v.items()},
        }

    # Save results
    output_path = os.path.join(base_dir, "adaptive_fusion_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
