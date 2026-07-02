"""
Unsupervised Alpha Discovery: Systematic exploration of methods to estimate
per-category fusion weight WITHOUT anomaly labels.

Core insight: The inverse error weighting formula works (r=0.83 with oracle)
when given true AUROC. The problem reduces to: "estimate AUROC without labels."

Approaches explored:
  Group A - Pseudo-label methods (estimate AUROC indirectly)
    1. GMM pseudo-labels: Fit 2-GMM, assign pseudo-labels, compute pseudo-AUROC
    2. Otsu pseudo-labels: Otsu threshold on scores, compute pseudo-AUROC
    3. Quantile pseudo-labels: Top-q% treated as anomalous, compute pseudo-AUROC

  Group B - Distribution shape (does score distribution look discriminative?)
    4. Dip test: Hartigan's dip statistic for unimodality
    5. KL from uniform: More concentrated = more confident
    6. Wasserstein from reference normal

  Group C - Cross-model signals (use the OTHER model's scores as signal)
    7. Rank disagreement: Spearman correlation between models
    8. Score disagreement variance: Variance of (clip_rank - dino_rank)
    9. Conditional confidence: When one model says anomaly, does the other agree?

  Group D - Ensemble-theoretic
    10. Oracle surface curvature: How peaked is the alpha-AUROC curve?
        (Not directly usable, but diagnostic)
    11. Score calibration: Reliability diagram without labels (via GMM)
    12. Diversity-weighted: Weight inversely proportional to correlation
"""

import os
import sys
import numpy as np
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fusion import (
    load_anomalyclip_scores,
    load_anomalydino_scores,
    align_samples,
    normalize_scores,
    compute_auroc,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_btad_dino_scores(csv_path):
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
    import csv as csv_mod
    with open(csv_path, 'r') as f:
        reader = csv_mod.reader(f)
        next(reader)
        for row in reader:
            obj = row[0]
            sample = row[1]
            score = float(row[2])
            is_anomaly = 0 if sample.startswith("ok/") else 1
            data[obj]['scores'].append(score)
            data[obj]['labels'].append(is_anomaly)
            data[obj]['paths'].append(sample)
    return dict(data)


# ============================================================================
# Group A: Pseudo-label methods
# ============================================================================

def method_gmm_pseudo_auroc(scores):
    """Fit 2-component GMM, assign pseudo-labels, compute pseudo-AUROC."""
    from sklearn.mixture import GaussianMixture
    X = scores.reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
        gmm.fit(X)
        # The component with higher mean is "anomaly"
        means = gmm.means_.flatten()
        anomaly_comp = np.argmax(means)
        probs = gmm.predict_proba(X)[:, anomaly_comp]
        # pseudo-AUROC: how well do original scores separate GMM components?
        pseudo_labels = (probs > 0.5).astype(int)
        if len(np.unique(pseudo_labels)) < 2:
            return 0.5
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(pseudo_labels, scores)
    except Exception:
        return 0.5


def method_otsu_pseudo_auroc(scores):
    """Otsu's method: find threshold that maximizes between-class variance."""
    sorted_s = np.sort(scores)
    n = len(sorted_s)
    best_t = sorted_s[n // 2]
    best_var = 0
    for i in range(1, n):
        t = sorted_s[i]
        w0 = i / n
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = sorted_s[:i].mean()
        m1 = sorted_s[i:].mean()
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    pseudo_labels = (scores > best_t).astype(int)
    if len(np.unique(pseudo_labels)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(pseudo_labels, scores)


def method_quantile_pseudo_auroc(scores, q=0.8):
    """Top-(1-q)% are pseudo-anomalies."""
    threshold = np.quantile(scores, q)
    pseudo_labels = (scores > threshold).astype(int)
    if len(np.unique(pseudo_labels)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(pseudo_labels, scores)


def method_kmeans_pseudo_auroc(scores):
    """K-means (k=2) clustering on scores, compute pseudo-AUROC."""
    from sklearn.cluster import KMeans
    X = scores.reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    # Cluster with higher mean center = anomaly
    c0_mean = scores[labels == 0].mean()
    c1_mean = scores[labels == 1].mean()
    if c1_mean > c0_mean:
        pseudo_labels = labels
    else:
        pseudo_labels = 1 - labels
    if len(np.unique(pseudo_labels)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(pseudo_labels, scores)


# ============================================================================
# Group B: Distribution shape methods
# ============================================================================

def method_bimodality_coefficient(scores):
    """BC = (skew^2 + 1) / kurtosis. BC > 5/9 suggests bimodality."""
    from scipy.stats import skew, kurtosis
    s = skew(scores)
    k = kurtosis(scores, fisher=False)  # Pearson kurtosis
    if k < 1e-10:
        return 0.0
    return (s ** 2 + 1) / k


def method_kl_from_uniform(scores):
    """KL divergence from uniform distribution. More concentrated = more confident."""
    n_bins = 20
    hist, _ = np.histogram(scores, bins=n_bins, density=True)
    bin_width = (scores.max() - scores.min() + 1e-10) / n_bins
    p = hist * bin_width
    p = p / (p.sum() + 1e-10)
    q = np.ones(n_bins) / n_bins
    kl = np.sum(p * np.log((p + 1e-10) / q))
    return kl


def method_tail_ratio(scores, q_low=0.1, q_high=0.9):
    """Ratio of tail weight to center weight. Heavy tails = anomalies detected."""
    low = np.quantile(scores, q_low)
    high = np.quantile(scores, q_high)
    center = scores[(scores >= low) & (scores <= high)]
    tails = scores[(scores < low) | (scores > high)]
    if len(center) == 0:
        return 0
    return np.std(tails) / (np.std(center) + 1e-10)


def method_score_range_ratio(scores):
    """Range of top 10% scores / total range. Focused top = confident detection."""
    sorted_s = np.sort(scores)
    n = len(sorted_s)
    total_range = sorted_s[-1] - sorted_s[0]
    if total_range < 1e-10:
        return 0
    top10 = sorted_s[int(n * 0.9):]
    top_range = top10[-1] - top10[0]
    return 1 - (top_range / total_range)  # High = top scores are concentrated = confident


# ============================================================================
# Group C: Cross-model methods
# ============================================================================

def method_rank_disagreement(clip_scores, dino_scores):
    """Spearman rank correlation between models.
    Returns per-model signal: low correlation = models disagree = complementary."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(clip_scores, dino_scores)
    return r


def method_score_disagreement_pattern(clip_norm, dino_norm):
    """Analyze disagreement patterns.
    Where DINO >> CLIP: DINO is more confident on those samples.
    The VARIANCE of (dino-clip) tells us how much they disagree."""
    diff = dino_norm - clip_norm
    # Positive diff = DINO thinks more anomalous
    # If mean diff is high and variance is low: DINO consistently better
    return np.mean(diff), np.std(diff)


def method_conditional_top_agreement(clip_norm, dino_norm, top_pct=0.2):
    """When one model says 'top anomaly', does the other agree?
    High agreement on top candidates = both models work well = weight evenly.
    Low agreement = one model is noisy on these = weight the other."""
    n = len(clip_norm)
    k = max(1, int(n * top_pct))

    clip_top_idx = set(np.argsort(clip_norm)[-k:])
    dino_top_idx = set(np.argsort(dino_norm)[-k:])

    overlap = len(clip_top_idx & dino_top_idx) / k
    return overlap


# ============================================================================
# Group D: Novel combination methods
# ============================================================================

def method_pseudo_auroc_inverse_error(clip_scores, dino_scores, pseudo_method='gmm'):
    """
    The key idea: estimate AUROC without labels, then use inverse error weighting.
    This connects the unsupervised signal to the proven formula.
    """
    if pseudo_method == 'gmm':
        pseudo_auroc_clip = method_gmm_pseudo_auroc(clip_scores)
        pseudo_auroc_dino = method_gmm_pseudo_auroc(dino_scores)
    elif pseudo_method == 'otsu':
        pseudo_auroc_clip = method_otsu_pseudo_auroc(clip_scores)
        pseudo_auroc_dino = method_otsu_pseudo_auroc(dino_scores)
    elif pseudo_method == 'quantile':
        pseudo_auroc_clip = method_quantile_pseudo_auroc(clip_scores)
        pseudo_auroc_dino = method_quantile_pseudo_auroc(dino_scores)
    elif pseudo_method == 'kmeans':
        pseudo_auroc_clip = method_kmeans_pseudo_auroc(clip_scores)
        pseudo_auroc_dino = method_kmeans_pseudo_auroc(dino_scores)
    else:
        return 0.5, 0.5, 0.5

    err_c = max(1 - pseudo_auroc_clip, 1e-6)
    err_d = max(1 - pseudo_auroc_dino, 1e-6)
    alpha = (1 / err_d) / (1 / err_c + 1 / err_d)
    return alpha, pseudo_auroc_clip, pseudo_auroc_dino


def method_agreement_weighted_alpha(clip_norm, dino_norm, base_alpha=0.65):
    """
    Adjust alpha based on model agreement.
    When models strongly agree on anomaly ranking, use base_alpha.
    When they disagree, shift alpha toward the model with higher score spread.
    """
    from scipy.stats import spearmanr
    rho, _ = spearmanr(clip_norm, dino_norm)

    # High agreement (rho close to 1): keep base_alpha
    # Low agreement: shift toward more discriminative model
    clip_spread = np.std(clip_norm)
    dino_spread = np.std(dino_norm)
    spread_ratio = dino_spread / (clip_spread + dino_spread + 1e-10)

    # Interpolate: at rho=1, alpha=base_alpha; at rho=0, alpha=spread_ratio
    agreement = max(0, rho)
    alpha = agreement * base_alpha + (1 - agreement) * spread_ratio
    return np.clip(alpha, 0.05, 0.95)


def method_combined_pseudo_signal(clip_scores, dino_scores):
    """
    Combine multiple pseudo-AUROC estimates for robustness.
    Average of GMM, Otsu, and K-means pseudo-AUROC.
    """
    methods = [method_gmm_pseudo_auroc, method_otsu_pseudo_auroc,
               method_kmeans_pseudo_auroc]
    clip_estimates = [m(clip_scores) for m in methods]
    dino_estimates = [m(dino_scores) for m in methods]

    avg_clip = np.mean(clip_estimates)
    avg_dino = np.mean(dino_estimates)

    err_c = max(1 - avg_clip, 1e-6)
    err_d = max(1 - avg_dino, 1e-6)
    alpha = (1 / err_d) / (1 / err_c + 1 / err_d)
    return alpha, avg_clip, avg_dino


def method_median_score_ratio(clip_scores, dino_scores):
    """
    Simple but potentially effective: ratio of median scores.
    Higher median anomaly score = model detects more = more useful.
    Use ratio as alpha directly (after normalization).
    """
    clip_norm = normalize_scores(clip_scores, 'minmax')
    dino_norm = normalize_scores(dino_scores, 'minmax')

    # Use upper quartile instead of median (more sensitive to anomaly detection)
    clip_q75 = np.percentile(clip_norm, 75)
    dino_q75 = np.percentile(dino_norm, 75)

    alpha = dino_q75 / (clip_q75 + dino_q75 + 1e-10)
    return np.clip(alpha, 0.05, 0.95)


def method_normal_deviation(clip_scores, dino_scores, normal_pct=0.7):
    """
    Assume the majority of test samples are normal (common in AD).
    Use the bottom normal_pct as 'reference normal', rest as potential anomalies.
    Compute separation = (mean_potential - mean_normal) / pooled_std.
    This is essentially Cohen's d with pseudo-labels.
    """
    n = len(clip_scores)
    k = int(n * normal_pct)

    clip_norm = normalize_scores(clip_scores, 'minmax')
    dino_norm = normalize_scores(dino_scores, 'minmax')

    # Sort and split
    clip_sorted_idx = np.argsort(clip_norm)
    dino_sorted_idx = np.argsort(dino_norm)

    clip_normal = clip_norm[clip_sorted_idx[:k]]
    clip_anom = clip_norm[clip_sorted_idx[k:]]
    dino_normal = dino_norm[dino_sorted_idx[:k]]
    dino_anom = dino_norm[dino_sorted_idx[k:]]

    def pseudo_d(normal, anom):
        if len(normal) < 2 or len(anom) < 2:
            return 0
        pooled = np.sqrt((np.var(normal) + np.var(anom)) / 2)
        if pooled < 1e-10:
            return 0
        return abs(np.mean(anom) - np.mean(normal)) / pooled

    d_clip = pseudo_d(clip_normal, clip_anom)
    d_dino = pseudo_d(dino_normal, dino_anom)

    if d_clip + d_dino < 1e-10:
        return 0.5
    alpha = d_dino / (d_clip + d_dino)
    return np.clip(alpha, 0.05, 0.95)


# ============================================================================
# Evaluation framework
# ============================================================================

def evaluate_all_methods(datasets):
    """
    Evaluate all unsupervised alpha estimation methods.
    For each method, compute predicted alpha per category, then compute
    fusion AUROC and compare with fixed 0.65 and oracle.
    """
    results = {}
    alpha_steps = [i / 100.0 for i in range(0, 101, 1)]

    # Collect per-category data
    all_categories = []
    for ds_name, aligned in datasets.items():
        for obj in sorted(aligned.keys()):
            d = aligned[obj]
            if len(np.unique(d['labels'])) < 2:
                continue
            all_categories.append((ds_name, obj, d))

    # Method definitions
    methods = {}

    # Group A: Pseudo-label methods → inverse error weighting
    for pseudo in ['gmm', 'otsu', 'quantile', 'kmeans']:
        methods[f'PseudoAUROC-{pseudo}'] = lambda d, p=pseudo: method_pseudo_auroc_inverse_error(
            normalize_scores(d['clip_scores'], 'minmax'),
            normalize_scores(d['dino_scores'], 'minmax'),
            p
        )[0]

    # Combined pseudo
    methods['PseudoAUROC-ensemble'] = lambda d: method_combined_pseudo_signal(
        normalize_scores(d['clip_scores'], 'minmax'),
        normalize_scores(d['dino_scores'], 'minmax')
    )[0]

    # Group B: Distribution shape (use ratio as alpha)
    methods['Bimodality-ratio'] = lambda d: (
        method_bimodality_coefficient(normalize_scores(d['dino_scores'], 'minmax')) /
        (method_bimodality_coefficient(normalize_scores(d['clip_scores'], 'minmax')) +
         method_bimodality_coefficient(normalize_scores(d['dino_scores'], 'minmax')) + 1e-10)
    )
    methods['KL-ratio'] = lambda d: (
        method_kl_from_uniform(normalize_scores(d['dino_scores'], 'minmax')) /
        (method_kl_from_uniform(normalize_scores(d['clip_scores'], 'minmax')) +
         method_kl_from_uniform(normalize_scores(d['dino_scores'], 'minmax')) + 1e-10)
    )

    # Group C: Cross-model methods
    methods['Agreement-weighted'] = lambda d: method_agreement_weighted_alpha(
        normalize_scores(d['clip_scores'], 'minmax'),
        normalize_scores(d['dino_scores'], 'minmax')
    )
    methods['MedianScore-ratio'] = lambda d: method_median_score_ratio(
        d['clip_scores'], d['dino_scores']
    )

    # Group D: Novel
    methods['NormalDeviation-70'] = lambda d: method_normal_deviation(
        d['clip_scores'], d['dino_scores'], 0.7
    )
    methods['NormalDeviation-80'] = lambda d: method_normal_deviation(
        d['clip_scores'], d['dino_scores'], 0.8
    )
    methods['NormalDeviation-90'] = lambda d: method_normal_deviation(
        d['clip_scores'], d['dino_scores'], 0.9
    )

    # Baselines
    methods['Fixed-0.50'] = lambda d: 0.50
    methods['Fixed-0.65'] = lambda d: 0.65

    # Evaluate each method
    print(f"\n{'=' * 110}")
    print(f"  SYSTEMATIC EVALUATION: {len(methods)} methods x {len(all_categories)} categories")
    print(f"{'=' * 110}")

    method_results = {}

    for method_name, method_fn in methods.items():
        aurocs = []
        alphas_pred = []
        alphas_oracle = []
        per_cat_detail = []

        for ds_name, obj, d in all_categories:
            labels = d['labels']
            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')

            # Predicted alpha
            try:
                a_pred = method_fn(d)
                a_pred = float(np.clip(a_pred, 0.0, 1.0))
            except Exception as e:
                a_pred = 0.65
                print(f"    WARN {method_name} failed on {obj}: {type(e).__name__}: {e} -> alpha=0.65")

            # Fusion with predicted alpha
            fused = (1 - a_pred) * clip_norm + a_pred * dino_norm
            auroc = compute_auroc(labels, fused)

            # Oracle alpha
            best_auroc = 0
            best_alpha = 0.5
            for a in alpha_steps:
                f = (1 - a) * clip_norm + a * dino_norm
                au = compute_auroc(labels, f)
                if au > best_auroc:
                    best_auroc = au
                    best_alpha = a

            aurocs.append(auroc)
            alphas_pred.append(a_pred)
            alphas_oracle.append(best_alpha)
            per_cat_detail.append({
                'ds': ds_name, 'cat': obj,
                'a_pred': round(a_pred, 3), 'a_oracle': best_alpha,
                'auroc': round(auroc, 2), 'auroc_oracle': round(best_auroc, 2)
            })

        mean_auroc = np.mean(aurocs)
        # Correlation with oracle alpha
        if len(set(alphas_pred)) > 1 and len(set(alphas_oracle)) > 1:
            pred_arr = np.array(alphas_pred)
            orc_arr = np.array(alphas_oracle)
            mx, my = np.mean(pred_arr), np.mean(orc_arr)
            cov = np.sum((pred_arr - mx) * (orc_arr - my))
            sx = np.sqrt(np.sum((pred_arr - mx) ** 2))
            sy = np.sqrt(np.sum((orc_arr - my) ** 2))
            pearson_r = cov / (sx * sy) if sx > 1e-10 and sy > 1e-10 else 0
        else:
            pearson_r = 0

        mae_alpha = np.mean(np.abs(np.array(alphas_pred) - np.array(alphas_oracle)))
        regret = np.mean(np.array([pc['auroc_oracle'] for pc in per_cat_detail]) -
                         np.array([pc['auroc'] for pc in per_cat_detail]))

        method_results[method_name] = {
            'mean_auroc': round(mean_auroc, 3),
            'pearson_r': round(pearson_r, 3),
            'mae_alpha': round(mae_alpha, 3),
            'mean_regret': round(regret, 3),
            'per_category': per_cat_detail,
        }

    # Print summary table sorted by mean AUROC
    print(f"\n  {'Method':<28} {'Mean AUROC':>10} {'r(a,oracle)':>12} "
          f"{'MAE(a)':>8} {'Regret':>8} {'vs 0.65':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

    fixed_auroc = method_results['Fixed-0.65']['mean_auroc']
    sorted_methods = sorted(method_results.items(), key=lambda x: -x[1]['mean_auroc'])

    for name, r in sorted_methods:
        diff = r['mean_auroc'] - fixed_auroc
        marker = " ***" if diff > 0.05 else (" **" if diff > 0.01 else (" *" if diff > 0 else ""))
        print(f"  {name:<28} {r['mean_auroc']:>10.3f} {r['pearson_r']:>12.3f} "
              f"{r['mae_alpha']:>8.3f} {r['mean_regret']:>8.3f} {diff:>+8.3f}{marker}")

    # Detailed view of top 5 methods
    print(f"\n{'=' * 110}")
    print(f"  TOP 5 METHODS: Per-Category Detail")
    print(f"{'=' * 110}")

    top5 = sorted_methods[:5]
    for method_name, r in top5:
        if method_name.startswith('Fixed'):
            continue
        print(f"\n  --- {method_name} (Mean AUROC: {r['mean_auroc']:.3f}, r={r['pearson_r']:.3f}) ---")
        print(f"  {'Category':<14} {'a_pred':>7} {'a_oracle':>8} {'|da|':>6} "
              f"{'AUROC':>7} {'Oracle':>7} {'diff':>7}")
        for pc in sorted(r['per_category'], key=lambda x: x['cat']):
            da = abs(pc['a_pred'] - pc['a_oracle'])
            diff = pc['auroc'] - pc['auroc_oracle']
            print(f"  {pc['cat']:<14} {pc['a_pred']:>7.3f} {pc['a_oracle']:>8.2f} {da:>6.3f} "
                  f"{pc['auroc']:>7.1f} {pc['auroc_oracle']:>7.1f} {diff:>+7.2f}")

    # Save
    output = {
        'summary': {name: {k: v for k, v in r.items() if k != 'per_category'}
                     for name, r in sorted_methods},
        'best_method': sorted_methods[0][0],
        'best_non_fixed': next((n for n, _ in sorted_methods if not n.startswith('Fixed')), None),
    }
    out_path = os.path.join(os.path.dirname(BASE_DIR), "results", "unsupervised_alpha_discovery.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {out_path}")

    return method_results


def main():
    print("=" * 110)
    print("  UNSUPERVISED ALPHA DISCOVERY")
    print("  Goal: Find a method to estimate per-category alpha WITHOUT anomaly labels")
    print("=" * 110)

    datasets = {}

    # MVTec
    clip_mvtec = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv")
    dino_mvtec_weak = os.path.join(BASE_DIR, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                    "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv")
    dino_mvtec_strong = os.path.join(BASE_DIR, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                      "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")
    if os.path.exists(clip_mvtec):
        clip_data = load_anomalyclip_scores(clip_mvtec)
        dino_data = {}
        for dp in [dino_mvtec_weak, dino_mvtec_strong]:
            if os.path.exists(dp):
                dino_data.update(load_anomalydino_scores(dp))
        if dino_data:
            print("\n--- Aligning MVTec ---")
            datasets['MVTec'] = align_samples(clip_data, dino_data)

    # VisA
    clip_visa = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv")
    dino_visa = os.path.join(BASE_DIR, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv")
    if os.path.exists(clip_visa) and os.path.exists(dino_visa):
        clip_data = load_anomalyclip_scores(clip_visa)
        dino_data = load_anomalydino_scores(dino_visa)
        print("\n--- Aligning VisA ---")
        datasets['VisA'] = align_samples(clip_data, dino_data)

    # BTAD
    clip_btad = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv")
    dino_btad = os.path.join(BASE_DIR, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv")
    if os.path.exists(clip_btad) and os.path.exists(dino_btad):
        clip_data = load_anomalyclip_scores(clip_btad)
        dino_data = load_btad_dino_scores(dino_btad)
        print("\n--- Aligning BTAD ---")
        datasets['BTAD'] = align_samples(clip_data, dino_data)
    else:
        print(f"[WARN] BTAD scores not found -- dataset will be EXCLUDED "
              f"(run would cover only 27 categories):\n  clip={clip_btad}\n  dino={dino_btad}")

    evaluate_all_methods(datasets)


if __name__ == "__main__":
    main()
