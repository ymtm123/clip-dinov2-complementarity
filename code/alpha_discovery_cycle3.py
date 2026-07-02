"""
Alpha Discovery Cycle 3: Direct Optimization of Unsupervised Fusion Quality

Key insight from Cycle 1-2 failures:
  Estimating alpha from individual model signals FAILS because:
  - Score distributions look similar after normalization
  - Cross-model signals can't distinguish quality differences

NEW APPROACH: Don't estimate alpha. OPTIMIZE it.
  For each alpha, compute the fused scores and measure their
  "anomaly-detection quality" using unsupervised metrics.
  Choose the alpha that maximizes this quality.

Unsupervised quality metrics for fused scores:
  1. Bimodality coefficient: higher = clearer normal/anomaly separation
  2. GMM separation (Ashman's D): higher = better 2-cluster separation
  3. Silhouette score (2-cluster): higher = tighter/more separated clusters
  4. Score dynamic range (IQR): may or may not correlate with quality
  5. Negative entropy: lower entropy = more peaked = more confident
  6. Calinski-Harabasz index: cluster separation quality
  7. Top-k gap: separation of top anomaly candidates
  8. Fisher criterion (2-cluster): between-class / within-class variance
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
            obj, sample, score = row[0], row[1], float(row[2])
            is_anomaly = 0 if sample.startswith("ok/") else 1
            data[obj]['scores'].append(score)
            data[obj]['labels'].append(is_anomaly)
            data[obj]['paths'].append(sample)
    return dict(data)


# ============================================================================
# Unsupervised quality metrics for fused scores
# ============================================================================

def metric_bimodality(scores):
    """Bimodality coefficient: BC = (skew^2 + 1) / kurtosis."""
    from scipy.stats import skew, kurtosis
    s = skew(scores)
    k = kurtosis(scores, fisher=False)
    if k < 1e-10:
        return 0.0
    return (s ** 2 + 1) / k


def metric_gmm_ashman_d(scores):
    """Ashman's D from 2-component GMM."""
    from sklearn.mixture import GaussianMixture
    X = scores.reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
        gmm.fit(X)
        mu1, mu2 = gmm.means_.flatten()
        s1, s2 = np.sqrt(gmm.covariances_.flatten())
        denom = np.sqrt(s1 ** 2 + s2 ** 2)
        if denom < 1e-10:
            return 0.0
        return np.sqrt(2) * abs(mu1 - mu2) / denom
    except Exception:
        return 0.0


def metric_silhouette(scores):
    """Silhouette score with 2-cluster K-means."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = scores.reshape(-1, 1)
    if len(X) < 4:
        return 0.0
    try:
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            return 0.0
        return silhouette_score(X, labels)
    except Exception:
        return 0.0


def metric_calinski_harabasz(scores):
    """Calinski-Harabasz index for 2-cluster."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score
    X = scores.reshape(-1, 1)
    if len(X) < 4:
        return 0.0
    try:
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            return 0.0
        return calinski_harabasz_score(X, labels)
    except Exception:
        return 0.0


def metric_topk_gap(scores, k_ratio=0.2):
    """Gap between top-k scores and the rest."""
    sorted_s = np.sort(scores)[::-1]
    k = max(1, int(len(scores) * k_ratio))
    return sorted_s[:k].mean() - sorted_s[k:].mean()


def metric_negentropy(scores):
    """Negative entropy (higher = more peaked = more confident)."""
    n_bins = 20
    hist, _ = np.histogram(scores, bins=n_bins, density=True)
    bin_width = (scores.max() - scores.min() + 1e-10) / n_bins
    p = hist * bin_width
    p = p / (p.sum() + 1e-10)
    entropy = -np.sum(p * np.log(p + 1e-10))
    return np.log(n_bins) - entropy


def metric_fisher_criterion(scores, n_components=2):
    """Fisher criterion: between-class variance / within-class variance."""
    from sklearn.mixture import GaussianMixture
    X = scores.reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)

        # Between-class variance
        overall_mean = scores.mean()
        between = 0
        within = 0
        for c in range(n_components):
            mask = labels == c
            if mask.sum() == 0:
                continue
            class_mean = scores[mask].mean()
            class_var = scores[mask].var()
            n_c = mask.sum()
            between += n_c * (class_mean - overall_mean) ** 2
            within += n_c * class_var

        if within < 1e-10:
            return 0.0
        return between / within
    except Exception:
        return 0.0


def metric_dip_statistic(scores):
    """Approximate Hartigan's dip test for unimodality.
    Higher dip = more multimodal = better anomaly separation."""
    sorted_s = np.sort(scores)
    n = len(sorted_s)
    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n
    # Fit uniform CDF between min and max
    uniform_cdf = (sorted_s - sorted_s[0]) / (sorted_s[-1] - sorted_s[0] + 1e-10)
    # Dip = max difference between ECDF and best-fitting unimodal CDF
    # Simplified: just use uniform as reference
    dip = np.max(np.abs(ecdf - uniform_cdf))
    return dip


def metric_score_separation(scores, split_pct=0.7):
    """
    Pseudo Cohen's d: treat bottom split_pct as normal, top as anomaly.
    This is the same as NormalDeviation but applied to fused scores.
    """
    n = len(scores)
    k = int(n * split_pct)
    order = np.argsort(scores)
    normal = scores[order[:k]]
    anomaly = scores[order[k:]]
    if len(normal) < 2 or len(anomaly) < 2:
        return 0
    pooled = np.sqrt((np.var(normal) + np.var(anomaly)) / 2)
    if pooled < 1e-10:
        return 0
    return abs(np.mean(anomaly) - np.mean(normal)) / pooled


# ============================================================================
# Optimization: find alpha that maximizes each metric
# ============================================================================

def optimize_alpha(clip_scores, dino_scores, metric_fn, alpha_steps):
    """
    For each alpha, compute fused scores and evaluate metric.
    Return alpha that maximizes the metric.
    """
    clip_norm = normalize_scores(clip_scores, 'minmax')
    dino_norm = normalize_scores(dino_scores, 'minmax')

    best_alpha = 0.5
    best_metric = -float('inf')
    metric_curve = []

    for alpha in alpha_steps:
        fused = (1 - alpha) * clip_norm + alpha * dino_norm
        m = metric_fn(fused)
        metric_curve.append(m)
        if m > best_metric:
            best_metric = m
            best_alpha = alpha

    return best_alpha, best_metric, metric_curve


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(datasets):
    alpha_steps = [i / 100.0 for i in range(0, 101, 5)]  # 0.05 step for speed

    all_categories = []
    for ds_name, aligned in datasets.items():
        for obj in sorted(aligned.keys()):
            d = aligned[obj]
            if len(np.unique(d['labels'])) < 2:
                continue
            all_categories.append((ds_name, obj, d))

    metrics = {
        'Bimodality': metric_bimodality,
        'GMM-AshmanD': metric_gmm_ashman_d,
        'Silhouette': metric_silhouette,
        'CalinskiHarabasz': metric_calinski_harabasz,
        'TopkGap-20%': lambda s: metric_topk_gap(s, 0.2),
        'TopkGap-10%': lambda s: metric_topk_gap(s, 0.1),
        'Negentropy': metric_negentropy,
        'Fisher': metric_fisher_criterion,
        'DipStat': metric_dip_statistic,
        'ScoreSep-70': lambda s: metric_score_separation(s, 0.7),
        'ScoreSep-80': lambda s: metric_score_separation(s, 0.8),
        'ScoreSep-90': lambda s: metric_score_separation(s, 0.9),
    }

    # Also test combined metrics
    def combined_metric(scores, weights=None):
        """Weighted combination of normalized metrics."""
        vals = {
            'gmm': metric_gmm_ashman_d(scores),
            'fisher': metric_fisher_criterion(scores),
            'sep80': metric_score_separation(scores, 0.8),
        }
        if weights is None:
            weights = {'gmm': 1, 'fisher': 1, 'sep80': 1}
        total = sum(weights[k] * vals[k] for k in weights)
        return total

    metrics['Combined-GFS'] = combined_metric

    # Add baselines
    print(f"\n{'=' * 110}")
    print(f"  CYCLE 3: Optimize alpha via {len(metrics)} unsupervised quality metrics")
    print(f"  Categories: {len(all_categories)}")
    print(f"{'=' * 110}")

    # Fine alpha steps for oracle
    fine_steps = [i / 100.0 for i in range(0, 101, 1)]

    method_results = {}

    for metric_name, metric_fn in metrics.items():
        aurocs = []
        alphas_pred = []
        alphas_oracle = []
        per_cat = []

        for ds_name, obj, d in all_categories:
            labels = d['labels']
            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')

            # Optimize alpha via unsupervised metric
            a_pred, _, _ = optimize_alpha(
                d['clip_scores'], d['dino_scores'], metric_fn, alpha_steps
            )

            # AUROC with predicted alpha
            fused = (1 - a_pred) * clip_norm + a_pred * dino_norm
            auroc = compute_auroc(labels, fused)

            # Oracle
            best_auroc, best_alpha = 0, 0.5
            for a in fine_steps:
                f = (1 - a) * clip_norm + a * dino_norm
                au = compute_auroc(labels, f)
                if au > best_auroc:
                    best_auroc = au
                    best_alpha = a

            aurocs.append(auroc)
            alphas_pred.append(a_pred)
            alphas_oracle.append(best_alpha)
            per_cat.append({
                'ds': ds_name, 'cat': obj,
                'a_pred': round(a_pred, 2), 'a_oracle': best_alpha,
                'auroc': round(auroc, 2), 'auroc_oracle': round(best_auroc, 2)
            })

        mean_auroc = np.mean(aurocs)
        pred_arr, orc_arr = np.array(alphas_pred), np.array(alphas_oracle)

        if len(set(alphas_pred)) > 1:
            mx, my = np.mean(pred_arr), np.mean(orc_arr)
            cov = np.sum((pred_arr - mx) * (orc_arr - my))
            sx = np.sqrt(np.sum((pred_arr - mx) ** 2))
            sy = np.sqrt(np.sum((orc_arr - my) ** 2))
            pearson_r = cov / (sx * sy) if sx > 1e-10 and sy > 1e-10 else 0
        else:
            pearson_r = 0

        mae = np.mean(np.abs(pred_arr - orc_arr))
        regret = np.mean(np.array([p['auroc_oracle'] for p in per_cat]) -
                         np.array([p['auroc'] for p in per_cat]))

        method_results[metric_name] = {
            'mean_auroc': round(mean_auroc, 3),
            'pearson_r': round(pearson_r, 3),
            'mae_alpha': round(mae, 3),
            'regret': round(regret, 3),
            'per_category': per_cat,
        }

    # Add fixed baselines
    for fixed_a in [0.50, 0.65]:
        aurocs, per_cat = [], []
        for ds_name, obj, d in all_categories:
            labels = d['labels']
            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')
            fused = (1 - fixed_a) * clip_norm + fixed_a * dino_norm
            auroc = compute_auroc(labels, fused)

            best_auroc, best_alpha = 0, 0.5
            for a in fine_steps:
                f = (1 - a) * clip_norm + a * dino_norm
                au = compute_auroc(labels, f)
                if au > best_auroc:
                    best_auroc = au
                    best_alpha = a

            aurocs.append(auroc)
            per_cat.append({
                'ds': ds_name, 'cat': obj,
                'a_pred': fixed_a, 'a_oracle': best_alpha,
                'auroc': round(auroc, 2), 'auroc_oracle': round(best_auroc, 2)
            })
        method_results[f'Fixed-{fixed_a:.2f}'] = {
            'mean_auroc': round(np.mean(aurocs), 3),
            'pearson_r': 0,
            'mae_alpha': round(np.mean([abs(fixed_a - p['a_oracle']) for p in per_cat]), 3),
            'regret': round(np.mean([p['auroc_oracle'] - p['auroc'] for p in per_cat]), 3),
            'per_category': per_cat,
        }

    # Sort and print
    fixed_auroc = method_results['Fixed-0.65']['mean_auroc']
    sorted_methods = sorted(method_results.items(), key=lambda x: -x[1]['mean_auroc'])

    print(f"\n  {'Method':<28} {'AUROC':>8} {'r':>8} {'MAE(a)':>8} {'Regret':>8} {'vs0.65':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, r in sorted_methods:
        diff = r['mean_auroc'] - fixed_auroc
        marker = " ***" if diff > 0.05 else (" **" if diff > 0.01 else (" *" if diff > 0 else ""))
        print(f"  {name:<28} {r['mean_auroc']:>8.3f} {r['pearson_r']:>8.3f} "
              f"{r['mae_alpha']:>8.3f} {r['regret']:>8.3f} {diff:>+8.3f}{marker}")

    # Top 3 per-category detail
    print(f"\n{'=' * 110}")
    print(f"  TOP METHODS: Per-Category Detail")
    print(f"{'=' * 110}")

    non_fixed = [(n, r) for n, r in sorted_methods if not n.startswith('Fixed')][:3]
    for method_name, r in non_fixed:
        print(f"\n  --- {method_name} (AUROC: {r['mean_auroc']:.3f}, r={r['pearson_r']:.3f}) ---")
        print(f"  {'Category':<14} {'a_pred':>7} {'a_orc':>6} {'|da|':>6} "
              f"{'AUROC':>7} {'Orc':>7} {'diff':>7}")
        for pc in sorted(r['per_category'], key=lambda x: x['cat']):
            da = abs(pc['a_pred'] - pc['a_oracle'])
            diff = pc['auroc'] - pc['auroc_oracle']
            print(f"  {pc['cat']:<14} {pc['a_pred']:>7.2f} {pc['a_oracle']:>6.2f} {da:>6.3f} "
                  f"{pc['auroc']:>7.1f} {pc['auroc_oracle']:>7.1f} {diff:>+7.2f}")

    # Analyze: which metrics' alpha curves correlate with AUROC curves?
    print(f"\n{'=' * 110}")
    print(f"  DIAGNOSTIC: Do unsupervised metric curves match AUROC curves?")
    print(f"{'=' * 110}")

    for metric_name in ['GMM-AshmanD', 'Fisher', 'ScoreSep-80']:
        metric_fn = metrics[metric_name]
        correlations = []
        for ds_name, obj, d in all_categories:
            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')
            labels = d['labels']

            auroc_curve = []
            metric_curve = []
            for alpha in alpha_steps:
                fused = (1 - alpha) * clip_norm + alpha * dino_norm
                auroc_curve.append(compute_auroc(labels, fused))
                metric_curve.append(metric_fn(fused))

            # Correlation between metric curve and AUROC curve
            ac = np.array(auroc_curve)
            mc = np.array(metric_curve)
            if np.std(ac) > 1e-10 and np.std(mc) > 1e-10:
                r = np.corrcoef(ac, mc)[0, 1]
            else:
                r = 0
            correlations.append(r)

        print(f"\n  {metric_name}:")
        print(f"    Mean curve correlation with AUROC: {np.mean(correlations):.3f}")
        print(f"    Std:  {np.std(correlations):.3f}")
        print(f"    Min:  {np.min(correlations):.3f}")
        print(f"    Max:  {np.max(correlations):.3f}")
        print(f"    >0.5: {sum(1 for r in correlations if r > 0.5)}/{len(correlations)}")
        print(f"    <0:   {sum(1 for r in correlations if r < 0)}/{len(correlations)}")

    # Save
    output = {name: {k: v for k, v in r.items() if k != 'per_category'}
              for name, r in sorted_methods}
    out_path = os.path.join(os.path.dirname(BASE_DIR), "results", "alpha_discovery_cycle3.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return method_results


def main():
    print("=" * 110)
    print("  ALPHA DISCOVERY CYCLE 3: Direct Optimization of Unsupervised Fusion Quality")
    print("=" * 110)

    datasets = {}
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
        print(f"[WARN] BTAD scores not found -- run would cover only 27 categories:\n  clip={clip_btad}\n  dino={dino_btad}")

    evaluate(datasets)


if __name__ == "__main__":
    main()
