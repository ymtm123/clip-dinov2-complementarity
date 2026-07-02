"""
Alpha Discovery Cycle 4: Conditional Score Analysis

3-cycle failure summary:
  Independent model signals, cross-model signals, and direct metric
  optimization ALL fail to beat fixed alpha=0.65.

NEW INSIGHT: We need to look at CONDITIONAL behavior:
  "Among the images that LOOK anomalous, which model contributes more?"

Approaches:
  1. Score gradient: When combined score is high, does DINO or CLIP
     contribute more to that height? Measured via regression slope.

  2. Top-k contribution: Among the top-k highest fused scores,
     what fraction of "anomalousness" comes from each model?

  3. Conditional variance: Variance of each model's scores AMONG
     high-fused-score images vs low-fused-score images.

  4. Sample-level adaptive: Per-image alpha based on local confidence.

  5. PCA direction: Principal component of (clip, dino) score space
     → weight = loading of the direction of maximum variance.

  6. Elbow clarity: How sharp is the transition from low to high scores?
     Sharper elbow = better discriminator.

  7. Score-ratio at extremes: Among top-10% fused scores,
     alpha = mean(dino / (clip + dino)).
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
# New methods
# ============================================================================

def method_score_gradient(d):
    """
    Regression of score difference on combined score.
    diff_i = dino_i - clip_i
    combined_i = (clip_i + dino_i) / 2
    Fit: diff = a + b * combined

    b > 0: DINO contributes more at high scores → alpha should be high
    b < 0: CLIP contributes more at high scores → alpha should be low

    Map slope to alpha via sigmoid.
    """
    clip_norm = normalize_scores(d['clip_scores'], 'minmax')
    dino_norm = normalize_scores(d['dino_scores'], 'minmax')

    diff = dino_norm - clip_norm
    combined = (clip_norm + dino_norm) / 2

    # Linear regression: diff = a + b * combined
    n = len(diff)
    mx, my = np.mean(combined), np.mean(diff)
    cov = np.sum((combined - mx) * (diff - my))
    var_x = np.sum((combined - mx) ** 2)
    if var_x < 1e-10:
        return 0.5
    b = cov / var_x

    # Map slope to alpha via sigmoid: alpha = sigmoid(b * scale)
    # b > 0 → alpha > 0.5 (favor DINO), b < 0 → alpha < 0.5
    scale = 3.0  # controls sensitivity
    alpha = 1 / (1 + np.exp(-b * scale))

    return np.clip(alpha, 0.05, 0.95)


def method_topk_contribution(d, k_pct=0.2):
    """
    Among the top-k% highest EQUAL-weight fused scores,
    what fraction of the score comes from each model?

    alpha = mean(dino_norm[topk] / (clip_norm[topk] + dino_norm[topk]))
    """
    clip_norm = normalize_scores(d['clip_scores'], 'minmax')
    dino_norm = normalize_scores(d['dino_scores'], 'minmax')

    fused_equal = 0.5 * clip_norm + 0.5 * dino_norm
    n = len(fused_equal)
    k = max(1, int(n * k_pct))

    top_idx = np.argsort(fused_equal)[-k:]

    clip_top = clip_norm[top_idx]
    dino_top = dino_norm[top_idx]

    total = clip_top + dino_top + 1e-10
    ratios = dino_top / total
    alpha = np.mean(ratios)

    return np.clip(alpha, 0.05, 0.95)


def method_conditional_variance(d, split_pct=0.5):
    """
    Split images by combined score into "high" and "low" groups.
    Compare each model's variance in the "high" group.
    Higher variance in high group = more discriminative for anomalies.
    """
    clip_norm = normalize_scores(d['clip_scores'], 'minmax')
    dino_norm = normalize_scores(d['dino_scores'], 'minmax')

    combined = 0.5 * clip_norm + 0.5 * dino_norm
    n = len(combined)
    k = int(n * split_pct)

    high_idx = np.argsort(combined)[-k:]

    clip_var_high = np.std(clip_norm[high_idx])
    dino_var_high = np.std(dino_norm[high_idx])

    total = clip_var_high + dino_var_high + 1e-10
    alpha = dino_var_high / total

    return np.clip(alpha, 0.05, 0.95)


def method_pca_weights(d):
    """
    PCA on (clip_norm, dino_norm) score space.
    The first principal component direction gives the optimal
    linear combination for maximum variance.
    alpha = |loading_dino| / (|loading_clip| + |loading_dino|)
    """
    clip_norm = normalize_scores(d['clip_scores'], 'minmax')
    dino_norm = normalize_scores(d['dino_scores'], 'minmax')

    X = np.column_stack([clip_norm, dino_norm])
    X_centered = X - X.mean(axis=0)

    # Covariance matrix
    cov = X_centered.T @ X_centered / (len(X) - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # PC1 is the last (largest eigenvalue)
    pc1 = eigenvectors[:, -1]

    # Ensure positive direction (higher scores = more anomalous)
    if pc1.sum() < 0:
        pc1 = -pc1

    w_clip, w_dino = abs(pc1[0]), abs(pc1[1])
    alpha = w_dino / (w_clip + w_dino + 1e-10)

    return np.clip(alpha, 0.05, 0.95)


def method_elbow_clarity(d):
    """
    Sort scores, compute second derivative (curvature).
    Sharper elbow = clearer transition = better discriminator.

    Compare elbow sharpness between CLIP and DINO.
    """
    results = {}
    for name, scores in [('clip', d['clip_scores']), ('dino', d['dino_scores'])]:
        sorted_s = np.sort(scores)
        n = len(sorted_s)
        # Normalize to [0, 1] for comparable curvature
        s_norm = (sorted_s - sorted_s[0]) / (sorted_s[-1] - sorted_s[0] + 1e-10)
        x = np.linspace(0, 1, n)

        # Second derivative (curvature approximation)
        if n < 5:
            results[name] = 0
            continue
        # Use window-based max curvature
        window = max(3, n // 20)
        curvatures = []
        for i in range(window, n - window):
            d2 = s_norm[i + window] + s_norm[i - window] - 2 * s_norm[i]
            curvatures.append(abs(d2))
        results[name] = max(curvatures) if curvatures else 0

    total = results['clip'] + results['dino'] + 1e-10
    alpha = results['dino'] / total

    return np.clip(alpha, 0.05, 0.95)


def method_score_ratio_extremes(d, top_pct=0.1, bot_pct=0.3):
    """
    Among extreme images (both top and bottom of combined score),
    compute the alpha that "explains" the score pattern.

    For top images: alpha = mean(dino / (clip + dino))
    For bottom images: alpha is less relevant (both should be low)

    Use only the top images' ratio.
    """
    clip_norm = normalize_scores(d['clip_scores'], 'minmax')
    dino_norm = normalize_scores(d['dino_scores'], 'minmax')

    combined = 0.5 * clip_norm + 0.5 * dino_norm
    n = len(combined)
    k_top = max(1, int(n * top_pct))

    top_idx = np.argsort(combined)[-k_top:]

    clip_top = clip_norm[top_idx]
    dino_top = dino_norm[top_idx]

    total = clip_top + dino_top + 1e-10
    alpha = np.mean(dino_top / total)

    return np.clip(alpha, 0.05, 0.95)


def method_weighted_pca(d):
    """
    Weighted PCA: weight each sample by its combined score.
    Higher-scoring samples (likely anomalies) should influence
    the principal direction more.
    """
    clip_norm = normalize_scores(d['clip_scores'], 'minmax')
    dino_norm = normalize_scores(d['dino_scores'], 'minmax')

    combined = 0.5 * clip_norm + 0.5 * dino_norm
    # Weight: emphasize high-scoring samples
    weights = combined ** 2  # Quadratic emphasis on high scores
    weights = weights / weights.sum()

    X = np.column_stack([clip_norm, dino_norm])
    weighted_mean = (weights[:, np.newaxis] * X).sum(axis=0)
    X_centered = X - weighted_mean

    # Weighted covariance
    cov = (X_centered * weights[:, np.newaxis]).T @ X_centered

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    pc1 = eigenvectors[:, -1]
    if pc1.sum() < 0:
        pc1 = -pc1

    w_clip, w_dino = abs(pc1[0]), abs(pc1[1])
    alpha = w_dino / (w_clip + w_dino + 1e-10)

    return np.clip(alpha, 0.05, 0.95)


def method_rank_concentration(d):
    """
    Compare how "concentrated" each model's high scores are.
    If DINO assigns high scores to a tight cluster while CLIP spreads
    them out, DINO is more decisive → higher alpha.

    Measure: coefficient of variation of top-30% scores.
    LOWER CV = more concentrated = more confident.
    """
    clip_norm = normalize_scores(d['clip_scores'], 'minmax')
    dino_norm = normalize_scores(d['dino_scores'], 'minmax')

    n = len(clip_norm)
    k = max(1, int(n * 0.3))

    clip_top = np.sort(clip_norm)[-k:]
    dino_top = np.sort(dino_norm)[-k:]

    # Use mean of top-k as signal (higher = more confident detection)
    clip_signal = np.mean(clip_top)
    dino_signal = np.mean(dino_top)

    total = clip_signal + dino_signal + 1e-10
    alpha = dino_signal / total

    return np.clip(alpha, 0.05, 0.95)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(datasets):
    fine_steps = [i / 100.0 for i in range(0, 101, 1)]

    all_categories = []
    for ds_name, aligned in datasets.items():
        for obj in sorted(aligned.keys()):
            d = aligned[obj]
            if len(np.unique(d['labels'])) < 2:
                continue
            all_categories.append((ds_name, obj, d))

    methods = {
        'Fixed-0.65': lambda d: 0.65,
        'Fixed-0.50': lambda d: 0.50,
        'ScoreGradient': method_score_gradient,
        'TopkContrib-10%': lambda d: method_topk_contribution(d, 0.10),
        'TopkContrib-20%': lambda d: method_topk_contribution(d, 0.20),
        'TopkContrib-30%': lambda d: method_topk_contribution(d, 0.30),
        'CondVar-30%': lambda d: method_conditional_variance(d, 0.30),
        'CondVar-50%': lambda d: method_conditional_variance(d, 0.50),
        'PCA-weights': method_pca_weights,
        'WeightedPCA': method_weighted_pca,
        'ElbowClarity': method_elbow_clarity,
        'ScoreRatio-10%': lambda d: method_score_ratio_extremes(d, 0.10),
        'ScoreRatio-20%': lambda d: method_score_ratio_extremes(d, 0.20),
        'RankConcentration': method_rank_concentration,
    }

    print(f"\n{'=' * 110}")
    print(f"  CYCLE 4: {len(methods)} methods x {len(all_categories)} categories")
    print(f"{'=' * 110}")

    method_results = {}

    for method_name, method_fn in methods.items():
        aurocs, alphas_pred, alphas_oracle, per_cat = [], [], [], []

        for ds_name, obj, d in all_categories:
            labels = d['labels']
            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')

            try:
                a_pred = float(np.clip(method_fn(d), 0.0, 1.0))
            except Exception:
                a_pred = 0.65

            fused = (1 - a_pred) * clip_norm + a_pred * dino_norm
            auroc = compute_auroc(labels, fused)

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
                'a_pred': round(a_pred, 3), 'a_oracle': best_alpha,
                'auroc': round(auroc, 2), 'auroc_oracle': round(best_auroc, 2)
            })

        mean_auroc = np.mean(aurocs)
        pred_arr, orc_arr = np.array(alphas_pred), np.array(alphas_oracle)

        if len(set(alphas_pred)) > 1:
            mx, my = np.mean(pred_arr), np.mean(orc_arr)
            cov_val = np.sum((pred_arr - mx) * (orc_arr - my))
            sx = np.sqrt(np.sum((pred_arr - mx) ** 2))
            sy = np.sqrt(np.sum((orc_arr - my) ** 2))
            pearson_r = cov_val / (sx * sy) if sx > 1e-10 and sy > 1e-10 else 0
        else:
            pearson_r = 0

        mae = np.mean(np.abs(pred_arr - orc_arr))
        regret = np.mean(np.array([p['auroc_oracle'] for p in per_cat]) -
                         np.array([p['auroc'] for p in per_cat]))

        method_results[method_name] = {
            'mean_auroc': round(mean_auroc, 3),
            'pearson_r': round(pearson_r, 3),
            'mae_alpha': round(mae, 3),
            'regret': round(regret, 3),
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

    # Top 3 per-category
    print(f"\n{'=' * 110}")
    print(f"  TOP NON-FIXED METHODS")
    print(f"{'=' * 110}")

    non_fixed = [(n, r) for n, r in sorted_methods if not n.startswith('Fixed')]
    for method_name, r in non_fixed[:3]:
        print(f"\n  --- {method_name} (AUROC: {r['mean_auroc']:.3f}, r={r['pearson_r']:.3f}) ---")
        print(f"  {'Category':<14} {'a_pred':>7} {'a_orc':>6} {'|da|':>6} "
              f"{'AUROC':>7} {'Orc':>7} {'diff':>7}")
        for pc in sorted(r['per_category'], key=lambda x: x['cat']):
            da = abs(pc['a_pred'] - pc['a_oracle'])
            diff = pc['auroc'] - pc['auroc_oracle']
            print(f"  {pc['cat']:<14} {pc['a_pred']:>7.3f} {pc['a_oracle']:>6.2f} {da:>6.3f} "
                  f"{pc['auroc']:>7.1f} {pc['auroc_oracle']:>7.1f} {diff:>+7.2f}")

    # Save
    output = {name: {k: v for k, v in r.items() if k != 'per_category'}
              for name, r in sorted_methods}
    out_path = os.path.join(os.path.dirname(BASE_DIR), "results", "alpha_discovery_cycle4.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


def main():
    print("=" * 110)
    print("  ALPHA DISCOVERY CYCLE 4: Conditional Score Analysis")
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
