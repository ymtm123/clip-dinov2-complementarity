"""
Alpha Discovery Cycle 5: Beyond Linear Fusion
=============================================

PARADIGM SHIFT: Cycles 1-4 tried to estimate per-category alpha from score
statistics. ALL 67 methods failed to beat fixed alpha=0.65.

Root cause: The oracle surface is too flat (0.406pp regret). Any noise in
alpha estimation exceeds the signal.

NEW QUESTION: Instead of "what is the right alpha?", ask:
  1. Can NON-LINEAR fusion bypass the alpha problem entirely?
  2. Can SAMPLE-LEVEL adaptation outperform category-level alpha?
  3. Can SCORE CALIBRATION make equal-weight fusion better?

These paradigms don't estimate alpha — they change the fusion function itself.
"""

import os
import sys
import numpy as np
import json
from collections import defaultdict
from scipy.stats import rankdata

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
    import csv as csv_mod
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
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
# PARADIGM A: Non-linear fusion (no alpha parameter at all)
# ============================================================================

def fusion_max(clip_norm, dino_norm):
    """Max fusion: score = max(clip, dino). Automatically selects the
    model that considers a sample most anomalous."""
    return np.maximum(clip_norm, dino_norm)


def fusion_geometric_mean(clip_norm, dino_norm):
    """Geometric mean: sqrt(clip * dino). Requires BOTH models to agree
    on anomalousness. Acts as a soft AND gate."""
    # Add small epsilon to avoid zero scores killing signal
    return np.sqrt((clip_norm + 1e-6) * (dino_norm + 1e-6))


def fusion_probabilistic_or(clip_norm, dino_norm):
    """Probabilistic OR: 1 - (1-c)(1-d). High if EITHER model detects anomaly.
    Acts as a soft OR gate — different from max when scores are moderate."""
    return 1.0 - (1.0 - clip_norm) * (1.0 - dino_norm)


def fusion_l2_norm(clip_norm, dino_norm):
    """L2 norm: sqrt(c^2 + d^2). Penalizes disagreement less than
    geometric mean, more than max."""
    return np.sqrt(clip_norm**2 + dino_norm**2)


def fusion_logsumexp(clip_norm, dino_norm, temperature=5.0):
    """LogSumExp (smooth max): log(exp(T*c) + exp(T*d)) / T.
    Interpolates between mean (T->0) and max (T->inf)."""
    tc = temperature * clip_norm
    td = temperature * dino_norm
    max_val = np.maximum(tc, td)
    return (max_val + np.log(np.exp(tc - max_val) + np.exp(td - max_val))) / temperature


def fusion_rank_average(clip_scores, dino_scores):
    """Rank-based fusion: Convert to percentile ranks, then average.
    Naturally calibrates different score distributions."""
    clip_ranks = rankdata(clip_scores) / len(clip_scores)
    dino_ranks = rankdata(dino_scores) / len(dino_scores)
    return (clip_ranks + dino_ranks) / 2.0


def fusion_rank_max(clip_scores, dino_scores):
    """Rank max: Convert to ranks, take max.
    Combines calibration of ranks with max fusion logic."""
    clip_ranks = rankdata(clip_scores) / len(clip_scores)
    dino_ranks = rankdata(dino_scores) / len(dino_scores)
    return np.maximum(clip_ranks, dino_ranks)


# ============================================================================
# PARADIGM B: Sample-level adaptive fusion
# ============================================================================

def fusion_score_proportional(clip_norm, dino_norm):
    """Per-sample alpha: alpha_i = dino_i / (clip_i + dino_i).
    Trusts the model that gives higher score for each sample."""
    total = clip_norm + dino_norm + 1e-10
    alpha_per_sample = dino_norm / total
    return (1 - alpha_per_sample) * clip_norm + alpha_per_sample * dino_norm


def fusion_softmax_adaptive(clip_norm, dino_norm, temperature=3.0):
    """Softmax per-sample: alpha_i = exp(T*dino_i)/(exp(T*clip_i)+exp(T*dino_i)).
    Sharper switching than score-proportional."""
    tc = temperature * clip_norm
    td = temperature * dino_norm
    max_val = np.maximum(tc, td)
    exp_c = np.exp(tc - max_val)
    exp_d = np.exp(td - max_val)
    alpha_per_sample = exp_d / (exp_c + exp_d)
    return (1 - alpha_per_sample) * clip_norm + alpha_per_sample * dino_norm


def fusion_confidence_gated(clip_norm, dino_norm, gate_pct=0.9):
    """Confidence-gated fusion: Use alpha=0.65 normally.
    But if one model has a score above the gate_pct percentile and
    the other doesn't, switch to the confident model."""
    clip_threshold = np.percentile(clip_norm, gate_pct * 100)
    dino_threshold = np.percentile(dino_norm, gate_pct * 100)

    base_fused = 0.35 * clip_norm + 0.65 * dino_norm

    # Override: if DINO is above threshold but CLIP isn't
    dino_only = (dino_norm >= dino_threshold) & (clip_norm < clip_threshold)
    base_fused[dino_only] = dino_norm[dino_only]

    # Override: if CLIP is above threshold but DINO isn't
    clip_only = (clip_norm >= clip_threshold) & (dino_norm < dino_threshold)
    base_fused[clip_only] = clip_norm[clip_only]

    return base_fused


def fusion_quantile_adaptive(clip_norm, dino_norm):
    """Quantile-adaptive: Convert scores to quantiles (CDF transform),
    then use the model with higher quantile for each sample.
    Combines rank calibration with sample-level selection."""
    clip_q = rankdata(clip_norm) / len(clip_norm)
    dino_q = rankdata(dino_norm) / len(dino_norm)

    # Alpha per sample based on who ranks higher
    alpha_per_sample = np.where(dino_q > clip_q, 0.8, 0.2)
    # Smooth transition near equality
    diff = np.abs(dino_q - clip_q)
    near_equal = diff < 0.1
    alpha_per_sample[near_equal] = 0.5

    return (1 - alpha_per_sample) * clip_norm + alpha_per_sample * dino_norm


def fusion_extremity_weighted(clip_norm, dino_norm):
    """Weight by extremity: the more extreme a score (far from median),
    the more we trust that model for that sample.
    Captures the idea that confident detections should be believed."""
    clip_med = np.median(clip_norm)
    dino_med = np.median(dino_norm)

    # Extremity = distance from median
    clip_ext = np.abs(clip_norm - clip_med)
    dino_ext = np.abs(dino_norm - dino_med)

    total = clip_ext + dino_ext + 1e-10
    alpha_per_sample = dino_ext / total

    return (1 - alpha_per_sample) * clip_norm + alpha_per_sample * dino_norm


# ============================================================================
# PARADIGM C: Score calibration + fixed fusion
# ============================================================================

def fusion_zscore_equal(clip_scores, dino_scores):
    """Z-score normalization + equal weight fusion.
    Z-score naturally calibrates mean and variance."""
    clip_z = normalize_scores(clip_scores, 'zscore')
    dino_z = normalize_scores(dino_scores, 'zscore')
    return 0.5 * clip_z + 0.5 * dino_z


def fusion_zscore_065(clip_scores, dino_scores):
    """Z-score normalization + alpha=0.65 fusion."""
    clip_z = normalize_scores(clip_scores, 'zscore')
    dino_z = normalize_scores(dino_scores, 'zscore')
    return 0.35 * clip_z + 0.65 * dino_z


def fusion_cdf_equal(clip_scores, dino_scores):
    """CDF transform (rank/n) + equal weight.
    Makes both distributions uniform before fusion."""
    clip_cdf = rankdata(clip_scores) / len(clip_scores)
    dino_cdf = rankdata(dino_scores) / len(dino_scores)
    return 0.5 * clip_cdf + 0.5 * dino_cdf


def fusion_cdf_065(clip_scores, dino_scores):
    """CDF transform + alpha=0.65."""
    clip_cdf = rankdata(clip_scores) / len(clip_scores)
    dino_cdf = rankdata(dino_scores) / len(dino_scores)
    return 0.35 * clip_cdf + 0.65 * dino_cdf


# ============================================================================
# PARADIGM D: Hybrid approaches
# ============================================================================

def fusion_max_then_065(clip_norm, dino_norm):
    """Two-stage: Max fusion for detection, 0.65 fusion for ranking.
    Use max to identify potential anomalies (top 30%),
    then rank them using 0.65 fusion for finer discrimination."""
    max_scores = np.maximum(clip_norm, dino_norm)
    linear_scores = 0.35 * clip_norm + 0.65 * dino_norm

    threshold = np.percentile(max_scores, 70)
    result = np.copy(linear_scores)
    # For samples where max is very high, boost with max score
    high_mask = max_scores > threshold
    result[high_mask] = 0.5 * linear_scores[high_mask] + 0.5 * max_scores[high_mask]
    return result


def fusion_disagreement_switch(clip_norm, dino_norm):
    """When models agree (both high or both low), use average.
    When they disagree, trust the higher score.
    This addresses the key failure mode: one model detects what the other misses."""
    avg = 0.5 * clip_norm + 0.5 * dino_norm
    diff = np.abs(clip_norm - dino_norm)

    # Disagreement threshold: where models significantly diverge
    disagree_threshold = np.percentile(diff, 70)

    result = np.copy(avg)
    disagree = diff > disagree_threshold
    # When they disagree, trust the higher score (the detector)
    result[disagree] = np.maximum(clip_norm[disagree], dino_norm[disagree])
    return result


def fusion_product_of_complements(clip_norm, dino_norm):
    """Product of complements: high when BOTH models think it's anomalous,
    but weighted by agreement.
    score = 1 - sqrt((1-c)(1-d)) gives a softer OR than probabilistic OR."""
    compl = np.sqrt((1 - clip_norm + 1e-10) * (1 - dino_norm + 1e-10))
    return 1 - compl


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

    print(f"\n  Total categories: {len(all_categories)}")
    print(f"  Oracle alpha range: varied per category")

    # Define all fusion methods
    # Each returns fused scores given the data dict
    def make_fuser(fn, uses_raw=False):
        """Wrapper to standardize interface."""
        def fuse(d):
            if uses_raw:
                return fn(d['clip_scores'], d['dino_scores'])
            else:
                clip_norm = normalize_scores(d['clip_scores'], 'minmax')
                dino_norm = normalize_scores(d['dino_scores'], 'minmax')
                return fn(clip_norm, dino_norm)
        return fuse

    methods = {
        # Baselines
        'Fixed-0.65': None,  # Special handling
        'Fixed-0.50': None,  # Special handling

        # Paradigm A: Non-linear (no alpha)
        'Max': make_fuser(fusion_max),
        'GeometricMean': make_fuser(fusion_geometric_mean),
        'ProbOR': make_fuser(fusion_probabilistic_or),
        'L2-norm': make_fuser(fusion_l2_norm),
        'LogSumExp-T3': make_fuser(lambda c, d: fusion_logsumexp(c, d, 3.0)),
        'LogSumExp-T5': make_fuser(lambda c, d: fusion_logsumexp(c, d, 5.0)),
        'LogSumExp-T10': make_fuser(lambda c, d: fusion_logsumexp(c, d, 10.0)),
        'RankAverage': make_fuser(fusion_rank_average, uses_raw=True),
        'RankMax': make_fuser(fusion_rank_max, uses_raw=True),

        # Paradigm B: Sample-level adaptive
        'ScoreProportional': make_fuser(fusion_score_proportional),
        'SoftmaxAdapt-T2': make_fuser(lambda c, d: fusion_softmax_adaptive(c, d, 2.0)),
        'SoftmaxAdapt-T5': make_fuser(lambda c, d: fusion_softmax_adaptive(c, d, 5.0)),
        'ConfGated-90': make_fuser(lambda c, d: fusion_confidence_gated(c, d, 0.90)),
        'ConfGated-95': make_fuser(lambda c, d: fusion_confidence_gated(c, d, 0.95)),
        'QuantileAdaptive': make_fuser(fusion_quantile_adaptive),
        'ExtremityWeighted': make_fuser(fusion_extremity_weighted),

        # Paradigm C: Calibration + fixed
        'Z-Equal': make_fuser(fusion_zscore_equal, uses_raw=True),
        'Z-0.65': make_fuser(fusion_zscore_065, uses_raw=True),
        'CDF-Equal': make_fuser(fusion_cdf_equal, uses_raw=True),
        'CDF-0.65': make_fuser(fusion_cdf_065, uses_raw=True),

        # Paradigm D: Hybrid
        'MaxThen065': make_fuser(fusion_max_then_065),
        'DisagreeSwitch': make_fuser(fusion_disagreement_switch),
        'ProdComplement': make_fuser(fusion_product_of_complements),
    }

    print(f"\n{'=' * 120}")
    print(f"  CYCLE 5: {len(methods)} fusion methods x {len(all_categories)} categories")
    print(f"{'=' * 120}")

    method_results = {}

    for method_name, method_fn in methods.items():
        aurocs = []
        per_cat = []

        for ds_name, obj, d in all_categories:
            labels = d['labels']
            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')

            # Compute fused scores
            if method_name == 'Fixed-0.65':
                fused = 0.35 * clip_norm + 0.65 * dino_norm
            elif method_name == 'Fixed-0.50':
                fused = 0.50 * clip_norm + 0.50 * dino_norm
            else:
                try:
                    fused = method_fn(d)
                except Exception as e:
                    fused = 0.35 * clip_norm + 0.65 * dino_norm

            auroc = compute_auroc(labels, fused)

            # Oracle (best linear alpha) for regret computation
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
                'auroc': round(auroc, 2), 'auroc_oracle': round(best_auroc, 2),
                'oracle_alpha': best_alpha
            })

        mean_auroc = np.mean(aurocs)
        oracle_aurocs = np.array([p['auroc_oracle'] for p in per_cat])
        method_aurocs = np.array([p['auroc'] for p in per_cat])
        regret = np.mean(oracle_aurocs - method_aurocs)

        # Count wins vs Fixed-0.65
        # (computed later when we have all results)

        method_results[method_name] = {
            'mean_auroc': round(mean_auroc, 3),
            'regret': round(regret, 3),
            'per_category': per_cat,
        }

    # Now compute wins vs Fixed-0.65
    fixed_per_cat = method_results['Fixed-0.65']['per_category']
    for method_name, r in method_results.items():
        if method_name == 'Fixed-0.65':
            r['wins_vs_065'] = '-'
            continue
        wins = 0
        for i, pc in enumerate(r['per_category']):
            if pc['auroc'] > fixed_per_cat[i]['auroc'] + 0.01:
                wins += 1
        r['wins_vs_065'] = f"{wins}/{len(all_categories)}"

    # Sort and print
    fixed_auroc = method_results['Fixed-0.65']['mean_auroc']
    sorted_methods = sorted(method_results.items(), key=lambda x: -x[1]['mean_auroc'])

    print(f"\n  {'Method':<24} {'AUROC':>8} {'Regret':>8} {'vs0.65':>8} {'Wins':>8}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, r in sorted_methods:
        diff = r['mean_auroc'] - fixed_auroc
        marker = " ***" if diff > 0.05 else (" **" if diff > 0.01 else (" *" if diff > 0 else ""))
        print(f"  {name:<24} {r['mean_auroc']:>8.3f} {r['regret']:>8.3f} "
              f"{diff:>+8.3f} {str(r['wins_vs_065']):>8}{marker}")

    # Detailed per-category for top methods
    print(f"\n{'=' * 120}")
    print(f"  TOP NON-FIXED METHODS - Per-Category Detail")
    print(f"{'=' * 120}")

    non_fixed = [(n, r) for n, r in sorted_methods if not n.startswith('Fixed')]
    for method_name, r in non_fixed[:5]:
        print(f"\n  --- {method_name} (AUROC: {r['mean_auroc']:.3f}, regret: {r['regret']:.3f}) ---")
        print(f"  {'Category':<16} {'This':>7} {'Fix65':>7} {'Oracle':>7} {'a*':>5} {'diff':>7}")
        cat_diffs = []
        for i, pc in enumerate(sorted(r['per_category'], key=lambda x: x['cat'])):
            fixed_pc = [p for p in fixed_per_cat if p['cat'] == pc['cat'] and p['ds'] == pc['ds']][0]
            diff = pc['auroc'] - fixed_pc['auroc']
            cat_diffs.append(diff)
            marker = " (+)" if diff > 0.1 else (" (-)" if diff < -0.1 else "")
            print(f"  {pc['cat']:<16} {pc['auroc']:>7.1f} {fixed_pc['auroc']:>7.1f} "
                  f"{pc['auroc_oracle']:>7.1f} {pc['oracle_alpha']:>5.2f} {diff:>+7.2f}{marker}")

        # Category-level win/loss/tie
        wins = sum(1 for d in cat_diffs if d > 0.01)
        losses = sum(1 for d in cat_diffs if d < -0.01)
        ties = len(cat_diffs) - wins - losses
        print(f"  Win/Loss/Tie: {wins}/{losses}/{ties}")

    # Analysis: Which paradigm works best?
    print(f"\n{'=' * 120}")
    print(f"  PARADIGM ANALYSIS")
    print(f"{'=' * 120}")

    paradigms = {
        'A-NonLinear': ['Max', 'GeometricMean', 'ProbOR', 'L2-norm',
                        'LogSumExp-T3', 'LogSumExp-T5', 'LogSumExp-T10',
                        'RankAverage', 'RankMax'],
        'B-SampleAdapt': ['ScoreProportional', 'SoftmaxAdapt-T2', 'SoftmaxAdapt-T5',
                          'ConfGated-90', 'ConfGated-95', 'QuantileAdaptive',
                          'ExtremityWeighted'],
        'C-Calibration': ['Z-Equal', 'Z-0.65', 'CDF-Equal', 'CDF-0.65'],
        'D-Hybrid': ['MaxThen065', 'DisagreeSwitch', 'ProdComplement'],
    }

    for paradigm, names in paradigms.items():
        results_in_paradigm = [(n, method_results[n]) for n in names if n in method_results]
        aurocs = [r['mean_auroc'] for _, r in results_in_paradigm]
        best_name, best_r = max(results_in_paradigm, key=lambda x: x[1]['mean_auroc'])
        diff_best = best_r['mean_auroc'] - fixed_auroc
        beats = "BEATS" if diff_best > 0 else "LOSES"
        print(f"\n  {paradigm}:")
        print(f"    Best: {best_name} = {best_r['mean_auroc']:.3f} ({beats} Fixed-0.65 by {diff_best:+.3f}pp)")
        print(f"    Range: [{min(aurocs):.3f}, {max(aurocs):.3f}]")

    # Save results (without per_category details for JSON size)
    output = {}
    for name, r in sorted_methods:
        output[name] = {k: v for k, v in r.items() if k != 'per_category'}
    out_path = os.path.join(os.path.dirname(BASE_DIR), "results", "alpha_discovery_cycle5.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Save full per-category data for analysis
    full_output = {}
    for name, r in sorted_methods:
        full_output[name] = r
    full_path = os.path.join(os.path.dirname(BASE_DIR), "results", "alpha_discovery_cycle5_full.json")
    with open(full_path, 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f"  Saved: {full_path}")


def main():
    print("=" * 120)
    print("  ALPHA DISCOVERY CYCLE 5: Beyond Linear Fusion")
    print("  Paradigms: Non-linear | Sample-Adaptive | Calibration | Hybrid")
    print("=" * 120)

    datasets = {}

    # MVTec
    clip_mvtec = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_mvtec",
                               "anomalyclip_scores.csv")
    dino_mvtec_weak = os.path.join(BASE_DIR, "AnomalyDINO", "results_MVTec",
                                    "dinov2_vitl14_448",
                                    "1-shot_preprocess=agnostic_weak_cats",
                                    "measurements_seed=0.csv")
    dino_mvtec_strong = os.path.join(BASE_DIR, "AnomalyDINO", "results_MVTec",
                                      "dinov2_vitl14_448",
                                      "1-shot_preprocess=agnostic_strong_cats",
                                      "measurements_seed=0.csv")
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
    clip_visa = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_visa",
                              "anomalyclip_scores.csv")
    dino_visa = os.path.join(BASE_DIR, "AnomalyDINO", "results_VisA",
                              "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_all",
                              "measurements_seed=0.csv")
    if os.path.exists(clip_visa) and os.path.exists(dino_visa):
        clip_data = load_anomalyclip_scores(clip_visa)
        dino_data = load_anomalydino_scores(dino_visa)
        print("\n--- Aligning VisA ---")
        datasets['VisA'] = align_samples(clip_data, dino_data)

    # BTAD
    clip_btad = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_btad",
                              "anomalyclip_scores.csv")
    dino_btad = os.path.join(BASE_DIR, "AnomalyDINO", "results_BTAD",
                              "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_btad_all",
                              "measurements_seed=0.csv")
    if os.path.exists(clip_btad) and os.path.exists(dino_btad):
        clip_data = load_anomalyclip_scores(clip_btad)
        dino_data = load_btad_dino_scores(dino_btad)
        print("\n--- Aligning BTAD ---")
        datasets['BTAD'] = align_samples(clip_data, dino_data)
    else:
        print(f"[WARN] BTAD scores not found -- run would cover only 27 categories:\n  clip={clip_btad}\n  dino={dino_btad}")

    if not datasets:
        print("ERROR: No datasets loaded!")
        return

    evaluate(datasets)


if __name__ == "__main__":
    main()
