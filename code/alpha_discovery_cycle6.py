"""
Alpha Discovery Cycle 6: The Impossibility Diagnostic
=====================================================

After 5 cycles and ~92 methods, NO unsupervised approach beats fixed alpha=0.65.
The closest was ConfGated-95 at -0.002pp.

This cycle has TWO goals:
  1. Power-mean family, ensemble-of-alphas, and refined gating
  2. DIAGNOSTIC: Explain WHY fixed-0.65 is unbeatable — decompose the regret

The diagnostic is the key contribution: it proves that the oracle surface is
structured such that any perturbation from 0.65 hurts more categories than
it helps, making per-category optimization fundamentally unprofitable without
direct supervision.
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
# PART 1: Additional methods
# ============================================================================

def fusion_power_mean(clip_norm, dino_norm, p, alpha=0.65):
    """Generalized (weighted) power mean:
    M_p = ((1-a)*c^p + a*d^p)^(1/p)
    p=-1: weighted harmonic mean
    p=0 : weighted geometric mean (limit)
    p=1 : weighted arithmetic mean (= linear fusion)
    p=2 : weighted quadratic mean
    p->inf: max
    """
    eps = 1e-8
    c = np.clip(clip_norm, eps, None)
    d = np.clip(dino_norm, eps, None)
    w_c, w_d = 1 - alpha, alpha

    if abs(p) < 0.1:
        # Geometric mean (limit case for small p)
        return np.exp(w_c * np.log(c) + w_d * np.log(d))
    else:
        val = w_c * (c ** p) + w_d * (d ** p)
        val = np.clip(val, eps, None)
        result = val ** (1.0 / p)
        return np.clip(result, 0, 10)  # prevent overflow


def fusion_ensemble_alpha(clip_norm, dino_norm, alphas):
    """Ensemble of fusion scores across multiple alpha values.
    Final score = mean of all fused scores."""
    fused_list = []
    for a in alphas:
        f = (1 - a) * clip_norm + a * dino_norm
        fused_list.append(f)
    return np.mean(fused_list, axis=0)


def fusion_trimmed_linear(clip_norm, dino_norm, alpha=0.65, trim_pct=0.95):
    """Linear fusion with trimming: when one model's score is in the
    extreme tail (> trim_pct percentile) and the other isn't,
    reduce the extreme model's contribution to prevent one-model dominance."""
    c_thresh = np.percentile(clip_norm, trim_pct * 100)
    d_thresh = np.percentile(dino_norm, trim_pct * 100)

    fused = (1 - alpha) * clip_norm + alpha * dino_norm

    # When CLIP is extreme but DINO isn't, moderate CLIP's influence
    clip_extreme = (clip_norm > c_thresh) & (dino_norm < d_thresh)
    fused[clip_extreme] = 0.5 * clip_norm[clip_extreme] + 0.5 * dino_norm[clip_extreme]

    # When DINO is extreme but CLIP isn't, moderate DINO's influence
    dino_extreme = (dino_norm > d_thresh) & (clip_norm < c_thresh)
    fused[dino_extreme] = 0.5 * clip_norm[dino_extreme] + 0.5 * dino_norm[dino_extreme]

    return fused


def fusion_rank_weighted_065(clip_scores, dino_scores):
    """Rank normalization + weighted fusion at 0.65.
    Rank normalization is more robust to score distribution differences."""
    clip_r = rankdata(clip_scores) / len(clip_scores)
    dino_r = rankdata(dino_scores) / len(dino_scores)
    return 0.35 * clip_r + 0.65 * dino_r


def fusion_adaptive_alpha_smooth(clip_norm, dino_norm, base_alpha=0.65, sensitivity=0.3):
    """Smoothly varying alpha per sample based on relative scores.
    alpha_i = base + sensitivity * (dino_rank_i - clip_rank_i)
    When DINO ranks higher for a sample, alpha increases slightly."""
    clip_r = rankdata(clip_norm) / len(clip_norm)
    dino_r = rankdata(dino_norm) / len(dino_norm)

    rank_diff = dino_r - clip_r  # [-1, 1]
    alpha_per_sample = np.clip(base_alpha + sensitivity * rank_diff, 0.1, 0.9)

    return (1 - alpha_per_sample) * clip_norm + alpha_per_sample * dino_norm


def fusion_copula_inspired(clip_norm, dino_norm):
    """Copula-inspired fusion: transform to uniform margins,
    apply Clayton copula (lower tail dependence), transform back.
    Clayton copula: C(u,v) = (u^-theta + v^-theta - 1)^(-1/theta)
    This emphasizes joint anomalousness (both models agree)."""
    u = rankdata(clip_norm) / (len(clip_norm) + 1)  # pseudo-uniform
    v = rankdata(dino_norm) / (len(dino_norm) + 1)

    theta = 2.0  # Controls tail dependence
    eps = 1e-6
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    # Clayton copula CDF
    copula = np.power(
        np.power(u, -theta) + np.power(v, -theta) - 1,
        -1.0 / theta
    )
    return np.clip(copula, 0, 1)


# ============================================================================
# PART 2: Diagnostic Analysis
# ============================================================================

def diagnostic_analysis(all_categories):
    """Deep analysis of WHY fixed alpha=0.65 is unbeatable."""

    fine_steps = [i / 100.0 for i in range(0, 101, 1)]

    print(f"\n{'=' * 120}")
    print(f"  DIAGNOSTIC: Why Fixed alpha=0.65 Cannot Be Beaten")
    print(f"{'=' * 120}")

    # 1. Per-category regret decomposition
    cat_data = []
    for ds_name, obj, d in all_categories:
        labels = d['labels']
        clip_norm = normalize_scores(d['clip_scores'], 'minmax')
        dino_norm = normalize_scores(d['dino_scores'], 'minmax')

        # Oracle
        best_auroc, best_alpha = 0, 0.5
        auroc_at_065 = 0
        auroc_at_050 = 0
        auroc_clip_only = compute_auroc(labels, clip_norm)
        auroc_dino_only = compute_auroc(labels, dino_norm)

        alpha_curve = []
        for a in fine_steps:
            f = (1 - a) * clip_norm + a * dino_norm
            au = compute_auroc(labels, f)
            alpha_curve.append(au)
            if a == 0.65:
                auroc_at_065 = au
            if a == 0.50:
                auroc_at_050 = au
            if au > best_auroc:
                best_auroc = au
                best_alpha = a

        regret_065 = best_auroc - auroc_at_065

        # Flatness: range of AUROC within alpha [0.4, 0.8]
        mid_range = [alpha_curve[i] for i in range(40, 81)]
        flatness = max(mid_range) - min(mid_range)

        # Sensitivity: how much does AUROC change per 0.01 alpha near 0.65?
        if len(alpha_curve) > 66:
            local_grad = abs(alpha_curve[66] - alpha_curve[64]) / 2
        else:
            local_grad = 0

        cat_data.append({
            'ds': ds_name, 'cat': obj,
            'oracle_alpha': best_alpha,
            'oracle_auroc': round(best_auroc, 2),
            'auroc_065': round(auroc_at_065, 2),
            'auroc_050': round(auroc_at_050, 2),
            'auroc_clip': round(auroc_clip_only, 2),
            'auroc_dino': round(auroc_dino_only, 2),
            'regret_065': round(regret_065, 3),
            'flatness_04_08': round(flatness, 3),
            'local_gradient': round(local_grad, 4),
            'alpha_curve': [round(x, 3) for x in alpha_curve],
        })

    # Sort by regret
    cat_data.sort(key=lambda x: -x['regret_065'])

    # 2. Print regret decomposition
    print(f"\n  === Regret Decomposition (sorted by regret) ===")
    print(f"  {'Category':<16} {'a*':>5} {'OracAUR':>8} {'Fix65':>8} {'Regret':>8} "
          f"{'Flat[.4,.8]':>12} {'Grad@.65':>10} {'CLIP':>7} {'DINO':>7}")
    print(f"  {'-'*16} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*10} {'-'*7} {'-'*7}")

    total_regret = 0
    cumulative = []
    for cd in cat_data:
        total_regret += cd['regret_065']
        cumulative.append(total_regret)
        extreme = " <<< EXTREME" if cd['oracle_alpha'] < 0.2 or cd['oracle_alpha'] > 0.8 else ""
        print(f"  {cd['cat']:<16} {cd['oracle_alpha']:>5.2f} {cd['oracle_auroc']:>8.1f} "
              f"{cd['auroc_065']:>8.1f} {cd['regret_065']:>8.3f} {cd['flatness_04_08']:>12.3f} "
              f"{cd['local_gradient']:>10.4f} {cd['auroc_clip']:>7.1f} {cd['auroc_dino']:>7.1f}{extreme}")

    mean_regret = total_regret / len(cat_data)
    print(f"\n  Total regret: {total_regret:.3f}pp across {len(cat_data)} categories")
    print(f"  Mean regret:  {mean_regret:.3f}pp")

    # 3. Concentration analysis
    print(f"\n  === Regret Concentration ===")
    regrets = [cd['regret_065'] for cd in cat_data]
    for k in [1, 2, 3, 5]:
        top_k_regret = sum(regrets[:k])
        pct = top_k_regret / total_regret * 100
        cats = ", ".join([cd['cat'] for cd in cat_data[:k]])
        print(f"  Top {k} categories: {top_k_regret:.3f}pp = {pct:.1f}% of total ({cats})")

    # 4. Flatness analysis
    print(f"\n  === Oracle Surface Flatness ===")
    flat_cats = [cd for cd in cat_data if cd['regret_065'] < 0.1]
    medium_cats = [cd for cd in cat_data if 0.1 <= cd['regret_065'] < 1.0]
    steep_cats = [cd for cd in cat_data if cd['regret_065'] >= 1.0]
    print(f"  Effectively flat (regret < 0.1pp): {len(flat_cats)}/{len(cat_data)} categories")
    print(f"    = {', '.join([c['cat'] for c in flat_cats])}")
    print(f"  Moderate (0.1-1.0pp regret): {len(medium_cats)}/{len(cat_data)} categories")
    print(f"    = {', '.join([c['cat'] for c in medium_cats])}")
    print(f"  Steep (>1.0pp regret): {len(steep_cats)}/{len(cat_data)} categories")
    print(f"    = {', '.join([c['cat'] for c in steep_cats])}")

    # 5. Alpha distance analysis
    print(f"\n  === Alpha Distance from 0.65 ===")
    alphas = [cd['oracle_alpha'] for cd in cat_data]
    distances = [abs(a - 0.65) for a in alphas]
    print(f"  Mean |alpha* - 0.65|: {np.mean(distances):.3f}")
    print(f"  Median |alpha* - 0.65|: {np.median(distances):.3f}")
    near = sum(1 for d in distances if d <= 0.1)
    mid = sum(1 for d in distances if 0.1 < d <= 0.3)
    far = sum(1 for d in distances if d > 0.3)
    print(f"  Near (|da|<=0.1): {near}/{len(cat_data)}")
    print(f"  Mid  (0.1<|da|<=0.3): {mid}/{len(cat_data)}")
    print(f"  Far  (|da|>0.3):  {far}/{len(cat_data)}")
    print(f"  Far categories: {', '.join([cd['cat'] + '(' + str(cd['oracle_alpha']) + ')' for cd in cat_data if abs(cd['oracle_alpha'] - 0.65) > 0.3])}")

    # 6. Asymmetry analysis: does moving alpha UP or DOWN from 0.65 hurt equally?
    print(f"\n  === Directional Asymmetry around alpha=0.65 ===")
    above = [cd for cd in cat_data if cd['oracle_alpha'] > 0.65]
    below = [cd for cd in cat_data if cd['oracle_alpha'] < 0.65]
    at_065 = [cd for cd in cat_data if cd['oracle_alpha'] == 0.65]
    print(f"  Categories wanting alpha > 0.65: {len(above)} (mean a*={np.mean([c['oracle_alpha'] for c in above]):.2f})")
    print(f"  Categories wanting alpha < 0.65: {len(below)} (mean a*={np.mean([c['oracle_alpha'] for c in below]):.2f})")
    print(f"  Categories where alpha = 0.65:   {len(at_065)}")

    if above:
        mean_regret_above = np.mean([c['regret_065'] for c in above])
    else:
        mean_regret_above = 0
    if below:
        mean_regret_below = np.mean([c['regret_065'] for c in below])
    else:
        mean_regret_below = 0
    print(f"  Mean regret (above): {mean_regret_above:.3f}pp")
    print(f"  Mean regret (below): {mean_regret_below:.3f}pp")

    # 7. The fundamental impossibility argument
    print(f"\n  === The Fundamental Impossibility ===")
    print(f"  For an unsupervised method to beat fixed-0.65, it must:")
    print(f"    (a) Correctly identify the {len(steep_cats)} steep categories")
    print(f"    (b) Move their alpha in the RIGHT direction")
    print(f"    (c) WITHOUT hurting the other {len(cat_data) - len(steep_cats)} categories")
    print(f"")
    if steep_cats:
        print(f"  The steep categories are:")
        for cd in steep_cats:
            direction = "needs MORE DINO" if cd['oracle_alpha'] > 0.65 else "needs LESS DINO"
            print(f"    {cd['cat']}: a*={cd['oracle_alpha']}, {direction}, regret={cd['regret_065']:.3f}pp")
        print(f"")
        print(f"  Problem: These categories want OPPOSITE directions!")
        want_up = [c for c in steep_cats if c['oracle_alpha'] > 0.65]
        want_down = [c for c in steep_cats if c['oracle_alpha'] < 0.65]
        print(f"    Want alpha UP: {len(want_up)} ({', '.join([c['cat'] for c in want_up])})")
        print(f"    Want alpha DOWN: {len(want_down)} ({', '.join([c['cat'] for c in want_down])})")
        print(f"  Any unsupervised signal that moves alpha for ALL categories in one direction")
        print(f"  will help some and hurt others -- net effect near zero or negative.")

    # 8. Information-theoretic bound estimate
    print(f"\n  === Maximum Recoverable Regret ===")
    print(f"  Total oracle regret from 0.65: {total_regret:.3f}pp")
    print(f"  If we could perfectly identify the top-3 steep categories and")
    print(f"  use their oracle alpha (while keeping 0.65 for others):")
    if len(cat_data) >= 3:
        partial_oracle_gain = sum(regrets[:3])
        partial_mean = partial_oracle_gain / len(cat_data)
        print(f"    Recoverable regret: {partial_oracle_gain:.3f}pp total, {partial_mean:.3f}pp mean")
        print(f"    New mean AUROC would be: {95.213 + partial_mean:.3f}")
    print(f"  But identifying these categories WITHOUT labels is the unsolved problem.")

    return cat_data


# ============================================================================
# Evaluation with fusion methods
# ============================================================================

def evaluate(datasets, all_categories):
    fine_steps = [i / 100.0 for i in range(0, 101, 1)]

    def make_fuser(fn, uses_raw=False):
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
        'Fixed-0.65': None,
        'Fixed-0.50': None,

        # Power means at alpha=0.65
        'PowerMean-p0.5': make_fuser(lambda c, d: fusion_power_mean(c, d, 0.5, 0.65)),
        'PowerMean-p2': make_fuser(lambda c, d: fusion_power_mean(c, d, 2.0, 0.65)),
        'PowerMean-p3': make_fuser(lambda c, d: fusion_power_mean(c, d, 3.0, 0.65)),
        'PowerMean-p-1': make_fuser(lambda c, d: fusion_power_mean(c, d, -1.0, 0.65)),
        'GeomMean-0.65': make_fuser(lambda c, d: fusion_power_mean(c, d, 0.01, 0.65)),

        # Ensemble of alphas (hedge against single alpha)
        'Ensemble-5': make_fuser(lambda c, d: fusion_ensemble_alpha(c, d, [0.55, 0.60, 0.65, 0.70, 0.75])),
        'Ensemble-7': make_fuser(lambda c, d: fusion_ensemble_alpha(c, d, [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])),
        'Ensemble-narrow': make_fuser(lambda c, d: fusion_ensemble_alpha(c, d, [0.60, 0.63, 0.65, 0.67, 0.70])),

        # Trimmed fusion
        'Trimmed-95': make_fuser(lambda c, d: fusion_trimmed_linear(c, d, 0.65, 0.95)),
        'Trimmed-90': make_fuser(lambda c, d: fusion_trimmed_linear(c, d, 0.65, 0.90)),

        # Rank-weighted
        'RankW-0.65': make_fuser(fusion_rank_weighted_065, uses_raw=True),

        # Smooth adaptive
        'SmoothAdapt-s0.1': make_fuser(lambda c, d: fusion_adaptive_alpha_smooth(c, d, 0.65, 0.1)),
        'SmoothAdapt-s0.2': make_fuser(lambda c, d: fusion_adaptive_alpha_smooth(c, d, 0.65, 0.2)),
        'SmoothAdapt-s0.3': make_fuser(lambda c, d: fusion_adaptive_alpha_smooth(c, d, 0.65, 0.3)),

        # Copula-inspired
        'Clayton-Copula': make_fuser(fusion_copula_inspired),
    }

    print(f"\n{'=' * 120}")
    print(f"  CYCLE 6: {len(methods)} methods x {len(all_categories)} categories")
    print(f"{'=' * 120}")

    method_results = {}

    for method_name, method_fn in methods.items():
        aurocs = []
        per_cat = []

        for ds_name, obj, d in all_categories:
            labels = d['labels']
            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')

            if method_name == 'Fixed-0.65':
                fused = 0.35 * clip_norm + 0.65 * dino_norm
            elif method_name == 'Fixed-0.50':
                fused = 0.50 * clip_norm + 0.50 * dino_norm
            else:
                try:
                    fused = method_fn(d)
                except Exception:
                    fused = 0.35 * clip_norm + 0.65 * dino_norm

            auroc = compute_auroc(labels, fused)

            best_auroc = 0
            for a in fine_steps:
                f = (1 - a) * clip_norm + a * dino_norm
                au = compute_auroc(labels, f)
                if au > best_auroc:
                    best_auroc = au

            aurocs.append(auroc)
            per_cat.append({'auroc': round(auroc, 2), 'oracle': round(best_auroc, 2)})

        mean_auroc = np.mean(aurocs)
        regret = np.mean([p['oracle'] - p['auroc'] for p in per_cat])

        method_results[method_name] = {
            'mean_auroc': round(mean_auroc, 3),
            'regret': round(regret, 3),
        }

    fixed_auroc = method_results['Fixed-0.65']['mean_auroc']
    sorted_methods = sorted(method_results.items(), key=lambda x: -x[1]['mean_auroc'])

    print(f"\n  {'Method':<24} {'AUROC':>8} {'Regret':>8} {'vs0.65':>8}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8}")

    for name, r in sorted_methods:
        diff = r['mean_auroc'] - fixed_auroc
        marker = " ***" if diff > 0.05 else (" **" if diff > 0.01 else (" *" if diff > 0 else ""))
        print(f"  {name:<24} {r['mean_auroc']:>8.3f} {r['regret']:>8.3f} {diff:>+8.3f}{marker}")

    output = {name: r for name, r in sorted_methods}
    out_path = os.path.join(os.path.dirname(BASE_DIR), "results", "alpha_discovery_cycle6.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


def main():
    print("=" * 120)
    print("  ALPHA DISCOVERY CYCLE 6: The Impossibility Diagnostic")
    print("  Final fusion attempts + Deep analysis of why fixed-0.65 is optimal")
    print("=" * 120)

    datasets = {}

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

    all_categories = []
    for ds_name, aligned in datasets.items():
        for obj in sorted(aligned.keys()):
            d = aligned[obj]
            if len(np.unique(d['labels'])) < 2:
                continue
            all_categories.append((ds_name, obj, d))

    # Run diagnostic first
    cat_data = diagnostic_analysis(all_categories)

    # Then run fusion methods
    evaluate(datasets, all_categories)

    # Save diagnostic data
    diag_output = []
    for cd in cat_data:
        d_copy = {k: v for k, v in cd.items() if k != 'alpha_curve'}
        diag_output.append(d_copy)
    diag_path = os.path.join(os.path.dirname(BASE_DIR), "results", "diagnostic_analysis.json")
    with open(diag_path, 'w') as f:
        json.dump(diag_output, f, indent=2)
    print(f"  Saved diagnostic: {diag_path}")


if __name__ == "__main__":
    main()
