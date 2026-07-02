"""
Fine-grained Alpha Sweep (0.01 step) using score_fusion.py's alignment logic.

Questions:
1. What is the true global optimal alpha at 0.01 granularity?
2. How much does per-category optimal alpha vary?
3. Does the inverse error weighting formula predict per-category alpha?
"""

import os
import sys
import numpy as np
import json
from collections import defaultdict

# Import from score_fusion.py
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
    """Load BTAD DINOv2 scores (uses ok/ko instead of good/bad)."""
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            obj = row[0]
            sample = row[1]
            score = float(row[2])
            is_anomaly = 0 if sample.startswith("ok/") else 1
            data[obj]['scores'].append(score)
            data[obj]['labels'].append(is_anomaly)
            data[obj]['paths'].append(sample)
    return dict(data)


def fine_alpha_sweep(aligned_data, dataset_name, alpha_steps):
    """
    Perform alpha sweep at given granularity.
    Returns: {category: {alpha: auroc}} and global mean per alpha
    """
    per_cat = {}
    for obj in sorted(aligned_data.keys()):
        d = aligned_data[obj]
        labels = d['labels']
        if len(np.unique(labels)) < 2:
            continue

        per_cat[obj] = {}
        clip_norm = normalize_scores(d['clip_scores'], 'minmax')
        dino_norm = normalize_scores(d['dino_scores'], 'minmax')

        for alpha in alpha_steps:
            fused = (1 - alpha) * clip_norm + alpha * dino_norm
            per_cat[obj][alpha] = compute_auroc(labels, fused)

    # Global mean per alpha
    global_mean = {}
    for alpha in alpha_steps:
        vals = [per_cat[obj][alpha] for obj in per_cat if not np.isnan(per_cat[obj][alpha])]
        global_mean[alpha] = np.mean(vals) if vals else 0

    return per_cat, global_mean


def cohens_d(scores, labels):
    """Cohen's d = |mu_anomaly - mu_normal| / pooled_std"""
    s = np.array(scores)
    l = np.array(labels)
    normal = s[l == 0]
    anomaly = s[l == 1]
    if len(normal) < 2 or len(anomaly) < 2:
        return None
    mu_n, mu_a = np.mean(normal), np.mean(anomaly)
    var_n, var_a = np.var(normal, ddof=1), np.var(anomaly, ddof=1)
    pooled = np.sqrt((var_n + var_a) / 2)
    if pooled < 1e-10:
        return None
    return abs(mu_a - mu_n) / pooled


def inverse_error_weight(aligned_data):
    """
    Compute per-category predicted alpha using inverse error weighting.
    alpha = (1/err_D) / (1/err_C + 1/err_D)
    where err = 1 - AUROC/100
    """
    predictions = {}
    for obj in sorted(aligned_data.keys()):
        d = aligned_data[obj]
        labels = d['labels']
        if len(np.unique(labels)) < 2:
            continue

        auroc_clip = compute_auroc(labels, d['clip_scores'])
        auroc_dino = compute_auroc(labels, d['dino_scores'])

        err_c = max(1 - auroc_clip / 100, 1e-6)
        err_d = max(1 - auroc_dino / 100, 1e-6)

        alpha_pred = (1 / err_d) / (1 / err_c + 1 / err_d)
        predictions[obj] = {
            'auroc_clip': auroc_clip,
            'auroc_dino': auroc_dino,
            'err_clip': err_c,
            'err_dino': err_d,
            'alpha_pred': alpha_pred
        }
    return predictions


def main():
    print("=" * 80)
    print("  FINE-GRAINED ALPHA SWEEP (0.01 step)")
    print("=" * 80)

    # Alpha steps: 0.00, 0.01, 0.02, ..., 1.00
    alpha_steps = [i / 100.0 for i in range(0, 101, 1)]

    # =====================================================
    # Load all datasets
    # =====================================================
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
        for dino_path in [dino_mvtec_weak, dino_mvtec_strong]:
            if os.path.exists(dino_path):
                dino_data.update(load_anomalydino_scores(dino_path))
        if dino_data:
            print("\n--- Aligning MVTec samples ---")
            datasets['MVTec'] = align_samples(clip_data, dino_data)

    # VisA
    clip_visa = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv")
    dino_visa = os.path.join(BASE_DIR, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv")

    if os.path.exists(clip_visa) and os.path.exists(dino_visa):
        clip_data = load_anomalyclip_scores(clip_visa)
        dino_data = load_anomalydino_scores(dino_visa)
        print("\n--- Aligning VisA samples ---")
        datasets['VisA'] = align_samples(clip_data, dino_data)

    # BTAD
    clip_btad = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv")
    dino_btad = os.path.join(BASE_DIR, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv")

    if os.path.exists(clip_btad) and os.path.exists(dino_btad):
        clip_data = load_anomalyclip_scores(clip_btad)
        dino_data = load_btad_dino_scores(dino_btad)
        print("\n--- Aligning BTAD samples ---")
        datasets['BTAD'] = align_samples(clip_data, dino_data)
    else:
        print(f"[WARN] BTAD scores not found -- dataset will be EXCLUDED from sweep "
              f"(run would cover only 27 categories):\n  clip={clip_btad}\n  dino={dino_btad}")

    # =====================================================
    # Fine-grained sweep per dataset
    # =====================================================
    all_per_cat = {}  # {cat: {alpha: auroc}}
    all_global_means = {}  # {dataset: {alpha: mean_auroc}}

    missing = [d for d in ('MVTec', 'VisA', 'BTAD') if d not in datasets]
    if missing:
        raise FileNotFoundError("missing score CSVs for: %s" % missing)
    for ds_name, aligned in datasets.items():
        per_cat, global_mean = fine_alpha_sweep(aligned, ds_name, alpha_steps)
        all_per_cat.update(per_cat)
        all_global_means[ds_name] = global_mean

    # =====================================================
    # Combined global mean
    # =====================================================
    combined_global = {}
    for alpha in alpha_steps:
        vals = []
        for ds in all_global_means.values():
            vals.append(ds[alpha])
        combined_global[alpha] = np.mean(vals) if vals else 0

    # Find best global alpha
    best_alpha = max(combined_global, key=combined_global.get)
    best_auroc = combined_global[best_alpha]

    print(f"\n{'=' * 80}")
    print(f"  RESULT 1: Global Optimal Alpha (0.01 granularity)")
    print(f"{'=' * 80}")
    print(f"\n  Best global alpha:  {best_alpha:.2f}")
    print(f"  Best mean AUROC:    {best_auroc:.2f}%")

    # Show top 10 alphas around peak
    sorted_alphas = sorted(combined_global.items(), key=lambda x: -x[1])
    print(f"\n  Top 10 alpha values:")
    for a, m in sorted_alphas[:10]:
        marker = " <-- BEST" if a == best_alpha else ""
        print(f"    alpha={a:.2f}: {m:.2f}%{marker}")

    # Show plateau: alphas within 0.1pp of best
    plateau = [(a, m) for a, m in combined_global.items() if best_auroc - m < 0.1]
    plateau.sort()
    print(f"\n  Alpha plateau (within 0.1pp of best): [{plateau[0][0]:.2f}, {plateau[-1][0]:.2f}]")

    # Per-dataset best
    print(f"\n  Per-dataset best alpha:")
    for ds_name, gm in all_global_means.items():
        best_a = max(gm, key=gm.get)
        best_v = gm[best_a]
        print(f"    {ds_name}: alpha={best_a:.2f} -> {best_v:.2f}%")

    # Compare 0.64 vs 0.65
    print(f"\n  Direct comparison:")
    print(f"    alpha=0.64: {combined_global[0.64]:.3f}%")
    print(f"    alpha=0.65: {combined_global[0.65]:.3f}%")
    print(f"    Difference: {combined_global[0.65] - combined_global[0.64]:.4f}pp")

    # =====================================================
    # Per-category oracle alpha at 0.01 granularity
    # =====================================================
    print(f"\n{'=' * 80}")
    print(f"  RESULT 2: Per-Category Optimal Alpha Distribution")
    print(f"{'=' * 80}")

    cat_best = {}
    for obj in sorted(all_per_cat.keys()):
        best_a = max(all_per_cat[obj], key=all_per_cat[obj].get)
        best_v = all_per_cat[obj][best_a]
        clip_v = all_per_cat[obj][0.0]
        dino_v = all_per_cat[obj][1.0]
        cat_best[obj] = best_a
        print(f"  {obj:<16} alpha*={best_a:.2f}  AUROC={best_v:.1f}%  "
              f"(CLIP={clip_v:.1f}%, DINO={dino_v:.1f}%)")

    alphas = list(cat_best.values())
    print(f"\n  Distribution of per-category alpha*:")
    print(f"    Mean:    {np.mean(alphas):.3f}")
    print(f"    Median:  {np.median(alphas):.3f}")
    print(f"    Std:     {np.std(alphas):.3f}")
    print(f"    Min:     {np.min(alphas):.2f}")
    print(f"    Max:     {np.max(alphas):.2f}")
    print(f"    IQR:     [{np.percentile(alphas, 25):.2f}, {np.percentile(alphas, 75):.2f}]")

    # Histogram
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist, _ = np.histogram(alphas, bins=bins)
    print(f"\n  Histogram of alpha*:")
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        bar = "#" * hist[i]
        print(f"    [{lo:.1f}, {hi:.1f}): {hist[i]:2d} {bar}")

    # =====================================================
    # Inverse Error Weighting Predictions
    # =====================================================
    print(f"\n{'=' * 80}")
    print(f"  RESULT 3: Inverse Error Weighting Formula Prediction")
    print(f"{'=' * 80}")

    all_preds = {}
    for ds_name, aligned in datasets.items():
        preds = inverse_error_weight(aligned)
        for obj, p in preds.items():
            all_preds[obj] = p

    # Compare predicted vs oracle (fine-grained)
    pred_list, oracle_list = [], []
    print(f"\n  {'Category':<16} {'a_pred':>8} {'a_oracle':>8} {'|err|':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}")

    for obj in sorted(set(all_preds.keys()) & set(cat_best.keys())):
        a_pred = all_preds[obj]['alpha_pred']
        a_oracle = cat_best[obj]
        err = abs(a_pred - a_oracle)
        print(f"  {obj:<16} {a_pred:8.3f} {a_oracle:8.2f} {err:8.3f}")
        pred_list.append(a_pred)
        oracle_list.append(a_oracle)

    pred_arr = np.array(pred_list)
    oracle_arr = np.array(oracle_list)
    n = len(pred_arr)

    # Pearson correlation
    mx, my = np.mean(pred_arr), np.mean(oracle_arr)
    cov = np.sum((pred_arr - mx) * (oracle_arr - my))
    sx = np.sqrt(np.sum((pred_arr - mx) ** 2))
    sy = np.sqrt(np.sum((oracle_arr - my) ** 2))
    r_pearson = cov / (sx * sy) if sx > 1e-10 and sy > 1e-10 else 0

    mae = np.mean(np.abs(pred_arr - oracle_arr))
    rmse = np.sqrt(np.mean((pred_arr - oracle_arr) ** 2))

    # Derived global alpha
    global_pred = np.mean(pred_arr)

    print(f"\n  Prediction quality:")
    print(f"    Pearson r:         {r_pearson:.4f}")
    print(f"    MAE:               {mae:.4f}")
    print(f"    RMSE:              {rmse:.4f}")
    print(f"    Mean predicted:    {global_pred:.4f}")
    print(f"    Mean oracle:       {np.mean(oracle_arr):.4f}")

    # =====================================================
    # KEY QUESTION: Does the formula-derived alpha actually WORK?
    # Compare AUROC of alpha=formula vs alpha=0.65 vs alpha=oracle
    # =====================================================
    print(f"\n{'=' * 80}")
    print(f"  RESULT 4: Does predicted alpha improve performance?")
    print(f"{'=' * 80}")

    auroc_fixed065 = []
    auroc_predicted = []
    auroc_oracle = []
    auroc_clip_only = []
    auroc_dino_only = []

    for ds_name, aligned in datasets.items():
        for obj in sorted(aligned.keys()):
            d = aligned[obj]
            labels = d['labels']
            if len(np.unique(labels)) < 2:
                continue
            if obj not in all_preds or obj not in cat_best:
                continue

            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')

            # Fixed alpha=0.65
            fused_065 = (1 - 0.65) * clip_norm + 0.65 * dino_norm
            auroc_fixed065.append(compute_auroc(labels, fused_065))

            # Predicted alpha per category
            a_pred = all_preds[obj]['alpha_pred']
            fused_pred = (1 - a_pred) * clip_norm + a_pred * dino_norm
            auroc_predicted.append(compute_auroc(labels, fused_pred))

            # Oracle alpha
            a_oracle = cat_best[obj]
            fused_oracle = (1 - a_oracle) * clip_norm + a_oracle * dino_norm
            auroc_oracle.append(compute_auroc(labels, fused_oracle))

            # Single models
            auroc_clip_only.append(compute_auroc(labels, d['clip_scores']))
            auroc_dino_only.append(compute_auroc(labels, d['dino_scores']))

    print(f"\n  Mean AUROC across all {len(auroc_fixed065)} categories:")
    print(f"    CLIP only:              {np.mean(auroc_clip_only):.2f}%")
    print(f"    DINOv2 only:            {np.mean(auroc_dino_only):.2f}%")
    print(f"    Fixed alpha=0.65:       {np.mean(auroc_fixed065):.2f}%")
    print(f"    Per-cat predicted:      {np.mean(auroc_predicted):.2f}%")
    print(f"    Per-cat oracle:         {np.mean(auroc_oracle):.2f}%")

    print(f"\n  Improvement over fixed alpha=0.65:")
    print(f"    Predicted alpha:  {np.mean(auroc_predicted) - np.mean(auroc_fixed065):+.3f}pp")
    print(f"    Oracle alpha:     {np.mean(auroc_oracle) - np.mean(auroc_fixed065):+.3f}pp")

    # Regret analysis
    regret_065 = np.array(auroc_oracle) - np.array(auroc_fixed065)
    regret_pred = np.array(auroc_oracle) - np.array(auroc_predicted)

    print(f"\n  Regret (oracle - method):")
    print(f"    Fixed alpha=0.65:  mean={np.mean(regret_065):.3f}pp, max={np.max(regret_065):.3f}pp")
    print(f"    Predicted alpha:   mean={np.mean(regret_pred):.3f}pp, max={np.max(regret_pred):.3f}pp")

    # Save results
    output = {
        "fine_grained_sweep": {
            "best_global_alpha": best_alpha,
            "best_global_auroc": round(best_auroc, 3),
            "alpha_0.64": round(combined_global[0.64], 3),
            "alpha_0.65": round(combined_global[0.65], 3),
        },
        "per_category_oracle_alpha": {
            obj: round(cat_best[obj], 2) for obj in sorted(cat_best.keys())
        },
        "per_category_alpha_stats": {
            "mean": round(float(np.mean(alphas)), 3),
            "median": round(float(np.median(alphas)), 3),
            "std": round(float(np.std(alphas)), 3),
            "min": round(float(np.min(alphas)), 2),
            "max": round(float(np.max(alphas)), 2),
        },
        "inverse_error_weighting": {
            "pearson_r": round(r_pearson, 4),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "global_predicted_alpha": round(float(global_pred), 4),
        },
        "performance_comparison": {
            "clip_only": round(float(np.mean(auroc_clip_only)), 2),
            "dino_only": round(float(np.mean(auroc_dino_only)), 2),
            "fixed_065": round(float(np.mean(auroc_fixed065)), 2),
            "predicted_per_cat": round(float(np.mean(auroc_predicted)), 2),
            "oracle_per_cat": round(float(np.mean(auroc_oracle)), 2),
        },
        "regret": {
            "fixed_065_mean": round(float(np.mean(regret_065)), 3),
            "fixed_065_max": round(float(np.max(regret_065)), 3),
            "predicted_mean": round(float(np.mean(regret_pred)), 3),
            "predicted_max": round(float(np.max(regret_pred)), 3),
        }
    }

    out_path = os.path.join(os.path.dirname(BASE_DIR), "results", "fine_grained_alpha_sweep.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
