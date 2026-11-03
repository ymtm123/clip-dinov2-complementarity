"""
Per-Category Alpha Analysis: Is the inverse error weighting formula
practically useful for per-category weight selection?

Key question: If a practitioner knows their category, can the formula
give them a better alpha than the fixed 0.65?
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
    """Load BTAD DINOv2 scores (uses ok/ko instead of good/bad)."""
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
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


def main():
    print("=" * 100)
    print("  PER-CATEGORY ALPHA ANALYSIS: Is the formula practically useful?")
    print("=" * 100)

    # Load and align all datasets
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
            datasets['MVTec'] = align_samples(clip_data, dino_data)

    # VisA
    clip_visa = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv")
    dino_visa = os.path.join(BASE_DIR, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv")
    if os.path.exists(clip_visa) and os.path.exists(dino_visa):
        clip_data = load_anomalyclip_scores(clip_visa)
        dino_data = load_anomalydino_scores(dino_visa)
        datasets['VisA'] = align_samples(clip_data, dino_data)

    # BTAD
    clip_btad = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv")
    dino_btad = os.path.join(BASE_DIR, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv")
    if os.path.exists(clip_btad) and os.path.exists(dino_btad):
        clip_data = load_anomalyclip_scores(clip_btad)
        dino_data = load_btad_dino_scores(dino_btad)
        datasets['BTAD'] = align_samples(clip_data, dino_data)
    else:
        print(f"[WARN] BTAD scores not found -- dataset will be EXCLUDED "
              f"(run would cover only 27 categories):\n  clip={clip_btad}\n  dino={dino_btad}")

    # Alpha steps at 0.01 granularity
    alpha_steps = [i / 100.0 for i in range(0, 101, 1)]

    # =====================================================
    # Per-category detailed analysis
    # =====================================================
    rows = []

    for ds_name, aligned in datasets.items():
        for obj in sorted(aligned.keys()):
            d = aligned[obj]
            labels = d['labels']
            if len(np.unique(labels)) < 2:
                continue

            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')

            # AUROC for each alpha
            aurocs = {}
            for alpha in alpha_steps:
                fused = (1 - alpha) * clip_norm + alpha * dino_norm
                aurocs[alpha] = compute_auroc(labels, fused)

            # Oracle alpha (best at 0.01)
            a_oracle = max(aurocs, key=aurocs.get)
            auroc_oracle = aurocs[a_oracle]

            # Fixed alpha=0.65
            auroc_065 = aurocs[0.65]

            # Inverse error weighting prediction
            auroc_clip = compute_auroc(labels, d['clip_scores'])
            auroc_dino = compute_auroc(labels, d['dino_scores'])
            err_c = max(1 - auroc_clip / 100, 1e-6)
            err_d = max(1 - auroc_dino / 100, 1e-6)
            a_pred = (1 / err_d) / (1 / err_c + 1 / err_d)

            # Clamp prediction to [0, 1]
            a_pred = max(0.0, min(1.0, a_pred))

            # AUROC with predicted alpha
            fused_pred = (1 - a_pred) * clip_norm + a_pred * dino_norm
            auroc_pred = compute_auroc(labels, fused_pred)

            # Sensitivity: how much AUROC changes per 0.01 alpha around oracle
            # (measures flatness of the surface)
            if a_oracle > 0.05 and a_oracle < 0.95:
                sensitivity = abs(aurocs[round(a_oracle + 0.05, 2)] - aurocs[round(a_oracle - 0.05, 2)]) / 0.10
            else:
                sensitivity = 0

            rows.append({
                'dataset': ds_name,
                'category': obj,
                'a_oracle': a_oracle,
                'a_pred': round(a_pred, 3),
                'a_err': round(abs(a_pred - a_oracle), 3),
                'auroc_clip': round(auroc_clip, 2),
                'auroc_dino': round(auroc_dino, 2),
                'auroc_065': round(auroc_065, 2),
                'auroc_pred': round(auroc_pred, 2),
                'auroc_oracle': round(auroc_oracle, 2),
                'gain_pred_vs_065': round(auroc_pred - auroc_065, 3),
                'gain_oracle_vs_065': round(auroc_oracle - auroc_065, 3),
                'sensitivity': round(sensitivity, 2),
            })

    # =====================================================
    # Print detailed table
    # =====================================================
    print(f"\n{'=' * 100}")
    print(f"  TABLE: Per-Category Comparison")
    print(f"{'=' * 100}")
    print(f"\n  {'Category':<14} {'a_orc':>5} {'a_pred':>6} {'|da|':>5} "
          f"{'CLIP':>6} {'DINO':>6} {'a=.65':>6} {'a=pred':>6} {'a=orc':>6} "
          f"{'pred-0.65':>9} {'orc-0.65':>9}")
    print(f"  {'-'*14} {'-'*5} {'-'*6} {'-'*5} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*9} {'-'*9}")

    n_pred_better = 0
    n_pred_worse = 0
    n_pred_same = 0

    for r in sorted(rows, key=lambda x: -abs(x['gain_pred_vs_065'])):
        gain_str = f"{r['gain_pred_vs_065']:+.3f}"
        orc_str = f"{r['gain_oracle_vs_065']:+.3f}"
        marker = ""
        if r['gain_pred_vs_065'] > 0.01:
            marker = " (+)"
            n_pred_better += 1
        elif r['gain_pred_vs_065'] < -0.01:
            marker = " (-)"
            n_pred_worse += 1
        else:
            n_pred_same += 1

        print(f"  {r['category']:<14} {r['a_oracle']:5.2f} {r['a_pred']:6.3f} {r['a_err']:5.3f} "
              f"{r['auroc_clip']:6.1f} {r['auroc_dino']:6.1f} {r['auroc_065']:6.1f} "
              f"{r['auroc_pred']:6.1f} {r['auroc_oracle']:6.1f} "
              f"{gain_str:>9}{marker} {orc_str:>9}")

    print(f"\n  Summary:")
    print(f"    Categories where predicted α beats fixed 0.65:  {n_pred_better}/{len(rows)}")
    print(f"    Categories where predicted α hurts vs fixed:    {n_pred_worse}/{len(rows)}")
    print(f"    Categories where difference < 0.01pp:           {n_pred_same}/{len(rows)}")

    # =====================================================
    # Aggregate statistics
    # =====================================================
    gains_pred = [r['gain_pred_vs_065'] for r in rows]
    gains_oracle = [r['gain_oracle_vs_065'] for r in rows]
    a_errs = [r['a_err'] for r in rows]

    print(f"\n{'=' * 100}")
    print(f"  AGGREGATE STATISTICS")
    print(f"{'=' * 100}")

    print(f"\n  Alpha prediction error |a_pred - a_oracle|:")
    print(f"    Mean:    {np.mean(a_errs):.3f}")
    print(f"    Median:  {np.median(a_errs):.3f}")
    print(f"    < 0.05:  {sum(1 for e in a_errs if e < 0.05)}/{len(a_errs)} categories")
    print(f"    < 0.10:  {sum(1 for e in a_errs if e < 0.10)}/{len(a_errs)} categories")
    print(f"    < 0.15:  {sum(1 for e in a_errs if e < 0.15)}/{len(a_errs)} categories")
    print(f"    > 0.20:  {sum(1 for e in a_errs if e > 0.20)}/{len(a_errs)} categories")

    print(f"\n  AUROC gain of predicted α over fixed 0.65:")
    print(f"    Mean:    {np.mean(gains_pred):+.3f}pp")
    print(f"    Median:  {np.median(gains_pred):+.3f}pp")
    print(f"    Max improvement:  {max(gains_pred):+.3f}pp")
    print(f"    Max degradation:  {min(gains_pred):+.3f}pp")

    print(f"\n  AUROC gain of oracle α over fixed 0.65:")
    print(f"    Mean:    {np.mean(gains_oracle):+.3f}pp")
    print(f"    Max:     {max(gains_oracle):+.3f}pp")

    # =====================================================
    # The critical question: is the formula PRACTICALLY useful?
    # =====================================================
    print(f"\n{'=' * 100}")
    print(f"  PRACTICAL ASSESSMENT")
    print(f"{'=' * 100}")

    # Group by error magnitude
    print(f"\n  Categories by alpha prediction error:")
    for threshold, label in [(0.05, "excellent"), (0.10, "good"), (0.15, "acceptable"), (0.20, "marginal"), (1.0, "poor")]:
        cats = [r for r in rows if r['a_err'] <= threshold]
        if cats:
            prev_threshold = [0.05, 0.10, 0.15, 0.20, 1.0]
            idx = [0.05, 0.10, 0.15, 0.20, 1.0].index(threshold)
            lo = prev_threshold[idx - 1] if idx > 0 else 0
            in_range = [r for r in rows if lo < r['a_err'] <= threshold]
            if in_range:
                cat_names = [r['category'] for r in in_range]
                print(f"    |da| in ({lo:.2f}, {threshold:.2f}] ({label}): {', '.join(cat_names)}")

    # The key insight: does alpha error translate to AUROC loss?
    print(f"\n  Relationship between alpha error and AUROC impact:")
    print(f"  (Large α error doesn't always mean large AUROC loss due to flat oracle surface)")
    print()

    # Sort by alpha error, show corresponding AUROC impact
    for r in sorted(rows, key=lambda x: -x['a_err']):
        if r['a_err'] > 0.15:
            print(f"    {r['category']:<14}: |da|={r['a_err']:.3f}, "
                  f"AUROC impact={r['gain_pred_vs_065']:+.3f}pp "
                  f"(oracle surface flat? {'yes' if abs(r['gain_oracle_vs_065']) < 0.5 else 'no'})")

    # =====================================================
    # Practical scenario: "I have product X, what alpha should I use?"
    # =====================================================
    print(f"\n{'=' * 100}")
    print(f"  SCENARIO: I have a product category. Which strategy is best?")
    print(f"{'=' * 100}")

    # For each category, compare strategies:
    # A: Use fixed alpha=0.65 (no information needed)
    # B: Use formula-predicted alpha (needs AUROC of both models = needs labels)
    # C: Use oracle alpha (needs exhaustive sweep with labels)

    print(f"\n  {'Category':<14} {'Best strategy':>14} {'Why':>50}")
    print(f"  {'-'*14} {'-'*14} {'-'*50}")

    for r in sorted(rows, key=lambda x: x['category']):
        if r['gain_oracle_vs_065'] < 0.1:
            best = "Fixed α=0.65"
            why = f"Oracle surface flat (oracle only +{r['gain_oracle_vs_065']:.2f}pp)"
        elif r['gain_pred_vs_065'] > 0 and r['gain_pred_vs_065'] >= r['gain_oracle_vs_065'] * 0.5:
            best = "Formula"
            why = f"Captures {r['gain_pred_vs_065']/max(r['gain_oracle_vs_065'],0.001)*100:.0f}% of oracle gain"
        else:
            best = "Need oracle"
            why = f"Formula: {r['gain_pred_vs_065']:+.2f}pp, oracle: +{r['gain_oracle_vs_065']:.2f}pp"
        print(f"  {r['category']:<14} {best:>14} {why:>50}")

    # Count
    n_flat = sum(1 for r in rows if r['gain_oracle_vs_065'] < 0.1)
    n_formula_ok = sum(1 for r in rows if r['gain_oracle_vs_065'] >= 0.1 and
                       r['gain_pred_vs_065'] > 0 and
                       r['gain_pred_vs_065'] >= r['gain_oracle_vs_065'] * 0.5)
    n_need_oracle = len(rows) - n_flat - n_formula_ok

    print(f"\n  Summary:")
    print(f"    Fixed 0.65 is sufficient (flat surface): {n_flat}/{len(rows)} ({n_flat/len(rows)*100:.0f}%)")
    print(f"    Formula gives useful improvement:        {n_formula_ok}/{len(rows)} ({n_formula_ok/len(rows)*100:.0f}%)")
    print(f"    Formula insufficient, need oracle:       {n_need_oracle}/{len(rows)} ({n_need_oracle/len(rows)*100:.0f}%)")

    # Save
    output = {
        "per_category": rows,
        "aggregate": {
            "alpha_error_mean": round(float(np.mean(a_errs)), 3),
            "alpha_error_median": round(float(np.median(a_errs)), 3),
            "auroc_gain_pred_mean": round(float(np.mean(gains_pred)), 3),
            "auroc_gain_oracle_mean": round(float(np.mean(gains_oracle)), 3),
            "n_pred_better": n_pred_better,
            "n_pred_worse": n_pred_worse,
            "n_pred_same": n_pred_same,
            "n_total": len(rows),
            "n_surface_flat": n_flat,
            "n_formula_useful": n_formula_ok,
            "n_need_oracle": n_need_oracle,
        }
    }

    out_path = os.path.join(os.path.dirname(BASE_DIR), "results", "per_category_alpha_analysis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
