"""
Experiment 13: Latent Capacity Gap Analysis
=============================================

Goal: Quantify per-category "latent capacity gap" - how much more
each method extracts from its backbone compared to kNN baseline.

kNN AUROC measures backbone capacity.
Native AUROC measures backbone + method capacity.
Gap = Native - kNN = method extraction efficiency.

Key questions:
1. Which categories have the largest method premium for each model?
2. Is the method premium correlated with backbone type (texture vs object)?
3. Does the method premium correlate with the backbone's relative weakness?
   (i.e., does AnomalyDINO compensate more for DINOv2's weak categories?)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

def main():
    src_dir = Path(__file__).parent

    # Load kNN backbone isolation results
    with open(src_dir / 'deep_analysis_results.json') as f:
        knn_data = json.load(f)

    # Load comprehensive evaluation (native mode) results
    with open(src_dir / 'comprehensive_eval_results.json') as f:
        eval_data = json.load(f)

    # Load k-shot data for kNN baselines
    with open(src_dir / 'deep_analysis_cycle3_results.json') as f:
        c3 = json.load(f)

    kshot = c3['experiment_8_kshot']

    texture_cats = ['carpet', 'grid', 'leather', 'tile', 'wood']
    object_cats = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                   'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    # Get per-category native AUROC
    native_clip = eval_data['MVTec AD']['CLIP only']['per_category']
    native_dino = eval_data['MVTec AD']['DINOv2 only']['per_category']

    # Get per-category kNN AUROC (k=1)
    knn_clip = {}
    knn_dino = {}
    for cat in kshot:
        if '1' in kshot[cat]:
            knn_clip[cat] = kshot[cat]['1']['clip_auroc']
            knn_dino[cat] = kshot[cat]['1']['dino_auroc']

    # ============================================================
    # Analysis 1: Per-category method premium
    # ============================================================
    print("=" * 90)
    print("Analysis 1: Per-Category Method Premium (Native - kNN)")
    print("  CLIP premium = AnomalyCLIP(0-shot) - CLIP kNN(1-shot)")
    print("  DINOv2 premium = AnomalyDINO(1-shot) - DINOv2 kNN(1-shot)")
    print("=" * 90)

    common_cats = sorted(set(knn_clip.keys()) & set(native_clip.keys()))

    results = []
    print(f"\n{'Category':15s} {'CLIP kNN':>9s} {'CLIP nat':>9s} {'CLIP+':>7s} "
          f"{'DINO kNN':>9s} {'DINO nat':>9s} {'DINO+':>7s} {'Type':>6s}")
    print("-" * 80)

    clip_premiums = []
    dino_premiums = []
    clip_premiums_tex = []
    clip_premiums_obj = []
    dino_premiums_tex = []
    dino_premiums_obj = []

    for cat in common_cats:
        clip_knn = knn_clip[cat]
        clip_nat = native_clip[cat]
        dino_knn = knn_dino[cat]
        dino_nat = native_dino[cat]

        clip_premium = clip_nat - clip_knn
        dino_premium = dino_nat - dino_knn

        cat_type = 'Tex' if cat in texture_cats else 'Obj'

        print(f"{cat:15s} {clip_knn:8.1f}% {clip_nat:8.1f}% {clip_premium:+6.1f} "
              f"{dino_knn:8.1f}% {dino_nat:8.1f}% {dino_premium:+6.1f} {cat_type:>6s}")

        clip_premiums.append(clip_premium)
        dino_premiums.append(dino_premium)

        if cat in texture_cats:
            clip_premiums_tex.append(clip_premium)
            dino_premiums_tex.append(dino_premium)
        else:
            clip_premiums_obj.append(clip_premium)
            dino_premiums_obj.append(dino_premium)

        results.append({
            'category': cat,
            'type': cat_type,
            'clip_knn': clip_knn,
            'clip_native': clip_nat,
            'clip_premium': clip_premium,
            'dino_knn': dino_knn,
            'dino_native': dino_nat,
            'dino_premium': dino_premium,
        })

    print(f"\n{'Mean':15s} {'':>9s} {'':>9s} {np.mean(clip_premiums):+6.1f} "
          f"{'':>9s} {'':>9s} {np.mean(dino_premiums):+6.1f}")

    # ============================================================
    # Analysis 2: Method premium by category type
    # ============================================================
    print("\n" + "=" * 90)
    print("Analysis 2: Method Premium by Category Type")
    print("=" * 90)

    print(f"\n{'':15s} {'CLIP premium':>15s} {'DINOv2 premium':>15s} {'Diff':>8s}")
    print("-" * 55)
    print(f"{'Texture':15s} {np.mean(clip_premiums_tex):+14.1f} {np.mean(dino_premiums_tex):+14.1f} "
          f"{np.mean(dino_premiums_tex) - np.mean(clip_premiums_tex):+7.1f}")
    print(f"{'Object':15s} {np.mean(clip_premiums_obj):+14.1f} {np.mean(dino_premiums_obj):+14.1f} "
          f"{np.mean(dino_premiums_obj) - np.mean(clip_premiums_obj):+7.1f}")
    print(f"{'All':15s} {np.mean(clip_premiums):+14.1f} {np.mean(dino_premiums):+14.1f} "
          f"{np.mean(dino_premiums) - np.mean(clip_premiums):+7.1f}")

    # t-test: is DINOv2 premium significantly larger than CLIP premium?
    t, p = stats.ttest_rel(dino_premiums, clip_premiums)
    print(f"\nPaired t-test (DINOv2 premium > CLIP premium):")
    print(f"  t={t:.3f}, p={p:.4f} {'*' if p < 0.05 else 'n.s.'}")

    # ============================================================
    # Analysis 3: Does method premium compensate for weakness?
    # ============================================================
    print("\n" + "=" * 90)
    print("Analysis 3: Does Method Premium Compensate for Backbone Weakness?")
    print("=" * 90)

    # For CLIP: correlation between CLIP kNN rank and CLIP premium
    clip_knn_vals = [r['clip_knn'] for r in results]
    clip_prem_vals = [r['clip_premium'] for r in results]
    r_clip, p_clip = stats.pearsonr(clip_knn_vals, clip_prem_vals)
    print(f"\nCLIP: r(kNN AUROC, method premium) = {r_clip:.3f}, p={p_clip:.4f}")
    if r_clip < 0:
        print("  -> Negative: AnomalyCLIP compensates more for weak backbone categories")
    else:
        print("  -> Positive: AnomalyCLIP amplifies backbone strengths")

    # For DINOv2
    dino_knn_vals = [r['dino_knn'] for r in results]
    dino_prem_vals = [r['dino_premium'] for r in results]
    r_dino, p_dino = stats.pearsonr(dino_knn_vals, dino_prem_vals)
    print(f"\nDINOv2: r(kNN AUROC, method premium) = {r_dino:.3f}, p={p_dino:.4f}")
    if r_dino < 0:
        print("  -> Negative: AnomalyDINO compensates more for weak backbone categories")
    else:
        print("  -> Positive: AnomalyDINO amplifies backbone strengths")

    # ============================================================
    # Analysis 4: Method premium vs native AUROC gap
    # ============================================================
    print("\n" + "=" * 90)
    print("Analysis 4: Method Premium vs Native Performance Gap")
    print("=" * 90)

    native_gaps = [r['dino_native'] - r['clip_native'] for r in results]
    premium_gaps = [r['dino_premium'] - r['clip_premium'] for r in results]

    r_gap, p_gap = stats.pearsonr(native_gaps, premium_gaps)
    print(f"\nr(native gap, premium gap) = {r_gap:.3f}, p={p_gap:.4f}")
    print(f"  -> {'Native performance gap is driven by method premium differences' if abs(r_gap) > 0.5 else 'Native gap is NOT primarily driven by method premium'}")

    # ============================================================
    # Analysis 5: Top/bottom categories by method premium
    # ============================================================
    print("\n" + "=" * 90)
    print("Analysis 5: Most/Least Extracted Categories")
    print("=" * 90)

    # Sort by CLIP premium
    sorted_clip = sorted(results, key=lambda r: r['clip_premium'], reverse=True)
    print("\nTop 5 CLIP premium (AnomalyCLIP extracts the most):")
    for r in sorted_clip[:5]:
        print(f"  {r['category']:15s} kNN={r['clip_knn']:5.1f}% -> native={r['clip_native']:5.1f}% premium={r['clip_premium']:+5.1f}")

    print("\nBottom 5 CLIP premium (AnomalyCLIP extracts the least):")
    for r in sorted_clip[-5:]:
        print(f"  {r['category']:15s} kNN={r['clip_knn']:5.1f}% -> native={r['clip_native']:5.1f}% premium={r['clip_premium']:+5.1f}")

    # Sort by DINOv2 premium
    sorted_dino = sorted(results, key=lambda r: r['dino_premium'], reverse=True)
    print("\nTop 5 DINOv2 premium (AnomalyDINO extracts the most):")
    for r in sorted_dino[:5]:
        print(f"  {r['category']:15s} kNN={r['dino_knn']:5.1f}% -> native={r['dino_native']:5.1f}% premium={r['dino_premium']:+5.1f}")

    print("\nBottom 5 DINOv2 premium (AnomalyDINO extracts the least):")
    for r in sorted_dino[-5:]:
        print(f"  {r['category']:15s} kNN={r['dino_knn']:5.1f}% -> native={r['dino_native']:5.1f}% premium={r['dino_premium']:+5.1f}")

    # ============================================================
    # Analysis 6: Untapped potential - where is CLIP's capacity wasted?
    # ============================================================
    print("\n" + "=" * 90)
    print("Analysis 6: Untapped Potential")
    print("  Categories where kNN CLIP > kNN DINOv2 but native DINOv2 > native CLIP")
    print("  = AnomalyDINO overcomes DINOv2's backbone deficit")
    print("=" * 90)

    for r in results:
        knn_clip_better = r['clip_knn'] > r['dino_knn']
        native_dino_better = r['dino_native'] > r['clip_native']

        if knn_clip_better and native_dino_better:
            print(f"  {r['category']:15s}: kNN CLIP={r['clip_knn']:.1f}% > kNN DINOv2={r['dino_knn']:.1f}% "
                  f"BUT native DINOv2={r['dino_native']:.1f}% > native CLIP={r['clip_native']:.1f}% "
                  f"(method overcome: {r['dino_premium'] - r['clip_premium']:+.1f})")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        'experiment_13_latent_capacity': {
            'per_category': results,
            'aggregates': {
                'clip_premium_mean': round(np.mean(clip_premiums), 2),
                'dino_premium_mean': round(np.mean(dino_premiums), 2),
                'clip_premium_texture': round(np.mean(clip_premiums_tex), 2),
                'clip_premium_object': round(np.mean(clip_premiums_obj), 2),
                'dino_premium_texture': round(np.mean(dino_premiums_tex), 2),
                'dino_premium_object': round(np.mean(dino_premiums_obj), 2),
                'paired_ttest_p': round(p, 4),
            },
            'compensation': {
                'clip_r': round(r_clip, 4),
                'clip_p': round(p_clip, 4),
                'dino_r': round(r_dino, 4),
                'dino_p': round(p_dino, 4),
            },
        }
    }

    output_path = src_dir / 'latent_capacity_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
