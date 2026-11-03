"""
Experiment 14: Cross-Dataset Method Premium & Difficulty Analysis
=================================================================

Goal: Extend Experiment 13 (MVTec-only latent capacity gap) to both
MVTec AD and VisA datasets, testing whether:

1. The compensatory pattern (r=-0.92, r=-0.62) generalizes across datasets
2. Negative premium categories exist in VisA
3. Category difficulty (absolute AUROC) predicts fusion benefit
4. Method premium patterns are consistent across datasets
5. The "fusion as insurance" message is supported by difficulty analysis

Key insight: If the compensatory pattern holds across datasets,
it becomes a general property of VFM anomaly detection methods,
not a dataset-specific artifact.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats


def main():
    src_dir = Path(__file__).parent

    # Load kNN backbone isolation results (both MVTec and VisA)
    with open(src_dir / 'clip_knn_results.json') as f:
        knn_data = json.load(f)

    # Load native mode results (both MVTec and VisA)
    with open(src_dir / 'comprehensive_eval_results.json') as f:
        eval_data = json.load(f)

    # Load fusion/oracle results for difficulty analysis
    with open(src_dir / 'category_taxonomy_results.json') as f:
        taxonomy_data = json.load(f)

    # ============================================================
    # Part A: Cross-Dataset Method Premium Analysis
    # ============================================================
    print("=" * 95)
    print("EXPERIMENT 14: Cross-Dataset Method Premium & Difficulty Analysis")
    print("=" * 95)

    datasets = {
        'MVTec AD': {
            'knn': knn_data['mvtec'],
            'native_clip': eval_data['MVTec AD']['CLIP only']['per_category'],
            'native_dino': eval_data['MVTec AD']['DINOv2 only']['per_category'],
        },
        'VisA': {
            'knn': knn_data['visa'],
            'native_clip': eval_data['VisA']['CLIP only']['per_category'],
            'native_dino': eval_data['VisA']['DINOv2 only']['per_category'],
        }
    }

    all_results = []

    for ds_name, ds in datasets.items():
        print(f"\n{'=' * 90}")
        print(f"Analysis 1-{ds_name}: Per-Category Method Premium")
        print(f"  CLIP premium = AnomalyCLIP(0-shot) - CLIP kNN(1-shot)")
        print(f"  DINOv2 premium = AnomalyDINO(1-shot) - DINOv2 kNN(1-shot)")
        print(f"{'=' * 90}")

        common_cats = sorted(set(ds['knn'].keys()) &
                             set(ds['native_clip'].keys()) &
                             set(ds['native_dino'].keys()))

        clip_premiums = []
        dino_premiums = []
        cat_results = []

        print(f"\n{'Category':15s} {'CLIP kNN':>9s} {'CLIP nat':>9s} {'CLIP+':>7s} "
              f"{'DINO kNN':>9s} {'DINO nat':>9s} {'DINO+':>7s}")
        print("-" * 75)

        for cat in common_cats:
            clip_knn = ds['knn'][cat]['clip_knn']
            dino_knn = ds['knn'][cat]['dinov2_knn']
            clip_nat = ds['native_clip'][cat]
            dino_nat = ds['native_dino'][cat]

            clip_premium = clip_nat - clip_knn
            dino_premium = dino_nat - dino_knn

            clip_premiums.append(clip_premium)
            dino_premiums.append(dino_premium)

            print(f"{cat:15s} {clip_knn:8.1f}% {clip_nat:8.1f}% {clip_premium:+6.1f} "
                  f"{dino_knn:8.1f}% {dino_nat:8.1f}% {dino_premium:+6.1f}")

            cat_results.append({
                'category': cat,
                'dataset': ds_name,
                'clip_knn': clip_knn,
                'clip_native': clip_nat,
                'clip_premium': clip_premium,
                'dino_knn': dino_knn,
                'dino_native': dino_nat,
                'dino_premium': dino_premium,
            })

        print(f"\n{'Mean':15s} {'':>9s} {'':>9s} {np.mean(clip_premiums):+6.1f} "
              f"{'':>9s} {'':>9s} {np.mean(dino_premiums):+6.1f}")

        # Compensatory correlation for this dataset
        clip_knn_vals = [r['clip_knn'] for r in cat_results]
        clip_prem_vals = [r['clip_premium'] for r in cat_results]
        dino_knn_vals = [r['dino_knn'] for r in cat_results]
        dino_prem_vals = [r['dino_premium'] for r in cat_results]

        r_clip, p_clip = stats.pearsonr(clip_knn_vals, clip_prem_vals)
        r_dino, p_dino = stats.pearsonr(dino_knn_vals, dino_prem_vals)

        print(f"\n  Compensatory pattern ({ds_name}):")
        print(f"    CLIP: r(kNN, premium) = {r_clip:.3f}, p={p_clip:.4f} "
              f"{'*' if p_clip < 0.05 else '†' if p_clip < 0.10 else 'n.s.'}")
        print(f"    DINOv2: r(kNN, premium) = {r_dino:.3f}, p={p_dino:.4f} "
              f"{'*' if p_dino < 0.05 else '†' if p_dino < 0.10 else 'n.s.'}")

        # Count negative premiums
        neg_clip = sum(1 for p in clip_premiums if p < 0)
        neg_dino = sum(1 for p in dino_premiums if p < 0)
        print(f"\n  Negative premium (wasted capacity):")
        print(f"    CLIP: {neg_clip}/{len(clip_premiums)} categories")
        print(f"    DINOv2: {neg_dino}/{len(dino_premiums)} categories")

        if neg_clip > 0:
            print(f"    CLIP neg premium categories:")
            for r in cat_results:
                if r['clip_premium'] < 0:
                    print(f"      {r['category']:15s}: kNN={r['clip_knn']:.1f}% → "
                          f"native={r['clip_native']:.1f}% (premium={r['clip_premium']:+.1f})")
        if neg_dino > 0:
            print(f"    DINOv2 neg premium categories:")
            for r in cat_results:
                if r['dino_premium'] < 0:
                    print(f"      {r['category']:15s}: kNN={r['dino_knn']:.1f}% → "
                          f"native={r['dino_native']:.1f}% (premium={r['dino_premium']:+.1f})")

        all_results.extend(cat_results)

    # ============================================================
    # Part B: Combined Cross-Dataset Compensatory Analysis
    # ============================================================
    print(f"\n\n{'=' * 90}")
    print("Analysis 2: Cross-Dataset Compensatory Pattern (MVTec + VisA combined)")
    print("=" * 90)

    all_clip_knn = [r['clip_knn'] for r in all_results]
    all_clip_prem = [r['clip_premium'] for r in all_results]
    all_dino_knn = [r['dino_knn'] for r in all_results]
    all_dino_prem = [r['dino_premium'] for r in all_results]

    r_clip_all, p_clip_all = stats.pearsonr(all_clip_knn, all_clip_prem)
    r_dino_all, p_dino_all = stats.pearsonr(all_dino_knn, all_dino_prem)

    print(f"\n  Combined (N={len(all_results)} categories):")
    print(f"    CLIP: r(kNN, premium) = {r_clip_all:.3f}, p={p_clip_all:.6f} "
          f"{'***' if p_clip_all < 0.001 else '**' if p_clip_all < 0.01 else '*' if p_clip_all < 0.05 else 'n.s.'}")
    print(f"    DINOv2: r(kNN, premium) = {r_dino_all:.3f}, p={p_dino_all:.6f} "
          f"{'***' if p_dino_all < 0.001 else '**' if p_dino_all < 0.01 else '*' if p_dino_all < 0.05 else 'n.s.'}")

    # Compare per-dataset correlations
    print(f"\n  Per-dataset comparison:")
    mvtec_results = [r for r in all_results if r['dataset'] == 'MVTec AD']
    visa_results = [r for r in all_results if r['dataset'] == 'VisA']

    r_clip_m, p_clip_m = stats.pearsonr(
        [r['clip_knn'] for r in mvtec_results],
        [r['clip_premium'] for r in mvtec_results])
    r_dino_m, p_dino_m = stats.pearsonr(
        [r['dino_knn'] for r in mvtec_results],
        [r['dino_premium'] for r in mvtec_results])
    r_clip_v, p_clip_v = stats.pearsonr(
        [r['clip_knn'] for r in visa_results],
        [r['clip_premium'] for r in visa_results])
    r_dino_v, p_dino_v = stats.pearsonr(
        [r['dino_knn'] for r in visa_results],
        [r['dino_premium'] for r in visa_results])

    print(f"    {'Dataset':12s} {'CLIP r':>8s} {'CLIP p':>10s} {'DINO r':>8s} {'DINO p':>10s}")
    print(f"    {'-' * 45}")
    print(f"    {'MVTec (15)':12s} {r_clip_m:+7.3f} {p_clip_m:10.4f} {r_dino_m:+7.3f} {p_dino_m:10.4f}")
    print(f"    {'VisA (12)':12s} {r_clip_v:+7.3f} {p_clip_v:10.4f} {r_dino_v:+7.3f} {p_dino_v:10.4f}")
    print(f"    {'Combined':12s} {r_clip_all:+7.3f} {p_clip_all:10.6f} {r_dino_all:+7.3f} {p_dino_all:10.6f}")

    # ============================================================
    # Part C: Category Difficulty vs Fusion Benefit
    # ============================================================
    print(f"\n\n{'=' * 90}")
    print("Analysis 3: Category Difficulty vs Fusion Benefit")
    print("=" * 90)

    # Get fusion results for difficulty analysis
    fusion_methods = {}
    for ds_key in ['MVTec AD', 'VisA']:
        for method_key in eval_data[ds_key]:
            if method_key not in fusion_methods:
                fusion_methods[method_key] = {}
            fusion_methods[method_key][ds_key] = eval_data[ds_key][method_key]['per_category']

    # Compute per-category difficulty and fusion benefit
    difficulty_analysis = []

    for r in all_results:
        ds = r['dataset']
        cat = r['category']

        best_single = max(r['clip_native'], r['dino_native'])
        worst_single = min(r['clip_native'], r['dino_native'])
        gap = abs(r['clip_native'] - r['dino_native'])

        # Get equal fusion AUROC
        if cat in fusion_methods.get('Equal (a=0.5)', {}).get(ds, {}):
            equal_fusion = fusion_methods['Equal (a=0.5)'][ds][cat]
        else:
            equal_fusion = None

        # Get oracle AUROC
        if cat in fusion_methods.get('Oracle', {}).get(ds, {}):
            oracle = fusion_methods['Oracle'][ds][cat]
        else:
            oracle = None

        if equal_fusion is not None:
            fusion_gain = equal_fusion - best_single  # vs best single model
            fusion_gain_worst = equal_fusion - worst_single  # vs worst single
        else:
            fusion_gain = None
            fusion_gain_worst = None

        difficulty_analysis.append({
            'category': cat,
            'dataset': ds,
            'best_single': best_single,
            'worst_single': worst_single,
            'gap': gap,
            'equal_fusion': equal_fusion,
            'oracle': oracle,
            'fusion_gain_vs_best': fusion_gain,
            'fusion_gain_vs_worst': fusion_gain_worst,
        })

    # Correlation: difficulty (best_single AUROC) vs fusion gain
    valid = [d for d in difficulty_analysis if d['fusion_gain_vs_best'] is not None]

    best_singles = [d['best_single'] for d in valid]
    fusion_gains = [d['fusion_gain_vs_best'] for d in valid]
    gaps = [d['gap'] for d in valid]

    r_diff_fusion, p_diff_fusion = stats.pearsonr(best_singles, fusion_gains)
    r_gap_fusion, p_gap_fusion = stats.pearsonr(gaps, fusion_gains)

    print(f"\n  N = {len(valid)} categories")
    print(f"\n  Correlation: best single AUROC vs fusion gain over best single:")
    print(f"    r = {r_diff_fusion:.3f}, p = {p_diff_fusion:.4f} "
          f"{'*' if p_diff_fusion < 0.05 else 'n.s.'}")
    if r_diff_fusion < 0:
        print(f"    → Harder categories benefit MORE from fusion")
    else:
        print(f"    → Easier categories benefit MORE from fusion")

    print(f"\n  Correlation: CLIP-DINOv2 gap vs fusion gain over best single:")
    print(f"    r = {r_gap_fusion:.3f}, p = {p_gap_fusion:.4f} "
          f"{'*' if p_gap_fusion < 0.05 else 'n.s.'}")

    # Quintile analysis by difficulty
    sorted_by_diff = sorted(valid, key=lambda d: d['best_single'])
    n = len(sorted_by_diff)
    quintile_size = n // 5

    print(f"\n  Difficulty quintile analysis (sorted by best single AUROC):")
    print(f"  {'Quintile':>10s} {'Best single':>12s} {'Gap':>6s} {'Fusion gain':>12s} {'N':>4s}")
    print(f"  {'-' * 50}")

    for q in range(5):
        start = q * quintile_size
        end = start + quintile_size if q < 4 else n
        quintile = sorted_by_diff[start:end]

        mean_best = np.mean([d['best_single'] for d in quintile])
        mean_gap = np.mean([d['gap'] for d in quintile])
        mean_gain = np.mean([d['fusion_gain_vs_best'] for d in quintile])

        label = ['Hardest', 'Hard', 'Medium', 'Easy', 'Easiest'][q]
        print(f"  {label:>10s} {mean_best:11.1f}% {mean_gap:5.1f} {mean_gain:+11.2f}pp {len(quintile):>4d}")

    # ============================================================
    # Part D: Wrong Model Risk vs Difficulty
    # ============================================================
    print(f"\n\n{'=' * 90}")
    print("Analysis 4: Wrong Model Penalty vs Category Difficulty")
    print("=" * 90)

    wrong_model_penalties = []
    for d in valid:
        penalty = d['best_single'] - d['worst_single']  # = gap
        wrong_model_penalties.append(penalty)

    # Correlation: difficulty vs wrong model penalty
    r_diff_penalty, p_diff_penalty = stats.pearsonr(best_singles, wrong_model_penalties)
    print(f"\n  r(best single AUROC, wrong model penalty) = {r_diff_penalty:.3f}, p={p_diff_penalty:.4f}")

    if r_diff_penalty < 0:
        print(f"  → Harder categories have LARGER wrong model penalties")
    else:
        print(f"  → Easier categories have LARGER wrong model penalties")

    # Quintile analysis
    print(f"\n  {'Quintile':>10s} {'Best single':>12s} {'Penalty':>8s} {'Max penalty':>12s}")
    print(f"  {'-' * 48}")

    for q in range(5):
        start = q * quintile_size
        end = start + quintile_size if q < 4 else n
        quintile = sorted_by_diff[start:end]

        mean_best = np.mean([d['best_single'] for d in quintile])
        mean_penalty = np.mean([d['gap'] for d in quintile])
        max_penalty = max([d['gap'] for d in quintile])

        label = ['Hardest', 'Hard', 'Medium', 'Easy', 'Easiest'][q]
        print(f"  {label:>10s} {mean_best:11.1f}% {mean_penalty:7.1f}pp {max_penalty:11.1f}pp")

    # ============================================================
    # Part E: Method Premium vs Fusion Benefit
    # ============================================================
    print(f"\n\n{'=' * 90}")
    print("Analysis 5: Method Premium Asymmetry vs Fusion Benefit")
    print("=" * 90)

    # For each category, compute premium asymmetry = |DINO premium - CLIP premium|
    # And correlate with fusion gain
    premium_asymmetries = []
    premium_ratios = []

    for d_a, r in zip(difficulty_analysis, all_results):
        if d_a['fusion_gain_vs_best'] is not None:
            asym = abs(r['dino_premium'] - r['clip_premium'])
            premium_asymmetries.append(asym)
            premium_ratios.append(r['dino_premium'] - r['clip_premium'])

    r_asym_fusion, p_asym_fusion = stats.pearsonr(premium_asymmetries, fusion_gains)
    print(f"\n  r(premium asymmetry, fusion gain) = {r_asym_fusion:.3f}, p={p_asym_fusion:.4f}")

    r_ratio_fusion, p_ratio_fusion = stats.pearsonr(premium_ratios, fusion_gains)
    print(f"  r(DINOv2-CLIP premium diff, fusion gain) = {r_ratio_fusion:.3f}, p={p_ratio_fusion:.4f}")

    # ============================================================
    # Part F: Cross-Dataset Method Premium Consistency
    # ============================================================
    print(f"\n\n{'=' * 90}")
    print("Analysis 6: Method Premium Pattern Summary")
    print("=" * 90)

    # Summary statistics
    for ds_name in ['MVTec AD', 'VisA']:
        ds_results = [r for r in all_results if r['dataset'] == ds_name]
        clip_p = [r['clip_premium'] for r in ds_results]
        dino_p = [r['dino_premium'] for r in ds_results]

        print(f"\n  {ds_name}:")
        print(f"    CLIP premium:  mean={np.mean(clip_p):+6.2f}, "
              f"std={np.std(clip_p):5.2f}, min={min(clip_p):+6.1f}, max={max(clip_p):+6.1f}")
        print(f"    DINOv2 premium: mean={np.mean(dino_p):+6.2f}, "
              f"std={np.std(dino_p):5.2f}, min={min(dino_p):+6.1f}, max={max(dino_p):+6.1f}")

        # Paired test within dataset
        t, p = stats.ttest_rel(dino_p, clip_p)
        print(f"    Paired t-test (DINO > CLIP): t={t:.3f}, p={p:.4f} "
              f"{'*' if p < 0.05 else 'n.s.'}")

    # Combined
    all_clip_p = [r['clip_premium'] for r in all_results]
    all_dino_p = [r['dino_premium'] for r in all_results]
    t_all, p_all = stats.ttest_rel(all_dino_p, all_clip_p)

    print(f"\n  Combined (N={len(all_results)}):")
    print(f"    CLIP premium:  mean={np.mean(all_clip_p):+6.2f}")
    print(f"    DINOv2 premium: mean={np.mean(all_dino_p):+6.2f}")
    print(f"    Paired t-test: t={t_all:.3f}, p={p_all:.4f} "
          f"{'*' if p_all < 0.05 else 'n.s.'}")

    # ============================================================
    # Part G: Which categories does AnomalyCLIP "waste" CLIP's potential?
    # ============================================================
    print(f"\n\n{'=' * 90}")
    print("Analysis 7: CLIP Capacity Waste (negative premium categories)")
    print("=" * 90)

    neg_premium = [r for r in all_results if r['clip_premium'] < 0]
    neg_premium.sort(key=lambda r: r['clip_premium'])

    print(f"\n  Categories where AnomalyCLIP HURTS CLIP backbone ({len(neg_premium)}/{len(all_results)}):")
    print(f"  {'Category':15s} {'Dataset':>8s} {'CLIP kNN':>9s} {'CLIP nat':>9s} {'Premium':>8s} "
          f"{'DINO nat':>9s} {'DINOv2 wins?':>12s}")
    print(f"  {'-' * 75}")

    for r in neg_premium:
        dino_wins = "YES" if r['dino_native'] > r['clip_native'] else "no"
        print(f"  {r['category']:15s} {r['dataset']:>8s} {r['clip_knn']:8.1f}% {r['clip_native']:8.1f}% "
              f"{r['clip_premium']:+7.1f} {r['dino_native']:8.1f}% {dino_wins:>12s}")

    # Characterize: are these the same categories where CLIP kNN > DINOv2 kNN?
    print(f"\n  Of these {len(neg_premium)} negative-premium categories:")
    clip_knn_better = sum(1 for r in neg_premium if r['clip_knn'] > r['dino_knn'])
    print(f"    CLIP kNN > DINOv2 kNN: {clip_knn_better}/{len(neg_premium)}")
    print(f"    → {'AnomalyCLIP is wasting CLIPs backbone advantage' if clip_knn_better > len(neg_premium)/2 else 'Mixed pattern'}")

    # ============================================================
    # Part H: Method-Driven Backbone Reversals
    # ============================================================
    print(f"\n\n{'=' * 90}")
    print("Analysis 8: Method-Driven Backbone Reversals (cross-dataset)")
    print("=" * 90)

    reversals = []
    for r in all_results:
        knn_clip_better = r['clip_knn'] > r['dino_knn']
        native_dino_better = r['dino_native'] > r['clip_native']

        if knn_clip_better and native_dino_better:
            method_overcome = r['dino_premium'] - r['clip_premium']
            reversals.append({**r, 'method_overcome': method_overcome})

    print(f"\n  Categories where kNN CLIP > kNN DINOv2 BUT native DINOv2 > native CLIP:")
    print(f"  ({len(reversals)}/{len(all_results)} = {100*len(reversals)/len(all_results):.0f}%)\n")

    for r in sorted(reversals, key=lambda x: -x['method_overcome']):
        print(f"  {r['category']:15s} ({r['dataset']:>5s}): "
              f"kNN CLIP={r['clip_knn']:.1f}% > kNN DINO={r['dino_knn']:.1f}% "
              f"BUT native DINO={r['dino_native']:.1f}% > native CLIP={r['clip_native']:.1f}% "
              f"(method overcome: {r['method_overcome']:+.1f})")

    # Reverse direction: kNN DINOv2 > kNN CLIP BUT native CLIP > native DINOv2
    rev2 = []
    for r in all_results:
        if r['dino_knn'] > r['clip_knn'] and r['clip_native'] > r['dino_native']:
            rev2.append(r)

    print(f"\n  Reverse: kNN DINOv2 > kNN CLIP BUT native CLIP > native DINOv2:")
    print(f"  ({len(rev2)}/{len(all_results)} = {100*len(rev2)/len(all_results):.0f}%)\n")
    for r in rev2:
        print(f"  {r['category']:15s} ({r['dataset']:>5s}): "
              f"kNN DINO={r['dino_knn']:.1f}% > kNN CLIP={r['clip_knn']:.1f}% "
              f"BUT native CLIP={r['clip_native']:.1f}% > native DINO={r['dino_native']:.1f}%")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        'experiment_14_cross_dataset_premium': {
            'per_category': all_results,
            'compensatory_pattern': {
                'mvtec': {
                    'clip_r': round(r_clip_m, 4),
                    'clip_p': round(p_clip_m, 4),
                    'dino_r': round(r_dino_m, 4),
                    'dino_p': round(p_dino_m, 4),
                    'n': len(mvtec_results),
                },
                'visa': {
                    'clip_r': round(r_clip_v, 4),
                    'clip_p': round(p_clip_v, 4),
                    'dino_r': round(r_dino_v, 4),
                    'dino_p': round(p_dino_v, 4),
                    'n': len(visa_results),
                },
                'combined': {
                    'clip_r': round(r_clip_all, 4),
                    'clip_p': round(float(p_clip_all), 6),
                    'dino_r': round(r_dino_all, 4),
                    'dino_p': round(float(p_dino_all), 6),
                    'n': len(all_results),
                },
            },
            'premium_summary': {
                'mvtec': {
                    'clip_mean': round(np.mean([r['clip_premium'] for r in mvtec_results]), 2),
                    'dino_mean': round(np.mean([r['dino_premium'] for r in mvtec_results]), 2),
                },
                'visa': {
                    'clip_mean': round(np.mean([r['clip_premium'] for r in visa_results]), 2),
                    'dino_mean': round(np.mean([r['dino_premium'] for r in visa_results]), 2),
                },
                'combined': {
                    'clip_mean': round(np.mean(all_clip_p), 2),
                    'dino_mean': round(np.mean(all_dino_p), 2),
                    'paired_ttest_p': round(float(p_all), 4),
                },
            },
            'difficulty_analysis': {
                'r_difficulty_fusion': round(r_diff_fusion, 4),
                'p_difficulty_fusion': round(p_diff_fusion, 4),
                'r_gap_fusion': round(r_gap_fusion, 4),
                'p_gap_fusion': round(p_gap_fusion, 4),
                'r_difficulty_penalty': round(r_diff_penalty, 4),
                'p_difficulty_penalty': round(p_diff_penalty, 4),
            },
            'backbone_reversals': {
                'clip_to_dino': len(reversals),
                'dino_to_clip': len(rev2),
                'total': len(all_results),
            },
            'negative_premium': {
                'clip_count': len(neg_premium),
                'categories': [r['category'] + ' (' + r['dataset'] + ')' for r in neg_premium],
            },
        }
    }

    output_path = src_dir / 'cross_dataset_premium_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
