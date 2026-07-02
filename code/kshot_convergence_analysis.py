"""
Experiment 12: k-shot Convergence and Complementarity Evolution
================================================================

Goal: Analyze how complementarity patterns change as k increases (1,2,4,8).

Key questions:
1. Does increasing k reduce the CLIP-DINOv2 gap?
2. Does fusion benefit persist at higher k?
3. Do CLIP-advantage categories switch to DINOv2 at higher k?
4. Is complementarity a property of low-shot settings only?
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

def main():
    src_dir = Path(__file__).parent

    with open(src_dir / 'deep_analysis_cycle3_results.json') as f:
        data = json.load(f)

    kshot = data['experiment_8_kshot']

    texture_cats = ['carpet', 'grid', 'leather', 'tile', 'wood']
    object_cats = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                   'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    k_values = ['1', '2', '4', '8']

    # ============================================================
    # Analysis 1: Aggregate convergence
    # ============================================================
    print("=" * 80)
    print("Analysis 1: Aggregate k-shot Convergence")
    print("=" * 80)

    print(f"\n{'k':>3s} {'CLIP mean':>10s} {'DINO mean':>10s} {'Gap':>7s} {'Fusion':>7s} {'Gain':>6s} {'CLIP wins':>9s}")
    print("-" * 60)

    for k in k_values:
        clips = []
        dinos = []
        fusions = []
        gains = []
        clip_wins = 0
        total = 0

        for cat in sorted(kshot.keys()):
            if k in kshot[cat]:
                d = kshot[cat][k]
                clips.append(d['clip_auroc'])
                dinos.append(d['dino_auroc'])
                fusions.append(d['fusion_auroc'])
                gains.append(d['fusion_gain'])
                if d['winner'] == 'CLIP':
                    clip_wins += 1
                total += 1

        gap = np.mean(dinos) - np.mean(clips)
        print(f"k={k:1s} {np.mean(clips):9.2f}% {np.mean(dinos):9.2f}% {gap:+6.2f} "
              f"{np.mean(fusions):6.2f}% {np.mean(gains):+5.2f} {clip_wins}/{total}")

    # ============================================================
    # Analysis 2: By category type
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 2: k-shot Convergence by Category Type")
    print("=" * 80)

    for group_name, group_cats in [('Texture', texture_cats), ('Object', object_cats)]:
        print(f"\n--- {group_name} categories ---")
        print(f"{'k':>3s} {'CLIP':>8s} {'DINO':>8s} {'Gap':>7s} {'Fusion':>8s} {'Gain':>6s} {'CLIP%':>6s}")
        print("-" * 50)

        for k in k_values:
            clips = []
            dinos = []
            fusions = []
            gains = []
            clip_wins = 0
            total = 0

            for cat in group_cats:
                if cat in kshot and k in kshot[cat]:
                    d = kshot[cat][k]
                    clips.append(d['clip_auroc'])
                    dinos.append(d['dino_auroc'])
                    fusions.append(d['fusion_auroc'])
                    gains.append(d['fusion_gain'])
                    if d['winner'] == 'CLIP':
                        clip_wins += 1
                    total += 1

            if total > 0:
                gap = np.mean(dinos) - np.mean(clips)
                print(f"k={k:1s} {np.mean(clips):7.2f}% {np.mean(dinos):7.2f}% {gap:+6.2f} "
                      f"{np.mean(fusions):7.2f}% {np.mean(gains):+5.2f} {clip_wins}/{total}")

    # ============================================================
    # Analysis 3: Per-category winner stability
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 3: Winner Stability Across k Values")
    print("=" * 80)

    print(f"\n{'Category':15s} {'k=1':>6s} {'k=2':>6s} {'k=4':>6s} {'k=8':>6s} {'Switches':>8s} {'Type':>8s}")
    print("-" * 65)

    unstable_cats = []
    for cat in sorted(kshot.keys()):
        cat_type = 'Tex' if cat in texture_cats else 'Obj'
        winners = []
        for k in k_values:
            if k in kshot[cat]:
                winners.append(kshot[cat][k]['winner'])

        # Count switches
        switches = sum(1 for i in range(1, len(winners)) if winners[i] != winners[i-1])
        if switches > 0:
            unstable_cats.append(cat)

        print(f"{cat:15s} {'C' if winners[0] == 'CLIP' else 'D':>6s} "
              f"{'C' if len(winners) > 1 and winners[1] == 'CLIP' else 'D':>6s} "
              f"{'C' if len(winners) > 2 and winners[2] == 'CLIP' else 'D':>6s} "
              f"{'C' if len(winners) > 3 and winners[3] == 'CLIP' else 'D':>6s} "
              f"{switches:>8d} {cat_type:>8s}")

    print(f"\nUnstable categories (winner switches): {len(unstable_cats)}/{len(kshot)}")
    print(f"  {unstable_cats}")

    # ============================================================
    # Analysis 4: Gap convergence rate
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 4: Gap Convergence Rate (does gap shrink with k?)")
    print("=" * 80)

    print(f"\n{'Category':15s} {'Gap k=1':>8s} {'Gap k=8':>8s} {'Change':>8s} {'Direction':>12s}")
    print("-" * 55)

    gap_changes = []
    for cat in sorted(kshot.keys()):
        if '1' in kshot[cat] and '8' in kshot[cat]:
            gap1 = kshot[cat]['1']['gap']
            gap8 = kshot[cat]['8']['gap']
            change = gap8 - gap1
            gap_changes.append(change)

            direction = 'Converge' if change < 0 else 'Diverge'
            print(f"{cat:15s} {gap1:7.1f} {gap8:7.1f} {change:+7.1f} {direction:>12s}")

    converging = sum(1 for g in gap_changes if g < 0)
    diverging = sum(1 for g in gap_changes if g > 0)
    print(f"\nConverging (gap shrinks): {converging}/{len(gap_changes)}")
    print(f"Diverging (gap grows): {diverging}/{len(gap_changes)}")
    print(f"Mean gap change: {np.mean(gap_changes):+.2f}")

    # ============================================================
    # Analysis 5: Fusion benefit persistence
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 5: Fusion Benefit Persistence at Higher k")
    print("=" * 80)

    print(f"\n{'k':>3s} {'Mean gain':>10s} {'Positive':>9s} {'Mean when +':>11s} {'Mean when -':>11s}")
    print("-" * 50)

    for k in k_values:
        gains_pos = []
        gains_neg = []
        for cat in kshot:
            if k in kshot[cat]:
                g = kshot[cat][k]['fusion_gain']
                if g > 0:
                    gains_pos.append(g)
                else:
                    gains_neg.append(g)

        n_pos = len(gains_pos)
        n_total = n_pos + len(gains_neg)
        print(f"k={k:1s} {np.mean(gains_pos + gains_neg):+9.2f} {n_pos}/{n_total:>6d} "
              f"{np.mean(gains_pos) if gains_pos else 0:+10.2f} "
              f"{np.mean(gains_neg) if gains_neg else 0:+10.2f}")

    # ============================================================
    # Analysis 6: Statistical test - does gap significantly decrease with k?
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 6: Statistical Test - Gap vs k")
    print("=" * 80)

    gaps_k1 = [kshot[cat]['1']['gap'] for cat in kshot if '1' in kshot[cat]]
    gaps_k8 = [kshot[cat]['8']['gap'] for cat in kshot if '8' in kshot[cat]]

    # Paired test (same categories)
    common_cats = [cat for cat in kshot if '1' in kshot[cat] and '8' in kshot[cat]]
    paired_k1 = [kshot[cat]['1']['gap'] for cat in common_cats]
    paired_k8 = [kshot[cat]['8']['gap'] for cat in common_cats]

    t, p_paired = stats.ttest_rel(paired_k1, paired_k8)
    print(f"\nPaired t-test (gap at k=1 vs k=8):")
    print(f"  Mean gap at k=1: {np.mean(paired_k1):.2f}")
    print(f"  Mean gap at k=8: {np.mean(paired_k8):.2f}")
    print(f"  t={t:.3f}, p={p_paired:.4f} {'*' if p_paired < 0.05 else 'n.s.'}")

    # Wilcoxon (non-parametric)
    w, p_wilcox = stats.wilcoxon(paired_k1, paired_k8)
    print(f"\nWilcoxon signed-rank test:")
    print(f"  W={w:.0f}, p={p_wilcox:.4f} {'*' if p_wilcox < 0.05 else 'n.s.'}")

    # ============================================================
    # Analysis 7: Complementarity persistence
    # ============================================================
    print("\n" + "=" * 80)
    print("Analysis 7: Does Complementarity Persist at k=8?")
    print("=" * 80)

    for k in k_values:
        clips = []
        dinos = []
        for cat in sorted(kshot.keys()):
            if k in kshot[cat]:
                clips.append(kshot[cat][k]['clip_auroc'])
                dinos.append(kshot[cat][k]['dino_auroc'])

        # Correlation between CLIP and DINOv2 across categories
        r, p = stats.pearsonr(clips, dinos)
        # Complementarity = 1 - correlation
        print(f"k={k}: r(CLIP, DINOv2) = {r:.3f} (p={p:.4f}), Complementarity = {1-r:.3f}")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        'experiment_12_kshot_convergence': {
            'aggregate': {},
            'winner_stability': {
                'unstable_categories': unstable_cats,
                'n_unstable': len(unstable_cats),
                'n_total': len(kshot),
            },
            'gap_convergence': {
                'mean_change': round(np.mean(gap_changes), 3),
                'n_converging': converging,
                'n_diverging': diverging,
                'paired_ttest_p': round(p_paired, 4),
                'wilcoxon_p': round(p_wilcox, 4),
            },
        }
    }

    # Add aggregate data
    for k in k_values:
        clips = [kshot[cat][k]['clip_auroc'] for cat in kshot if k in kshot[cat]]
        dinos = [kshot[cat][k]['dino_auroc'] for cat in kshot if k in kshot[cat]]
        fusions = [kshot[cat][k]['fusion_auroc'] for cat in kshot if k in kshot[cat]]

        output['experiment_12_kshot_convergence']['aggregate'][f'k={k}'] = {
            'clip_mean': round(np.mean(clips), 2),
            'dino_mean': round(np.mean(dinos), 2),
            'fusion_mean': round(np.mean(fusions), 2),
            'gap': round(np.mean(dinos) - np.mean(clips), 2),
        }

    output_path = src_dir / 'kshot_convergence_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
