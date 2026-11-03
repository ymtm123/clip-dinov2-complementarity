"""
Experiment 10: Native Mode Per-Defect-Type Analysis
=====================================================

Goal: Verify whether the kNN-based taxonomy findings generalize to native mode
(AnomalyCLIP 0-shot + AnomalyDINO 1-shot).

Key question: Does "color → CLIP" and "semantic → DINOv2" pattern hold
when using the native methods instead of controlled kNN?
"""

import csv
import json
import os
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score

# ============================================================
# Defect taxonomy (same as deep_analysis experiments)
# ============================================================
defect_taxonomy = {
    ('bottle', 'broken_large'): 'structural', ('bottle', 'broken_small'): 'structural',
    ('bottle', 'contamination'): 'foreign_object',
    ('cable', 'bent_wire'): 'geometric', ('cable', 'cable_swap'): 'semantic',
    ('cable', 'combined'): 'mixed', ('cable', 'cut_inner_insulation'): 'geometric',
    ('cable', 'cut_outer_insulation'): 'geometric', ('cable', 'missing_cable'): 'semantic',
    ('cable', 'missing_wire'): 'semantic', ('cable', 'poke_insulation'): 'structural',
    ('capsule', 'crack'): 'structural', ('capsule', 'faulty_imprint'): 'color',
    ('capsule', 'poke'): 'structural', ('capsule', 'scratch'): 'texture_surface',
    ('capsule', 'squeeze'): 'geometric',
    ('carpet', 'color'): 'color', ('carpet', 'cut'): 'structural',
    ('carpet', 'hole'): 'structural', ('carpet', 'metal_contamination'): 'foreign_object',
    ('carpet', 'thread'): 'texture_surface',
    ('grid', 'bent'): 'geometric', ('grid', 'broken'): 'structural',
    ('grid', 'glue'): 'foreign_object', ('grid', 'metal_contamination'): 'foreign_object',
    ('grid', 'thread'): 'texture_surface',
    ('hazelnut', 'crack'): 'structural', ('hazelnut', 'cut'): 'structural',
    ('hazelnut', 'hole'): 'structural', ('hazelnut', 'print'): 'color',
    ('leather', 'color'): 'color', ('leather', 'cut'): 'structural',
    ('leather', 'fold'): 'geometric', ('leather', 'glue'): 'foreign_object',
    ('leather', 'poke'): 'structural',
    ('metal_nut', 'bent'): 'geometric', ('metal_nut', 'color'): 'color',
    ('metal_nut', 'flip'): 'semantic', ('metal_nut', 'scratch'): 'texture_surface',
    ('pill', 'color'): 'color', ('pill', 'combined'): 'mixed',
    ('pill', 'contamination'): 'foreign_object', ('pill', 'crack'): 'structural',
    ('pill', 'faulty_imprint'): 'color', ('pill', 'pill_type'): 'semantic',
    ('pill', 'scratch'): 'texture_surface',
    ('screw', 'manipulated_front'): 'geometric', ('screw', 'scratch_head'): 'texture_surface',
    ('screw', 'scratch_neck'): 'texture_surface', ('screw', 'thread_side'): 'texture_surface',
    ('screw', 'thread_top'): 'texture_surface',
    ('tile', 'crack'): 'structural', ('tile', 'glue_strip'): 'foreign_object',
    ('tile', 'gray_stroke'): 'color', ('tile', 'oil'): 'foreign_object',
    ('tile', 'rough'): 'texture_surface',
    ('toothbrush', 'defective'): 'structural',
    ('transistor', 'bent_lead'): 'geometric', ('transistor', 'cut_lead'): 'geometric',
    ('transistor', 'damaged_case'): 'structural', ('transistor', 'misplaced'): 'semantic',
    ('wood', 'color'): 'color', ('wood', 'combined'): 'mixed',
    ('wood', 'hole'): 'structural', ('wood', 'liquid'): 'foreign_object',
    ('wood', 'scratch'): 'texture_surface',
    ('zipper', 'broken_teeth'): 'structural', ('zipper', 'combined'): 'mixed',
    ('zipper', 'fabric_border'): 'texture_surface', ('zipper', 'fabric_interior'): 'texture_surface',
    ('zipper', 'rough'): 'texture_surface', ('zipper', 'split_teeth'): 'structural',
    ('zipper', 'squeezed_teeth'): 'geometric',
}

# ============================================================
# Data loading
# ============================================================

def load_anomalyclip_scores(csv_path):
    """Load AnomalyCLIP scores, returns {category: {defect_type: [scores], 'good': [scores]}}"""
    data = defaultdict(lambda: defaultdict(list))
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            obj = row[0]
            img_path = row[1].replace('\\', '/')
            score = float(row[2])
            gt = int(row[3])

            # Extract defect type from path: .../test/defect_type/image.png
            match = re.search(r'/test/([^/]+)/', img_path)
            if match:
                defect_type = match.group(1)
            else:
                defect_type = 'good' if gt == 0 else 'unknown'

            data[obj][defect_type].append(score)

    return dict(data)


def load_anomalydino_scores(csv_path):
    """Load AnomalyDINO scores, returns {category: {defect_type: [scores], 'good': [scores]}}"""
    data = defaultdict(lambda: defaultdict(list))
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            obj = row[0]
            sample = row[1]  # e.g., "contamination/000.png"
            score = float(row[2])

            # Extract defect type from sample path
            defect_type = sample.split('/')[0]

            data[obj][defect_type].append(score)

    return dict(data)


def compute_auroc_safe(normal_scores, anomaly_scores):
    """Compute AUROC from normal and anomaly score arrays."""
    if len(normal_scores) == 0 or len(anomaly_scores) == 0:
        return float('nan')
    labels = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    scores = list(normal_scores) + list(anomaly_scores)
    if len(set(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, scores) * 100


# ============================================================
# Main analysis
# ============================================================

def main():
    src_dir = Path(__file__).parent

    # Load scores
    clip_path = src_dir / 'AnomalyCLIP' / 'results' / 'scores_mvtec' / 'anomalyclip_scores.csv'
    # AnomalyDINO has scores split into strong_cats (12) and weak_cats (3)
    dino_path_strong = src_dir / 'AnomalyDINO' / 'results_MVTec' / 'dinov2_vitl14_448' / \
                       '1-shot_preprocess=agnostic_strong_cats' / 'measurements_seed=0.csv'
    dino_path_weak = src_dir / 'AnomalyDINO' / 'results_MVTec' / 'dinov2_vitl14_448' / \
                     '1-shot_preprocess=agnostic_weak_cats' / 'measurements_seed=0.csv'

    print("Loading AnomalyCLIP scores...")
    clip_data = load_anomalyclip_scores(clip_path)
    print(f"  Loaded {len(clip_data)} categories")

    print("Loading AnomalyDINO scores (strong_cats + weak_cats)...")
    dino_data_strong = load_anomalydino_scores(dino_path_strong)
    dino_data_weak = load_anomalydino_scores(dino_path_weak)
    # Merge
    dino_data = {}
    dino_data.update(dino_data_strong)
    dino_data.update(dino_data_weak)
    print(f"  Loaded {len(dino_data)} categories (strong={len(dino_data_strong)}, weak={len(dino_data_weak)})")

    # Verify categories match
    common_cats = sorted(set(clip_data.keys()) & set(dino_data.keys()))
    print(f"\nCommon categories: {len(common_cats)}")
    print(f"  {common_cats}")

    # ============================================================
    # Per-defect-type AUROC computation
    # ============================================================
    print("\n" + "=" * 90)
    print("Experiment 10: Native Mode Per-Defect-Type AUROC")
    print("=" * 90)

    results = []

    print(f"\n{'Category/Defect':35s} {'CLIP':>7s} {'DINO':>7s} {'Diff':>7s} {'Winner':>8s} {'Nature':>15s}")
    print("-" * 85)

    for cat in common_cats:
        # Get normal scores for this category
        clip_normal = clip_data[cat].get('good', [])
        dino_normal = dino_data[cat].get('good', [])

        if len(clip_normal) == 0 or len(dino_normal) == 0:
            print(f"  SKIP {cat}: no normal scores (CLIP={len(clip_normal)}, DINO={len(dino_normal)})")
            continue

        # Get all defect types (exclude 'good')
        all_defect_types = set()
        for dt in clip_data[cat]:
            if dt != 'good':
                all_defect_types.add(dt)
        for dt in dino_data[cat]:
            if dt != 'good':
                all_defect_types.add(dt)

        for dt in sorted(all_defect_types):
            clip_anom = clip_data[cat].get(dt, [])
            dino_anom = dino_data[cat].get(dt, [])

            if len(clip_anom) == 0 and len(dino_anom) == 0:
                continue

            # Compute per-defect-type AUROC
            clip_auroc = compute_auroc_safe(clip_normal, clip_anom) if len(clip_anom) > 0 else float('nan')
            dino_auroc = compute_auroc_safe(dino_normal, dino_anom) if len(dino_anom) > 0 else float('nan')

            if np.isnan(clip_auroc) or np.isnan(dino_auroc):
                continue

            diff = clip_auroc - dino_auroc
            winner = 'CLIP' if diff > 0 else 'DINOv2'
            nature = defect_taxonomy.get((cat, dt), 'unknown')

            results.append({
                'category': cat,
                'defect_type': dt,
                'clip_auroc': clip_auroc,
                'dino_auroc': dino_auroc,
                'diff': diff,
                'winner': winner,
                'nature': nature,
            })

            prefix = f"  {cat}/{dt}"
            print(f"{prefix:35s} {clip_auroc:6.1f}% {dino_auroc:6.1f}% {diff:+6.1f}% {winner:>8s} {nature:>15s}")

    # ============================================================
    # Summary statistics
    # ============================================================
    print("\n" + "=" * 90)
    print("Summary: Per-Defect-Type Analysis")
    print("=" * 90)

    clip_wins = sum(1 for r in results if r['winner'] == 'CLIP')
    dino_wins = sum(1 for r in results if r['winner'] == 'DINOv2')
    total = len(results)

    print(f"\nTotal defect types analyzed: {total}")
    print(f"CLIP wins: {clip_wins} ({clip_wins/total*100:.1f}%)")
    print(f"DINOv2 wins: {dino_wins} ({dino_wins/total*100:.1f}%)")

    diffs = [r['diff'] for r in results]
    print(f"Mean AUROC diff (CLIP - DINOv2): {np.mean(diffs):+.2f}")
    print(f"Median AUROC diff: {np.median(diffs):+.2f}")

    # ============================================================
    # Taxonomy analysis (same as kNN experiment 7)
    # ============================================================
    print("\n" + "=" * 90)
    print("Taxonomy Analysis: Native Mode")
    print("=" * 90)

    nature_groups = defaultdict(list)
    for r in results:
        if r['nature'] != 'unknown':
            nature_groups[r['nature']].append(r['diff'])

    print(f"\n{'Nature':20s} {'N':>3s} {'Mean Diff':>10s} {'CLIP Win%':>10s} {'p-value':>10s} {'Sig':>5s}")
    print("-" * 65)

    taxonomy_results = {}

    for nature in ['color', 'texture_surface', 'structural', 'geometric', 'foreign_object', 'semantic', 'mixed']:
        if nature in nature_groups:
            diffs_n = nature_groups[nature]
            n = len(diffs_n)
            mean_diff = np.mean(diffs_n)
            clip_win_rate = sum(1 for d in diffs_n if d > 0) / n * 100

            # One-sample t-test: is the mean diff significantly different from 0?
            if n >= 3:
                t_stat, p_val = stats.ttest_1samp(diffs_n, 0)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '†' if p_val < 0.1 else 'n.s.'
            else:
                p_val = float('nan')
                sig = 'n/a'

            taxonomy_results[nature] = {
                'n': n,
                'mean_diff': round(mean_diff, 2),
                'clip_win_rate': round(clip_win_rate, 1),
                'p_value': round(p_val, 4) if not np.isnan(p_val) else None,
                'diffs': [round(d, 2) for d in diffs_n],
            }

            print(f"{nature:20s} {n:3d} {mean_diff:+9.2f} {clip_win_rate:9.1f}% {p_val:9.4f} {sig:>5s}")

    # ============================================================
    # ANOVA across nature categories
    # ============================================================
    print("\n--- ANOVA across nature categories ---")
    anova_groups = [nature_groups[n] for n in ['color', 'texture_surface', 'structural',
                                                 'geometric', 'foreign_object', 'semantic']
                    if n in nature_groups and len(nature_groups[n]) >= 2]

    if len(anova_groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*anova_groups)
        print(f"F-statistic: {f_stat:.3f}")
        print(f"p-value: {p_anova:.4f}")
        print(f"Significant: {'YES' if p_anova < 0.05 else 'NO'}")

    # ============================================================
    # Pairwise comparisons (Welch's t-test)
    # ============================================================
    print("\n--- Pairwise t-tests (key comparisons) ---")

    pairwise_results = {}
    pairs = [
        ('color', 'structural'),
        ('color', 'geometric'),
        ('color', 'semantic'),
        ('semantic', 'structural'),
        ('texture_surface', 'geometric'),
    ]

    for n1, n2 in pairs:
        if n1 in nature_groups and n2 in nature_groups:
            g1, g2 = nature_groups[n1], nature_groups[n2]
            if len(g1) >= 2 and len(g2) >= 2:
                t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '†' if p_val < 0.1 else 'n.s.'
                pairwise_results[f"{n1}_vs_{n2}"] = {
                    't_stat': round(t_stat, 3),
                    'p_value': round(p_val, 4),
                }
                print(f"  {n1} vs {n2}: t={t_stat:.3f}, p={p_val:.4f} ({sig})")

    # ============================================================
    # kNN vs Native mode comparison
    # ============================================================
    print("\n" + "=" * 90)
    print("Comparison: kNN vs Native Mode Taxonomy")
    print("=" * 90)

    # Load kNN results
    knn_results_path = src_dir / 'deep_analysis_cycle2_results.json'
    if knn_results_path.exists():
        with open(knn_results_path) as f:
            knn_data = json.load(f)

        knn_taxonomy = knn_data.get('experiment_7_defect_taxonomy', {})

        print(f"\n{'Nature':20s} {'kNN Diff':>10s} {'Native Diff':>12s} {'kNN CLIP%':>10s} {'Native CLIP%':>12s} {'Consistent?':>12s}")
        print("-" * 80)

        consistency_count = 0
        for nature in ['color', 'texture_surface', 'structural', 'geometric', 'foreign_object', 'semantic']:
            if nature in knn_taxonomy and nature in taxonomy_results:
                knn_d = knn_taxonomy[nature]
                nat_d = taxonomy_results[nature]

                knn_diff = knn_d['mean_diff']
                nat_diff = nat_d['mean_diff']
                knn_win = knn_d['clip_win_rate']
                nat_win = nat_d['clip_win_rate']

                # Consistent = same sign of mean_diff
                consistent = 'YES' if (knn_diff > 0) == (nat_diff > 0) else 'NO'
                if consistent == 'YES':
                    consistency_count += 1

                print(f"{nature:20s} {knn_diff:+9.2f} {nat_diff:+11.2f} {knn_win:9.1f}% {nat_win:11.1f}% {consistent:>12s}")

        print(f"\nDirection consistency: {consistency_count}/6")

    # ============================================================
    # Per-category level comparison: native vs kNN alpha
    # ============================================================
    print("\n" + "=" * 90)
    print("Per-Category: Native AUROC Comparison")
    print("=" * 90)

    cat_summary = defaultdict(lambda: {'clip_aurocs': [], 'dino_aurocs': []})
    for r in results:
        cat_summary[r['category']]['clip_aurocs'].append(r['clip_auroc'])
        cat_summary[r['category']]['dino_aurocs'].append(r['dino_auroc'])

    print(f"\n{'Category':15s} {'CLIP Mean':>10s} {'DINO Mean':>10s} {'Diff':>8s} {'Winner':>8s} {'N_defects':>10s}")
    print("-" * 65)

    for cat in sorted(cat_summary.keys()):
        cs = cat_summary[cat]
        clip_m = np.mean(cs['clip_aurocs'])
        dino_m = np.mean(cs['dino_aurocs'])
        diff_m = clip_m - dino_m
        winner = 'CLIP' if diff_m > 0 else 'DINOv2'
        n_def = len(cs['clip_aurocs'])
        print(f"{cat:15s} {clip_m:9.1f}% {dino_m:9.1f}% {diff_m:+7.1f}% {winner:>8s} {n_def:>10d}")

    # ============================================================
    # Top exclusive defect types (native mode)
    # ============================================================
    print("\n" + "=" * 90)
    print("Top Defect Types: CLIP vs DINOv2 Advantage (Native Mode)")
    print("=" * 90)

    sorted_by_diff = sorted(results, key=lambda r: r['diff'])

    print("\nTop 10 DINOv2 advantage (CLIP - DINOv2 most negative):")
    for r in sorted_by_diff[:10]:
        print(f"  {r['category']}/{r['defect_type']:25s} CLIP={r['clip_auroc']:.1f} DINO={r['dino_auroc']:.1f} diff={r['diff']:+.1f} [{r['nature']}]")

    print("\nTop 10 CLIP advantage (CLIP - DINOv2 most positive):")
    for r in sorted_by_diff[-10:][::-1]:
        print(f"  {r['category']}/{r['defect_type']:25s} CLIP={r['clip_auroc']:.1f} DINO={r['dino_auroc']:.1f} diff={r['diff']:+.1f} [{r['nature']}]")

    # ============================================================
    # Effect size (Cohen's d) per nature
    # ============================================================
    print("\n" + "=" * 90)
    print("Effect Size Analysis")
    print("=" * 90)

    for nature in ['color', 'texture_surface', 'structural', 'geometric', 'foreign_object', 'semantic']:
        if nature in nature_groups and len(nature_groups[nature]) >= 3:
            diffs_n = nature_groups[nature]
            mean_d = np.mean(diffs_n)
            std_d = np.std(diffs_n, ddof=1)
            cohens_d = mean_d / std_d if std_d > 0 else float('inf')
            print(f"  {nature:20s}: Cohen's d = {cohens_d:+.3f} (mean={mean_d:+.2f}, sd={std_d:.2f})")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        'experiment_10_native_defect_type': {
            'description': 'Per-defect-type AUROC comparison in native mode (AnomalyCLIP 0-shot vs AnomalyDINO 1-shot)',
            'total_defect_types': total,
            'clip_wins': clip_wins,
            'dino_wins': dino_wins,
            'mean_diff': round(np.mean(diffs), 2),
            'per_defect_type': results,
            'taxonomy': taxonomy_results,
            'anova_p': round(p_anova, 4) if 'p_anova' in dir() else None,
            'pairwise': pairwise_results,
        }
    }

    output_path = src_dir / 'native_defect_type_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
