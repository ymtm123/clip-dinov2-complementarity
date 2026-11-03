"""
Experiment 15: Per-Image Score Correlation & Detection Agreement (Native Mode)
================================================================================

Goal: Analyze the per-image relationship between CLIP and DINOv2 anomaly
scores to understand the nature of complementarity at the instance level.

Key questions:
1. How correlated are CLIP and DINOv2 anomaly scores per category?
2. What fraction of detections are exclusive to one model?
3. Does low score correlation predict high fusion benefit?
4. Are there categories where models disagree on which images are anomalous?
5. How does native mode score correlation compare to fusion effectiveness?

This complements Experiment 9 (kNN agreement) by examining native-mode behavior.
"""

import json
import csv
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score


def load_anomalyclip_scores(csv_path):
    """Load AnomalyCLIP per-image scores."""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row['Object']
            img = row['ImagePath'].replace('\\', '/')
            # Extract relative path after test/
            parts = img.split('/test/')
            if len(parts) == 2:
                rel = parts[1]
            else:
                rel = img.split('/')[-1]
            score = float(row['AnomalyScore'])
            gt = int(row['GroundTruth'])

            if cat not in data:
                data[cat] = []
            data[cat].append({
                'rel_path': rel,
                'score': score,
                'gt': gt,
            })
    return data


def load_anomalydino_scores(csv_path):
    """Load AnomalyDINO per-image scores."""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row['Object']
            sample = row['Sample']
            score = float(row['Anomaly_Score'])
            # Determine ground truth from path
            gt = 0 if sample.startswith('good/') else 1

            if cat not in data:
                data[cat] = []
            data[cat].append({
                'rel_path': sample,
                'score': score,
                'gt': gt,
            })
    return data


def optimal_threshold(scores, labels):
    """Compute Youden's J optimal threshold."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx]


def main():
    src_dir = Path(__file__).parent

    # Load AnomalyCLIP scores (MVTec)
    clip_path = src_dir / 'AnomalyCLIP' / 'results' / 'scores_mvtec' / 'anomalyclip_scores.csv'
    clip_data = load_anomalyclip_scores(clip_path)

    # Load AnomalyDINO scores (MVTec, both strong and weak cats)
    dino_strong = src_dir / 'AnomalyDINO' / 'results_MVTec' / 'dinov2_vitl14_448' / \
                  '1-shot_preprocess=agnostic_strong_cats' / 'measurements_seed=0.csv'
    dino_weak = src_dir / 'AnomalyDINO' / 'results_MVTec' / 'dinov2_vitl14_448' / \
                '1-shot_preprocess=agnostic_weak_cats' / 'measurements_seed=0.csv'

    dino_data = load_anomalydino_scores(dino_strong)
    dino_data_weak = load_anomalydino_scores(dino_weak)
    for cat, entries in dino_data_weak.items():
        if cat not in dino_data:
            dino_data[cat] = entries
        else:
            dino_data[cat].extend(entries)

    # Load fusion results to get fusion gains
    with open(src_dir / 'comprehensive_eval_results.json') as f:
        eval_data = json.load(f)

    print("=" * 90)
    print("EXPERIMENT 15: Per-Image Score Correlation & Detection Agreement (Native Mode)")
    print("=" * 90)

    # ============================================================
    # Part A: Per-category score correlation
    # ============================================================
    print(f"\n{'=' * 85}")
    print("Analysis 1: Per-Category Score Correlation (Spearman rho)")
    print("=" * 85)

    common_cats = sorted(set(clip_data.keys()) & set(dino_data.keys()))

    results = []

    print(f"\n{'Category':15s} {'N_anom':>6s} {'N_norm':>6s} {'rho_all':>8s} {'rho_anom':>9s} "
          f"{'CLIP AUC':>9s} {'DINO AUC':>9s} {'Gap':>6s}")
    print("-" * 75)

    for cat in common_cats:
        # Build index for CLIP
        clip_entries = clip_data[cat]
        dino_entries = dino_data[cat]

        # Match by defect type and image number
        # CLIP: rel_path like "color/000.png" or "good/000.png"
        # DINO: rel_path like "color/000.png" or "good/000.png"

        clip_by_path = {}
        for e in clip_entries:
            # Normalize: remove any leading slash
            p = e['rel_path'].strip('/')
            clip_by_path[p] = e

        matched_clip = []
        matched_dino = []
        matched_gt = []

        for de in dino_entries:
            p = de['rel_path'].strip('/')
            if p in clip_by_path:
                ce = clip_by_path[p]
                matched_clip.append(ce['score'])
                matched_dino.append(de['score'])
                matched_gt.append(ce['gt'])

        if len(matched_clip) < 5:
            continue

        matched_clip = np.array(matched_clip)
        matched_dino = np.array(matched_dino)
        matched_gt = np.array(matched_gt)

        n_anom = int(np.sum(matched_gt))
        n_norm = len(matched_gt) - n_anom

        # Spearman correlation (all images)
        rho_all, p_all = stats.spearmanr(matched_clip, matched_dino)

        # Spearman correlation (anomaly images only)
        if n_anom >= 5:
            anom_mask = matched_gt == 1
            rho_anom, p_anom = stats.spearmanr(matched_clip[anom_mask], matched_dino[anom_mask])
        else:
            rho_anom = np.nan
            p_anom = np.nan

        # Per-model AUROC
        if n_anom > 0 and n_norm > 0:
            clip_auroc = roc_auc_score(matched_gt, matched_clip) * 100
            dino_auroc = roc_auc_score(matched_gt, matched_dino) * 100
        else:
            clip_auroc = dino_auroc = np.nan

        gap = dino_auroc - clip_auroc

        print(f"{cat:15s} {n_anom:>6d} {n_norm:>6d} {rho_all:+7.3f} "
              f"{'  ' + f'{rho_anom:+.3f}' if not np.isnan(rho_anom) else '    N/A':>9s} "
              f"{clip_auroc:8.1f}% {dino_auroc:8.1f}% {gap:+5.1f}")

        # Detection agreement using optimal thresholds
        clip_thresh = optimal_threshold(matched_clip, matched_gt)
        dino_thresh = optimal_threshold(matched_dino, matched_gt)

        clip_pred = (matched_clip >= clip_thresh).astype(int)
        dino_pred = (matched_dino >= dino_thresh).astype(int)

        # For anomalous images only
        anom_mask = matched_gt == 1
        clip_detects = clip_pred[anom_mask]
        dino_detects = dino_pred[anom_mask]

        both = int(np.sum((clip_detects == 1) & (dino_detects == 1)))
        clip_only = int(np.sum((clip_detects == 1) & (dino_detects == 0)))
        dino_only = int(np.sum((clip_detects == 0) & (dino_detects == 1)))
        neither = int(np.sum((clip_detects == 0) & (dino_detects == 0)))

        total_anom = n_anom
        exclusive_rate = (clip_only + dino_only) / total_anom if total_anom > 0 else 0

        # Get fusion result
        fusion_auroc = None
        if cat in eval_data.get('MVTec AD', {}).get('Equal (a=0.5)', {}).get('per_category', {}):
            fusion_auroc = eval_data['MVTec AD']['Equal (a=0.5)']['per_category'][cat]

        best_single = max(clip_auroc, dino_auroc) if not np.isnan(clip_auroc) else None
        fusion_gain = (fusion_auroc - best_single) if (fusion_auroc and best_single) else None

        results.append({
            'category': cat,
            'n_matched': len(matched_clip),
            'n_anom': n_anom,
            'n_norm': n_norm,
            'rho_all': round(rho_all, 4),
            'rho_anom': round(rho_anom, 4) if not np.isnan(rho_anom) else None,
            'clip_auroc': round(clip_auroc, 2),
            'dino_auroc': round(dino_auroc, 2),
            'gap': round(gap, 2),
            'both_detect': both,
            'clip_only': clip_only,
            'dino_only': dino_only,
            'neither_detect': neither,
            'exclusive_rate': round(exclusive_rate, 4),
            'fusion_auroc': fusion_auroc,
            'fusion_gain': round(fusion_gain, 2) if fusion_gain is not None else None,
        })

    # ============================================================
    # Part B: Detection Agreement Summary
    # ============================================================
    print(f"\n{'=' * 85}")
    print("Analysis 2: Detection Agreement (Anomalous Images Only)")
    print("=" * 85)

    print(f"\n{'Category':15s} {'Both':>5s} {'CLIP':>6s} {'DINO':>6s} {'None':>6s} {'Excl%':>6s} {'Fus gain':>9s}")
    print("-" * 60)

    for r in results:
        fg = f"{r['fusion_gain']:+8.2f}" if r['fusion_gain'] is not None else "    N/A"
        print(f"{r['category']:15s} {r['both_detect']:>5d} {r['clip_only']:>6d} {r['dino_only']:>6d} "
              f"{r['neither_detect']:>6d} {r['exclusive_rate']*100:5.1f}% {fg}")

    total_both = sum(r['both_detect'] for r in results)
    total_clip = sum(r['clip_only'] for r in results)
    total_dino = sum(r['dino_only'] for r in results)
    total_neither = sum(r['neither_detect'] for r in results)
    total = total_both + total_clip + total_dino + total_neither

    print(f"\n{'TOTAL':15s} {total_both:>5d} {total_clip:>6d} {total_dino:>6d} "
          f"{total_neither:>6d} {(total_clip+total_dino)/total*100:5.1f}%")
    print(f"\n  Both detect: {total_both}/{total} ({100*total_both/total:.1f}%)")
    print(f"  CLIP only:   {total_clip}/{total} ({100*total_clip/total:.1f}%)")
    print(f"  DINOv2 only: {total_dino}/{total} ({100*total_dino/total:.1f}%)")
    print(f"  Neither:     {total_neither}/{total} ({100*total_neither/total:.1f}%)")

    # ============================================================
    # Part C: Score Correlation vs Fusion Benefit
    # ============================================================
    print(f"\n{'=' * 85}")
    print("Analysis 3: Does Score Correlation Predict Fusion Benefit?")
    print("=" * 85)

    valid = [r for r in results if r['fusion_gain'] is not None and r['rho_all'] is not None]

    rho_vals = [r['rho_all'] for r in valid]
    fg_vals = [r['fusion_gain'] for r in valid]

    r_corr, p_corr = stats.pearsonr(rho_vals, fg_vals)
    print(f"\n  r(score correlation, fusion gain) = {r_corr:.3f}, p={p_corr:.4f} "
          f"{'*' if p_corr < 0.05 else 'n.s.'}")

    if r_corr < 0:
        print(f"  → Lower score correlation → higher fusion benefit (as expected)")
    else:
        print(f"  → Higher score correlation → higher fusion benefit (unexpected)")

    # Exclusive rate vs fusion gain
    excl_vals = [r['exclusive_rate'] for r in valid]
    r_excl, p_excl = stats.pearsonr(excl_vals, fg_vals)
    print(f"\n  r(exclusive detection rate, fusion gain) = {r_excl:.3f}, p={p_excl:.4f} "
          f"{'*' if p_excl < 0.05 else 'n.s.'}")

    # Score correlation vs AUROC gap
    gap_vals = [abs(r['gap']) for r in valid]
    r_gap_corr, p_gap_corr = stats.pearsonr(rho_vals, gap_vals)
    print(f"\n  r(score correlation, |AUROC gap|) = {r_gap_corr:.3f}, p={p_gap_corr:.4f}")

    # ============================================================
    # Part D: Anomaly-only score correlation
    # ============================================================
    print(f"\n{'=' * 85}")
    print("Analysis 4: Anomaly-Only Score Correlation")
    print("=" * 85)

    valid_anom = [r for r in results if r['rho_anom'] is not None]

    print(f"\n  {'Category':15s} {'rho_all':>8s} {'rho_anom':>9s} {'Diff':>6s} {'Meaning':>30s}")
    print(f"  {'-' * 72}")

    for r in valid_anom:
        diff = r['rho_anom'] - r['rho_all']
        meaning = "Anomalies more independent" if diff < -0.1 else \
                  "Anomalies more correlated" if diff > 0.1 else "Similar"
        print(f"  {r['category']:15s} {r['rho_all']:+7.3f} {r['rho_anom']:+8.3f} {diff:+5.2f} "
              f"{meaning:>30s}")

    rho_all_mean = np.mean([r['rho_all'] for r in valid_anom])
    rho_anom_mean = np.mean([r['rho_anom'] for r in valid_anom])
    print(f"\n  Mean rho (all images): {rho_all_mean:.3f}")
    print(f"  Mean rho (anomalies only): {rho_anom_mean:.3f}")
    print(f"  → {'Anomaly scores are MORE independent than overall scores' if rho_anom_mean < rho_all_mean else 'Anomaly scores are MORE correlated than overall scores'}")

    # ============================================================
    # Part E: Categories with highest complementarity
    # ============================================================
    print(f"\n{'=' * 85}")
    print("Analysis 5: Complementarity Ranking (by score independence)")
    print("=" * 85)

    sorted_by_rho = sorted(results, key=lambda r: r['rho_all'])

    print(f"\n  Top 5 most complementary (lowest score correlation):")
    for r in sorted_by_rho[:5]:
        print(f"    {r['category']:15s}: rho={r['rho_all']:+.3f}, excl={r['exclusive_rate']*100:.0f}%, "
              f"gap={r['gap']:+.1f}, fusion_gain={r['fusion_gain'] if r['fusion_gain'] else 'N/A'}")

    print(f"\n  Top 5 least complementary (highest score correlation):")
    for r in sorted_by_rho[-5:]:
        print(f"    {r['category']:15s}: rho={r['rho_all']:+.3f}, excl={r['exclusive_rate']*100:.0f}%, "
              f"gap={r['gap']:+.1f}, fusion_gain={r['fusion_gain'] if r['fusion_gain'] else 'N/A'}")

    # ============================================================
    # Part F: Score rank agreement analysis
    # ============================================================
    print(f"\n{'=' * 85}")
    print("Analysis 6: Per-Image Fusion Score Analysis")
    print("=" * 85)

    # For each category, compute fusion scores and check if fusion helps
    print(f"\n  How often does fusion fix a wrong detection?")
    total_fixed = 0
    total_broken = 0
    total_unchanged = 0

    for cat in common_cats:
        clip_entries = clip_data[cat]
        dino_entries = dino_data[cat]

        clip_by_path = {}
        for e in clip_entries:
            p = e['rel_path'].strip('/')
            clip_by_path[p] = e

        clips = []
        dinos = []
        gts = []

        for de in dino_entries:
            p = de['rel_path'].strip('/')
            if p in clip_by_path:
                ce = clip_by_path[p]
                clips.append(ce['score'])
                dinos.append(de['score'])
                gts.append(ce['gt'])

        if len(clips) < 5:
            continue

        clips = np.array(clips)
        dinos = np.array(dinos)
        gts = np.array(gts)

        # Z-normalize per category
        clip_z = (clips - clips.mean()) / (clips.std() + 1e-10)
        dino_z = (dinos - dinos.mean()) / (dinos.std() + 1e-10)
        fuse_z = 0.5 * clip_z + 0.5 * dino_z

        # For each anomalous image, check ranking
        anom_indices = np.where(gts == 1)[0]
        norm_indices = np.where(gts == 0)[0]

        if len(norm_indices) == 0 or len(anom_indices) == 0:
            continue

        # Best single model scores
        best_single_z = np.where(
            roc_auc_score(gts, clip_z) > roc_auc_score(gts, dino_z),
            clip_z, dino_z
        )

        # Check: is fusion AUROC better than best single?
        fuse_auc = roc_auc_score(gts, fuse_z) * 100
        clip_auc = roc_auc_score(gts, clip_z) * 100
        dino_auc = roc_auc_score(gts, dino_z) * 100
        best_auc = max(clip_auc, dino_auc)

        if fuse_auc > best_auc + 0.1:
            total_fixed += 1
        elif fuse_auc < best_auc - 0.1:
            total_broken += 1
        else:
            total_unchanged += 1

    print(f"\n  Categories where fusion improves over best single: {total_fixed}/{total_fixed+total_broken+total_unchanged}")
    print(f"  Categories where fusion hurts: {total_broken}/{total_fixed+total_broken+total_unchanged}")
    print(f"  Categories where fusion is neutral: {total_unchanged}/{total_fixed+total_broken+total_unchanged}")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        'experiment_15_score_correlation': {
            'per_category': results,
            'summary': {
                'mean_rho_all': round(np.mean([r['rho_all'] for r in results]), 4),
                'mean_rho_anom': round(rho_anom_mean, 4),
                'total_both_detect': total_both,
                'total_clip_only': total_clip,
                'total_dino_only': total_dino,
                'total_neither': total_neither,
                'exclusive_rate': round((total_clip + total_dino) / total, 4),
            },
            'correlations': {
                'rho_vs_fusion_gain': {
                    'r': round(r_corr, 4),
                    'p': round(p_corr, 4),
                },
                'excl_vs_fusion_gain': {
                    'r': round(r_excl, 4),
                    'p': round(p_excl, 4),
                },
                'rho_vs_gap': {
                    'r': round(r_gap_corr, 4),
                    'p': round(p_gap_corr, 4),
                },
            },
        }
    }

    output_path = src_dir / 'score_correlation_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
