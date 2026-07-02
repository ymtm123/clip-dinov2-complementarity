"""
Experiment 16: Level Reversal Deep Analysis

Investigate WHY 43% of categories exhibit level reversal (optimal model switches
between image-level classification and pixel-level segmentation).

Analyses:
A. Level Consistency Score and reversal characterization
B. Reversal predictors (category type, method premium, kNN AUROC)
C. Per-level compensatory pattern comparison
D. Alpha shift patterns: what drives image→pixel alpha changes?
E. Reversal vs fusion benefit
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

src_dir = Path(__file__).parent

# Load data
with open(src_dir / 'comprehensive_eval_results.json') as f:
    eval_data = json.load(f)

with open(src_dir / 'category_taxonomy_results.json') as f:
    tax_data = json.load(f)

with open(src_dir / 'cross_dataset_premium_results.json') as f:
    premium_data = json.load(f)

taxonomy = tax_data['taxonomy']
oracle_alphas = tax_data['oracle_alphas']
premium_cats = premium_data['experiment_14_cross_dataset_premium']['per_category']

# Build premium lookup
premium_lookup = {}
for r in premium_cats:
    premium_lookup[r['category']] = r


def part_a_level_consistency():
    """Level Consistency Score and reversal characterization."""
    print("=" * 70)
    print("Part A: Level Consistency Score & Reversal Characterization")
    print("=" * 70)

    results = []
    for dataset_prefix, img_key, pix_key in [
        ('MVTec', 'mvtec_image', 'mvtec_pixel'),
        ('VisA', 'visa_image', 'visa_pixel'),
        ('BTAD', 'btad_image', 'btad_pixel'),
    ]:
        img_alphas = oracle_alphas[img_key]
        pix_alphas = oracle_alphas[pix_key]

        for cat in sorted(img_alphas.keys()):
            if cat not in pix_alphas:
                continue
            img_a = img_alphas[cat]
            pix_a = pix_alphas[cat]

            # Level Consistency Score
            lcs = 1.0 - abs(img_a - pix_a)

            # Reversal: image prefers one model, pixel prefers the other
            img_prefers_dino = img_a > 0.5
            pix_prefers_dino = pix_a > 0.5
            reversal = img_prefers_dino != pix_prefers_dino

            # Alpha shift
            alpha_shift = pix_a - img_a  # positive = pixel more DINOv2-leaning

            cat_type = taxonomy.get(cat, {}).get('type', 'unknown')
            semantic = taxonomy.get(cat, {}).get('semantic', 'unknown')

            results.append({
                'category': cat,
                'dataset': dataset_prefix,
                'type': cat_type,
                'semantic': semantic,
                'img_alpha': img_a,
                'pix_alpha': pix_a,
                'lcs': lcs,
                'reversal': reversal,
                'alpha_shift': alpha_shift,
            })

    # Print summary
    n_reversal = sum(1 for r in results if r['reversal'])
    n_total = len(results)
    print(f"\nTotal categories: {n_total}")
    print(f"Reversals: {n_reversal}/{n_total} ({n_reversal/n_total*100:.0f}%)")
    print(f"Mean LCS: {np.mean([r['lcs'] for r in results]):.3f}")

    # Reversal breakdown by type
    for t in ['texture', 'object']:
        subset = [r for r in results if r['type'] == t]
        n_rev = sum(1 for r in subset if r['reversal'])
        print(f"  {t}: {n_rev}/{len(subset)} ({n_rev/len(subset)*100:.0f}%) reverse")

    # Reversal breakdown by dataset
    for ds in ['MVTec', 'VisA', 'BTAD']:
        subset = [r for r in results if r['dataset'] == ds]
        n_rev = sum(1 for r in subset if r['reversal'])
        print(f"  {ds}: {n_rev}/{len(subset)} ({n_rev/len(subset)*100:.0f}%) reverse")

    # Mean alpha shift
    shifts = [r['alpha_shift'] for r in results]
    print(f"\nMean alpha shift (pixel - image): {np.mean(shifts):+.3f}")
    print(f"  Positive shift (pixel more DINOv2): {sum(1 for s in shifts if s > 0)}/{n_total}")
    print(f"  Negative shift (pixel more CLIP): {sum(1 for s in shifts if s < 0)}/{n_total}")

    # Detailed reversal categories
    print(f"\n--- Reversal categories ---")
    for r in sorted(results, key=lambda x: abs(x['alpha_shift']), reverse=True):
        if r['reversal']:
            direction = "DINOv2→CLIP" if r['img_alpha'] > 0.5 else "CLIP→DINOv2"
            print(f"  {r['category']:15s} ({r['dataset']:5s}, {r['type']:7s}): "
                  f"img α={r['img_alpha']:.2f} → pix α={r['pix_alpha']:.2f}  "
                  f"|shift|={abs(r['alpha_shift']):.2f}  [{direction}]")

    return results


def part_b_reversal_predictors(results):
    """Can we predict which categories will reverse?"""
    print("\n" + "=" * 70)
    print("Part B: Reversal Predictors")
    print("=" * 70)

    # Filter to categories with premium data (MVTec + VisA only, N=27)
    with_premium = []
    for r in results:
        cat = r['category']
        if cat in premium_lookup:
            p = premium_lookup[cat]
            r['clip_knn'] = p['clip_knn']
            r['dino_knn'] = p['dino_knn']
            r['clip_premium'] = p['clip_premium']
            r['dino_premium'] = p['dino_premium']
            r['knn_gap'] = p['dino_knn'] - p['clip_knn']  # positive = DINOv2 better at kNN
            r['premium_diff'] = p['dino_premium'] - p['clip_premium']  # DINOv2 extracts more
            with_premium.append(r)

    n = len(with_premium)
    print(f"\nCategories with premium data: {n}")

    # Binary prediction: what distinguishes reversal from non-reversal?
    rev_cats = [r for r in with_premium if r['reversal']]
    non_rev_cats = [r for r in with_premium if not r['reversal']]

    print(f"Reversal: {len(rev_cats)}, Non-reversal: {len(non_rev_cats)}")

    # Compare distributions
    for feature_name, feature_key in [
        ('kNN gap (DINOv2-CLIP)', 'knn_gap'),
        ('CLIP premium', 'clip_premium'),
        ('DINOv2 premium', 'dino_premium'),
        ('Premium diff (DINOv2-CLIP)', 'premium_diff'),
        ('CLIP kNN AUROC', 'clip_knn'),
        ('DINOv2 kNN AUROC', 'dino_knn'),
    ]:
        rev_vals = [r[feature_key] for r in rev_cats]
        non_rev_vals = [r[feature_key] for r in non_rev_cats]

        u_stat, u_p = stats.mannwhitneyu(rev_vals, non_rev_vals, alternative='two-sided')
        print(f"\n  {feature_name}:")
        print(f"    Reversal mean={np.mean(rev_vals):+.2f}, Non-reversal mean={np.mean(non_rev_vals):+.2f}")
        print(f"    Mann-Whitney U p={u_p:.4f} {'*' if u_p < 0.05 else 'n.s.'}")

    # Continuous prediction: what predicts LCS?
    print("\n--- Correlation with Level Consistency Score (LCS) ---")
    lcs_vals = [r['lcs'] for r in with_premium]
    for feature_name, feature_key in [
        ('kNN gap (DINOv2-CLIP)', 'knn_gap'),
        ('|kNN gap|', None),
        ('CLIP premium', 'clip_premium'),
        ('DINOv2 premium', 'dino_premium'),
        ('Premium diff', 'premium_diff'),
        ('Mean kNN AUROC', None),
    ]:
        if feature_key is None:
            if feature_name == '|kNN gap|':
                feat_vals = [abs(r['knn_gap']) for r in with_premium]
            elif feature_name == 'Mean kNN AUROC':
                feat_vals = [(r['clip_knn'] + r['dino_knn']) / 2 for r in with_premium]
            else:
                continue
        else:
            feat_vals = [r[feature_key] for r in with_premium]

        r_val, p_val = stats.spearmanr(feat_vals, lcs_vals)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        print(f"  {feature_name:30s}: r={r_val:+.3f}, p={p_val:.4f} {sig}")

    # Continuous prediction: what predicts alpha_shift?
    print("\n--- Correlation with Alpha Shift (pixel - image) ---")
    shift_vals = [r['alpha_shift'] for r in with_premium]
    for feature_name, feature_key in [
        ('kNN gap (DINOv2-CLIP)', 'knn_gap'),
        ('CLIP premium', 'clip_premium'),
        ('DINOv2 premium', 'dino_premium'),
        ('Premium diff', 'premium_diff'),
        ('Image-level alpha', 'img_alpha'),
    ]:
        feat_vals = [r[feature_key] for r in with_premium]
        r_val, p_val = stats.spearmanr(feat_vals, shift_vals)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        print(f"  {feature_name:30s}: r={r_val:+.3f}, p={p_val:.4f} {sig}")

    return with_premium


def part_c_per_level_compensatory(with_premium):
    """Compare compensatory pattern at image vs pixel level."""
    print("\n" + "=" * 70)
    print("Part C: Per-Level Compensatory Pattern Comparison")
    print("=" * 70)

    # Load pixel-level data from pixel_fusion_results.json
    with open(src_dir / 'pixel_fusion_results.json') as f:
        pixel_results = json.load(f)

    pixel_data = {}
    for ds_key in ['mvtec', 'visa']:
        clip_pix = pixel_results[ds_key]['strategies']['CLIP only']
        dino_pix = pixel_results[ds_key]['strategies']['DINOv2 only']

        for cat in clip_pix:
            if cat in dino_pix:
                pixel_data[cat] = {
                    'clip_pixel': clip_pix[cat],
                    'dino_pixel': dino_pix[cat],
                    'pixel_gap': dino_pix[cat] - clip_pix[cat],
                }

    # Now compute image-level gap and pixel-level gap for each category
    print(f"\nCategories with both image and pixel data: {len([c for c in with_premium if c['category'] in pixel_data])}")

    # Compare image-level preference with pixel-level preference
    categories_both = []
    for r in with_premium:
        cat = r['category']
        if cat in pixel_data:
            r_copy = dict(r)
            r_copy.update(pixel_data[cat])
            # Image-level gap (from native mode)
            p = premium_lookup.get(cat, {})
            if isinstance(p, dict):
                r_copy['clip_native'] = p.get('clip_native', 0)
                r_copy['dino_native'] = p.get('dino_native', 0)
                r_copy['image_gap'] = p.get('dino_native', 0) - p.get('clip_native', 0)
            categories_both.append(r_copy)

    if len(categories_both) < 5:
        print("  Insufficient data for per-level analysis")
        return

    # Correlation between image gap and pixel gap
    img_gaps = [r['image_gap'] for r in categories_both]
    pix_gaps = [r['pixel_gap'] for r in categories_both]
    r_val, p_val = stats.spearmanr(img_gaps, pix_gaps)
    print(f"\nImage gap vs Pixel gap correlation:")
    print(f"  Spearman r={r_val:.3f}, p={p_val:.4f}")
    print(f"  → {'Consistent' if r_val > 0.5 else 'Inconsistent'} model preferences across levels")

    # Categories where gap sign reverses
    sign_reversal = sum(1 for ig, pg in zip(img_gaps, pix_gaps) if ig * pg < 0)
    print(f"\n  Gap sign reversal: {sign_reversal}/{len(categories_both)} categories")
    for r in categories_both:
        if r['image_gap'] * r['pixel_gap'] < 0:
            img_better = "DINOv2" if r['image_gap'] > 0 else "CLIP"
            pix_better = "DINOv2" if r['pixel_gap'] > 0 else "CLIP"
            print(f"    {r['category']:15s}: Image→{img_better} (gap={r['image_gap']:+.1f}), "
                  f"Pixel→{pix_better} (gap={r['pixel_gap']:+.1f})")

    # Does compensatory pattern hold at pixel level?
    # (kNN is image-level only, so this compares kNN backbone AUROC vs native pixel AUROC)
    print("\n--- kNN AUROC vs Pixel-Level Native Gap ---")
    knn_means = [(r['clip_knn'] + r['dino_knn']) / 2 for r in categories_both]
    pix_gaps_abs = [abs(r['pixel_gap']) for r in categories_both]

    r_val, p_val = stats.spearmanr(knn_means, pix_gaps_abs)
    print(f"  Mean kNN AUROC vs |pixel gap|: r={r_val:.3f}, p={p_val:.4f}")

    return categories_both


def part_d_alpha_shift_patterns(results):
    """Characterize the direction and magnitude of alpha shifts."""
    print("\n" + "=" * 70)
    print("Part D: Alpha Shift Patterns")
    print("=" * 70)

    # Alpha shift = pixel_alpha - image_alpha
    # Positive shift: pixel is more DINOv2-leaning than image
    # Negative shift: pixel is more CLIP-leaning than image

    shifts = [(r['category'], r['dataset'], r['type'], r['alpha_shift'],
               r['img_alpha'], r['pix_alpha']) for r in results]

    pos_shifts = [s for s in shifts if s[3] > 0]
    neg_shifts = [s for s in shifts if s[3] < 0]
    zero_shifts = [s for s in shifts if s[3] == 0]

    print(f"\nPositive shift (pixel more DINOv2): {len(pos_shifts)}/{len(shifts)}")
    print(f"Negative shift (pixel more CLIP): {len(neg_shifts)}/{len(shifts)}")
    print(f"No shift: {len(zero_shifts)}/{len(shifts)}")
    print(f"Mean shift: {np.mean([s[3] for s in shifts]):+.3f}")

    # Mean shift by category type
    for t in ['texture', 'object']:
        t_shifts = [s[3] for s in shifts if s[2] == t]
        if t_shifts:
            print(f"\n  {t}: mean shift = {np.mean(t_shifts):+.3f} "
                  f"(pos: {sum(1 for s in t_shifts if s > 0)}, "
                  f"neg: {sum(1 for s in t_shifts if s < 0)})")

    # Test: is the mean shift significantly different from 0?
    all_shifts = [s[3] for s in shifts]
    t_stat, t_p = stats.ttest_1samp(all_shifts, 0)
    print(f"\nOne-sample t-test (shift ≠ 0): t={t_stat:.3f}, p={t_p:.4f}")

    # Test: does texture shift differently from object?
    text_shifts = [s[3] for s in shifts if s[2] == 'texture']
    obj_shifts = [s[3] for s in shifts if s[2] == 'object']
    if text_shifts and obj_shifts:
        u_stat, u_p = stats.mannwhitneyu(text_shifts, obj_shifts, alternative='two-sided')
        print(f"Texture vs Object shift: U p={u_p:.4f}")

    # Top 5 largest shifts
    print("\n--- Largest Alpha Shifts ---")
    for cat, ds, typ, shift, img_a, pix_a in sorted(shifts, key=lambda x: abs(x[3]), reverse=True)[:10]:
        direction = "→DINOv2" if shift > 0 else "→CLIP"
        print(f"  {cat:15s} ({ds:5s}, {typ:7s}): {shift:+.2f} [{direction}] "
              f"(img α={img_a:.2f}, pix α={pix_a:.2f})")


def part_e_reversal_vs_fusion(with_premium):
    """Do reversal categories benefit more from fusion?"""
    print("\n" + "=" * 70)
    print("Part E: Reversal vs Fusion Benefit")
    print("=" * 70)

    # Get fusion gain for each category (image level)
    fusion_gains = {}
    for eval_key in ['MVTec AD', 'VisA']:
        clip_per = eval_data[eval_key]['CLIP only']['per_category']
        dino_per = eval_data[eval_key]['DINOv2 only']['per_category']
        equal_per = eval_data[eval_key]['Equal (a=0.5)']['per_category']

        for cat in clip_per:
            best_single = max(clip_per[cat], dino_per[cat])
            equal_val = equal_per.get(cat, 0)
            fusion_gains[cat] = equal_val - best_single

    # Compare fusion gain for reversal vs non-reversal categories
    rev_gains = [fusion_gains[r['category']] for r in with_premium
                 if r['reversal'] and r['category'] in fusion_gains]
    non_rev_gains = [fusion_gains[r['category']] for r in with_premium
                     if not r['reversal'] and r['category'] in fusion_gains]

    print(f"\nMean fusion gain (image-level):")
    print(f"  Reversal categories: {np.mean(rev_gains):+.2f}pp (N={len(rev_gains)})")
    print(f"  Non-reversal categories: {np.mean(non_rev_gains):+.2f}pp (N={len(non_rev_gains)})")

    if rev_gains and non_rev_gains:
        u_stat, u_p = stats.mannwhitneyu(rev_gains, non_rev_gains, alternative='two-sided')
        print(f"  Mann-Whitney U p={u_p:.4f}")

    # Correlation: LCS vs fusion gain
    lcs_vals = [r['lcs'] for r in with_premium if r['category'] in fusion_gains]
    fg_vals = [fusion_gains[r['category']] for r in with_premium if r['category'] in fusion_gains]

    r_val, p_val = stats.spearmanr(lcs_vals, fg_vals)
    print(f"\nLCS vs Fusion Gain correlation:")
    print(f"  r={r_val:.3f}, p={p_val:.4f}")

    # Alpha shift magnitude vs fusion gain
    shift_mags = [abs(r['alpha_shift']) for r in with_premium if r['category'] in fusion_gains]
    fg_vals2 = [fusion_gains[r['category']] for r in with_premium if r['category'] in fusion_gains]

    r_val2, p_val2 = stats.spearmanr(shift_mags, fg_vals2)
    print(f"|Alpha shift| vs Fusion Gain: r={r_val2:.3f}, p={p_val2:.4f}")

    # Reversal categories: do they need different alpha for image vs pixel?
    print("\n--- Practical implication for reversal categories ---")
    print("  If using fixed α for both levels:")
    for r in sorted(with_premium, key=lambda x: abs(x['alpha_shift']), reverse=True):
        if r['reversal']:
            # Loss from using image-optimal alpha at pixel level (or vice versa)
            print(f"  {r['category']:15s}: img α*={r['img_alpha']:.2f}, pix α*={r['pix_alpha']:.2f}  "
                  f"→ using either would be suboptimal for the other task")


def main():
    print("Experiment 16: Level Reversal Deep Analysis")
    print("=" * 70)

    results = part_a_level_consistency()
    with_premium = part_b_reversal_predictors(results)
    categories_both = part_c_per_level_compensatory(with_premium)
    part_d_alpha_shift_patterns(results)
    part_e_reversal_vs_fusion(with_premium)

    # Save key results
    output = {
        'experiment_16_level_reversal': {
            'n_total': len(results),
            'n_reversal': sum(1 for r in results if r['reversal']),
            'reversal_rate': sum(1 for r in results if r['reversal']) / len(results),
            'mean_lcs': float(np.mean([r['lcs'] for r in results])),
            'mean_alpha_shift': float(np.mean([r['alpha_shift'] for r in results])),
            'per_category': [{
                'category': r['category'],
                'dataset': r['dataset'],
                'type': r['type'],
                'img_alpha': r['img_alpha'],
                'pix_alpha': r['pix_alpha'],
                'lcs': r['lcs'],
                'reversal': r['reversal'],
                'alpha_shift': r['alpha_shift'],
            } for r in results],
        }
    }

    out_path = src_dir / 'level_reversal_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
