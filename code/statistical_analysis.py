"""
Phase 2-1: Statistical Analysis of VFM Complementarity
=======================================================
- Wilcoxon signed-rank test: CLIP vs DINOv2 AUROC difference
- Cohen's d effect size
- Correlation analysis: image-level vs pixel-level Oracle alphas
- Level Consistency Score
"""

import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict


def load_all_results(base_dir):
    """Load all fusion results from JSON files."""
    results = {}

    # --- Image-level results ---
    # fusion_results.json: {mvtec: {strategies: {...}, oracle: {...}}, visa: {...}}
    fusion_path = os.path.join(base_dir, "fusion_results.json")
    if os.path.exists(fusion_path):
        with open(fusion_path) as f:
            fusion_data = json.load(f)
        if 'mvtec' in fusion_data:
            results['mvtec_image'] = fusion_data['mvtec']
        if 'visa' in fusion_data:
            results['visa_image'] = fusion_data['visa']

    # adaptive_fusion_results.json has oracle_alphas for image-level
    adaptive_path = os.path.join(base_dir, "adaptive_fusion_results.json")
    if os.path.exists(adaptive_path):
        with open(adaptive_path) as f:
            adaptive_data = json.load(f)
        if 'mvtec' in adaptive_data and 'mvtec_image' in results:
            results['mvtec_image']['oracle_alphas'] = adaptive_data['mvtec'].get('oracle_alphas', {})
        if 'visa' in adaptive_data and 'visa_image' in results:
            results['visa_image']['oracle_alphas'] = adaptive_data['visa'].get('oracle_alphas', {})

    # btad_fusion_results.json: {btad_image_level: {strategies: {...}, oracle_alphas: {...}}}
    btad_path = os.path.join(base_dir, "btad_fusion_results.json")
    if os.path.exists(btad_path):
        with open(btad_path) as f:
            btad_data = json.load(f)
        if 'btad_image_level' in btad_data:
            results['btad_image'] = btad_data['btad_image_level']

    # Merge 'oracle' into 'strategies' for fusion_results.json format
    # (fusion_results.json stores Oracle separately, others include it in strategies)
    for key in ['mvtec_image', 'visa_image']:
        if key in results and 'oracle' in results[key] and 'strategies' in results[key]:
            if 'Oracle' not in results[key]['strategies']:
                results[key]['strategies']['Oracle'] = results[key]['oracle']

    # --- Pixel-level results ---
    # pixel_fusion_results.json: {mvtec: {...}, visa: {...}, btad: {...}}
    pix_path = os.path.join(base_dir, "pixel_fusion_results.json")
    if os.path.exists(pix_path):
        with open(pix_path) as f:
            pix_data = json.load(f)
        if 'mvtec' in pix_data:
            results['mvtec_pixel'] = pix_data['mvtec']
        if 'visa' in pix_data:
            results['visa_pixel'] = pix_data['visa']
        if 'btad' in pix_data:
            results['btad_pixel'] = pix_data['btad']

    return results


def wilcoxon_test(results):
    """Run Wilcoxon signed-rank tests comparing CLIP vs DINOv2."""
    print("\n" + "=" * 80)
    print("  WILCOXON SIGNED-RANK TEST: CLIP vs DINOv2")
    print("=" * 80)

    all_tests = {}

    for key, data in sorted(results.items()):
        strats = data.get('strategies', data.get('image_level', {}).get('strategies', {}))
        if not strats:
            continue

        clip_scores = strats.get('CLIP only', {})
        dino_scores = strats.get('DINOv2 only', {})
        equal_scores = strats.get('Equal (a=0.5)', {})

        if not clip_scores or not dino_scores:
            continue

        # Get common categories
        cats = sorted(set(clip_scores.keys()) & set(dino_scores.keys()))
        if len(cats) < 5:
            print(f"\n  {key}: Only {len(cats)} categories, skipping Wilcoxon (need >= 5)")
            continue

        clip_vals = np.array([clip_scores[c] for c in cats])
        dino_vals = np.array([dino_scores[c] for c in cats])
        equal_vals = np.array([equal_scores[c] for c in cats]) if equal_scores else None

        # Test 1: CLIP vs DINOv2
        diff = dino_vals - clip_vals
        stat, p_val = stats.wilcoxon(diff, alternative='two-sided')

        # Cohen's d
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

        print(f"\n  --- {key} ({len(cats)} categories) ---")
        print(f"  CLIP mean: {np.mean(clip_vals):.2f}%, DINOv2 mean: {np.mean(dino_vals):.2f}%")
        print(f"  Mean difference (DINO - CLIP): {np.mean(diff):.2f}%")
        print(f"  Wilcoxon statistic: {stat:.1f}, p-value: {p_val:.6f}")
        print(f"  Cohen's d: {d:.3f} ", end="")
        if abs(d) < 0.2:
            print("(negligible)")
        elif abs(d) < 0.5:
            print("(small)")
        elif abs(d) < 0.8:
            print("(medium)")
        else:
            print("(large)")

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  Significance: {sig}")

        all_tests[key] = {
            'n': len(cats),
            'clip_mean': round(float(np.mean(clip_vals)), 2),
            'dino_mean': round(float(np.mean(dino_vals)), 2),
            'mean_diff': round(float(np.mean(diff)), 2),
            'wilcoxon_stat': round(float(stat), 2),
            'p_value': round(float(p_val), 6),
            'cohens_d': round(float(d), 3),
            'significance': sig,
        }

        # Test 2: Equal fusion vs best single model
        if equal_vals is not None:
            best_single = np.maximum(clip_vals, dino_vals)
            diff2 = equal_vals - best_single
            if np.any(diff2 != 0):
                stat2, p_val2 = stats.wilcoxon(diff2, alternative='two-sided')
                d2 = np.mean(diff2) / np.std(diff2, ddof=1) if np.std(diff2, ddof=1) > 0 else 0
                sig2 = "***" if p_val2 < 0.001 else "**" if p_val2 < 0.01 else "*" if p_val2 < 0.05 else "n.s."
                print(f"\n  Equal fusion vs Best single:")
                print(f"  Equal mean: {np.mean(equal_vals):.2f}%, Best single mean: {np.mean(best_single):.2f}%")
                print(f"  Mean improvement: {np.mean(diff2):.2f}%")
                print(f"  Wilcoxon p-value: {p_val2:.6f} ({sig2}), Cohen's d: {d2:.3f}")

                all_tests[f"{key}_fusion_vs_single"] = {
                    'equal_mean': round(float(np.mean(equal_vals)), 2),
                    'best_single_mean': round(float(np.mean(best_single)), 2),
                    'mean_improvement': round(float(np.mean(diff2)), 2),
                    'p_value': round(float(p_val2), 6),
                    'cohens_d': round(float(d2), 3),
                    'significance': sig2,
                }

    return all_tests


def level_consistency_analysis(results):
    """Analyze consistency between image-level and pixel-level Oracle alphas."""
    print("\n\n" + "=" * 80)
    print("  LEVEL CONSISTENCY ANALYSIS (RQ2)")
    print("=" * 80)

    datasets = {
        'MVTec AD': ('mvtec_image', 'mvtec_pixel'),
        'VisA': ('visa_image', 'visa_pixel'),
        'BTAD': ('btad_image', 'btad_pixel'),
    }

    all_image_alphas = []
    all_pixel_alphas = []
    level_consistency = {}

    for ds_name, (img_key, pix_key) in datasets.items():
        if img_key not in results or pix_key not in results:
            print(f"\n  {ds_name}: Missing data (img={img_key in results}, pix={pix_key in results})")
            continue

        img_data = results[img_key]
        pix_data = results[pix_key]

        img_alphas = img_data.get('oracle_alphas', {})
        pix_alphas = pix_data.get('oracle_alphas', {})

        if not img_alphas or not pix_alphas:
            # Try nested structure
            if 'strategies' not in img_data and 'image_level' in img_data:
                img_alphas = img_data['image_level'].get('oracle_alphas', {})
            continue

        common_cats = sorted(set(img_alphas.keys()) & set(pix_alphas.keys()))
        if not common_cats:
            print(f"\n  {ds_name}: No common categories")
            continue

        print(f"\n  --- {ds_name} ({len(common_cats)} categories) ---")
        print(f"  {'Category':<15} {'Image alpha':>12} {'Pixel alpha':>12} {'LCS':>8} {'Reversal':>10}")
        print(f"  {'-'*60}")

        ds_lcs = []
        reversals = 0

        for cat in common_cats:
            ia = img_alphas[cat]
            pa = pix_alphas[cat]
            lcs = 1.0 - abs(ia - pa)
            ds_lcs.append(lcs)

            img_winner = "DINOv2" if ia > 0.5 else ("CLIP" if ia < 0.5 else "equal")
            pix_winner = "DINOv2" if pa > 0.5 else ("CLIP" if pa < 0.5 else "equal")
            is_reversal = (ia > 0.5 and pa < 0.5) or (ia < 0.5 and pa > 0.5)
            if is_reversal:
                reversals += 1

            all_image_alphas.append(ia)
            all_pixel_alphas.append(pa)

            print(f"  {cat:<15} {ia:>10.2f}   {pa:>10.2f}   {lcs:>6.2f}   {'** YES **' if is_reversal else ''}")

        mean_lcs = np.mean(ds_lcs)
        print(f"\n  Mean LCS: {mean_lcs:.3f}")
        print(f"  Reversals: {reversals}/{len(common_cats)} ({100*reversals/len(common_cats):.0f}%)")

        level_consistency[ds_name] = {
            'mean_lcs': round(float(mean_lcs), 3),
            'reversals': reversals,
            'total': len(common_cats),
            'reversal_rate': round(100 * reversals / len(common_cats), 1),
        }

    # Overall correlation
    if all_image_alphas and all_pixel_alphas:
        corr, corr_p = stats.pearsonr(all_image_alphas, all_pixel_alphas)
        spearman_corr, spearman_p = stats.spearmanr(all_image_alphas, all_pixel_alphas)

        print(f"\n  --- Overall Correlation (N={len(all_image_alphas)}) ---")
        print(f"  Pearson r = {corr:.3f} (p = {corr_p:.6f})")
        print(f"  Spearman rho = {spearman_corr:.3f} (p = {spearman_p:.6f})")

        total_reversals = sum(v['reversals'] for v in level_consistency.values())
        total_cats = sum(v['total'] for v in level_consistency.values())
        overall_lcs = np.mean([1.0 - abs(ia - pa) for ia, pa in zip(all_image_alphas, all_pixel_alphas)])

        print(f"  Overall Mean LCS: {overall_lcs:.3f}")
        print(f"  Total Reversals: {total_reversals}/{total_cats} ({100*total_reversals/total_cats:.0f}%)")

        level_consistency['overall'] = {
            'pearson_r': round(float(corr), 3),
            'pearson_p': round(float(corr_p), 6),
            'spearman_rho': round(float(spearman_corr), 3),
            'spearman_p': round(float(spearman_p), 6),
            'mean_lcs': round(float(overall_lcs), 3),
            'total_reversals': total_reversals,
            'total_cats': total_cats,
        }

    return level_consistency


def dominance_pattern_analysis(results):
    """Analyze CLIP vs DINOv2 dominance patterns across datasets and levels."""
    print("\n\n" + "=" * 80)
    print("  VFM DOMINANCE PATTERN ANALYSIS")
    print("=" * 80)

    patterns = {}

    for key, data in sorted(results.items()):
        alphas = data.get('oracle_alphas', {})
        if not alphas:
            continue

        cats = sorted(alphas.keys())
        alpha_vals = [alphas[c] for c in cats]

        clip_dom = sum(1 for a in alpha_vals if a < 0.5)
        dino_dom = sum(1 for a in alpha_vals if a > 0.5)
        equal = sum(1 for a in alpha_vals if a == 0.5)

        mean_alpha = np.mean(alpha_vals)
        std_alpha = np.std(alpha_vals)
        spread = max(alpha_vals) - min(alpha_vals)

        print(f"\n  {key}:")
        print(f"    CLIP-dominant: {clip_dom}/{len(cats)} ({100*clip_dom/len(cats):.0f}%)")
        print(f"    DINOv2-dominant: {dino_dom}/{len(cats)} ({100*dino_dom/len(cats):.0f}%)")
        print(f"    Mean alpha: {mean_alpha:.3f} (std: {std_alpha:.3f})")
        print(f"    Alpha spread: {spread:.2f} (min={min(alpha_vals):.2f}, max={max(alpha_vals):.2f})")

        patterns[key] = {
            'clip_dominant': clip_dom,
            'dino_dominant': dino_dom,
            'equal': equal,
            'total': len(cats),
            'mean_alpha': round(float(mean_alpha), 3),
            'std_alpha': round(float(std_alpha), 3),
            'alpha_spread': round(float(spread), 2),
        }

    return patterns


def fusion_benefit_analysis(results):
    """Analyze the benefit of fusion: Oracle Gap Reduction Rate (OGRR)."""
    print("\n\n" + "=" * 80)
    print("  FUSION BENEFIT ANALYSIS (OGRR)")
    print("=" * 80)

    print(f"\n  OGRR = (Equal - BestSingle) / (Oracle - BestSingle)")
    print(f"  Measures how much of the Oracle gap Equal fusion closes\n")

    ogrr_results = {}

    for key, data in sorted(results.items()):
        strats = data.get('strategies', {})
        if not strats:
            continue

        clip = strats.get('CLIP only', {})
        dino = strats.get('DINOv2 only', {})
        equal = strats.get('Equal (a=0.5)', {})
        oracle = strats.get('Oracle', {})

        if not all([clip, dino, equal, oracle]):
            continue

        cats = sorted(set(clip.keys()) & set(dino.keys()) & set(equal.keys()) & set(oracle.keys()))

        print(f"\n  --- {key} ({len(cats)} categories) ---")
        print(f"  {'Category':<15} {'BestSingle':>10} {'Equal':>8} {'Oracle':>8} {'OGRR':>8} {'Gap Closed':>10}")
        print(f"  {'-'*65}")

        ogrrs = []
        for cat in cats:
            best_single = max(clip[cat], dino[cat])
            eq = equal[cat]
            orc = oracle[cat]

            gap = orc - best_single
            if gap > 0.01:
                ogrr = (eq - best_single) / gap
            elif eq >= orc - 0.01:
                ogrr = 1.0  # No gap, fusion matches oracle
            else:
                ogrr = 0.0

            ogrrs.append(ogrr)
            print(f"  {cat:<15} {best_single:>9.1f}% {eq:>7.1f}% {orc:>7.1f}% {ogrr:>7.2f}   {gap:>+7.1f}%")

        mean_ogrr = np.mean(ogrrs)
        print(f"\n  Mean OGRR: {mean_ogrr:.3f}")

        ogrr_results[key] = {
            'mean_ogrr': round(float(mean_ogrr), 3),
            'per_category': {cat: round(float(ogrr), 3) for cat, ogrr in zip(cats, ogrrs)},
        }

    return ogrr_results


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading results...")
    results = load_all_results(base_dir)
    print(f"Loaded: {list(results.keys())}")

    # 1. Wilcoxon tests
    wilcoxon_results = wilcoxon_test(results)

    # 2. Level consistency analysis
    level_results = level_consistency_analysis(results)

    # 3. Dominance patterns
    dominance_results = dominance_pattern_analysis(results)

    # 4. Fusion benefit (OGRR)
    ogrr_results = fusion_benefit_analysis(results)

    # Save all results
    output = {
        'wilcoxon_tests': wilcoxon_results,
        'level_consistency': level_results,
        'dominance_patterns': dominance_results,
        'ogrr': ogrr_results,
    }

    output_path = os.path.join(base_dir, "statistical_analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
