"""
LOCO Validation: Leave-One-Category-Out Taxonomy Prediction
=============================================================

Goal: Quantify the practical utility of category taxonomy for VFM selection.

For each of 30 categories:
1. Use remaining 29 categories to learn taxonomy → α mapping
2. Predict optimal α for the held-out category
3. Compute regret vs oracle α
4. Also compute "wrong model penalty" (choosing worst single model)

This validates whether product attributes can guide VFM selection
without needing anomaly detection results.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def main():
    src_dir = Path(__file__).parent

    # Load taxonomy and oracle alphas
    with open(src_dir / 'category_taxonomy_results.json') as f:
        data = json.load(f)

    taxonomy = data['taxonomy']
    oracle_alphas = data['oracle_alphas']

    # Load comprehensive evaluation results for per-category AUROCs
    with open(src_dir / 'comprehensive_eval_results.json') as f:
        eval_data = json.load(f)

    # Build per-category data across datasets
    categories = []
    for cat, attrs in taxonomy.items():
        dataset = attrs['dataset']
        # Find the relevant eval key
        if dataset == 'MVTec':
            eval_key = 'MVTec AD'
            alpha_key = 'mvtec_image'
        elif dataset == 'VisA':
            eval_key = 'VisA'
            alpha_key = 'visa_image'
        elif dataset == 'BTAD':
            eval_key = 'BTAD'
            alpha_key = 'btad_image'
        else:
            continue

        # Get per-category AUROCs
        clip_auroc = eval_data[eval_key]['CLIP only']['per_category'].get(cat)
        dino_auroc = eval_data[eval_key]['DINOv2 only']['per_category'].get(cat)
        equal_auroc = eval_data[eval_key]['Equal (a=0.5)']['per_category'].get(cat)

        # For alpha-specific AUROCs, try to get from various fixed alphas
        alpha_aurocs = {}
        for strategy_name, strategy_data in eval_data[eval_key].items():
            if 'per_category' in strategy_data and cat in strategy_data['per_category']:
                alpha_aurocs[strategy_name] = strategy_data['per_category'][cat]

        oracle_alpha = oracle_alphas[alpha_key].get(cat)

        if clip_auroc is None or dino_auroc is None or oracle_alpha is None:
            continue

        # Compute oracle AUROC (from eval_data if available)
        oracle_auroc = eval_data[eval_key].get('Oracle', {}).get('per_category', {}).get(cat)
        if oracle_auroc is None:
            # Estimate from best of the strategies
            oracle_auroc = max(alpha_aurocs.values()) if alpha_aurocs else max(clip_auroc, dino_auroc)

        categories.append({
            'name': cat,
            'dataset': dataset,
            'type': attrs['type'],
            'semantic': attrs['semantic'],
            'structure': attrs['structure'],
            'defect_scale': attrs['defect_scale'],
            'oracle_alpha': oracle_alpha,
            'clip_auroc': clip_auroc,
            'dino_auroc': dino_auroc,
            'equal_auroc': equal_auroc,
            'oracle_auroc': oracle_auroc,
            'alpha_aurocs': alpha_aurocs,
        })

    print(f"Total categories: {len(categories)}")

    # ============================================================
    # Strategy 1: Simple taxonomy rule
    # ============================================================
    print("\n" + "=" * 90)
    print("Strategy 1: Simple Taxonomy Rule")
    print("  Texture+Color → α=0.3 (CLIP-heavy)")
    print("  Object+Shape → α=0.7 (DINOv2-heavy)")
    print("  Otherwise → α=0.5 (Equal)")
    print("=" * 90)

    def simple_rule(cat_attrs):
        """Simple taxonomy-based α prediction"""
        if cat_attrs['type'] == 'texture' and cat_attrs['semantic'] == 'color':
            return 0.30  # CLIP-heavy
        elif cat_attrs['type'] == 'object' and cat_attrs['semantic'] == 'shape':
            return 0.70  # DINOv2-heavy
        else:
            return 0.50  # Equal

    evaluate_strategy(categories, simple_rule, "Simple Rule")

    # ============================================================
    # Strategy 2: LOCO Mean Rule (use mean oracle α from same-attribute group)
    # ============================================================
    print("\n" + "=" * 90)
    print("Strategy 2: LOCO Mean Rule")
    print("  For each category, predict α as mean oracle α of same-type categories (excluding self)")
    print("=" * 90)

    def loco_mean_rule(cat_attrs, all_cats, exclude_name):
        """Use mean oracle α from same product-type group, excluding self"""
        same_type = [c for c in all_cats if c['type'] == cat_attrs['type'] and c['name'] != exclude_name]
        if len(same_type) == 0:
            return 0.5
        return np.mean([c['oracle_alpha'] for c in same_type])

    results_loco = []
    for i, cat in enumerate(categories):
        predicted_alpha = loco_mean_rule(cat, categories, cat['name'])
        results_loco.append({
            'name': cat['name'],
            'predicted_alpha': predicted_alpha,
            'oracle_alpha': cat['oracle_alpha'],
        })

    evaluate_strategy_from_results(categories, results_loco, "LOCO Type-Mean")

    # ============================================================
    # Strategy 3: LOCO Semantic Rule (use semantic attribute)
    # ============================================================
    print("\n" + "=" * 90)
    print("Strategy 3: LOCO Semantic-Mean Rule")
    print("  Predict α as mean oracle α of same semantic anomaly type (excluding self)")
    print("=" * 90)

    results_sem = []
    for cat in categories:
        same_semantic = [c for c in categories if c['semantic'] == cat['semantic'] and c['name'] != cat['name']]
        if len(same_semantic) > 0:
            predicted = np.mean([c['oracle_alpha'] for c in same_semantic])
        else:
            predicted = 0.5
        results_sem.append({
            'name': cat['name'],
            'predicted_alpha': predicted,
            'oracle_alpha': cat['oracle_alpha'],
        })

    evaluate_strategy_from_results(categories, results_sem, "LOCO Semantic-Mean")

    # ============================================================
    # Strategy 4: LOCO Combined Rule (type + semantic)
    # ============================================================
    print("\n" + "=" * 90)
    print("Strategy 4: LOCO Combined Rule (type × semantic)")
    print("  Predict α as mean oracle α of same (type, semantic) group (excluding self)")
    print("=" * 90)

    results_combined = []
    for cat in categories:
        same_group = [c for c in categories
                      if c['type'] == cat['type'] and c['semantic'] == cat['semantic']
                      and c['name'] != cat['name']]
        if len(same_group) > 0:
            predicted = np.mean([c['oracle_alpha'] for c in same_group])
        else:
            # Fallback to type-only
            same_type = [c for c in categories if c['type'] == cat['type'] and c['name'] != cat['name']]
            predicted = np.mean([c['oracle_alpha'] for c in same_type]) if same_type else 0.5
        results_combined.append({
            'name': cat['name'],
            'predicted_alpha': predicted,
            'oracle_alpha': cat['oracle_alpha'],
        })

    evaluate_strategy_from_results(categories, results_combined, "LOCO Combined")

    # ============================================================
    # Baseline comparisons
    # ============================================================
    print("\n" + "=" * 90)
    print("Baseline Comparisons (no LOCO, fixed strategies)")
    print("=" * 90)

    # Always CLIP
    print("\nBaseline: Always CLIP (α=0.0)")
    evaluate_strategy(categories, lambda _: 0.0, "Always CLIP")

    # Always DINOv2
    print("\nBaseline: Always DINOv2 (α=1.0)")
    evaluate_strategy(categories, lambda _: 1.0, "Always DINOv2")

    # Always Equal
    print("\nBaseline: Always Equal (α=0.5)")
    evaluate_strategy(categories, lambda _: 0.5, "Always Equal")

    # Fixed (α=0.65)
    print("\nBaseline: Fixed α=0.65")
    evaluate_strategy(categories, lambda _: 0.65, "Fixed α=0.65")

    # ============================================================
    # Wrong Model Penalty
    # ============================================================
    print("\n" + "=" * 90)
    print("Wrong Model Penalty Analysis")
    print("=" * 90)

    penalties = []
    for cat in categories:
        best_single = max(cat['clip_auroc'], cat['dino_auroc'])
        worst_single = min(cat['clip_auroc'], cat['dino_auroc'])
        penalty = best_single - worst_single
        penalties.append({
            'name': cat['name'],
            'dataset': cat['dataset'],
            'clip_auroc': cat['clip_auroc'],
            'dino_auroc': cat['dino_auroc'],
            'best': 'CLIP' if cat['clip_auroc'] > cat['dino_auroc'] else 'DINOv2',
            'penalty': penalty,
        })

    penalties.sort(key=lambda x: -x['penalty'])

    print(f"\n{'Category':15s} {'Dataset':>8s} {'Best':>8s} {'CLIP':>7s} {'DINO':>7s} {'Penalty':>8s}")
    print("-" * 60)
    for p in penalties:
        print(f"{p['name']:15s} {p['dataset']:>8s} {p['best']:>8s} {p['clip_auroc']:6.1f}% {p['dino_auroc']:6.1f}% {p['penalty']:7.1f}pp")

    print(f"\nMean wrong model penalty: {np.mean([p['penalty'] for p in penalties]):.2f}pp")
    print(f"Median: {np.median([p['penalty'] for p in penalties]):.2f}pp")
    print(f"Max: {max(p['penalty'] for p in penalties):.2f}pp ({penalties[0]['name']})")
    print(f"Categories with penalty >5pp: {sum(1 for p in penalties if p['penalty'] > 5)}/30")

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n" + "=" * 90)
    print("Summary: Strategy Comparison")
    print("=" * 90)

    # (collected in evaluate functions, print summary here)

    # ============================================================
    # Save results
    # ============================================================
    output = {
        'loco_validation': {
            'total_categories': len(categories),
            'wrong_model_penalty': {
                'mean_pp': round(np.mean([p['penalty'] for p in penalties]), 2),
                'median_pp': round(np.median([p['penalty'] for p in penalties]), 2),
                'max_pp': round(max(p['penalty'] for p in penalties), 2),
                'per_category': penalties,
            },
        }
    }

    output_path = src_dir / 'loco_validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


def evaluate_strategy(categories, rule_fn, name):
    """Evaluate a strategy that takes category attributes and returns predicted α"""
    results = []
    for cat in categories:
        predicted_alpha = rule_fn(cat)
        results.append({
            'name': cat['name'],
            'predicted_alpha': predicted_alpha,
            'oracle_alpha': cat['oracle_alpha'],
        })

    evaluate_strategy_from_results(categories, results, name)


def evaluate_strategy_from_results(categories, results, name):
    """Evaluate given predicted alphas against oracle and compute various metrics"""
    cat_map = {c['name']: c for c in categories}

    regrets = []
    correct_direction = 0
    total = 0

    print(f"\n{'Category':15s} {'Pred α':>7s} {'Oracle α':>9s} {'|Δα|':>6s} {'Dir?':>5s} {'Pred AUROC':>11s} {'Oracle AUROC':>12s} {'Regret':>8s}")
    print("-" * 80)

    for r in results:
        cat = cat_map[r['name']]
        pred_alpha = r['predicted_alpha']
        oracle_alpha = r['oracle_alpha']

        # Alpha error
        alpha_err = abs(pred_alpha - oracle_alpha)

        # Direction check: does predicted alpha correctly pick the better model?
        # oracle_alpha > 0.5 means DINOv2 is better; < 0.5 means CLIP is better
        oracle_prefers_dino = oracle_alpha > 0.5
        pred_prefers_dino = pred_alpha > 0.5
        direction_correct = (oracle_prefers_dino == pred_prefers_dino) or abs(oracle_alpha - 0.5) < 0.05
        if direction_correct:
            correct_direction += 1
        total += 1

        # Compute predicted AUROC from linear interpolation
        # AUROC(α) ≈ α * dino_auroc + (1-α) * clip_auroc (approximate)
        # Actually, fusion AUROC is NOT linear in α, but we can approximate
        # Using the actual eval data if available
        clip_a = cat['clip_auroc']
        dino_a = cat['dino_auroc']
        equal_a = cat['equal_auroc']
        oracle_a = cat['oracle_auroc']

        # Better approximation: use quadratic interpolation through (0, CLIP), (0.5, Equal), (1, DINOv2)
        # AUROC(α) = a*α^2 + b*α + c where c=CLIP, a+b+c=DINO, 0.25a+0.5b+c=Equal
        # → c = CLIP
        # → a + b = DINO - CLIP
        # → 0.25a + 0.5b = Equal - CLIP
        # → a = 2*(DINO - CLIP) - 4*(Equal - CLIP) = 2*DINO - 2*CLIP - 4*Equal + 4*CLIP = 2*DINO + 2*CLIP - 4*Equal
        # → b = DINO - CLIP - a = DINO - CLIP - 2*DINO - 2*CLIP + 4*Equal = -DINO - 3*CLIP + 4*Equal
        c = clip_a
        a = 2*dino_a + 2*clip_a - 4*equal_a
        b = dino_a - clip_a - a

        pred_auroc = a * pred_alpha**2 + b * pred_alpha + c
        oracle_auroc_est = a * oracle_alpha**2 + b * oracle_alpha + c

        regret = max(0, oracle_a - pred_auroc)
        regrets.append(regret)

        dir_str = 'OK' if direction_correct else 'NG'
        print(f"{r['name']:15s} {pred_alpha:6.2f} {oracle_alpha:8.2f} {alpha_err:5.2f} {dir_str:>5s} {pred_auroc:10.1f}% {oracle_a:11.1f}% {regret:7.2f}pp")

    print(f"\n--- {name} Summary ---")
    alpha_errors = [abs(r['predicted_alpha'] - r['oracle_alpha']) for r in results]
    print(f"  Mean |Δα|: {np.mean(alpha_errors):.3f}")
    print(f"  Direction accuracy: {correct_direction}/{total} ({correct_direction/total*100:.1f}%)")
    print(f"  Mean regret: {np.mean(regrets):.2f}pp")
    print(f"  Max regret: {max(regrets):.2f}pp")
    print(f"  Categories with regret > 1pp: {sum(1 for r in regrets if r > 1)}/{total}")


if __name__ == '__main__':
    main()
