"""
Experiment 17: Negative CLIP Premium Characterization

The 9/27 categories where AnomalyCLIP produces negative premium (hurts CLIP's kNN
baseline) are a key finding. This experiment deeply characterizes WHAT makes these
categories special and whether negative premium is predictable.

Analyses:
A. Taxonomy characterization of negative vs positive premium categories
B. kNN performance threshold analysis
C. DINOv2 premium in same categories (symmetry analysis)
D. Score distribution signatures in negative-premium categories
E. Practical prediction: can we identify negative-premium risk a priori?
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

src_dir = Path(__file__).parent

# Load data
with open(src_dir / 'cross_dataset_premium_results.json') as f:
    premium_data = json.load(f)

with open(src_dir / 'category_taxonomy_results.json') as f:
    tax_data = json.load(f)

with open(src_dir / 'comprehensive_eval_results.json') as f:
    eval_data = json.load(f)

taxonomy = tax_data['taxonomy']
oracle_alphas = tax_data['oracle_alphas']
per_cat = premium_data['experiment_14_cross_dataset_premium']['per_category']


def part_a_taxonomy():
    """Characterize negative premium categories by taxonomy."""
    print("=" * 70)
    print("Part A: Taxonomy of Negative vs Positive CLIP Premium")
    print("=" * 70)

    neg_cats = [r for r in per_cat if r['clip_premium'] < 0]
    pos_cats = [r for r in per_cat if r['clip_premium'] >= 0]

    print(f"\nNegative CLIP premium: {len(neg_cats)}/27")
    print(f"Positive CLIP premium: {len(pos_cats)}/27")

    # Type distribution
    for group_name, group in [('Negative', neg_cats), ('Positive', pos_cats)]:
        types = [taxonomy.get(r['category'], {}).get('type', 'unknown') for r in group]
        semantics = [taxonomy.get(r['category'], {}).get('semantic', 'unknown') for r in group]
        print(f"\n  {group_name} premium categories:")
        print(f"    Types: {dict((t, types.count(t)) for t in set(types))}")
        print(f"    Semantics: {dict((s, semantics.count(s)) for s in set(semantics))}")

    # List all negative premium categories with details
    print(f"\n--- Negative CLIP Premium Categories ---")
    print(f"{'Category':15s} {'Dataset':8s} {'Type':8s} {'CLIP_kNN':>9s} {'CLIP_nat':>9s} "
          f"{'CLIP_prem':>10s} {'DINO_kNN':>9s} {'DINO_nat':>9s} {'DINO_prem':>10s} "
          f"{'kNN_gap':>8s}")
    for r in sorted(neg_cats, key=lambda x: x['clip_premium']):
        cat_type = taxonomy.get(r['category'], {}).get('type', '?')
        knn_gap = r['dino_knn'] - r['clip_knn']
        print(f"{r['category']:15s} {r['dataset']:8s} {cat_type:8s} "
              f"{r['clip_knn']:9.2f} {r['clip_native']:9.2f} {r['clip_premium']:+10.2f} "
              f"{r['dino_knn']:9.2f} {r['dino_native']:9.2f} {r['dino_premium']:+10.2f} "
              f"{knn_gap:+8.2f}")


def part_b_knn_threshold():
    """Is there a kNN performance threshold that predicts negative premium?"""
    print("\n" + "=" * 70)
    print("Part B: kNN Performance and Negative Premium")
    print("=" * 70)

    clip_knns = [r['clip_knn'] for r in per_cat]
    clip_prems = [r['clip_premium'] for r in per_cat]
    dino_knns = [r['dino_knn'] for r in per_cat]
    dino_prems = [r['dino_premium'] for r in per_cat]

    # CLIP: kNN vs premium
    neg_clip_knns = [r['clip_knn'] for r in per_cat if r['clip_premium'] < 0]
    pos_clip_knns = [r['clip_knn'] for r in per_cat if r['clip_premium'] >= 0]

    print(f"\nCLIP kNN AUROC for negative premium categories:")
    print(f"  Mean: {np.mean(neg_clip_knns):.1f}%, Range: [{min(neg_clip_knns):.1f}, {max(neg_clip_knns):.1f}]")
    print(f"CLIP kNN AUROC for positive premium categories:")
    print(f"  Mean: {np.mean(pos_clip_knns):.1f}%, Range: [{min(pos_clip_knns):.1f}, {max(pos_clip_knns):.1f}]")

    u_stat, u_p = stats.mannwhitneyu(neg_clip_knns, pos_clip_knns, alternative='two-sided')
    print(f"  Mann-Whitney: p={u_p:.4f}")

    # Threshold analysis
    print(f"\n--- kNN Threshold Analysis ---")
    for threshold in [80, 85, 88, 90, 92, 95]:
        above = [r for r in per_cat if r['clip_knn'] >= threshold]
        above_neg = sum(1 for r in above if r['clip_premium'] < 0)
        below = [r for r in per_cat if r['clip_knn'] < threshold]
        below_neg = sum(1 for r in below if r['clip_premium'] < 0)
        print(f"  kNN >= {threshold}%: {above_neg}/{len(above)} negative ({above_neg/max(len(above),1)*100:.0f}%)"
              f"  |  kNN < {threshold}%: {below_neg}/{len(below)} negative ({below_neg/max(len(below),1)*100:.0f}%)")

    # kNN gap analysis: CLIP kNN > DINOv2 kNN?
    print(f"\n--- kNN Gap and Negative Premium ---")
    clip_better_knn = [r for r in per_cat if r['clip_knn'] > r['dino_knn']]
    dino_better_knn = [r for r in per_cat if r['dino_knn'] >= r['clip_knn']]
    print(f"CLIP kNN > DINOv2 kNN: {len(clip_better_knn)} categories")
    print(f"  Negative CLIP premium: {sum(1 for r in clip_better_knn if r['clip_premium'] < 0)}")
    print(f"DINOv2 kNN >= CLIP kNN: {len(dino_better_knn)} categories")
    print(f"  Negative CLIP premium: {sum(1 for r in dino_better_knn if r['clip_premium'] < 0)}")

    # Is "CLIP already strong at kNN" a risk factor for negative premium?
    # Compute precision/recall of rule: "CLIP kNN > DINOv2 kNN → negative premium"
    tp = sum(1 for r in per_cat if r['clip_knn'] > r['dino_knn'] and r['clip_premium'] < 0)
    fp = sum(1 for r in per_cat if r['clip_knn'] > r['dino_knn'] and r['clip_premium'] >= 0)
    fn = sum(1 for r in per_cat if r['clip_knn'] <= r['dino_knn'] and r['clip_premium'] < 0)
    tn = sum(1 for r in per_cat if r['clip_knn'] <= r['dino_knn'] and r['clip_premium'] >= 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"\nPrediction rule: 'CLIP kNN > DINOv2 kNN → negative premium'")
    print(f"  Precision: {precision:.2f} ({tp}/{tp+fp})")
    print(f"  Recall: {recall:.2f} ({tp}/{tp+fn})")
    print(f"  Accuracy: {(tp+tn)/27:.2f} ({tp+tn}/27)")


def part_c_symmetry():
    """Is the premium asymmetry symmetric? Compare CLIP and DINOv2 negative premiums."""
    print("\n" + "=" * 70)
    print("Part C: Premium Asymmetry -- CLIP vs DINOv2")
    print("=" * 70)

    clip_neg = sum(1 for r in per_cat if r['clip_premium'] < 0)
    dino_neg = sum(1 for r in per_cat if r['dino_premium'] < 0)

    print(f"\nNegative premium count:")
    print(f"  CLIP: {clip_neg}/27 ({clip_neg/27*100:.0f}%)")
    print(f"  DINOv2: {dino_neg}/27 ({dino_neg/27*100:.0f}%)")

    # Mean premium
    clip_mean = np.mean([r['clip_premium'] for r in per_cat])
    dino_mean = np.mean([r['dino_premium'] for r in per_cat])
    print(f"\nMean premium:")
    print(f"  CLIP: {clip_mean:+.2f}pp")
    print(f"  DINOv2: {dino_mean:+.2f}pp")

    # Premium variability
    clip_std = np.std([r['clip_premium'] for r in per_cat])
    dino_std = np.std([r['dino_premium'] for r in per_cat])
    print(f"\nPremium std:")
    print(f"  CLIP: {clip_std:.2f}pp")
    print(f"  DINOv2: {dino_std:.2f}pp")

    # Range
    clip_premiums = [r['clip_premium'] for r in per_cat]
    dino_premiums = [r['dino_premium'] for r in per_cat]
    print(f"\nPremium range:")
    print(f"  CLIP: [{min(clip_premiums):+.1f}, {max(clip_premiums):+.1f}]")
    print(f"  DINOv2: [{min(dino_premiums):+.1f}, {max(dino_premiums):+.1f}]")

    # Categories where BOTH have negative premium
    both_neg = [r for r in per_cat if r['clip_premium'] < 0 and r['dino_premium'] < 0]
    print(f"\nBoth negative: {len(both_neg)} categories")
    for r in both_neg:
        print(f"  {r['category']:15s}: CLIP {r['clip_premium']:+.2f}, DINOv2 {r['dino_premium']:+.2f}")

    # Categories where premiums have opposite sign
    opposite = [r for r in per_cat if r['clip_premium'] * r['dino_premium'] < 0]
    print(f"\nOpposite sign premiums: {len(opposite)}/27 ({len(opposite)/27*100:.0f}%)")

    # Correlation between CLIP and DINOv2 premiums
    r_val, p_val = stats.spearmanr(clip_premiums, dino_premiums)
    print(f"\nCLIP premium vs DINOv2 premium correlation:")
    print(f"  Spearman r={r_val:.3f}, p={p_val:.4f}")
    if r_val > 0:
        print(f"  → Positive: methods struggle/excel on similar categories")
    else:
        print(f"  → Negative: methods compensate for each other")


def part_d_oracle_alpha():
    """Relationship between negative premium and oracle fusion weight."""
    print("\n" + "=" * 70)
    print("Part D: Negative Premium and Oracle Fusion Weight")
    print("=" * 70)

    # Get image-level oracle alphas for MVTec + VisA
    all_alphas = {}
    for key in ['mvtec_image', 'visa_image']:
        all_alphas.update(oracle_alphas[key])

    matched = []
    for r in per_cat:
        cat = r['category']
        if cat in all_alphas:
            matched.append({
                'category': cat,
                'clip_premium': r['clip_premium'],
                'dino_premium': r['dino_premium'],
                'oracle_alpha': all_alphas[cat],
                'clip_knn': r['clip_knn'],
                'dino_knn': r['dino_knn'],
            })

    # Correlate premium with oracle alpha
    clip_prems = [m['clip_premium'] for m in matched]
    dino_prems = [m['dino_premium'] for m in matched]
    alphas = [m['oracle_alpha'] for m in matched]

    r_clip, p_clip = stats.spearmanr(clip_prems, alphas)
    r_dino, p_dino = stats.spearmanr(dino_prems, alphas)

    print(f"\nCorrelation: Premium vs Oracle Alpha (N={len(matched)})")
    print(f"  CLIP premium vs α: r={r_clip:+.3f}, p={p_clip:.4f}")
    print(f"  DINOv2 premium vs α: r={r_dino:+.3f}, p={p_dino:.4f}")

    # Premium difference vs alpha
    prem_diffs = [m['dino_premium'] - m['clip_premium'] for m in matched]
    r_diff, p_diff = stats.spearmanr(prem_diffs, alphas)
    print(f"  Premium diff (DINOv2-CLIP) vs α: r={r_diff:+.3f}, p={p_diff:.4f}")

    # Mean alpha for negative vs positive CLIP premium
    neg_alphas = [m['oracle_alpha'] for m in matched if m['clip_premium'] < 0]
    pos_alphas = [m['oracle_alpha'] for m in matched if m['clip_premium'] >= 0]
    print(f"\nMean oracle alpha:")
    print(f"  Negative CLIP premium: α={np.mean(neg_alphas):.3f} (N={len(neg_alphas)})")
    print(f"  Positive CLIP premium: α={np.mean(pos_alphas):.3f} (N={len(pos_alphas)})")
    u_stat, u_p = stats.mannwhitneyu(neg_alphas, pos_alphas, alternative='two-sided')
    print(f"  Mann-Whitney: p={u_p:.4f}")


def part_e_prediction_model():
    """Can we predict negative premium from multiple features?"""
    print("\n" + "=" * 70)
    print("Part E: Multi-Feature Negative Premium Prediction")
    print("=" * 70)

    # Features we can compute for each category
    features = []
    for r in per_cat:
        cat = r['category']
        tax = taxonomy.get(cat, {})
        features.append({
            'category': cat,
            'clip_knn': r['clip_knn'],
            'dino_knn': r['dino_knn'],
            'knn_gap': r['dino_knn'] - r['clip_knn'],
            'is_texture': 1 if tax.get('type') == 'texture' else 0,
            'is_color': 1 if tax.get('semantic') == 'color' else 0,
            'clip_neg_premium': 1 if r['clip_premium'] < 0 else 0,
            'clip_premium': r['clip_premium'],
        })

    # Simple rule-based prediction
    print("\n--- Simple Rule-Based Predictors ---")

    rules = [
        ("CLIP kNN > DINOv2 kNN", lambda f: f['knn_gap'] < 0),
        ("CLIP kNN > 85%", lambda f: f['clip_knn'] > 85),
        ("CLIP kNN > 89%", lambda f: f['clip_knn'] > 89),
        ("kNN gap < -5pp", lambda f: f['knn_gap'] < -5),
        ("Texture category", lambda f: f['is_texture'] == 1),
        ("CLIP kNN > DINOv2 kNN AND NOT texture",
         lambda f: f['knn_gap'] < 0 and f['is_texture'] == 0),
    ]

    for rule_name, rule_fn in rules:
        pred_pos = [f for f in features if rule_fn(f)]
        pred_neg = [f for f in features if not rule_fn(f)]
        tp = sum(1 for f in pred_pos if f['clip_neg_premium'])
        fp = sum(1 for f in pred_pos if not f['clip_neg_premium'])
        fn = sum(1 for f in pred_neg if f['clip_neg_premium'])
        tn = sum(1 for f in pred_neg if not f['clip_neg_premium'])

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        accuracy = (tp + tn) / len(features)

        print(f"\n  Rule: '{rule_name}'")
        print(f"    Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, Acc={accuracy:.2f}")
        print(f"    (TP={tp}, FP={fp}, FN={fn}, TN={tn})")

    # Logistic regression with multiple features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.array([[f['clip_knn'], f['dino_knn'], f['knn_gap']] for f in features])
    y = np.array([f['clip_neg_premium'] for f in features])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    acc = np.mean(preds == y)
    print(f"\n--- Logistic Regression (kNN features) ---")
    print(f"  Training accuracy: {acc:.2f}")
    print(f"  Coefficients: clip_knn={model.coef_[0][0]:.3f}, "
          f"dino_knn={model.coef_[0][1]:.3f}, knn_gap={model.coef_[0][2]:.3f}")

    # Leave-one-out cross-validation
    correct = 0
    for i in range(len(X)):
        X_train = np.delete(X_scaled, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X_scaled[i:i+1]
        model_cv = LogisticRegression(random_state=42, max_iter=1000)
        model_cv.fit(X_train, y_train)
        pred = model_cv.predict(X_test)
        if pred[0] == y[i]:
            correct += 1
    loo_acc = correct / len(X)
    print(f"  LOO-CV accuracy: {loo_acc:.2f}")


def main():
    print("Experiment 17: Negative CLIP Premium Characterization")
    print("=" * 70)

    part_a_taxonomy()
    part_b_knn_threshold()
    part_c_symmetry()
    part_d_oracle_alpha()
    part_e_prediction_model()

    # Save summary
    neg_cats = [r for r in per_cat if r['clip_premium'] < 0]
    pos_cats = [r for r in per_cat if r['clip_premium'] >= 0]

    output = {
        'experiment_17_negative_premium': {
            'n_clip_negative': len(neg_cats),
            'n_dino_negative': sum(1 for r in per_cat if r['dino_premium'] < 0),
            'clip_mean_premium': float(np.mean([r['clip_premium'] for r in per_cat])),
            'dino_mean_premium': float(np.mean([r['dino_premium'] for r in per_cat])),
            'negative_categories': [
                {
                    'category': r['category'],
                    'dataset': r['dataset'],
                    'clip_knn': r['clip_knn'],
                    'clip_premium': r['clip_premium'],
                    'dino_premium': r['dino_premium'],
                    'type': taxonomy.get(r['category'], {}).get('type', 'unknown'),
                } for r in neg_cats
            ],
        }
    }

    out_path = src_dir / 'negative_premium_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
