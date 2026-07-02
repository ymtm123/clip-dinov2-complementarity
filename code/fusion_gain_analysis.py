"""
Experiment 18: Fusion Gain Predictors & Oracle Gap Decomposition

Analyzes what predicts which categories benefit most from score-level fusion,
decomposes the oracle gap, and identifies fusion-friendly vs fusion-resistant categories.

Parts:
A. Fusion gain computation for all 30 categories
B. Univariate predictors of fusion gain
C. Oracle gap decomposition (which categories contribute most to Fixed-vs-Oracle gap)
D. Fusion-friendly vs fusion-resistant category characterization
E. Multi-variate regression model of fusion gain
"""

import json
import numpy as np
from scipy import stats
import os

# Output file
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "fusion_gain_results.json")

def load_data():
    """Load all required data files."""
    base = os.path.dirname(__file__)

    with open(os.path.join(base, "comprehensive_eval_results.json")) as f:
        comp = json.load(f)
    with open(os.path.join(base, "category_taxonomy_results.json")) as f:
        tax = json.load(f)
    with open(os.path.join(base, "cross_dataset_premium_results.json")) as f:
        prem = json.load(f)
    with open(os.path.join(base, "level_reversal_results.json")) as f:
        rev = json.load(f)

    return comp, tax, prem, rev


def build_category_table(comp, tax, prem, rev):
    """Build a unified table of per-category features and fusion gain."""
    rows = []

    # Map premium data by category
    prem_map = {}
    for p in prem["experiment_14_cross_dataset_premium"]["per_category"]:
        prem_map[p["category"]] = p

    # Map reversal data by category
    rev_map = {}
    for r in rev["experiment_16_level_reversal"]["per_category"]:
        key = r["category"]
        rev_map[key] = r

    # Process each dataset
    dataset_map = {
        "MVTec AD": ("MVTec AD", list(comp["MVTec AD"]["CLIP only"]["per_category"].keys())),
        "VisA": ("VisA", list(comp["VisA"]["CLIP only"]["per_category"].keys())),
        "BTAD": ("BTAD", list(comp["BTAD"]["CLIP only"]["per_category"].keys()))
    }

    for ds_name, (ds_key, categories) in dataset_map.items():
        for cat in categories:
            clip_auroc = comp[ds_key]["CLIP only"]["per_category"][cat]
            dino_auroc = comp[ds_key]["DINOv2 only"]["per_category"][cat]
            equal_auroc = comp[ds_key]["Equal (a=0.5)"]["per_category"][cat]
            fixed_auroc = comp[ds_key]["Fixed (a=0.65)"]["per_category"][cat]
            oracle_auroc = comp[ds_key]["Oracle"]["per_category"][cat]
            adafuse_auroc = comp[ds_key]["AdaFuse-PF"]["per_category"][cat]

            best_single = max(clip_auroc, dino_auroc)
            worst_single = min(clip_auroc, dino_auroc)
            auroc_gap = abs(clip_auroc - dino_auroc)
            fusion_gain = oracle_auroc - best_single
            equal_gain = equal_auroc - best_single
            fixed_regret = oracle_auroc - fixed_auroc
            equal_regret = oracle_auroc - equal_auroc

            # Taxonomy
            tax_data = tax["taxonomy"].get(cat, {})
            cat_type = tax_data.get("type", "unknown")
            cat_semantic = tax_data.get("semantic", "unknown")
            is_texture = 1 if cat_type == "texture" else 0

            # Oracle alpha
            ds_alpha_key = {"MVTec AD": "mvtec_image", "VisA": "visa_image", "BTAD": "btad_image"}[ds_name]
            oracle_alpha = tax["oracle_alphas"][ds_alpha_key].get(cat, 0.5)
            alpha_extremity = abs(oracle_alpha - 0.5)

            # Premium data (MVTec + VisA only, N=27)
            prem_data = prem_map.get(cat, None)
            clip_premium = prem_data["clip_premium"] if prem_data else None
            dino_premium = prem_data["dino_premium"] if prem_data else None
            premium_diff = (dino_premium - clip_premium) if (prem_data and clip_premium is not None and dino_premium is not None) else None

            # Reversal data
            rev_data = rev_map.get(cat, None)
            is_reversal = rev_data["reversal"] if rev_data else None
            lcs = rev_data["lcs"] if rev_data else None

            rows.append({
                "category": cat,
                "dataset": ds_name,
                "type": cat_type,
                "semantic": cat_semantic,
                "is_texture": is_texture,
                "clip_auroc": clip_auroc,
                "dino_auroc": dino_auroc,
                "best_single": best_single,
                "worst_single": worst_single,
                "auroc_gap": auroc_gap,
                "equal_auroc": equal_auroc,
                "fixed_auroc": fixed_auroc,
                "oracle_auroc": oracle_auroc,
                "adafuse_auroc": adafuse_auroc,
                "fusion_gain": fusion_gain,
                "equal_gain": equal_gain,
                "fixed_regret": fixed_regret,
                "equal_regret": equal_regret,
                "oracle_alpha": oracle_alpha,
                "alpha_extremity": alpha_extremity,
                "clip_premium": clip_premium,
                "dino_premium": dino_premium,
                "premium_diff": premium_diff,
                "is_reversal": is_reversal,
                "lcs": lcs
            })

    return rows


def part_a_fusion_gain(rows):
    """Part A: Fusion gain computation and summary statistics."""
    print("=" * 70)
    print("PART A: Fusion Gain Computation (30 categories)")
    print("=" * 70)

    gains = [r["fusion_gain"] for r in rows]
    equal_gains = [r["equal_gain"] for r in rows]

    print(f"\nOracle fusion gain (oracle - best single):")
    print(f"  Mean:   {np.mean(gains):.3f} pp")
    print(f"  Median: {np.median(gains):.3f} pp")
    print(f"  Std:    {np.std(gains):.3f} pp")
    print(f"  Min:    {np.min(gains):.3f} pp ({rows[np.argmin(gains)]['category']})")
    print(f"  Max:    {np.max(gains):.3f} pp ({rows[np.argmax(gains)]['category']})")

    # How many categories benefit from fusion?
    n_positive = sum(1 for g in gains if g > 0)
    n_zero = sum(1 for g in gains if g == 0)
    n_negative = sum(1 for g in gains if g < 0)
    print(f"\n  Positive gain: {n_positive}/30 ({n_positive/30*100:.0f}%)")
    print(f"  Zero gain:     {n_zero}/30 ({n_zero/30*100:.0f}%)")
    print(f"  Negative gain: {n_negative}/30 ({n_negative/30*100:.0f}%)")

    # Equal fusion gain
    eq_positive = sum(1 for g in equal_gains if g > 0)
    print(f"\nEqual fusion gain (equal - best single):")
    print(f"  Mean:   {np.mean(equal_gains):.3f} pp")
    print(f"  Positive: {eq_positive}/30 ({eq_positive/30*100:.0f}%)")

    # Top 5 fusion-friendly categories
    sorted_rows = sorted(rows, key=lambda x: x["fusion_gain"], reverse=True)
    print(f"\nTop 5 fusion-friendly categories:")
    for r in sorted_rows[:5]:
        print(f"  {r['category']:15s} ({r['dataset']:8s}): gain={r['fusion_gain']:+.2f}pp  "
              f"CLIP={r['clip_auroc']:.1f}  DINOv2={r['dino_auroc']:.1f}  Oracle={r['oracle_auroc']:.1f}  "
              f"alpha={r['oracle_alpha']:.2f}")

    # Bottom 5 (fusion-resistant)
    print(f"\nTop 5 fusion-resistant categories (lowest gain):")
    for r in sorted_rows[-5:]:
        print(f"  {r['category']:15s} ({r['dataset']:8s}): gain={r['fusion_gain']:+.2f}pp  "
              f"CLIP={r['clip_auroc']:.1f}  DINOv2={r['dino_auroc']:.1f}  Oracle={r['oracle_auroc']:.1f}  "
              f"alpha={r['oracle_alpha']:.2f}")

    return {
        "mean_gain": float(np.mean(gains)),
        "median_gain": float(np.median(gains)),
        "std_gain": float(np.std(gains)),
        "n_positive": n_positive,
        "n_zero": n_zero,
        "top5_friendly": [{"cat": r["category"], "dataset": r["dataset"],
                           "gain": r["fusion_gain"]} for r in sorted_rows[:5]],
        "top5_resistant": [{"cat": r["category"], "dataset": r["dataset"],
                            "gain": r["fusion_gain"]} for r in sorted_rows[-5:]]
    }


def part_b_univariate_predictors(rows):
    """Part B: Univariate predictors of fusion gain."""
    print("\n" + "=" * 70)
    print("PART B: Univariate Predictors of Fusion Gain")
    print("=" * 70)

    gains = np.array([r["fusion_gain"] for r in rows])
    results = {}

    # B1: AUROC gap vs fusion gain
    gaps = np.array([r["auroc_gap"] for r in rows])
    r_gap, p_gap = stats.spearmanr(gaps, gains)
    print(f"\n1. |AUROC gap| vs Fusion Gain:")
    print(f"   Spearman r = {r_gap:.3f}, p = {p_gap:.4f}")
    results["auroc_gap"] = {"r": float(r_gap), "p": float(p_gap)}

    # B2: Best single AUROC vs fusion gain (difficulty)
    best = np.array([r["best_single"] for r in rows])
    r_best, p_best = stats.spearmanr(best, gains)
    print(f"\n2. Best single AUROC vs Fusion Gain (difficulty effect):")
    print(f"   Spearman r = {r_best:.3f}, p = {p_best:.4f}")
    results["best_single"] = {"r": float(r_best), "p": float(p_best)}

    # B3: Alpha extremity vs fusion gain
    extremity = np.array([r["alpha_extremity"] for r in rows])
    r_ext, p_ext = stats.spearmanr(extremity, gains)
    print(f"\n3. Alpha extremity |alpha-0.5| vs Fusion Gain:")
    print(f"   Spearman r = {r_ext:.3f}, p = {p_ext:.4f}")
    results["alpha_extremity"] = {"r": float(r_ext), "p": float(p_ext)}

    # B4: Taxonomy type vs fusion gain
    texture_gains = [r["fusion_gain"] for r in rows if r["is_texture"] == 1]
    object_gains = [r["fusion_gain"] for r in rows if r["is_texture"] == 0]
    u_stat, p_type = stats.mannwhitneyu(texture_gains, object_gains, alternative='two-sided')
    print(f"\n4. Texture vs Object fusion gain:")
    print(f"   Texture mean: {np.mean(texture_gains):.3f}pp (N={len(texture_gains)})")
    print(f"   Object mean:  {np.mean(object_gains):.3f}pp (N={len(object_gains)})")
    print(f"   Mann-Whitney p = {p_type:.4f}")
    results["type_texture_vs_object"] = {
        "texture_mean": float(np.mean(texture_gains)),
        "object_mean": float(np.mean(object_gains)),
        "p": float(p_type)
    }

    # B5: Premium diff vs fusion gain (N=27 only, MVTec+VisA)
    prem_rows = [r for r in rows if r["premium_diff"] is not None]
    if prem_rows:
        prem_diffs = np.array([r["premium_diff"] for r in prem_rows])
        prem_gains = np.array([r["fusion_gain"] for r in prem_rows])
        r_prem, p_prem = stats.spearmanr(prem_diffs, prem_gains)
        print(f"\n5. Premium diff (DINOv2-CLIP) vs Fusion Gain (N={len(prem_rows)}):")
        print(f"   Spearman r = {r_prem:.3f}, p = {p_prem:.4f}")
        results["premium_diff"] = {"r": float(r_prem), "p": float(p_prem), "N": len(prem_rows)}

    # B6: Reversal status vs fusion gain
    rev_rows = [r for r in rows if r["is_reversal"] is not None]
    if rev_rows:
        rev_gains = [r["fusion_gain"] for r in rev_rows if r["is_reversal"]]
        nonrev_gains = [r["fusion_gain"] for r in rev_rows if not r["is_reversal"]]
        u_stat2, p_rev = stats.mannwhitneyu(rev_gains, nonrev_gains, alternative='two-sided')
        print(f"\n6. Reversal vs Non-reversal fusion gain:")
        print(f"   Reversal mean:     {np.mean(rev_gains):.3f}pp (N={len(rev_gains)})")
        print(f"   Non-reversal mean: {np.mean(nonrev_gains):.3f}pp (N={len(nonrev_gains)})")
        print(f"   Mann-Whitney p = {p_rev:.4f}")
        results["reversal_status"] = {
            "reversal_mean": float(np.mean(rev_gains)),
            "nonreversal_mean": float(np.mean(nonrev_gains)),
            "p": float(p_rev)
        }

    # B7: LCS vs fusion gain
    lcs_rows = [r for r in rows if r["lcs"] is not None]
    if lcs_rows:
        lcs_vals = np.array([r["lcs"] for r in lcs_rows])
        lcs_gains = np.array([r["fusion_gain"] for r in lcs_rows])
        r_lcs, p_lcs = stats.spearmanr(lcs_vals, lcs_gains)
        print(f"\n7. Level Consistency Score vs Fusion Gain (N={len(lcs_rows)}):")
        print(f"   Spearman r = {r_lcs:.3f}, p = {p_lcs:.4f}")
        results["lcs"] = {"r": float(r_lcs), "p": float(p_lcs), "N": len(lcs_rows)}

    # B8: Worst single AUROC vs fusion gain
    worst = np.array([r["worst_single"] for r in rows])
    r_worst, p_worst = stats.spearmanr(worst, gains)
    print(f"\n8. Worst single AUROC vs Fusion Gain:")
    print(f"   Spearman r = {r_worst:.3f}, p = {p_worst:.4f}")
    results["worst_single"] = {"r": float(r_worst), "p": float(p_worst)}

    # Summary of significant predictors
    print(f"\n--- Summary of Predictors (p < 0.05) ---")
    for name, vals in results.items():
        if "p" in vals and vals["p"] < 0.05:
            r_val = vals.get("r", "N/A")
            print(f"  {name}: r={r_val}, p={vals['p']:.4f} ***")

    return results


def part_c_oracle_gap_decomposition(rows):
    """Part C: Oracle gap decomposition."""
    print("\n" + "=" * 70)
    print("PART C: Oracle Gap Decomposition")
    print("=" * 70)

    # Fixed regret per category
    fixed_regrets = [(r["category"], r["dataset"], r["fixed_regret"], r["oracle_alpha"])
                     for r in rows]
    fixed_regrets.sort(key=lambda x: x[2], reverse=True)

    total_fixed_regret = sum(r[2] for r in fixed_regrets)
    mean_fixed_regret = total_fixed_regret / len(fixed_regrets)

    print(f"\nFixed (alpha=0.65) vs Oracle gap decomposition:")
    print(f"  Total regret: {total_fixed_regret:.2f}pp")
    print(f"  Mean regret:  {mean_fixed_regret:.3f}pp")

    # Which categories contribute most to the gap?
    print(f"\nTop 10 contributors to Fixed-vs-Oracle gap:")
    cumulative = 0
    for i, (cat, ds, regret, alpha) in enumerate(fixed_regrets[:10]):
        cumulative += regret
        pct = cumulative / total_fixed_regret * 100 if total_fixed_regret > 0 else 0
        print(f"  {i+1}. {cat:15s} ({ds:8s}): regret={regret:.2f}pp  "
              f"alpha*={alpha:.2f}  cumul={pct:.0f}%")

    # Categories with zero regret
    n_zero_regret = sum(1 for r in fixed_regrets if r[2] < 0.01)
    print(f"\n  Categories with <0.01pp regret: {n_zero_regret}/30 ({n_zero_regret/30*100:.0f}%)")

    # Equal fusion regret analysis
    equal_regrets = [(r["category"], r["dataset"], r["equal_regret"], r["oracle_alpha"])
                     for r in rows]
    equal_regrets.sort(key=lambda x: x[2], reverse=True)
    total_equal_regret = sum(r[2] for r in equal_regrets)

    print(f"\nEqual (alpha=0.5) vs Oracle gap decomposition:")
    print(f"  Total regret: {total_equal_regret:.2f}pp")
    print(f"  Mean regret:  {total_equal_regret/len(equal_regrets):.3f}pp")

    print(f"\nTop 5 contributors to Equal-vs-Oracle gap:")
    cumulative = 0
    for i, (cat, ds, regret, alpha) in enumerate(equal_regrets[:5]):
        cumulative += regret
        pct = cumulative / total_equal_regret * 100 if total_equal_regret > 0 else 0
        print(f"  {i+1}. {cat:15s} ({ds:8s}): regret={regret:.2f}pp  "
              f"alpha*={alpha:.2f}  cumul={pct:.0f}%")

    # Oracle alpha vs regret relationship
    alphas = np.array([r["oracle_alpha"] for r in rows])
    f_regrets = np.array([r["fixed_regret"] for r in rows])
    r_ar, p_ar = stats.spearmanr(alphas, f_regrets)
    print(f"\n  Oracle alpha vs Fixed regret: r={r_ar:.3f}, p={p_ar:.4f}")

    # Alpha distance from 0.65 (fixed point) vs regret
    alpha_dist = np.abs(alphas - 0.65)
    r_dist, p_dist = stats.spearmanr(alpha_dist, f_regrets)
    print(f"  |alpha - 0.65| vs Fixed regret: r={r_dist:.3f}, p={p_dist:.4f}")

    return {
        "fixed_total_regret": float(total_fixed_regret),
        "fixed_mean_regret": float(mean_fixed_regret),
        "equal_total_regret": float(total_equal_regret),
        "n_zero_regret": n_zero_regret,
        "alpha_vs_regret_r": float(r_ar),
        "alpha_vs_regret_p": float(p_ar),
        "alpha_dist_vs_regret_r": float(r_dist),
        "alpha_dist_vs_regret_p": float(p_dist),
        "top5_fixed_contributors": [
            {"cat": cat, "dataset": ds, "regret": float(regret), "alpha": float(alpha)}
            for cat, ds, regret, alpha in fixed_regrets[:5]
        ]
    }


def part_d_characterization(rows):
    """Part D: Fusion-friendly vs fusion-resistant category characterization."""
    print("\n" + "=" * 70)
    print("PART D: Fusion-Friendly vs Fusion-Resistant Characterization")
    print("=" * 70)

    sorted_rows = sorted(rows, key=lambda x: x["fusion_gain"], reverse=True)

    # Split into tertiles
    n = len(sorted_rows)
    top_third = sorted_rows[:n//3]
    bottom_third = sorted_rows[-(n//3):]

    print(f"\nTop tertile (fusion-friendly, N={len(top_third)}):")
    for r in top_third:
        print(f"  {r['category']:15s} ({r['dataset']:8s}): gain={r['fusion_gain']:+.2f}pp  "
              f"type={r['type']:8s}  gap={r['auroc_gap']:.1f}pp  alpha={r['oracle_alpha']:.2f}")

    print(f"\nBottom tertile (fusion-resistant, N={len(bottom_third)}):")
    for r in bottom_third:
        print(f"  {r['category']:15s} ({r['dataset']:8s}): gain={r['fusion_gain']:+.2f}pp  "
              f"type={r['type']:8s}  gap={r['auroc_gap']:.1f}pp  alpha={r['oracle_alpha']:.2f}")

    # Compare characteristics
    print(f"\n--- Characteristic Comparison ---")

    # AUROC gap
    top_gap = np.mean([r["auroc_gap"] for r in top_third])
    bot_gap = np.mean([r["auroc_gap"] for r in bottom_third])
    print(f"  Mean AUROC gap:  Top={top_gap:.1f}pp  Bottom={bot_gap:.1f}pp")

    # Best single
    top_best = np.mean([r["best_single"] for r in top_third])
    bot_best = np.mean([r["best_single"] for r in bottom_third])
    print(f"  Mean best single: Top={top_best:.1f}%  Bottom={bot_best:.1f}%")

    # Alpha extremity
    top_ext = np.mean([r["alpha_extremity"] for r in top_third])
    bot_ext = np.mean([r["alpha_extremity"] for r in bottom_third])
    print(f"  Mean |alpha-0.5|: Top={top_ext:.3f}  Bottom={bot_ext:.3f}")

    # Type distribution
    top_texture = sum(1 for r in top_third if r["is_texture"])
    bot_texture = sum(1 for r in bottom_third if r["is_texture"])
    print(f"  Texture count:   Top={top_texture}/{len(top_third)}  Bottom={bot_texture}/{len(bottom_third)}")

    # Reversal rate
    top_rev = [r for r in top_third if r["is_reversal"] is not None]
    bot_rev = [r for r in bottom_third if r["is_reversal"] is not None]
    if top_rev and bot_rev:
        top_rev_rate = sum(1 for r in top_rev if r["is_reversal"]) / len(top_rev)
        bot_rev_rate = sum(1 for r in bot_rev if r["is_reversal"]) / len(bot_rev)
        print(f"  Reversal rate:   Top={top_rev_rate:.0%}  Bottom={bot_rev_rate:.0%}")

    # Semantic type distribution
    for group_name, group in [("Top", top_third), ("Bottom", bottom_third)]:
        sem_counts = {}
        for r in group:
            s = r["semantic"]
            sem_counts[s] = sem_counts.get(s, 0) + 1
        print(f"  Semantic ({group_name}): {sem_counts}")

    return {
        "top_mean_gap": float(top_gap),
        "bottom_mean_gap": float(bot_gap),
        "top_mean_best": float(top_best),
        "bottom_mean_best": float(bot_best),
        "top_mean_extremity": float(top_ext),
        "bottom_mean_extremity": float(bot_ext)
    }


def part_e_multivariate(rows):
    """Part E: Multi-variate analysis of fusion gain."""
    print("\n" + "=" * 70)
    print("PART E: Multi-Variate Analysis of Fusion Gain")
    print("=" * 70)

    # Use all 30 categories with available features
    gains = np.array([r["fusion_gain"] for r in rows])

    # Feature matrix (available for all 30)
    features_all = {
        "auroc_gap": np.array([r["auroc_gap"] for r in rows]),
        "best_single": np.array([r["best_single"] for r in rows]),
        "alpha_extremity": np.array([r["alpha_extremity"] for r in rows]),
        "is_texture": np.array([r["is_texture"] for r in rows]),
    }

    # Step-wise: add features one at a time, compute R^2 improvement
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut

    print("\n--- Step-wise regression (N=30) ---")

    # Single feature regressions
    for name, feat in features_all.items():
        X = feat.reshape(-1, 1)
        model = LinearRegression().fit(X, gains)
        r2 = model.score(X, gains)
        print(f"  {name:20s}: R^2 = {r2:.3f}")

    # Best combination of 2
    names = list(features_all.keys())
    feats = list(features_all.values())
    print(f"\n--- Best 2-feature combinations ---")
    best_r2 = 0
    best_combo = None
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            X = np.column_stack([feats[i], feats[j]])
            model = LinearRegression().fit(X, gains)
            r2 = model.score(X, gains)
            if r2 > best_r2:
                best_r2 = r2
                best_combo = (names[i], names[j])
            print(f"  {names[i]:20s} + {names[j]:20s}: R^2 = {r2:.3f}")
    print(f"  Best: {best_combo} with R^2 = {best_r2:.3f}")

    # All 4 features
    X_all = np.column_stack(feats)
    model_all = LinearRegression().fit(X_all, gains)
    r2_all = model_all.score(X_all, gains)
    print(f"\n--- All 4 features: R^2 = {r2_all:.3f} ---")
    for name, coef in zip(names, model_all.coef_):
        print(f"  {name:20s}: coef = {coef:.4f}")
    print(f"  Intercept: {model_all.intercept_:.4f}")

    # Leave-one-out cross-validation
    loo = LeaveOneOut()
    loo_preds = []
    loo_true = []
    for train_idx, test_idx in loo.split(X_all):
        model = LinearRegression().fit(X_all[train_idx], gains[train_idx])
        pred = model.predict(X_all[test_idx])
        loo_preds.append(pred[0])
        loo_true.append(gains[test_idx][0])

    loo_preds = np.array(loo_preds)
    loo_true = np.array(loo_true)
    loo_r2 = 1 - np.sum((loo_true - loo_preds)**2) / np.sum((loo_true - np.mean(loo_true))**2)
    loo_mae = np.mean(np.abs(loo_true - loo_preds))
    print(f"\n--- LOO-CV Results ---")
    print(f"  LOO R^2 = {loo_r2:.3f}")
    print(f"  LOO MAE = {loo_mae:.3f}pp")

    # With premium diff (N=27 only)
    prem_rows = [r for r in rows if r["premium_diff"] is not None]
    if len(prem_rows) >= 10:
        print(f"\n--- With premium diff feature (N={len(prem_rows)}) ---")
        prem_gains = np.array([r["fusion_gain"] for r in prem_rows])
        prem_features = np.column_stack([
            np.array([r["auroc_gap"] for r in prem_rows]),
            np.array([r["best_single"] for r in prem_rows]),
            np.array([r["alpha_extremity"] for r in prem_rows]),
            np.array([r["is_texture"] for r in prem_rows]),
            np.array([r["premium_diff"] for r in prem_rows]),
        ])
        model_prem = LinearRegression().fit(prem_features, prem_gains)
        r2_prem = model_prem.score(prem_features, prem_gains)
        print(f"  5-feature R^2 = {r2_prem:.3f} (vs {r2_all:.3f} without premium_diff)")

        prem_names = names + ["premium_diff"]
        for name, coef in zip(prem_names, model_prem.coef_):
            print(f"  {name:20s}: coef = {coef:.4f}")

    # Practical question: can we predict which categories will benefit from adaptive fusion?
    # i.e., categories where oracle > fixed
    print(f"\n--- Fusion benefit thresholds ---")
    thresholds = [0, 0.5, 1.0, 2.0]
    for t in thresholds:
        n_above = sum(1 for g in gains if g > t)
        print(f"  Categories with fusion gain > {t}pp: {n_above}/30 ({n_above/30*100:.0f}%)")

    return {
        "single_r2": {name: float(LinearRegression().fit(
            feat.reshape(-1, 1), gains).score(feat.reshape(-1, 1), gains))
            for name, feat in features_all.items()},
        "all_features_r2": float(r2_all),
        "loo_r2": float(loo_r2),
        "loo_mae": float(loo_mae),
        "best_2feature_combo": list(best_combo) if best_combo else None,
        "best_2feature_r2": float(best_r2)
    }


def main():
    print("Experiment 18: Fusion Gain Predictors & Oracle Gap Decomposition")
    print("=" * 70)

    comp, tax, prem, rev = load_data()
    rows = build_category_table(comp, tax, prem, rev)

    print(f"\nBuilt table with {len(rows)} categories")

    results = {}
    results["part_a"] = part_a_fusion_gain(rows)
    results["part_b"] = part_b_univariate_predictors(rows)
    results["part_c"] = part_c_oracle_gap_decomposition(rows)
    results["part_d"] = part_d_characterization(rows)
    results["part_e"] = part_e_multivariate(rows)

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump({"experiment_18_fusion_gain_analysis": results}, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
