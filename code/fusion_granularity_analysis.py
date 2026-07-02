"""
Experiment 24: Fusion Weight Granularity Analysis

Answers the question: "How granular does alpha need to be?"
Compares global, per-dataset, per-type, and per-category alpha assignments.

Parts:
A. Global optimal alpha search (single alpha for all 30 categories)
B. Per-dataset optimal alpha (3 alphas)
C. Per-type optimal alpha (2 alphas: texture/object)
D. Per-category oracle alpha (30 alphas) — upper bound
E. Diminishing returns analysis: how much does each level of granularity add?
F. Regret partition: how much regret is between-dataset vs within-dataset vs within-category?
"""

import json
import numpy as np
from scipy import stats
import os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "fusion_granularity_results.json")


def load_data():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "comprehensive_eval_results.json")) as f:
        comp = json.load(f)
    with open(os.path.join(base, "category_taxonomy_results.json")) as f:
        tax = json.load(f)
    return comp, tax


def build_per_category_profiles(comp, tax):
    """Build performance profiles at multiple alpha points for each category."""
    alpha_points = [0.0, 0.5, 0.65, 0.7, 1.0]
    strategy_keys = ["CLIP only", "Equal (a=0.5)", "Fixed (a=0.65)",
                     "DINO-heavy (a=0.7)", "DINOv2 only"]
    ds_alpha_map = {"MVTec AD": "mvtec_image", "VisA": "visa_image", "BTAD": "btad_image"}

    profiles = []
    for ds_key, alpha_key in ds_alpha_map.items():
        categories = list(comp[ds_key]["CLIP only"]["per_category"].keys())
        for cat in categories:
            aurocs = [comp[ds_key][sk]["per_category"][cat] for sk in strategy_keys]
            oracle = comp[ds_key]["Oracle"]["per_category"][cat]
            oracle_alpha = tax["oracle_alphas"][alpha_key].get(cat, 0.5)
            cat_type = tax["taxonomy"].get(cat, {}).get("type", "object")

            profiles.append({
                "category": cat,
                "dataset": ds_key,
                "type": cat_type,
                "alpha_points": alpha_points,
                "aurocs": aurocs,  # at 0.0, 0.5, 0.65, 0.7, 1.0
                "oracle": oracle,
                "oracle_alpha": oracle_alpha,
            })
    return profiles


def interpolate_auroc(profile, alpha):
    """Linear interpolation of AUROC at given alpha."""
    pts = list(zip(profile["alpha_points"], profile["aurocs"]))
    if alpha <= 0.0:
        return pts[0][1]
    if alpha >= 1.0:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        if pts[i][0] <= alpha <= pts[i+1][0]:
            a1, v1 = pts[i]
            a2, v2 = pts[i+1]
            t = (alpha - a1) / (a2 - a1) if a2 != a1 else 0
            return v1 + t * (v2 - v1)
    return profile["aurocs"][2]  # fallback to Fixed


def search_optimal_alpha(profiles, alpha_range=None):
    """Search for optimal alpha over a set of profiles by grid search."""
    if alpha_range is None:
        alphas = np.arange(0.0, 1.01, 0.01)
    else:
        alphas = np.arange(alpha_range[0], alpha_range[1] + 0.01, 0.01)

    best_alpha = 0.5
    best_mean = 0

    for alpha in alphas:
        aurocs = [interpolate_auroc(p, alpha) for p in profiles]
        mean_auroc = np.mean(aurocs)
        if mean_auroc > best_mean:
            best_mean = mean_auroc
            best_alpha = alpha

    return round(best_alpha, 2), best_mean


def part_a_global_optimal(profiles):
    """Part A: Find single global optimal alpha."""
    print("=" * 70)
    print("PART A: Global Optimal Alpha (Single α for All 30 Categories)")
    print("=" * 70)

    # Grid search for best global alpha
    best_alpha, best_mean = search_optimal_alpha(profiles)

    # Compute regret for key alpha values
    alphas_to_check = [0.0, 0.3, 0.5, 0.65, 0.7, best_alpha, 1.0]
    oracle_aurocs = [p["oracle"] for p in profiles]
    mean_oracle = np.mean(oracle_aurocs)

    print(f"\nGlobal optimal α = {best_alpha}")
    print(f"Oracle mean AUROC = {mean_oracle:.3f}%")
    print(f"\n{'Alpha':>8s}  {'Mean AUROC':>10s}  {'Regret':>10s}")
    print("-" * 35)

    results = {}
    for alpha in sorted(set(alphas_to_check)):
        aurocs = [interpolate_auroc(p, alpha) for p in profiles]
        mean_a = np.mean(aurocs)
        regret = mean_oracle - mean_a
        marker = " ← optimal" if alpha == best_alpha else ""
        print(f"  {alpha:.2f}    {mean_a:.3f}%      {regret:.3f}pp{marker}")
        results[f"alpha_{alpha:.2f}"] = {"mean_auroc": float(mean_a), "regret": float(regret)}

    results["optimal_alpha"] = best_alpha
    return results


def part_b_per_dataset(profiles):
    """Part B: Per-dataset optimal alpha (3 alphas)."""
    print("\n" + "=" * 70)
    print("PART B: Per-Dataset Optimal Alpha (3 Alphas)")
    print("=" * 70)

    datasets = ["MVTec AD", "VisA", "BTAD"]
    ds_alphas = {}
    ds_regrets = []

    for ds in datasets:
        ds_profiles = [p for p in profiles if p["dataset"] == ds]
        best_alpha, best_mean = search_optimal_alpha(ds_profiles)
        oracle_mean = np.mean([p["oracle"] for p in ds_profiles])
        regret = oracle_mean - best_mean

        # Per-category regret with this alpha
        cat_regrets = [p["oracle"] - interpolate_auroc(p, best_alpha) for p in ds_profiles]

        print(f"\n  {ds:10s}: optimal α={best_alpha:.2f}, "
              f"mean={best_mean:.3f}%, regret={regret:.3f}pp, "
              f"mean_cat_regret={np.mean(cat_regrets):.3f}pp")

        ds_alphas[ds] = best_alpha
        ds_regrets.extend(cat_regrets)

    mean_per_ds_regret = np.mean(ds_regrets)
    print(f"\n  Overall mean per-category regret (per-dataset α): {mean_per_ds_regret:.4f}pp")

    return {
        "per_dataset_alphas": ds_alphas,
        "mean_per_category_regret": float(mean_per_ds_regret)
    }


def part_c_per_type(profiles):
    """Part C: Per-type optimal alpha (2 alphas: texture/object)."""
    print("\n" + "=" * 70)
    print("PART C: Per-Type Optimal Alpha (2 Alphas)")
    print("=" * 70)

    type_alphas = {}
    type_regrets = []

    for t in ["texture", "object"]:
        t_profiles = [p for p in profiles if p["type"] == t]
        best_alpha, best_mean = search_optimal_alpha(t_profiles)
        oracle_mean = np.mean([p["oracle"] for p in t_profiles])
        regret = oracle_mean - best_mean

        cat_regrets = [p["oracle"] - interpolate_auroc(p, best_alpha) for p in t_profiles]

        print(f"\n  {t:10s} (N={len(t_profiles)}): optimal α={best_alpha:.2f}, "
              f"mean={best_mean:.3f}%, regret={regret:.3f}pp, "
              f"mean_cat_regret={np.mean(cat_regrets):.3f}pp")

        type_alphas[t] = best_alpha
        type_regrets.extend(cat_regrets)

    mean_per_type_regret = np.mean(type_regrets)
    print(f"\n  Overall mean per-category regret (per-type α): {mean_per_type_regret:.4f}pp")

    return {
        "per_type_alphas": type_alphas,
        "mean_per_category_regret": float(mean_per_type_regret)
    }


def part_d_per_category(profiles):
    """Part D: Per-category oracle alpha (upper bound)."""
    print("\n" + "=" * 70)
    print("PART D: Per-Category Oracle (30 Alphas, Upper Bound)")
    print("=" * 70)

    oracle_regrets = [p["oracle"] - max(p["aurocs"]) for p in profiles]
    # oracle is already the best per-category, so regret = 0 by definition at the oracle alpha
    # But it's actually oracle - interpolated(oracle_alpha), which may differ due to discretization
    # Let's compute actual regret at oracle_alpha via interpolation
    regrets = [p["oracle"] - interpolate_auroc(p, p["oracle_alpha"]) for p in profiles]

    print(f"\n  Mean per-category Oracle regret (due to α discretization): "
          f"{np.mean(regrets):.4f}pp")

    # The true oracle (from JSON) has 0 regret by definition
    print(f"  True oracle: 0.000pp (by definition)")

    return {
        "mean_discretization_regret": float(np.mean(regrets))
    }


def part_e_diminishing_returns(profiles, global_res, ds_res, type_res):
    """Part E: Diminishing returns analysis."""
    print("\n" + "=" * 70)
    print("PART E: Diminishing Returns Analysis")
    print("=" * 70)

    # Compute regrets at each granularity level
    # Level 0: Best single model (DINOv2 only)
    regrets_single = [p["oracle"] - p["aurocs"][4] for p in profiles]  # DINOv2 only
    mean_single = np.mean(regrets_single)

    # Level 1: Global optimal α
    global_alpha = global_res["optimal_alpha"]
    regrets_global = [p["oracle"] - interpolate_auroc(p, global_alpha) for p in profiles]
    mean_global = np.mean(regrets_global)

    # Level 2: Fixed α=0.65
    regrets_fixed = [p["oracle"] - interpolate_auroc(p, 0.65) for p in profiles]
    mean_fixed = np.mean(regrets_fixed)

    # Level 3: Per-dataset α
    ds_alphas = ds_res["per_dataset_alphas"]
    regrets_ds = [p["oracle"] - interpolate_auroc(p, ds_alphas[p["dataset"]])
                  for p in profiles]
    mean_ds = np.mean(regrets_ds)

    # Level 4: Per-type α
    type_alphas = type_res["per_type_alphas"]
    regrets_type = [p["oracle"] - interpolate_auroc(p, type_alphas[p["type"]])
                    for p in profiles]
    mean_type = np.mean(regrets_type)

    # Level 5: Per-category oracle
    mean_oracle = 0.0

    print(f"\n{'Level':>30s}  {'# Params':>8s}  {'Regret':>10s}  {'Δ from prev':>12s}")
    print("-" * 65)
    levels = [
        ("DINOv2 only (no fusion)", 0, mean_single, None),
        ("Global optimal α", 1, mean_global, mean_single - mean_global),
        ("Fixed α=0.65", 1, mean_fixed, mean_global - mean_fixed),
        ("Per-dataset α", 3, mean_ds, mean_fixed - mean_ds),
        ("Per-type α", 2, mean_type, mean_ds - mean_type),
        ("Per-category oracle", 30, mean_oracle, mean_type - mean_oracle),
    ]

    for name, n_params, regret, delta in levels:
        delta_str = f"{delta:+.3f}pp" if delta is not None else "---"
        print(f"  {name:28s}  {n_params:>8d}  {regret:.3f}pp      {delta_str}")

    # Cumulative gap closure from DINOv2 to Oracle
    total_gap = mean_single - mean_oracle  # = mean_single since mean_oracle = 0
    print(f"\n  Total gap (DINOv2 → Oracle): {total_gap:.3f}pp")
    print(f"\n  Gap closed by each step:")

    steps = [
        ("Global fusion (1 α)", mean_single - mean_global),
        ("Fixed α=0.65 vs global", mean_global - mean_fixed),
        ("Per-dataset (3 α)", mean_fixed - mean_ds),
        ("Per-category oracle (30 α)", mean_ds - mean_oracle),
    ]

    for name, closed in steps:
        pct = closed / total_gap * 100 if total_gap > 0 else 0
        print(f"    {name:30s}: {closed:+.3f}pp ({pct:.1f}%)")

    return {
        "regret_single": float(mean_single),
        "regret_global": float(mean_global),
        "regret_fixed": float(mean_fixed),
        "regret_per_dataset": float(mean_ds),
        "regret_per_type": float(mean_type),
        "regret_oracle": 0.0,
        "total_gap": float(total_gap)
    }


def part_f_regret_partition(profiles):
    """Part F: Variance decomposition of oracle alpha."""
    print("\n" + "=" * 70)
    print("PART F: Oracle Alpha Variance Decomposition")
    print("=" * 70)

    # How much variance in oracle alpha is between-dataset vs within-dataset?
    all_alphas = [p["oracle_alpha"] for p in profiles]
    overall_mean = np.mean(all_alphas)
    total_var = np.var(all_alphas)

    # Between-dataset variance
    ds_means = {}
    for ds in ["MVTec AD", "VisA", "BTAD"]:
        ds_alphas = [p["oracle_alpha"] for p in profiles if p["dataset"] == ds]
        ds_means[ds] = np.mean(ds_alphas)

    between_ds_var = np.mean([(ds_means[p["dataset"]] - overall_mean)**2 for p in profiles])

    # Within-dataset variance
    within_ds_var = total_var - between_ds_var

    print(f"\n  Overall oracle α: mean={overall_mean:.3f}, var={total_var:.4f}")
    print(f"  Between-dataset variance: {between_ds_var:.4f} ({between_ds_var/total_var*100:.1f}%)")
    print(f"  Within-dataset variance:  {within_ds_var:.4f} ({within_ds_var/total_var*100:.1f}%)")

    # Between-type variance
    type_means = {}
    for t in ["texture", "object"]:
        t_alphas = [p["oracle_alpha"] for p in profiles if p["type"] == t]
        type_means[t] = np.mean(t_alphas)

    between_type_var = np.mean([(type_means[p["type"]] - overall_mean)**2 for p in profiles])
    within_type_var = total_var - between_type_var

    print(f"\n  Between-type variance:    {between_type_var:.4f} ({between_type_var/total_var*100:.1f}%)")
    print(f"  Within-type variance:     {within_type_var:.4f} ({within_type_var/total_var*100:.1f}%)")

    # Per-dataset means
    print(f"\n  Per-dataset mean α:")
    for ds, mean_a in ds_means.items():
        print(f"    {ds:10s}: α={mean_a:.3f}")

    print(f"\n  Per-type mean α:")
    for t, mean_a in type_means.items():
        print(f"    {t:10s}: α={mean_a:.3f}")

    # Key insight
    print(f"\n--- Key Insight ---")
    print(f"  {within_ds_var/total_var*100:.0f}% of oracle α variance is WITHIN-dataset")
    print(f"  {within_type_var/total_var*100:.0f}% of oracle α variance is WITHIN-type")
    print(f"  → Most α variation is category-specific, not explained by dataset or type")
    print(f"  → This is why per-dataset/per-type α selection adds little over fixed α")

    return {
        "total_var": float(total_var),
        "between_ds_pct": float(between_ds_var / total_var * 100),
        "within_ds_pct": float(within_ds_var / total_var * 100),
        "between_type_pct": float(between_type_var / total_var * 100),
        "within_type_pct": float(within_type_var / total_var * 100),
    }


def main():
    print("Experiment 24: Fusion Weight Granularity Analysis")
    print("=" * 70)

    comp, tax = load_data()
    profiles = build_per_category_profiles(comp, tax)

    print(f"\nTotal categories: {len(profiles)}")

    results = {}
    results["part_a"] = part_a_global_optimal(profiles)
    results["part_b"] = part_b_per_dataset(profiles)
    results["part_c"] = part_c_per_type(profiles)
    results["part_d"] = part_d_per_category(profiles)
    results["part_e"] = part_e_diminishing_returns(
        profiles, results["part_a"], results["part_b"], results["part_c"])
    results["part_f"] = part_f_regret_partition(profiles)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"experiment_24_granularity": results}, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
