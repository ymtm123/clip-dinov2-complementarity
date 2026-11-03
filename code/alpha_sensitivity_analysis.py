"""
Experiment 22: Alpha Sensitivity & Performance Plateau Analysis

Quantifies how sensitive per-category AUROC is to alpha perturbation,
supporting the practical recommendation that simple fixed fusion is robust.

Parts:
A. Per-category performance profiles across alpha values
B. Performance plateau width (how flat is the curve around optimal?)
C. "Safe zone" analysis: range of alpha values with < 1pp loss
D. Practical recommendation validation
"""

import json
import numpy as np
from scipy import stats
import os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "alpha_sensitivity_results.json")


def load_data():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "comprehensive_eval_results.json")) as f:
        comp = json.load(f)
    with open(os.path.join(base, "pixel_fusion_results.json")) as f:
        pix = json.load(f)
    with open(os.path.join(base, "category_taxonomy_results.json")) as f:
        tax = json.load(f)
    return comp, pix, tax


def build_alpha_profiles_image(comp):
    """Build per-category AUROC profiles at different alpha values."""
    # Available alpha points: 0 (CLIP), 0.5 (Equal), 0.65 (Fixed), 0.7 (DINO-heavy), 1.0 (DINOv2)
    alpha_points = [0.0, 0.5, 0.65, 0.7, 1.0]
    strategy_keys = ["CLIP only", "Equal (a=0.5)", "Fixed (a=0.65)",
                     "DINO-heavy (a=0.7)", "DINOv2 only"]

    profiles = []
    for ds_key in ["MVTec AD", "VisA", "BTAD"]:
        categories = list(comp[ds_key]["CLIP only"]["per_category"].keys())
        for cat in categories:
            aurocs = []
            for strat_key in strategy_keys:
                aurocs.append(comp[ds_key][strat_key]["per_category"][cat])

            oracle = comp[ds_key]["Oracle"]["per_category"][cat]
            best_auroc = max(aurocs)
            worst_auroc = min(aurocs)
            range_auroc = best_auroc - worst_auroc

            profiles.append({
                "category": cat,
                "dataset": ds_key,
                "alpha_points": alpha_points,
                "aurocs": aurocs,
                "oracle": oracle,
                "best": best_auroc,
                "worst": worst_auroc,
                "range": range_auroc,
            })
    return profiles


def build_alpha_profiles_pixel(pix):
    """Build per-category pixel AUROC profiles."""
    alpha_points = [0.0, 0.3, 0.5, 0.7, 1.0]
    strategy_keys = ["CLIP only", "CLIP-heavy (a=0.3)", "Equal (a=0.5)",
                     "DINO-heavy (a=0.7)", "DINOv2 only"]

    profiles = []
    ds_map = {"mvtec": "MVTec AD", "visa": "VisA", "btad": "BTAD"}
    for ds_key, ds_name in ds_map.items():
        strats = pix[ds_key]["strategies"]
        categories = list(strats["CLIP only"].keys())
        for cat in categories:
            aurocs = [strats[sk][cat] for sk in strategy_keys]
            oracle = strats["Oracle"][cat]
            best = max(aurocs)
            worst = min(aurocs)

            profiles.append({
                "category": cat,
                "dataset": ds_name,
                "aurocs": aurocs,
                "oracle": oracle,
                "best": best,
                "worst": worst,
                "range": best - worst,
            })
    return profiles


def part_a_profiles(img_profiles):
    """Part A: Per-category performance range analysis."""
    print("=" * 70)
    print("PART A: Image-Level Performance Profiles")
    print("=" * 70)

    ranges = [p["range"] for p in img_profiles]
    print(f"\nPerformance range (best alpha - worst alpha) across strategies:")
    print(f"  Mean range:   {np.mean(ranges):.2f}pp")
    print(f"  Median range: {np.median(ranges):.2f}pp")
    print(f"  Max range:    {np.max(ranges):.2f}pp")
    print(f"  Min range:    {np.min(ranges):.2f}pp")

    # Categories with very flat profiles (range < 2pp)
    n_flat = sum(1 for r in ranges if r < 2)
    n_medium = sum(1 for r in ranges if 2 <= r < 5)
    n_steep = sum(1 for r in ranges if r >= 5)
    print(f"\n  Flat (<2pp):    {n_flat}/30 ({n_flat/30*100:.0f}%)")
    print(f"  Medium (2-5pp): {n_medium}/30 ({n_medium/30*100:.0f}%)")
    print(f"  Steep (>=5pp):  {n_steep}/30 ({n_steep/30*100:.0f}%)")

    # Top 5 most sensitive categories
    sorted_profiles = sorted(img_profiles, key=lambda p: p["range"], reverse=True)
    print(f"\nTop 5 most alpha-sensitive categories:")
    for p in sorted_profiles[:5]:
        print(f"  {p['category']:15s} ({p['dataset']:8s}): range={p['range']:.2f}pp  "
              f"CLIP={p['aurocs'][0]:.1f}  Equal={p['aurocs'][1]:.1f}  "
              f"DINOv2={p['aurocs'][4]:.1f}")

    # Bottom 5 (least sensitive)
    print(f"\nTop 5 least alpha-sensitive categories:")
    for p in sorted_profiles[-5:]:
        print(f"  {p['category']:15s} ({p['dataset']:8s}): range={p['range']:.2f}pp  "
              f"CLIP={p['aurocs'][0]:.1f}  Equal={p['aurocs'][1]:.1f}  "
              f"DINOv2={p['aurocs'][4]:.1f}")

    return {
        "mean_range": float(np.mean(ranges)),
        "median_range": float(np.median(ranges)),
        "n_flat": n_flat,
        "n_medium": n_medium,
        "n_steep": n_steep
    }


def part_b_plateau_width(img_profiles):
    """Part B: Performance plateau width at each alpha point."""
    print("\n" + "=" * 70)
    print("PART B: Performance Plateau Analysis")
    print("=" * 70)

    # For each strategy, how many categories are within 1pp of oracle?
    strategies = {
        "CLIP only (0.0)": 0,
        "Equal (0.5)": 1,
        "Fixed (0.65)": 2,
        "DINO-heavy (0.7)": 3,
        "DINOv2 only (1.0)": 4,
    }

    print(f"\nCategories within Xpp of oracle (N=30):")
    thresholds = [0.5, 1.0, 2.0, 3.0]
    for name, idx in strategies.items():
        for thresh in thresholds:
            n_within = sum(1 for p in img_profiles
                          if (p["oracle"] - p["aurocs"][idx]) <= thresh)
            pct = n_within / len(img_profiles) * 100
            print(f"  {name:25s} within {thresh}pp: {n_within}/30 ({pct:.0f}%)")
        print()

    # Mean regret per strategy
    print(f"\nMean per-category regret (vs oracle):")
    for name, idx in strategies.items():
        regrets = [p["oracle"] - p["aurocs"][idx] for p in img_profiles]
        mean_r = np.mean(regrets)
        max_r = np.max(regrets)
        print(f"  {name:25s}: mean={mean_r:.3f}pp  max={max_r:.2f}pp")

    # Key insight: how many strategies are "safe" (< 1pp from oracle) for each category?
    print(f"\nPer-category: # strategies within 1pp of oracle:")
    fusion_strategies_idx = [1, 2, 3]  # Equal, Fixed, DINO-heavy
    n_all_safe = 0
    for p in img_profiles:
        n_safe = sum(1 for idx in fusion_strategies_idx
                     if (p["oracle"] - p["aurocs"][idx]) < 1.0)
        if n_safe == 3:
            n_all_safe += 1
    print(f"  All 3 fusion strategies within 1pp: {n_all_safe}/30 ({n_all_safe/30*100:.0f}%)")

    return {}


def part_c_safe_zone(img_profiles, tax):
    """Part C: 'Safe zone' analysis for alpha selection."""
    print("\n" + "=" * 70)
    print("PART C: Alpha Safe Zone Analysis")
    print("=" * 70)

    # For each category, the "safe zone" is the range of alpha where performance
    # is within 1pp of the oracle. We can estimate this from our discrete points.

    # With points at 0, 0.5, 0.65, 0.7, 1.0, we can check which are "safe"
    alpha_points = [0.0, 0.5, 0.65, 0.7, 1.0]

    print(f"\nPer-category alpha safe zone (within 1pp of oracle):")
    safe_ranges = []
    for p in img_profiles:
        safe_alphas = [alpha_points[i] for i, a in enumerate(p["aurocs"])
                      if (p["oracle"] - a) <= 1.0]
        if safe_alphas:
            safe_min = min(safe_alphas)
            safe_max = max(safe_alphas)
            safe_width = safe_max - safe_min
        else:
            safe_min = safe_max = safe_width = 0

        safe_ranges.append({
            "category": p["category"],
            "dataset": p["dataset"],
            "safe_min": safe_min,
            "safe_max": safe_max,
            "safe_width": safe_width,
            "n_safe": len(safe_alphas)
        })

    # Summary
    widths = [s["safe_width"] for s in safe_ranges]
    print(f"\n  Mean safe zone width: {np.mean(widths):.2f}")
    print(f"  Median safe zone width: {np.median(widths):.2f}")

    # How many categories have alpha=0.65 in their safe zone?
    n_65_safe = sum(1 for p in img_profiles
                    if (p["oracle"] - p["aurocs"][2]) <= 1.0)
    print(f"\n  alpha=0.65 is safe (within 1pp): {n_65_safe}/30 ({n_65_safe/30*100:.0f}%)")

    # alpha=0.5 safe?
    n_50_safe = sum(1 for p in img_profiles
                    if (p["oracle"] - p["aurocs"][1]) <= 1.0)
    print(f"  alpha=0.50 is safe (within 1pp): {n_50_safe}/30 ({n_50_safe/30*100:.0f}%)")

    # Both 0.5 and 0.65 safe?
    n_both = sum(1 for p in img_profiles
                 if (p["oracle"] - p["aurocs"][1]) <= 1.0
                 and (p["oracle"] - p["aurocs"][2]) <= 1.0)
    print(f"  Both 0.50 AND 0.65 safe: {n_both}/30 ({n_both/30*100:.0f}%)")

    # Categories where no fusion strategy is safe
    n_none = sum(1 for p in img_profiles
                 if all((p["oracle"] - p["aurocs"][i]) > 1.0 for i in [1, 2, 3]))
    print(f"  No fusion strategy is safe: {n_none}/30 ({n_none/30*100:.0f}%)")

    # Which categories are the hardest to cover?
    print(f"\nCategories where Fixed (0.65) is NOT safe (>1pp from oracle):")
    for p in img_profiles:
        regret = p["oracle"] - p["aurocs"][2]
        if regret > 1.0:
            oracle_alpha_key = {"MVTec AD": "mvtec_image", "VisA": "visa_image",
                               "BTAD": "btad_image"}[p["dataset"]]
            oalpha = tax["oracle_alphas"][oracle_alpha_key].get(p["category"], 0.5)
            print(f"  {p['category']:15s} ({p['dataset']:8s}): regret={regret:.2f}pp  "
                  f"oracle_alpha={oalpha:.2f}")

    return {
        "mean_safe_width": float(np.mean(widths)),
        "n_065_safe": n_65_safe,
        "n_050_safe": n_50_safe,
        "n_both_safe": n_both,
        "n_none_safe": n_none
    }


def part_d_practical_validation(img_profiles, pix_profiles):
    """Part D: Validate practical recommendation."""
    print("\n" + "=" * 70)
    print("PART D: Practical Recommendation Validation")
    print("=" * 70)

    # Key question: For a practitioner, how much does alpha choice matter?
    # Compare worst-case scenarios

    # Image level
    img_fixed_regrets = [p["oracle"] - p["aurocs"][2] for p in img_profiles]  # Fixed
    img_equal_regrets = [p["oracle"] - p["aurocs"][1] for p in img_profiles]  # Equal
    img_wrong_model = [max(p["aurocs"][0], p["aurocs"][4]) - min(p["aurocs"][0], p["aurocs"][4])
                      for p in img_profiles]

    print(f"\nImage-Level Risk Summary:")
    print(f"  Wrong model penalty:  mean={np.mean(img_wrong_model):.2f}pp  max={np.max(img_wrong_model):.2f}pp")
    print(f"  Equal fusion regret:  mean={np.mean(img_equal_regrets):.3f}pp  max={np.max(img_equal_regrets):.2f}pp")
    print(f"  Fixed fusion regret:  mean={np.mean(img_fixed_regrets):.3f}pp  max={np.max(img_fixed_regrets):.2f}pp")

    # Risk reduction ratio
    wrm = np.mean(img_wrong_model)
    frr = np.mean(img_fixed_regrets)
    risk_reduction = (1 - frr / wrm) * 100
    print(f"\n  Risk reduction: {risk_reduction:.1f}% "
          f"(wrong model {wrm:.2f}pp -> fixed regret {frr:.3f}pp)")

    # Pixel level
    pix_ranges = [p["range"] for p in pix_profiles]
    print(f"\nPixel-Level:")
    print(f"  Mean performance range across alpha: {np.mean(pix_ranges):.2f}pp")

    # Combined recommendation
    print(f"\n--- PRACTICAL RECOMMENDATION VALIDATION ---")
    print(f"  1. 'Always fuse' reduces risk by {risk_reduction:.0f}%")
    print(f"  2. Fixed alpha=0.65 is safe (<1pp from oracle) for "
          f"{sum(1 for r in img_fixed_regrets if r < 1.0)}/30 categories")
    print(f"  3. Mean fixed regret = {frr:.3f}pp -- negligible")
    print(f"  4. Alpha choice between [0.5, 0.7] changes mean AUROC by "
          f"{abs(np.mean([p['aurocs'][1] for p in img_profiles]) - np.mean([p['aurocs'][3] for p in img_profiles])):.2f}pp")

    return {
        "risk_reduction_pct": float(risk_reduction),
        "mean_wrong_model_penalty": float(wrm),
        "mean_fixed_regret": float(frr),
        "n_fixed_safe": int(sum(1 for r in img_fixed_regrets if r < 1.0))
    }


def main():
    print("Experiment 22: Alpha Sensitivity & Performance Plateau Analysis")
    print("=" * 70)

    comp, pix, tax = load_data()

    img_profiles = build_alpha_profiles_image(comp)
    pix_profiles = build_alpha_profiles_pixel(pix)

    results = {}
    results["part_a"] = part_a_profiles(img_profiles)
    results["part_b"] = part_b_plateau_width(img_profiles)
    results["part_c"] = part_c_safe_zone(img_profiles, tax)
    results["part_d"] = part_d_practical_validation(img_profiles, pix_profiles)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"experiment_22_alpha_sensitivity": results}, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
