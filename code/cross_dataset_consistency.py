"""
Experiment 19: Cross-Dataset Consistency & Transfer Analysis

Tests whether the key findings are consistent across datasets independently,
addressing RQ5 (generalization).

Parts:
A. Per-dataset fusion gain predictors: Does difficulty predict gain in each dataset?
B. Alpha stability: Are oracle alphas stable across alpha resolution?
C. Fusion strategy ranking consistency: Same ranking across datasets?
D. Taxonomy prediction consistency: Do category attributes predict the same way?
E. Transfer experiment: Predict VisA performance from MVTec patterns
"""

import json
import numpy as np
from scipy import stats
import os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "cross_dataset_consistency_results.json")


def load_data():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "comprehensive_eval_results.json")) as f:
        comp = json.load(f)
    with open(os.path.join(base, "category_taxonomy_results.json")) as f:
        tax = json.load(f)
    with open(os.path.join(base, "pixel_fusion_results.json")) as f:
        pix = json.load(f)
    return comp, tax, pix


def get_per_category_data(comp, dataset_key):
    """Extract per-category CLIP, DINOv2, Oracle, etc. for a dataset."""
    ds = comp[dataset_key]
    categories = list(ds["CLIP only"]["per_category"].keys())
    rows = []
    for cat in categories:
        clip = ds["CLIP only"]["per_category"][cat]
        dino = ds["DINOv2 only"]["per_category"][cat]
        oracle = ds["Oracle"]["per_category"][cat]
        equal = ds["Equal (a=0.5)"]["per_category"][cat]
        fixed = ds["Fixed (a=0.65)"]["per_category"][cat]
        adafuse = ds["AdaFuse-PF"]["per_category"][cat]

        best = max(clip, dino)
        fusion_gain = oracle - best
        equal_gain = equal - best
        auroc_gap = abs(clip - dino)

        rows.append({
            "category": cat,
            "clip": clip,
            "dino": dino,
            "oracle": oracle,
            "equal": equal,
            "fixed": fixed,
            "adafuse": adafuse,
            "best_single": best,
            "fusion_gain": fusion_gain,
            "equal_gain": equal_gain,
            "auroc_gap": auroc_gap,
        })
    return rows


def part_a_per_dataset_predictors(comp, tax):
    """Part A: Test difficulty-gain relationship in each dataset independently."""
    print("=" * 70)
    print("PART A: Per-Dataset Fusion Gain Predictors")
    print("=" * 70)

    results = {}
    for ds_key, alpha_key in [("MVTec AD", "mvtec_image"), ("VisA", "visa_image"), ("BTAD", "btad_image")]:
        rows = get_per_category_data(comp, ds_key)
        n = len(rows)

        gains = np.array([r["fusion_gain"] for r in rows])
        best = np.array([r["best_single"] for r in rows])
        gaps = np.array([r["auroc_gap"] for r in rows])

        # Oracle alphas
        alphas = [tax["oracle_alphas"][alpha_key].get(r["category"], 0.5) for r in rows]
        extremity = np.abs(np.array(alphas) - 0.5)

        print(f"\n--- {ds_key} (N={n}) ---")
        print(f"  Mean fusion gain: {np.mean(gains):.3f}pp")

        # Best single vs gain
        if n >= 5:
            r_best, p_best = stats.spearmanr(best, gains)
            print(f"  Best single vs gain: r={r_best:.3f}, p={p_best:.4f}")
        else:
            r_best, p_best = float('nan'), float('nan')
            print(f"  Best single vs gain: N too small")

        # Alpha extremity vs gain
        if n >= 5:
            r_ext, p_ext = stats.spearmanr(extremity, gains)
            print(f"  |alpha-0.5| vs gain: r={r_ext:.3f}, p={p_ext:.4f}")
        else:
            r_ext, p_ext = float('nan'), float('nan')
            print(f"  |alpha-0.5| vs gain: N too small")

        # AUROC gap vs gain
        if n >= 5:
            r_gap, p_gap = stats.spearmanr(gaps, gains)
            print(f"  |AUROC gap| vs gain: r={r_gap:.3f}, p={p_gap:.4f}")
        else:
            r_gap, p_gap = float('nan'), float('nan')

        results[ds_key] = {
            "N": n,
            "mean_gain": float(np.mean(gains)),
            "best_single_r": float(r_best) if not np.isnan(r_best) else None,
            "best_single_p": float(p_best) if not np.isnan(p_best) else None,
            "alpha_ext_r": float(r_ext) if not np.isnan(r_ext) else None,
            "alpha_ext_p": float(p_ext) if not np.isnan(p_ext) else None,
        }

    # Cross-dataset consistency check
    print(f"\n--- Consistency of difficulty-gain relationship ---")
    all_rs = [results[k]["best_single_r"] for k in ["MVTec AD", "VisA"]
              if results[k]["best_single_r"] is not None]
    if all_rs:
        print(f"  MVTec r={all_rs[0]:.3f}, VisA r={all_rs[1]:.3f}")
        print(f"  Both negative? {all(r < 0 for r in all_rs)}")
        print(f"  Direction consistent: {'YES' if all(r < 0 for r in all_rs) else 'NO'}")

    return results


def part_b_alpha_robustness(comp, tax):
    """Part B: How sensitive is performance to alpha perturbation?"""
    print("\n" + "=" * 70)
    print("PART B: Alpha Robustness Analysis")
    print("=" * 70)

    results = {}

    for ds_key, alpha_key in [("MVTec AD", "mvtec_image"), ("VisA", "visa_image"), ("BTAD", "btad_image")]:
        rows = get_per_category_data(comp, ds_key)
        n = len(rows)

        # For each category, compute how much performance drops
        # when using alpha=0.5, 0.65, 0.7 vs oracle
        strategies = {
            "Equal (0.5)": "Equal (a=0.5)",
            "Fixed (0.65)": "Fixed (a=0.65)",
            "DINO-heavy (0.7)": "DINO-heavy (a=0.7)",
            "AdaFuse-PF": "AdaFuse-PF"
        }

        print(f"\n--- {ds_key} (N={n}) ---")
        ds = comp[ds_key]
        oracle_mean = ds["Oracle"]["mean_auroc"]

        for name, key in strategies.items():
            strat_mean = ds[key]["mean_auroc"]
            gap = oracle_mean - strat_mean
            # Per-category max regret
            per_cat_regrets = []
            for cat in ds["Oracle"]["per_category"]:
                oracle_val = ds["Oracle"]["per_category"][cat]
                strat_val = ds[key]["per_category"][cat]
                per_cat_regrets.append(oracle_val - strat_val)
            max_regret = max(per_cat_regrets)
            mean_regret = np.mean(per_cat_regrets)
            print(f"  {name:20s}: mean_gap={gap:.3f}pp  max_regret={max_regret:.2f}pp  "
                  f"mean_regret={mean_regret:.3f}pp")

        results[ds_key] = {"oracle_mean": float(oracle_mean)}

    return results


def part_c_ranking_consistency(comp):
    """Part C: Are fusion strategy rankings consistent across datasets?"""
    print("\n" + "=" * 70)
    print("PART C: Fusion Strategy Ranking Consistency")
    print("=" * 70)

    strategies = ["CLIP only", "DINOv2 only", "Equal (a=0.5)", "Fixed (a=0.65)",
                  "DINO-heavy (a=0.7)", "AdaFuse-PF", "Max fusion"]

    rankings = {}
    for ds_key in ["MVTec AD", "VisA", "BTAD"]:
        means = [(s, comp[ds_key][s]["mean_auroc"]) for s in strategies]
        means.sort(key=lambda x: x[1], reverse=True)
        rankings[ds_key] = [s for s, _ in means]
        print(f"\n{ds_key} ranking:")
        for i, (s, m) in enumerate(means):
            print(f"  {i+1}. {s:25s}: {m:.2f}%")

    # Kendall's tau for ranking consistency
    print(f"\n--- Ranking Consistency (Kendall's tau) ---")
    dataset_pairs = [("MVTec AD", "VisA"), ("MVTec AD", "BTAD"), ("VisA", "BTAD")]
    for ds1, ds2 in dataset_pairs:
        # Get ranks for each strategy
        ranks1 = {s: i for i, s in enumerate(rankings[ds1])}
        ranks2 = {s: i for i, s in enumerate(rankings[ds2])}
        r1 = [ranks1[s] for s in strategies]
        r2 = [ranks2[s] for s in strategies]
        tau, p = stats.kendalltau(r1, r2)
        print(f"  {ds1} vs {ds2}: tau={tau:.3f}, p={p:.4f}")

    # Is DINOv2 > CLIP consistent?
    print(f"\n--- DINOv2 > CLIP consistency ---")
    for ds_key in ["MVTec AD", "VisA", "BTAD"]:
        clip = comp[ds_key]["CLIP only"]["mean_auroc"]
        dino = comp[ds_key]["DINOv2 only"]["mean_auroc"]
        print(f"  {ds_key}: CLIP={clip:.2f}  DINOv2={dino:.2f}  DINOv2>CLIP: {dino > clip}")

    # Is Fixed > Equal consistent?
    print(f"\n--- Fixed(0.65) > Equal(0.5) consistency ---")
    for ds_key in ["MVTec AD", "VisA", "BTAD"]:
        equal = comp[ds_key]["Equal (a=0.5)"]["mean_auroc"]
        fixed = comp[ds_key]["Fixed (a=0.65)"]["mean_auroc"]
        print(f"  {ds_key}: Equal={equal:.2f}  Fixed={fixed:.2f}  Fixed>Equal: {fixed > equal}")

    # Fusion > best single consistent?
    print(f"\n--- Fusion > best single consistency ---")
    for ds_key in ["MVTec AD", "VisA", "BTAD"]:
        clip = comp[ds_key]["CLIP only"]["mean_auroc"]
        dino = comp[ds_key]["DINOv2 only"]["mean_auroc"]
        best_single = max(clip, dino)
        equal = comp[ds_key]["Equal (a=0.5)"]["mean_auroc"]
        fixed = comp[ds_key]["Fixed (a=0.65)"]["mean_auroc"]
        adafuse = comp[ds_key]["AdaFuse-PF"]["mean_auroc"]
        print(f"  {ds_key}: BestSingle={best_single:.2f}  "
              f"Equal={equal:.2f}(+{equal-best_single:.2f})  "
              f"Fixed={fixed:.2f}(+{fixed-best_single:.2f})  "
              f"AdaFuse={adafuse:.2f}(+{adafuse-best_single:.2f})")

    return rankings


def part_d_taxonomy_consistency(comp, tax):
    """Part D: Do taxonomy predictions work consistently across datasets?"""
    print("\n" + "=" * 70)
    print("PART D: Taxonomy Prediction Consistency")
    print("=" * 70)

    # For each dataset, test if texture/object predicts oracle alpha
    for ds_key, alpha_key in [("MVTec AD", "mvtec_image"), ("VisA", "visa_image")]:
        categories = list(comp[ds_key]["CLIP only"]["per_category"].keys())
        texture_alphas = []
        object_alphas = []

        for cat in categories:
            t = tax["taxonomy"].get(cat, {})
            alpha = tax["oracle_alphas"][alpha_key].get(cat, 0.5)
            if t.get("type") == "texture":
                texture_alphas.append(alpha)
            else:
                object_alphas.append(alpha)

        if texture_alphas and object_alphas:
            u, p = stats.mannwhitneyu(texture_alphas, object_alphas, alternative='two-sided')
            print(f"\n{ds_key}:")
            print(f"  Texture alpha: mean={np.mean(texture_alphas):.3f} (N={len(texture_alphas)})")
            print(f"  Object alpha:  mean={np.mean(object_alphas):.3f} (N={len(object_alphas)})")
            print(f"  Mann-Whitney p={p:.4f}")
            print(f"  Direction: {'Texture favors CLIP' if np.mean(texture_alphas) < np.mean(object_alphas) else 'Object favors CLIP'}")

    # DINOv2 win rate consistency
    print(f"\n--- DINOv2 Win Rate Consistency ---")
    for ds_key in ["MVTec AD", "VisA", "BTAD"]:
        categories = list(comp[ds_key]["CLIP only"]["per_category"].keys())
        dino_wins = 0
        for cat in categories:
            clip = comp[ds_key]["CLIP only"]["per_category"][cat]
            dino = comp[ds_key]["DINOv2 only"]["per_category"][cat]
            if dino > clip:
                dino_wins += 1
        rate = dino_wins / len(categories)
        print(f"  {ds_key}: DINOv2 wins {dino_wins}/{len(categories)} ({rate:.0%})")

    return {}


def part_e_transfer_experiment(comp, tax):
    """Part E: Can we predict VisA oracle alphas from MVTec patterns?"""
    print("\n" + "=" * 70)
    print("PART E: Cross-Dataset Transfer Analysis")
    print("=" * 70)

    # Simple transfer test: use MVTec mean oracle alpha to predict VisA
    mvtec_alphas = list(tax["oracle_alphas"]["mvtec_image"].values())
    visa_alphas = list(tax["oracle_alphas"]["visa_image"].values())
    btad_alphas = list(tax["oracle_alphas"]["btad_image"].values())

    mvtec_mean_alpha = np.mean(mvtec_alphas)
    visa_mean_alpha = np.mean(visa_alphas)
    btad_mean_alpha = np.mean(btad_alphas)

    print(f"\nMean oracle alpha per dataset:")
    print(f"  MVTec: {mvtec_mean_alpha:.3f}")
    print(f"  VisA:  {visa_mean_alpha:.3f}")
    print(f"  BTAD:  {btad_mean_alpha:.3f}")

    # If we use MVTec mean alpha (0.54?) for all VisA categories, what's the regret?
    print(f"\n--- Transfer: MVTec mean alpha as VisA fixed weight ---")
    visa_categories = list(comp["VisA"]["CLIP only"]["per_category"].keys())

    # Simulate fusion with MVTec-derived alpha for each VisA category
    print(f"  MVTec-derived alpha = {mvtec_mean_alpha:.3f}")

    # Compute fusion at alpha = mvtec_mean for each VisA category
    # We can't directly compute this (we only have discrete alpha results)
    # But we can compare oracle regret patterns

    # More meaningful transfer: Do per-category characteristics predict
    # the same alpha patterns?
    print(f"\n--- Texture vs Object alpha patterns (transfer test) ---")

    datasets_info = {
        "MVTec": ("mvtec_image", "MVTec AD"),
        "VisA": ("visa_image", "VisA"),
        "BTAD": ("btad_image", "BTAD")
    }

    for ds_name, (alpha_key, comp_key) in datasets_info.items():
        cats = list(comp[comp_key]["CLIP only"]["per_category"].keys())
        tex_a, obj_a = [], []
        for cat in cats:
            t = tax["taxonomy"].get(cat, {})
            alpha = tax["oracle_alphas"][alpha_key].get(cat, 0.5)
            if t.get("type") == "texture":
                tex_a.append(alpha)
            else:
                obj_a.append(alpha)
        if tex_a and obj_a:
            print(f"  {ds_name}: Texture={np.mean(tex_a):.3f}  Object={np.mean(obj_a):.3f}  "
                  f"(Texture favors CLIP: {'YES' if np.mean(tex_a) < np.mean(obj_a) else 'NO'})")

    # Fusion improvement pattern consistency
    print(f"\n--- Fusion improvement pattern ---")
    for ds_key in ["MVTec AD", "VisA", "BTAD"]:
        clip_mean = comp[ds_key]["CLIP only"]["mean_auroc"]
        dino_mean = comp[ds_key]["DINOv2 only"]["mean_auroc"]
        best_mean = max(clip_mean, dino_mean)
        oracle_mean = comp[ds_key]["Oracle"]["mean_auroc"]
        equal_mean = comp[ds_key]["Equal (a=0.5)"]["mean_auroc"]
        fixed_mean = comp[ds_key]["Fixed (a=0.65)"]["mean_auroc"]
        adafuse_mean = comp[ds_key]["AdaFuse-PF"]["mean_auroc"]

        ogrr_equal = (equal_mean - best_mean) / (oracle_mean - best_mean) if oracle_mean > best_mean else 0
        ogrr_fixed = (fixed_mean - best_mean) / (oracle_mean - best_mean) if oracle_mean > best_mean else 0
        ogrr_ada = (adafuse_mean - best_mean) / (oracle_mean - best_mean) if oracle_mean > best_mean else 0

        print(f"  {ds_key}: OGRR_equal={ogrr_equal:.3f}  OGRR_fixed={ogrr_fixed:.3f}  "
              f"OGRR_ada={ogrr_ada:.3f}  oracle_gap={oracle_mean-best_mean:.2f}pp")

    # Key consistency test: Is DINOv2 advantage consistent?
    print(f"\n--- Summary: Key finding consistency ---")
    consistency_tests = {
        "DINOv2 > CLIP (mean)": [],
        "Fixed > Equal": [],
        "Fusion > Best single": [],
        "Texture alpha < Object alpha": []
    }

    for ds_key in ["MVTec AD", "VisA", "BTAD"]:
        clip = comp[ds_key]["CLIP only"]["mean_auroc"]
        dino = comp[ds_key]["DINOv2 only"]["mean_auroc"]
        equal = comp[ds_key]["Equal (a=0.5)"]["mean_auroc"]
        fixed = comp[ds_key]["Fixed (a=0.65)"]["mean_auroc"]
        best = max(clip, dino)

        consistency_tests["DINOv2 > CLIP (mean)"].append(dino > clip)
        consistency_tests["Fixed > Equal"].append(fixed > equal)
        consistency_tests["Fusion > Best single"].append(equal > best)

    for finding, results_list in consistency_tests.items():
        n_true = sum(results_list)
        print(f"  {finding}: {n_true}/3 datasets ({'CONSISTENT' if n_true == 3 else 'PARTIAL'})")

    return {
        "mean_alphas": {"mvtec": float(mvtec_mean_alpha),
                       "visa": float(visa_mean_alpha),
                       "btad": float(btad_mean_alpha)}
    }


def main():
    print("Experiment 19: Cross-Dataset Consistency & Transfer Analysis")
    print("=" * 70)

    comp, tax, pix = load_data()

    results = {}
    results["part_a"] = part_a_per_dataset_predictors(comp, tax)
    results["part_b"] = part_b_alpha_robustness(comp, tax)
    results["part_c"] = part_c_ranking_consistency(comp)
    results["part_d"] = part_d_taxonomy_consistency(comp, tax)
    results["part_e"] = part_e_transfer_experiment(comp, tax)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"experiment_19_cross_dataset_consistency": results}, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
