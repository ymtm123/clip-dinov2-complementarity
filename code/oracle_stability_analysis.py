"""
Experiment 25: Oracle Alpha Stability & Variance Analysis

Tests whether the oracle alpha is itself stable. If oracle alpha varies
substantially due to test set composition, then trying to estimate it
(vs using fixed alpha) is inherently limited.

Parts:
A. Bootstrap resampling of oracle alpha (N=1000 resamples per category)
B. Oracle alpha confidence intervals
C. Implication for adaptive methods: what's the "estimable" component?
D. Comparison of oracle stability by category difficulty
"""

import json
import numpy as np
from scipy import stats
import os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "oracle_stability_results.json")


def load_data():
    base = os.path.dirname(__file__)
    # Load the raw per-image scores for bootstrap resampling
    # We need the actual per-image scores, not just per-category AUROC
    with open(os.path.join(base, "comprehensive_eval_results.json")) as f:
        comp = json.load(f)
    with open(os.path.join(base, "category_taxonomy_results.json")) as f:
        tax = json.load(f)

    # Try to load per-image score data
    score_data = {}
    for ds_key in ["mvtec", "visa", "btad"]:
        score_file = os.path.join(base, f"fusion_results_{ds_key}.json")
        if os.path.exists(score_file):
            with open(score_file) as f:
                score_data[ds_key] = json.load(f)

    return comp, tax, score_data


def bootstrap_oracle_alpha_from_scores(clip_scores, dino_scores, labels,
                                        n_boot=1000, alpha_grid=None):
    """Bootstrap resample oracle alpha from per-image scores."""
    from sklearn.metrics import roc_auc_score

    if alpha_grid is None:
        alpha_grid = np.arange(0.0, 1.01, 0.05)

    clip_scores = np.array(clip_scores)
    dino_scores = np.array(dino_scores)
    labels = np.array(labels)

    n = len(labels)

    # Normalize scores (min-max per category)
    def normalize(s):
        smin, smax = s.min(), s.max()
        if smax - smin < 1e-10:
            return np.zeros_like(s)
        return (s - smin) / (smax - smin)

    clip_norm = normalize(clip_scores)
    dino_norm = normalize(dino_scores)

    boot_alphas = []
    rng = np.random.RandomState(42)

    for _ in range(n_boot):
        # Resample with replacement
        idx = rng.choice(n, size=n, replace=True)
        b_labels = labels[idx]

        # Skip if only one class
        if len(np.unique(b_labels)) < 2:
            continue

        b_clip = clip_norm[idx]
        b_dino = dino_norm[idx]

        # Search oracle alpha on bootstrap sample
        best_alpha = 0.5
        best_auroc = 0

        for alpha in alpha_grid:
            fused = alpha * b_dino + (1 - alpha) * b_clip
            try:
                auroc = roc_auc_score(b_labels, fused)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_alpha = alpha
            except:
                continue

        boot_alphas.append(best_alpha)

    return np.array(boot_alphas)


def part_a_bootstrap_analysis(score_data, comp, tax):
    """Part A: Bootstrap oracle alpha for available datasets."""
    print("=" * 70)
    print("PART A: Bootstrap Oracle Alpha Stability")
    print("=" * 70)

    results = {}
    alpha_grid = np.arange(0.0, 1.01, 0.05)

    ds_map = {
        "mvtec": ("MVTec AD", "mvtec_image"),
        "visa": ("VisA", "visa_image"),
        "btad": ("BTAD", "btad_image"),
    }

    all_stds = []
    all_ci_widths = []
    all_categories = []

    for ds_key, (ds_name, alpha_key) in ds_map.items():
        if ds_key not in score_data:
            print(f"\n  {ds_name}: No per-image score data available, skipping")
            continue

        ds = score_data[ds_key]
        print(f"\n  {ds_name}:")

        for cat_key, cat_data in ds.items():
            if "clip_scores" not in cat_data or "dino_scores" not in cat_data:
                continue

            clip_scores = cat_data["clip_scores"]
            dino_scores = cat_data["dino_scores"]
            labels = cat_data["labels"]

            if len(labels) < 10:
                continue

            # Get original oracle alpha from taxonomy
            original_alpha = tax["oracle_alphas"].get(alpha_key, {}).get(cat_key, 0.5)

            # Bootstrap
            boot_alphas = bootstrap_oracle_alpha_from_scores(
                clip_scores, dino_scores, labels, n_boot=1000, alpha_grid=alpha_grid
            )

            if len(boot_alphas) < 100:
                continue

            boot_mean = np.mean(boot_alphas)
            boot_std = np.std(boot_alphas)
            boot_ci = np.percentile(boot_alphas, [2.5, 97.5])
            ci_width = boot_ci[1] - boot_ci[0]

            all_stds.append(boot_std)
            all_ci_widths.append(ci_width)
            all_categories.append({
                "category": cat_key,
                "dataset": ds_name,
                "original_alpha": original_alpha,
                "boot_mean": float(boot_mean),
                "boot_std": float(boot_std),
                "ci_low": float(boot_ci[0]),
                "ci_high": float(boot_ci[1]),
                "ci_width": float(ci_width),
            })

            print(f"    {cat_key:15s}: α*={original_alpha:.2f}, "
                  f"boot_mean={boot_mean:.3f}±{boot_std:.3f}, "
                  f"95% CI=[{boot_ci[0]:.2f}, {boot_ci[1]:.2f}], "
                  f"width={ci_width:.2f}")

    if all_stds:
        print(f"\n  Summary across {len(all_stds)} categories:")
        print(f"    Mean bootstrap α std:     {np.mean(all_stds):.3f}")
        print(f"    Median bootstrap α std:   {np.median(all_stds):.3f}")
        print(f"    Mean 95% CI width:        {np.mean(all_ci_widths):.3f}")
        print(f"    Median 95% CI width:      {np.median(all_ci_widths):.3f}")
        print(f"    Max CI width:             {np.max(all_ci_widths):.3f}")

    results["categories"] = all_categories
    results["mean_std"] = float(np.mean(all_stds)) if all_stds else None
    results["mean_ci_width"] = float(np.mean(all_ci_widths)) if all_ci_widths else None
    return results


def part_b_ci_implications(boot_results, tax):
    """Part B: CI implications for fixed vs adaptive alpha."""
    print("\n" + "=" * 70)
    print("PART B: Confidence Interval Implications")
    print("=" * 70)

    cats = boot_results.get("categories", [])
    if not cats:
        print("  No bootstrap data available")
        return {}

    # How many categories have α=0.65 within their bootstrap CI?
    n_fixed_in_ci = 0
    n_equal_in_ci = 0
    for c in cats:
        if c["ci_low"] <= 0.65 <= c["ci_high"]:
            n_fixed_in_ci += 1
        if c["ci_low"] <= 0.50 <= c["ci_high"]:
            n_equal_in_ci += 1

    n_total = len(cats)
    print(f"\n  α=0.65 within 95% CI: {n_fixed_in_ci}/{n_total} "
          f"({n_fixed_in_ci/n_total*100:.0f}%)")
    print(f"  α=0.50 within 95% CI: {n_equal_in_ci}/{n_total} "
          f"({n_equal_in_ci/n_total*100:.0f}%)")

    # Overlap between CIs of different categories
    # (Do categories have distinguishable oracle alphas?)
    ci_overlap_count = 0
    ci_total_pairs = 0
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            overlap = min(cats[i]["ci_high"], cats[j]["ci_high"]) - \
                      max(cats[i]["ci_low"], cats[j]["ci_low"])
            ci_total_pairs += 1
            if overlap > 0:
                ci_overlap_count += 1

    print(f"\n  CI overlap between category pairs:")
    print(f"    {ci_overlap_count}/{ci_total_pairs} pairs have overlapping CIs "
          f"({ci_overlap_count/ci_total_pairs*100:.0f}%)")

    # Effective degrees of freedom: how many distinguishable alpha values?
    # Cluster categories by their CI overlap
    # Simple heuristic: how many categories have non-overlapping CIs with alpha=0.65?
    n_clearly_different = 0
    for c in cats:
        if 0.65 < c["ci_low"] or 0.65 > c["ci_high"]:
            n_clearly_different += 1
            print(f"    {c['category']:15s}: α*={c['original_alpha']:.2f}, "
                  f"CI=[{c['ci_low']:.2f}, {c['ci_high']:.2f}] — "
                  f"CLEARLY different from 0.65")

    print(f"\n  Categories clearly different from α=0.65: "
          f"{n_clearly_different}/{n_total} ({n_clearly_different/n_total*100:.0f}%)")

    return {
        "n_fixed_in_ci": n_fixed_in_ci,
        "n_equal_in_ci": n_equal_in_ci,
        "ci_overlap_pct": float(ci_overlap_count / ci_total_pairs * 100) if ci_total_pairs > 0 else None,
        "n_clearly_different": n_clearly_different,
    }


def part_c_difficulty_correlation(boot_results, comp):
    """Part C: Is oracle alpha stability related to category difficulty?"""
    print("\n" + "=" * 70)
    print("PART C: Oracle Alpha Stability vs Category Difficulty")
    print("=" * 70)

    cats = boot_results.get("categories", [])
    if not cats:
        print("  No bootstrap data available")
        return {}

    ci_widths = []
    best_singles = []

    for c in cats:
        ds = c["dataset"]
        cat = c["category"]
        try:
            clip = comp[ds]["CLIP only"]["per_category"][cat]
            dino = comp[ds]["DINOv2 only"]["per_category"][cat]
            best_single = max(clip, dino)
        except KeyError:
            continue
        ci_widths.append(c["ci_width"])
        best_singles.append(best_single)

    ci_widths = np.array(ci_widths)
    best_singles = np.array(best_singles)

    r, p = stats.spearmanr(ci_widths, best_singles)
    print(f"\n  CI width vs best single AUROC:")
    print(f"    Spearman r = {r:.3f}, p = {p:.4f}")
    if r < 0:
        print(f"    → Harder categories have WIDER CIs (less stable oracle α)")
    else:
        print(f"    → Easier categories have WIDER CIs")

    return {
        "ci_vs_difficulty_r": float(r),
        "ci_vs_difficulty_p": float(p),
    }


def part_d_analytical_estimate(comp, tax):
    """Part D: Analytical estimate of oracle alpha instability.

    Even without per-image scores, we can estimate instability from
    the performance profile: categories where the AUROC curve is flat
    near the oracle have unstable alpha (many alphas are equally good).
    """
    print("\n" + "=" * 70)
    print("PART D: Analytical Oracle Alpha Stability Estimate")
    print("=" * 70)

    alpha_points = [0.0, 0.5, 0.65, 0.7, 1.0]
    strategy_keys = ["CLIP only", "Equal (a=0.5)", "Fixed (a=0.65)",
                     "DINO-heavy (a=0.7)", "DINOv2 only"]
    ds_alpha_map = {"MVTec AD": "mvtec_image", "VisA": "visa_image", "BTAD": "btad_image"}

    results = []
    for ds_key, alpha_key in ds_alpha_map.items():
        categories = list(comp[ds_key]["CLIP only"]["per_category"].keys())
        for cat in categories:
            aurocs = [comp[ds_key][sk]["per_category"][cat] for sk in strategy_keys]
            oracle = comp[ds_key]["Oracle"]["per_category"][cat]
            oracle_alpha = tax["oracle_alphas"][alpha_key].get(cat, 0.5)

            # Count how many alpha points are within 0.5pp of oracle
            n_near_oracle = sum(1 for a in aurocs if (oracle - a) <= 0.5)

            # Performance gradient near oracle: how quickly does AUROC drop?
            # Use the AUROC drop from oracle to 2nd best
            sorted_aurocs = sorted(aurocs, reverse=True)
            if len(sorted_aurocs) > 1:
                oracle_margin = sorted_aurocs[0] - sorted_aurocs[1]
            else:
                oracle_margin = 0

            results.append({
                "category": cat,
                "dataset": ds_key,
                "oracle_alpha": oracle_alpha,
                "n_near_oracle": n_near_oracle,
                "oracle_margin": oracle_margin,
                "oracle": oracle,
                "best_single": max(aurocs[0], aurocs[4]),
            })

    # Analyze
    print(f"\n  Categories by number of alpha points within 0.5pp of oracle:")
    for n in range(1, 6):
        count = sum(1 for r in results if r["n_near_oracle"] == n)
        print(f"    {n} points near oracle: {count}/30 ({count/30*100:.0f}%)")

    # Oracle margin analysis
    margins = [r["oracle_margin"] for r in results]
    print(f"\n  Oracle margin (best AUROC - 2nd best AUROC):")
    print(f"    Mean:   {np.mean(margins):.3f}pp")
    print(f"    Median: {np.median(margins):.3f}pp")
    print(f"    Min:    {np.min(margins):.3f}pp")
    print(f"    Max:    {np.max(margins):.3f}pp")

    n_tiny_margin = sum(1 for m in margins if m < 0.1)
    n_small_margin = sum(1 for m in margins if m < 0.5)
    print(f"\n    Margin < 0.1pp (essentially flat): {n_tiny_margin}/30 ({n_tiny_margin/30*100:.0f}%)")
    print(f"    Margin < 0.5pp (nearly flat): {n_small_margin}/30 ({n_small_margin/30*100:.0f}%)")

    # Key insight
    print(f"\n--- Key Insight ---")
    print(f"  {n_small_margin}/30 categories have oracle margin < 0.5pp")
    print(f"  → For these categories, the 'oracle' alpha is effectively arbitrary")
    print(f"  → Trying to estimate it is chasing noise")
    print(f"  → Fixed alpha is the rational choice")

    return {
        "mean_margin": float(np.mean(margins)),
        "n_tiny_margin": n_tiny_margin,
        "n_small_margin": n_small_margin,
    }


def main():
    print("Experiment 25: Oracle Alpha Stability Analysis")
    print("=" * 70)

    comp, tax, score_data = load_data()

    print(f"\nAvailable per-image score data: {list(score_data.keys())}")

    results = {}

    if score_data:
        results["part_a"] = part_a_bootstrap_analysis(score_data, comp, tax)
        results["part_b"] = part_b_ci_implications(results["part_a"], tax)
        results["part_c"] = part_c_difficulty_correlation(results["part_a"], comp)
    else:
        print("\nNo per-image score data found. Running analytical analysis only.")
        results["part_a"] = {"note": "No per-image scores available"}
        results["part_b"] = {}
        results["part_c"] = {}

    results["part_d"] = part_d_analytical_estimate(comp, tax)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"experiment_25_oracle_stability": results}, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
