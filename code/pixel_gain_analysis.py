"""
Experiment 21: Pixel-Level Fusion Gain Analysis & Image-Pixel Comparison

Tests whether the same predictors (difficulty, balance) work at pixel level,
and compares image vs pixel fusion gain patterns.

Parts:
A. Pixel-level fusion gain statistics
B. Pixel-level fusion gain predictors
C. Image-level vs pixel-level fusion gain correlation
D. Level-specific fusion effectiveness comparison
"""

import json
import numpy as np
from scipy import stats
import os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "pixel_gain_results.json")


def load_data():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "pixel_fusion_results.json")) as f:
        pix = json.load(f)
    with open(os.path.join(base, "comprehensive_eval_results.json")) as f:
        comp = json.load(f)
    with open(os.path.join(base, "category_taxonomy_results.json")) as f:
        tax = json.load(f)
    return pix, comp, tax


def build_pixel_table(pix, tax):
    """Build per-category pixel-level data."""
    rows = []
    ds_map = {
        "mvtec": ("MVTec AD", "mvtec_pixel"),
        "visa": ("VisA", "visa_pixel"),
        "btad": ("BTAD", "btad_pixel")
    }

    for ds_key, (ds_name, alpha_key) in ds_map.items():
        strats = pix[ds_key]["strategies"]
        categories = list(strats["CLIP only"].keys())

        for cat in categories:
            clip = strats["CLIP only"][cat]
            dino = strats["DINOv2 only"][cat]
            oracle = strats["Oracle"][cat]
            equal = strats["Equal (a=0.5)"][cat]

            best = max(clip, dino)
            gain = oracle - best
            equal_gain = equal - best
            gap = abs(clip - dino)

            alpha = pix[ds_key]["oracle_alphas"].get(cat, 0.5)
            extremity = abs(alpha - 0.5)

            cat_type = tax["taxonomy"].get(cat, {}).get("type", "object")
            is_texture = 1 if cat_type == "texture" else 0

            rows.append({
                "category": cat, "dataset": ds_name, "type": cat_type,
                "is_texture": is_texture,
                "clip": clip, "dino": dino, "oracle": oracle, "equal": equal,
                "best_single": best, "fusion_gain": gain, "equal_gain": equal_gain,
                "auroc_gap": gap, "oracle_alpha": alpha, "alpha_extremity": extremity
            })
    return rows


def build_image_table(comp, tax):
    """Build per-category image-level data."""
    rows = []
    ds_map = {
        "MVTec AD": "mvtec_image",
        "VisA": "visa_image",
        "BTAD": "btad_image"
    }

    for ds_key, alpha_key in ds_map.items():
        ds = comp[ds_key]
        categories = list(ds["CLIP only"]["per_category"].keys())

        for cat in categories:
            clip = ds["CLIP only"]["per_category"][cat]
            dino = ds["DINOv2 only"]["per_category"][cat]
            oracle = ds["Oracle"]["per_category"][cat]
            best = max(clip, dino)
            gain = oracle - best
            alpha = tax["oracle_alphas"][alpha_key].get(cat, 0.5)
            extremity = abs(alpha - 0.5)

            rows.append({
                "category": cat, "dataset": ds_key,
                "best_single": best, "fusion_gain": gain,
                "oracle_alpha": alpha, "alpha_extremity": extremity
            })
    return rows


def part_a_pixel_gain_stats(pix_rows):
    """Part A: Pixel-level fusion gain statistics."""
    print("=" * 70)
    print("PART A: Pixel-Level Fusion Gain Statistics (30 categories)")
    print("=" * 70)

    gains = [r["fusion_gain"] for r in pix_rows]
    equal_gains = [r["equal_gain"] for r in pix_rows]

    print(f"\nOracle pixel fusion gain (oracle - best single):")
    print(f"  Mean:   {np.mean(gains):.3f}pp")
    print(f"  Median: {np.median(gains):.3f}pp")
    print(f"  Std:    {np.std(gains):.3f}pp")
    print(f"  Min:    {np.min(gains):.3f}pp")
    print(f"  Max:    {np.max(gains):.3f}pp")

    n_pos = sum(1 for g in gains if g > 0)
    n_zero = sum(1 for g in gains if g == 0)
    print(f"\n  Positive gain: {n_pos}/30 ({n_pos/30*100:.0f}%)")
    print(f"  Zero gain:     {n_zero}/30 ({n_zero/30*100:.0f}%)")

    print(f"\nEqual pixel fusion gain:")
    print(f"  Mean: {np.mean(equal_gains):.3f}pp")
    eq_pos = sum(1 for g in equal_gains if g > 0)
    print(f"  Positive: {eq_pos}/30 ({eq_pos/30*100:.0f}%)")

    # Per dataset breakdown
    for ds in ["MVTec AD", "VisA", "BTAD"]:
        ds_gains = [r["fusion_gain"] for r in pix_rows if r["dataset"] == ds]
        ds_eq = [r["equal_gain"] for r in pix_rows if r["dataset"] == ds]
        print(f"\n  {ds}: mean oracle gain={np.mean(ds_gains):.3f}pp, "
              f"mean equal gain={np.mean(ds_eq):.3f}pp")

    return {
        "mean_gain": float(np.mean(gains)),
        "n_positive": n_pos,
        "mean_equal_gain": float(np.mean(equal_gains))
    }


def part_b_pixel_predictors(pix_rows):
    """Part B: Pixel-level fusion gain predictors."""
    print("\n" + "=" * 70)
    print("PART B: Pixel-Level Fusion Gain Predictors")
    print("=" * 70)

    gains = np.array([r["fusion_gain"] for r in pix_rows])
    results = {}

    # B1: Best single vs gain
    best = np.array([r["best_single"] for r in pix_rows])
    r_b, p_b = stats.spearmanr(best, gains)
    print(f"\n1. Best single vs Pixel Fusion Gain:")
    print(f"   Spearman r = {r_b:.3f}, p = {p_b:.4f}")
    results["best_single"] = {"r": float(r_b), "p": float(p_b)}

    # B2: Alpha extremity vs gain
    ext = np.array([r["alpha_extremity"] for r in pix_rows])
    r_e, p_e = stats.spearmanr(ext, gains)
    print(f"\n2. |alpha-0.5| vs Pixel Fusion Gain:")
    print(f"   Spearman r = {r_e:.3f}, p = {p_e:.4f}")
    results["alpha_extremity"] = {"r": float(r_e), "p": float(p_e)}

    # B3: AUROC gap vs gain
    gaps = np.array([r["auroc_gap"] for r in pix_rows])
    r_g, p_g = stats.spearmanr(gaps, gains)
    print(f"\n3. |AUROC gap| vs Pixel Fusion Gain:")
    print(f"   Spearman r = {r_g:.3f}, p = {p_g:.4f}")
    results["auroc_gap"] = {"r": float(r_g), "p": float(p_g)}

    # B4: Texture vs Object
    tex_gains = [r["fusion_gain"] for r in pix_rows if r["is_texture"]]
    obj_gains = [r["fusion_gain"] for r in pix_rows if not r["is_texture"]]
    if tex_gains and obj_gains:
        u, p_t = stats.mannwhitneyu(tex_gains, obj_gains, alternative='two-sided')
        print(f"\n4. Texture vs Object pixel fusion gain:")
        print(f"   Texture mean: {np.mean(tex_gains):.3f}pp (N={len(tex_gains)})")
        print(f"   Object mean:  {np.mean(obj_gains):.3f}pp (N={len(obj_gains)})")
        print(f"   Mann-Whitney p = {p_t:.4f}")
        results["type"] = {"texture_mean": float(np.mean(tex_gains)),
                          "object_mean": float(np.mean(obj_gains)),
                          "p": float(p_t)}

    # Summary
    print(f"\n--- Summary of Pixel-Level Predictors (p < 0.05) ---")
    for name, vals in results.items():
        if "p" in vals and vals["p"] < 0.05:
            r_val = vals.get("r", "N/A")
            print(f"  {name}: r={r_val}, p={vals['p']:.4f} ***")

    return results


def part_c_image_pixel_comparison(pix_rows, img_rows):
    """Part C: Compare image-level vs pixel-level fusion gain."""
    print("\n" + "=" * 70)
    print("PART C: Image vs Pixel Fusion Gain Comparison")
    print("=" * 70)

    # Match categories across levels
    pix_map = {(r["category"], r["dataset"]): r for r in pix_rows}
    img_map = {(r["category"], r["dataset"]): r for r in img_rows}

    common_keys = set(pix_map.keys()) & set(img_map.keys())
    print(f"\nMatched categories: {len(common_keys)}")

    img_gains = []
    pix_gains = []
    img_best = []
    pix_best = []

    for key in sorted(common_keys):
        ig = img_map[key]["fusion_gain"]
        pg = pix_map[key]["fusion_gain"]
        img_gains.append(ig)
        pix_gains.append(pg)
        img_best.append(img_map[key]["best_single"])
        pix_best.append(pix_map[key]["best_single"])

    img_gains = np.array(img_gains)
    pix_gains = np.array(pix_gains)

    # Correlation between image and pixel fusion gains
    r_gain, p_gain = stats.spearmanr(img_gains, pix_gains)
    print(f"\nImage gain vs Pixel gain correlation:")
    print(f"  Spearman r = {r_gain:.3f}, p = {p_gain:.4f}")

    # Compare means
    print(f"\nMean fusion gain:")
    print(f"  Image level: {np.mean(img_gains):.3f}pp")
    print(f"  Pixel level: {np.mean(pix_gains):.3f}pp")
    print(f"  Difference:  {np.mean(img_gains) - np.mean(pix_gains):.3f}pp")

    # Wilcoxon test: is fusion gain different between levels?
    w, p_w = stats.wilcoxon(img_gains, pix_gains)
    print(f"  Wilcoxon p = {p_w:.4f}")

    # Categories where pixel gain > image gain
    n_pixel_higher = sum(1 for i, p in zip(img_gains, pix_gains) if p > i)
    print(f"\n  Pixel gain > Image gain: {n_pixel_higher}/{len(common_keys)}")
    print(f"  Image gain > Pixel gain: {len(common_keys) - n_pixel_higher}/{len(common_keys)}")

    # Oracle alpha correlation (image vs pixel)
    img_alphas = [img_map[key]["oracle_alpha"] for key in sorted(common_keys)]
    pix_alphas = [pix_map[key]["oracle_alpha"] for key in sorted(common_keys)]
    r_alpha, p_alpha = stats.spearmanr(img_alphas, pix_alphas)
    print(f"\nImage alpha vs Pixel alpha correlation:")
    print(f"  Spearman r = {r_alpha:.3f}, p = {p_alpha:.4f}")

    return {
        "gain_correlation_r": float(r_gain),
        "gain_correlation_p": float(p_gain),
        "mean_image_gain": float(np.mean(img_gains)),
        "mean_pixel_gain": float(np.mean(pix_gains)),
        "wilcoxon_p": float(p_w),
        "alpha_correlation_r": float(r_alpha),
        "alpha_correlation_p": float(p_alpha)
    }


def part_d_level_effectiveness(pix_rows, img_rows):
    """Part D: Level-specific fusion effectiveness comparison."""
    print("\n" + "=" * 70)
    print("PART D: Level-Specific Fusion Effectiveness")
    print("=" * 70)

    # Image level stats
    img_gains = [r["fusion_gain"] for r in img_rows]
    pix_gains = [r["fusion_gain"] for r in pix_rows]

    # Categories that benefit at BOTH levels
    pix_map = {(r["category"], r["dataset"]): r for r in pix_rows}
    img_map = {(r["category"], r["dataset"]): r for r in img_rows}
    common = set(pix_map.keys()) & set(img_map.keys())

    both_benefit = 0
    image_only = 0
    pixel_only = 0
    neither = 0

    for key in common:
        ig = img_map[key]["fusion_gain"] > 0.1
        pg = pix_map[key]["fusion_gain"] > 0.1
        if ig and pg:
            both_benefit += 1
        elif ig and not pg:
            image_only += 1
        elif not ig and pg:
            pixel_only += 1
        else:
            neither += 1

    print(f"\nFusion benefit (>0.1pp gain) by level:")
    print(f"  Both levels:     {both_benefit}/30 ({both_benefit/30*100:.0f}%)")
    print(f"  Image only:      {image_only}/30 ({image_only/30*100:.0f}%)")
    print(f"  Pixel only:      {pixel_only}/30 ({pixel_only/30*100:.0f}%)")
    print(f"  Neither:         {neither}/30 ({neither/30*100:.0f}%)")

    # Biggest discrepancies
    print(f"\nLargest image-pixel gain discrepancies:")
    discrepancies = []
    for key in sorted(common):
        ig = img_map[key]["fusion_gain"]
        pg = pix_map[key]["fusion_gain"]
        discrepancies.append((key[0], key[1], ig, pg, ig - pg))

    discrepancies.sort(key=lambda x: abs(x[4]), reverse=True)
    for cat, ds, ig, pg, diff in discrepancies[:10]:
        direction = "image > pixel" if diff > 0 else "pixel > image"
        print(f"  {cat:15s} ({ds:8s}): img={ig:+.2f}  pix={pg:+.2f}  diff={diff:+.2f} ({direction})")

    return {
        "both_benefit": both_benefit,
        "image_only": image_only,
        "pixel_only": pixel_only,
        "neither": neither
    }


def main():
    print("Experiment 21: Pixel-Level Fusion Gain Analysis")
    print("=" * 70)

    pix, comp, tax = load_data()

    pix_rows = build_pixel_table(pix, tax)
    img_rows = build_image_table(comp, tax)

    print(f"\nPixel table: {len(pix_rows)} categories")
    print(f"Image table: {len(img_rows)} categories")

    results = {}
    results["part_a"] = part_a_pixel_gain_stats(pix_rows)
    results["part_b"] = part_b_pixel_predictors(pix_rows)
    results["part_c"] = part_c_image_pixel_comparison(pix_rows, img_rows)
    results["part_d"] = part_d_level_effectiveness(pix_rows, img_rows)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"experiment_21_pixel_gain": results}, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
