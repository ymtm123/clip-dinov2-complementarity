"""
Experiment 23: Taxonomy-Guided Alpha Selection & Regret Decomposition

Tests whether the taxonomy (texture/object, color/shape) can improve
alpha selection beyond fixed α=0.65. Quantifies the practical value
of the taxonomy for fusion weight estimation.

Parts:
A. Group-mean alpha selection: texture→α_tex, object→α_obj
B. 2D taxonomy selection: combine type + semantic attributes
C. Leave-one-out taxonomy alpha estimation
D. Regret decomposition by taxonomy attributes
E. Complete effect size (Cohen's d) compilation
"""

import json
import numpy as np
from scipy import stats
import os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "taxonomy_alpha_results.json")


def load_data():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "comprehensive_eval_results.json")) as f:
        comp = json.load(f)
    with open(os.path.join(base, "category_taxonomy_results.json")) as f:
        tax = json.load(f)
    with open(os.path.join(base, "pixel_fusion_results.json")) as f:
        pix = json.load(f)
    return comp, tax, pix


def build_category_data(comp, tax):
    """Build per-category data with taxonomy attributes and all strategy AUROCs."""
    alpha_points = [0.0, 0.5, 0.65, 0.7, 1.0]
    strategy_keys = ["CLIP only", "Equal (a=0.5)", "Fixed (a=0.65)",
                     "DINO-heavy (a=0.7)", "DINOv2 only"]

    ds_alpha_map = {"MVTec AD": "mvtec_image", "VisA": "visa_image", "BTAD": "btad_image"}
    rows = []

    for ds_key, alpha_key in ds_alpha_map.items():
        categories = list(comp[ds_key]["CLIP only"]["per_category"].keys())
        for cat in categories:
            aurocs = {}
            for sk in strategy_keys:
                aurocs[sk] = comp[ds_key][sk]["per_category"][cat]

            oracle = comp[ds_key]["Oracle"]["per_category"][cat]
            oracle_alpha = tax["oracle_alphas"][alpha_key].get(cat, 0.5)
            cat_tax = tax["taxonomy"].get(cat, {})

            rows.append({
                "category": cat,
                "dataset": ds_key,
                "type": cat_tax.get("type", "object"),
                "semantic": cat_tax.get("semantic", "shape"),
                "structure": cat_tax.get("structure", "fine"),
                "scale": cat_tax.get("defect_scale", "small"),
                "aurocs": aurocs,
                "oracle": oracle,
                "oracle_alpha": oracle_alpha,
                "clip": aurocs["CLIP only"],
                "dino": aurocs["DINOv2 only"],
                "equal": aurocs["Equal (a=0.5)"],
                "fixed": aurocs["Fixed (a=0.65)"],
            })
    return rows


def interpolate_auroc(clip_auroc, dino_auroc, alpha, equal_auroc, fixed_auroc, dino_heavy_auroc):
    """Interpolate AUROC for arbitrary alpha using available data points.

    Points: (0.0, clip), (0.5, equal), (0.65, fixed), (0.7, dino_heavy), (1.0, dino)
    Uses linear interpolation between nearest known points.
    """
    points = [(0.0, clip_auroc), (0.5, equal_auroc), (0.65, fixed_auroc),
              (0.7, dino_heavy_auroc), (1.0, dino_auroc)]

    if alpha <= 0.0:
        return clip_auroc
    if alpha >= 1.0:
        return dino_auroc

    # Find bracketing points
    for i in range(len(points) - 1):
        if points[i][0] <= alpha <= points[i+1][0]:
            a1, v1 = points[i]
            a2, v2 = points[i+1]
            t = (alpha - a1) / (a2 - a1) if a2 != a1 else 0
            return v1 + t * (v2 - v1)

    return fixed_auroc  # fallback


def part_a_group_mean_alpha(rows, comp):
    """Part A: Group-mean alpha selection (texture → α_tex, object → α_obj)."""
    print("=" * 70)
    print("PART A: Group-Mean Taxonomy-Guided Alpha Selection")
    print("=" * 70)

    # Compute group mean oracle alphas
    type_groups = {}
    for r in rows:
        t = r["type"]
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(r["oracle_alpha"])

    print("\nOracle alpha by type:")
    type_mean_alpha = {}
    for t, alphas in type_groups.items():
        mean_a = np.mean(alphas)
        type_mean_alpha[t] = mean_a
        print(f"  {t:10s}: mean={mean_a:.3f}, std={np.std(alphas):.3f}, "
              f"N={len(alphas)}, range=[{min(alphas):.2f}, {max(alphas):.2f}]")

    # Compute semantic group means
    sem_groups = {}
    for r in rows:
        s = r["semantic"]
        if s not in sem_groups:
            sem_groups[s] = []
        sem_groups[s].append(r["oracle_alpha"])

    print("\nOracle alpha by semantic:")
    sem_mean_alpha = {}
    for s, alphas in sem_groups.items():
        mean_a = np.mean(alphas)
        sem_mean_alpha[s] = mean_a
        print(f"  {s:10s}: mean={mean_a:.3f}, std={np.std(alphas):.3f}, "
              f"N={len(alphas)}, range=[{min(alphas):.2f}, {max(alphas):.2f}]")

    # Strategy: assign α based on type group mean
    print("\n--- Strategy: type-based alpha ---")
    type_predicted = []
    fixed_aurocs = []
    oracle_aurocs = []
    type_regrets = []
    fixed_regrets = []

    for r in rows:
        predicted_alpha = type_mean_alpha[r["type"]]
        type_predicted.append(predicted_alpha)

        # Interpolate AUROC at predicted alpha
        dh = comp[r["dataset"]]["DINO-heavy (a=0.7)"]["per_category"][r["category"]]
        pred_auroc = interpolate_auroc(
            r["clip"], r["dino"], predicted_alpha,
            r["equal"], r["fixed"], dh
        )
        type_regrets.append(r["oracle"] - pred_auroc)
        fixed_regrets.append(r["oracle"] - r["fixed"])
        oracle_aurocs.append(r["oracle"])

    mean_type_regret = np.mean(type_regrets)
    mean_fixed_regret = np.mean(fixed_regrets)

    print(f"  Type-based mean regret:  {mean_type_regret:.4f}pp")
    print(f"  Fixed (0.65) mean regret: {mean_fixed_regret:.4f}pp")
    print(f"  Improvement: {mean_fixed_regret - mean_type_regret:.4f}pp")

    if mean_type_regret < mean_fixed_regret:
        print(f"  → Type-based BETTER by {mean_fixed_regret - mean_type_regret:.4f}pp")
    else:
        print(f"  → Fixed (0.65) BETTER by {mean_type_regret - mean_fixed_regret:.4f}pp")

    return {
        "type_mean_alpha": {k: float(v) for k, v in type_mean_alpha.items()},
        "sem_mean_alpha": {k: float(v) for k, v in sem_mean_alpha.items()},
        "type_based_regret": float(mean_type_regret),
        "fixed_regret": float(mean_fixed_regret),
    }


def part_b_2d_taxonomy(rows, comp):
    """Part B: 2D taxonomy alpha selection (type × semantic)."""
    print("\n" + "=" * 70)
    print("PART B: 2D Taxonomy Alpha Selection (Type × Semantic)")
    print("=" * 70)

    # Compute 2D group means
    groups_2d = {}
    for r in rows:
        key = (r["type"], r["semantic"])
        if key not in groups_2d:
            groups_2d[key] = []
        groups_2d[key].append(r["oracle_alpha"])

    print("\n2D taxonomy group means:")
    group_mean_2d = {}
    for key, alphas in sorted(groups_2d.items()):
        mean_a = np.mean(alphas)
        group_mean_2d[key] = mean_a
        print(f"  {key[0]:8s} × {key[1]:8s}: α={mean_a:.3f} (N={len(alphas)})")

    # Apply 2D taxonomy prediction
    regrets_2d = []
    for r in rows:
        key = (r["type"], r["semantic"])
        predicted_alpha = group_mean_2d[key]
        dh = comp[r["dataset"]]["DINO-heavy (a=0.7)"]["per_category"][r["category"]]
        pred_auroc = interpolate_auroc(
            r["clip"], r["dino"], predicted_alpha,
            r["equal"], r["fixed"], dh
        )
        regrets_2d.append(r["oracle"] - pred_auroc)

    mean_2d_regret = np.mean(regrets_2d)
    print(f"\n  2D taxonomy mean regret:  {mean_2d_regret:.4f}pp")

    return {
        "group_means_2d": {f"{k[0]}_{k[1]}": float(v) for k, v in group_mean_2d.items()},
        "mean_regret_2d": float(mean_2d_regret)
    }


def part_c_loo_taxonomy(rows, comp):
    """Part C: Leave-one-out taxonomy alpha estimation."""
    print("\n" + "=" * 70)
    print("PART C: Leave-One-Out Taxonomy Alpha Estimation")
    print("=" * 70)

    # For each category, compute group mean alpha EXCLUDING that category
    loo_regrets_type = []
    loo_regrets_sem = []
    loo_regrets_2d = []
    fixed_regrets = []

    for i, r in enumerate(rows):
        # Type-based LOO
        others_same_type = [rows[j]["oracle_alpha"] for j in range(len(rows))
                           if j != i and rows[j]["type"] == r["type"]]
        if others_same_type:
            loo_alpha_type = np.mean(others_same_type)
        else:
            loo_alpha_type = 0.65  # fallback

        # Semantic-based LOO
        others_same_sem = [rows[j]["oracle_alpha"] for j in range(len(rows))
                          if j != i and rows[j]["semantic"] == r["semantic"]]
        if others_same_sem:
            loo_alpha_sem = np.mean(others_same_sem)
        else:
            loo_alpha_sem = 0.65

        # 2D LOO
        others_same_2d = [rows[j]["oracle_alpha"] for j in range(len(rows))
                         if j != i and rows[j]["type"] == r["type"]
                         and rows[j]["semantic"] == r["semantic"]]
        if others_same_2d:
            loo_alpha_2d = np.mean(others_same_2d)
        else:
            loo_alpha_2d = 0.65

        dh = comp[r["dataset"]]["DINO-heavy (a=0.7)"]["per_category"][r["category"]]

        # Compute AUROC at each predicted alpha
        auroc_type = interpolate_auroc(r["clip"], r["dino"], loo_alpha_type,
                                       r["equal"], r["fixed"], dh)
        auroc_sem = interpolate_auroc(r["clip"], r["dino"], loo_alpha_sem,
                                      r["equal"], r["fixed"], dh)
        auroc_2d = interpolate_auroc(r["clip"], r["dino"], loo_alpha_2d,
                                     r["equal"], r["fixed"], dh)

        loo_regrets_type.append(r["oracle"] - auroc_type)
        loo_regrets_sem.append(r["oracle"] - auroc_sem)
        loo_regrets_2d.append(r["oracle"] - auroc_2d)
        fixed_regrets.append(r["oracle"] - r["fixed"])

    print(f"\nLeave-One-Out Regret Comparison (N=30):")
    print(f"  Fixed (α=0.65):     mean={np.mean(fixed_regrets):.4f}pp, "
          f"max={np.max(fixed_regrets):.2f}pp")
    print(f"  LOO Type-based:     mean={np.mean(loo_regrets_type):.4f}pp, "
          f"max={np.max(loo_regrets_type):.2f}pp")
    print(f"  LOO Semantic-based: mean={np.mean(loo_regrets_sem):.4f}pp, "
          f"max={np.max(loo_regrets_sem):.2f}pp")
    print(f"  LOO 2D (type×sem):  mean={np.mean(loo_regrets_2d):.4f}pp, "
          f"max={np.max(loo_regrets_2d):.2f}pp")

    # Which approach wins per category?
    methods = ["Fixed", "LOO-Type", "LOO-Sem", "LOO-2D"]
    regret_arrays = [fixed_regrets, loo_regrets_type, loo_regrets_sem, loo_regrets_2d]
    wins = {m: 0 for m in methods}
    for i in range(len(rows)):
        best_idx = np.argmin([regret_arrays[j][i] for j in range(4)])
        wins[methods[best_idx]] += 1

    print(f"\n  Per-category wins:")
    for m, w in wins.items():
        print(f"    {m:15s}: {w}/30 ({w/30*100:.0f}%)")

    # Directional accuracy: does LOO predict correct model preference?
    n_correct_type = 0
    n_correct_sem = 0
    for i, r in enumerate(rows):
        others_type = [rows[j]["oracle_alpha"] for j in range(len(rows))
                      if j != i and rows[j]["type"] == r["type"]]
        if others_type:
            pred_pref = "dino" if np.mean(others_type) > 0.5 else "clip"
            true_pref = "dino" if r["oracle_alpha"] > 0.5 else "clip"
            if pred_pref == true_pref:
                n_correct_type += 1

        others_sem = [rows[j]["oracle_alpha"] for j in range(len(rows))
                     if j != i and rows[j]["semantic"] == r["semantic"]]
        if others_sem:
            pred_pref = "dino" if np.mean(others_sem) > 0.5 else "clip"
            true_pref = "dino" if r["oracle_alpha"] > 0.5 else "clip"
            if pred_pref == true_pref:
                n_correct_sem += 1

    print(f"\n  LOO directional accuracy (correct model preference):")
    print(f"    Type-based: {n_correct_type}/30 ({n_correct_type/30*100:.0f}%)")
    print(f"    Semantic-based: {n_correct_sem}/30 ({n_correct_sem/30*100:.0f}%)")

    return {
        "fixed_regret": float(np.mean(fixed_regrets)),
        "loo_type_regret": float(np.mean(loo_regrets_type)),
        "loo_sem_regret": float(np.mean(loo_regrets_sem)),
        "loo_2d_regret": float(np.mean(loo_regrets_2d)),
        "directional_accuracy_type": n_correct_type,
        "directional_accuracy_sem": n_correct_sem,
    }


def part_d_regret_decomposition(rows):
    """Part D: Regret decomposition by taxonomy attributes."""
    print("\n" + "=" * 70)
    print("PART D: Fixed-Fusion Regret Decomposition by Taxonomy")
    print("=" * 70)

    fixed_regrets = [(r, r["oracle"] - r["fixed"]) for r in rows]

    # By type
    print("\nRegret by Type:")
    for t in ["texture", "object"]:
        t_regrets = [reg for r, reg in fixed_regrets if r["type"] == t]
        print(f"  {t:10s}: mean={np.mean(t_regrets):.3f}pp, "
              f"max={np.max(t_regrets):.2f}pp, N={len(t_regrets)}")

    # By semantic
    print("\nRegret by Semantic:")
    for s in ["color", "shape", "mixed"]:
        s_regrets = [reg for r, reg in fixed_regrets if r["semantic"] == s]
        if s_regrets:
            print(f"  {s:10s}: mean={np.mean(s_regrets):.3f}pp, "
                  f"max={np.max(s_regrets):.2f}pp, N={len(s_regrets)}")

    # By dataset
    print("\nRegret by Dataset:")
    for ds in ["MVTec AD", "VisA", "BTAD"]:
        ds_regrets = [reg for r, reg in fixed_regrets if r["dataset"] == ds]
        print(f"  {ds:10s}: mean={np.mean(ds_regrets):.3f}pp, "
              f"max={np.max(ds_regrets):.2f}pp, N={len(ds_regrets)}")

    # Total regret budget: which categories contribute most?
    total_regret = sum(reg for _, reg in fixed_regrets)
    sorted_regrets = sorted(fixed_regrets, key=lambda x: x[1], reverse=True)

    print(f"\nTotal regret budget: {total_regret:.2f}pp across 30 categories")
    print(f"\nTop 10 regret contributors:")
    cumulative = 0
    for r, reg in sorted_regrets[:10]:
        cumulative += reg
        pct = cumulative / total_regret * 100
        print(f"  {r['category']:15s} ({r['dataset']:8s}, {r['type']:7s}, "
              f"α*={r['oracle_alpha']:.2f}): {reg:.3f}pp  "
              f"cumulative={cumulative:.2f}pp ({pct:.0f}%)")

    # How many categories have zero regret (<0.01pp)?
    n_zero = sum(1 for _, reg in fixed_regrets if reg < 0.01)
    n_small = sum(1 for _, reg in fixed_regrets if reg < 0.1)
    n_medium = sum(1 for _, reg in fixed_regrets if 0.1 <= reg < 0.5)
    n_large = sum(1 for _, reg in fixed_regrets if reg >= 0.5)

    print(f"\nRegret distribution:")
    print(f"  Zero (<0.01pp):     {n_zero}/30 ({n_zero/30*100:.0f}%)")
    print(f"  Small (<0.1pp):     {n_small}/30 ({n_small/30*100:.0f}%)")
    print(f"  Medium (0.1-0.5pp): {n_medium}/30 ({n_medium/30*100:.0f}%)")
    print(f"  Large (>=0.5pp):    {n_large}/30 ({n_large/30*100:.0f}%)")

    return {
        "total_regret": float(total_regret),
        "n_zero_regret": n_zero,
        "n_small_regret": n_small,
        "n_large_regret": n_large,
    }


def part_e_effect_sizes(comp, pix):
    """Part E: Complete Cohen's d effect size compilation."""
    print("\n" + "=" * 70)
    print("PART E: Effect Size Compilation (Cohen's d)")
    print("=" * 70)

    results = {}

    # Image level: CLIP vs DINOv2 per dataset
    print("\nImage-Level: CLIP vs DINOv2 (paired)")
    for ds in ["MVTec AD", "VisA", "BTAD"]:
        clip_vals = list(comp[ds]["CLIP only"]["per_category"].values())
        dino_vals = list(comp[ds]["DINOv2 only"]["per_category"].values())
        diffs = [d - c for c, d in zip(clip_vals, dino_vals)]
        n = len(diffs)

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        w_stat, w_p = stats.wilcoxon(clip_vals, dino_vals)

        print(f"  {ds:10s} (N={n}): mean_diff={mean_diff:+.2f}pp, "
              f"Cohen's d={cohens_d:.3f}, Wilcoxon p={w_p:.4f}")

        results[f"image_{ds}"] = {
            "mean_diff": float(mean_diff),
            "cohens_d": float(cohens_d),
            "wilcoxon_p": float(w_p),
            "n": n
        }

    # Image level: Best single vs Oracle fusion
    print("\nImage-Level: Best Single vs Oracle Fusion")
    for ds in ["MVTec AD", "VisA", "BTAD"]:
        clip_vals = list(comp[ds]["CLIP only"]["per_category"].values())
        dino_vals = list(comp[ds]["DINOv2 only"]["per_category"].values())
        oracle_vals = list(comp[ds]["Oracle"]["per_category"].values())
        best_vals = [max(c, d) for c, d in zip(clip_vals, dino_vals)]

        diffs = [o - b for o, b in zip(oracle_vals, best_vals)]
        n = len(diffs)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        print(f"  {ds:10s} (N={n}): mean_gain={mean_diff:+.3f}pp, Cohen's d={cohens_d:.3f}")
        results[f"fusion_gain_{ds}"] = {
            "mean_gain": float(mean_diff),
            "cohens_d": float(cohens_d),
            "n": n
        }

    # Pixel level: CLIP vs DINOv2
    print("\nPixel-Level: CLIP vs DINOv2 (paired)")
    ds_map = {"mvtec": ("MVTec AD", 15), "visa": ("VisA", 12), "btad": ("BTAD", 3)}
    for ds_key, (ds_name, n_cat) in ds_map.items():
        strats = pix[ds_key]["strategies"]
        clip_vals = list(strats["CLIP only"].values())
        dino_vals = list(strats["DINOv2 only"].values())

        diffs = [d - c for c, d in zip(clip_vals, dino_vals)]
        n = len(diffs)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        if n >= 6:  # Wilcoxon needs enough pairs
            w_stat, w_p = stats.wilcoxon(clip_vals, dino_vals)
        else:
            w_p = float('nan')

        print(f"  {ds_name:10s} (N={n}): mean_diff={mean_diff:+.2f}pp, "
              f"Cohen's d={cohens_d:.3f}, Wilcoxon p={w_p:.4f}")

        results[f"pixel_{ds_name}"] = {
            "mean_diff": float(mean_diff),
            "cohens_d": float(cohens_d),
            "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
            "n": n
        }

    # Cross-dataset pooled: image
    print("\nCross-Dataset Pooled (Image, N=30):")
    all_clip = []
    all_dino = []
    for ds in ["MVTec AD", "VisA", "BTAD"]:
        all_clip.extend(comp[ds]["CLIP only"]["per_category"].values())
        all_dino.extend(comp[ds]["DINOv2 only"]["per_category"].values())

    diffs = [d - c for c, d in zip(all_clip, all_dino)]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    w_stat, w_p = stats.wilcoxon(all_clip, all_dino)

    print(f"  Pooled image (N=30): mean_diff={mean_diff:+.2f}pp, "
          f"Cohen's d={cohens_d:.3f}, Wilcoxon p={w_p:.4f}")

    results["pooled_image"] = {
        "mean_diff": float(mean_diff),
        "cohens_d": float(cohens_d),
        "wilcoxon_p": float(w_p),
        "n": 30
    }

    # Effect size interpretation
    print("\n--- Effect Size Interpretation ---")
    print("  Cohen's d < 0.2: negligible")
    print("  Cohen's d 0.2-0.5: small")
    print("  Cohen's d 0.5-0.8: medium")
    print("  Cohen's d > 0.8: large")
    print()
    for key, val in results.items():
        d = val["cohens_d"]
        interp = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"  {key:25s}: d={d:+.3f} ({interp})")

    return results


def main():
    print("Experiment 23: Taxonomy-Guided Alpha Selection & Regret Decomposition")
    print("=" * 70)

    comp, tax, pix = load_data()
    rows = build_category_data(comp, tax)

    print(f"\nTotal categories: {len(rows)}")

    results = {}
    results["part_a"] = part_a_group_mean_alpha(rows, comp)
    results["part_b"] = part_b_2d_taxonomy(rows, comp)
    results["part_c"] = part_c_loo_taxonomy(rows, comp)
    results["part_d"] = part_d_regret_decomposition(rows)
    results["part_e"] = part_e_effect_sizes(comp, pix)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"experiment_23_taxonomy_alpha": results}, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
