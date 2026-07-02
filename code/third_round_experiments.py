"""
Third-round reviewer response experiments.

Part 1: Naive baseline accuracy (always-predict-DINOv2)
Part 2: Bootstrap CI for regret values (fixed vs taxonomy)
Part 3: Rank-biserial correlation for Wilcoxon tests
Part 4: r-value verification for Figure 5
Part 5: Robust reversal count from Part A bootstrap data
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

src_dir = Path(__file__).parent

# Load data
with open(src_dir / 'category_taxonomy_results.json') as f:
    tax_data = json.load(f)

with open(src_dir / 'reviewer_response_results.json') as f:
    rr_data = json.load(f)

with open(src_dir / 'fusion_results.json') as f:
    fusion_data = json.load(f)

with open(src_dir / 'comprehensive_eval_results.json') as f:
    eval_data = json.load(f)

oracle_alphas = tax_data['oracle_alphas']
taxonomy = tax_data['taxonomy']


# ================================================================
# Part 1: Naive baseline (always predict DINOv2)
# ================================================================
print("=" * 60)
print("Part 1: Naive Baseline Accuracy")
print("=" * 60)

all_alphas = {}
for ds_key in ['mvtec_image', 'visa_image', 'btad_image']:
    for cat, alpha in oracle_alphas[ds_key].items():
        all_alphas[cat] = alpha

n_dino_preferred = sum(1 for a in all_alphas.values() if a >= 0.5)
n_clip_preferred = sum(1 for a in all_alphas.values() if a < 0.5)
n_total = len(all_alphas)

print(f"  DINOv2-preferred (a* >= 0.5): {n_dino_preferred}/{n_total}")
print(f"  CLIP-preferred (a* < 0.5):    {n_clip_preferred}/{n_total}")
print(f"  Naive baseline (always DINOv2): {n_dino_preferred}/{n_total} = {n_dino_preferred/n_total*100:.1f}%")

# Per-dataset breakdown
for ds_key, ds_name in [('mvtec_image', 'MVTec'), ('visa_image', 'VisA'), ('btad_image', 'BTAD')]:
    alphas = oracle_alphas[ds_key]
    n_d = sum(1 for a in alphas.values() if a >= 0.5)
    n_t = len(alphas)
    clip_cats = [cat for cat, a in alphas.items() if a < 0.5]
    print(f"  {ds_name}: {n_d}/{n_t} DINOv2-preferred ({n_d/n_t*100:.1f}%)")
    if clip_cats:
        print(f"    CLIP-preferred: {clip_cats}")

# Binomial test against naive baseline
tax_correct = sum(1 for cat, a in all_alphas.items()
                  if (taxonomy.get(cat, {}).get('type') == 'object') == (a >= 0.5))
p_vs_naive = 1 - stats.binom.cdf(tax_correct - 1, n_total, n_dino_preferred / n_total)
print(f"\n  Taxonomy accuracy: {tax_correct}/{n_total} = {tax_correct/n_total*100:.1f}%")
print(f"  Naive baseline:   {n_dino_preferred}/{n_total} = {n_dino_preferred/n_total*100:.1f}%")
print(f"  Improvement: {tax_correct - n_dino_preferred} categories ({(tax_correct - n_dino_preferred)/n_total*100:.1f}pp)")
p1 = dict(n_dino=n_dino_preferred, n_clip=n_clip_preferred, n_total=n_total, tax_correct=tax_correct)


# ================================================================
# Part 2: Bootstrap CI for regret values
# ================================================================
print("\n" + "=" * 60)
print("Part 2: Bootstrap CI for Regret Values")
print("=" * 60)

# Compute per-category regret under different strategies
# We need per-category AUROC for CLIP, DINOv2, Oracle
per_cat_data = []
for ds_eval_key, ds_alpha_key in [("MVTec AD", "mvtec_image"), ("VisA", "visa_image"), ("BTAD", "btad_image")]:
    ds = eval_data[ds_eval_key]
    for cat in ds["CLIP only"]["per_category"]:
        clip_auroc = ds["CLIP only"]["per_category"][cat]
        dino_auroc = ds["DINOv2 only"]["per_category"][cat]
        oracle_auroc = ds["Oracle"]["per_category"][cat]
        oracle_alpha = oracle_alphas[ds_alpha_key].get(cat, 0.5)
        cat_type = taxonomy.get(cat, {}).get("type", "object")

        # Regret under fixed alpha=0.65
        fused_fixed = clip_auroc * (1 - 0.65) + dino_auroc * 0.65
        regret_fixed = oracle_auroc - fused_fixed

        # Regret under equal alpha=0.5
        fused_equal = clip_auroc * 0.5 + dino_auroc * 0.5
        regret_equal = oracle_auroc - fused_equal

        per_cat_data.append({
            "category": cat,
            "dataset": ds_eval_key,
            "type": cat_type,
            "oracle_alpha": oracle_alpha,
            "oracle_auroc": oracle_auroc,
            "clip_auroc": clip_auroc,
            "dino_auroc": dino_auroc,
            "regret_fixed": max(0, regret_fixed),
            "regret_equal": max(0, regret_equal),
        })

# Compute per-category regret for taxonomy strategies
# Strategy (i): group-mean by product type
type_groups = {}
for d in per_cat_data:
    t = d["type"]
    if t not in type_groups:
        type_groups[t] = []
    type_groups[t].append(d["oracle_alpha"])

type_means = {t: np.mean(alphas) for t, alphas in type_groups.items()}

for d in per_cat_data:
    # LOO group-mean: exclude self from group
    same_type = [dd["oracle_alpha"] for dd in per_cat_data
                 if dd["type"] == d["type"] and dd["category"] != d["category"]]
    loo_alpha = np.mean(same_type) if same_type else 0.65
    fused = d["clip_auroc"] * (1 - loo_alpha) + d["dino_auroc"] * loo_alpha
    d["regret_taxonomy_group"] = max(0, d["oracle_auroc"] - fused)

# Print current regret values
regrets_fixed = [d["regret_fixed"] for d in per_cat_data]
regrets_taxonomy = [d["regret_taxonomy_group"] for d in per_cat_data]
print(f"  Mean regret (fixed α=0.65): {np.mean(regrets_fixed):.4f} pp")
print(f"  Mean regret (group-mean taxonomy): {np.mean(regrets_taxonomy):.4f} pp")

# Bootstrap CI for regret
np.random.seed(42)
n_boot = 2000
n_cats = len(per_cat_data)

boot_fixed = []
boot_taxonomy = []
boot_diff = []

for _ in range(n_boot):
    idx = np.random.choice(n_cats, size=n_cats, replace=True)
    rf = np.mean([regrets_fixed[i] for i in idx])
    rt = np.mean([regrets_taxonomy[i] for i in idx])
    boot_fixed.append(rf)
    boot_taxonomy.append(rt)
    boot_diff.append(rt - rf)

ci_fixed = (np.percentile(boot_fixed, 2.5), np.percentile(boot_fixed, 97.5))
ci_taxonomy = (np.percentile(boot_taxonomy, 2.5), np.percentile(boot_taxonomy, 97.5))
ci_diff = (np.percentile(boot_diff, 2.5), np.percentile(boot_diff, 97.5))

print(f"\n  Bootstrap 95% CI (2000 resamples):")
print(f"  Fixed α=0.65 regret:    [{ci_fixed[0]:.3f}, {ci_fixed[1]:.3f}] pp")
print(f"  Group-mean taxonomy:    [{ci_taxonomy[0]:.3f}, {ci_taxonomy[1]:.3f}] pp")
print(f"  Difference (tax-fixed): [{ci_diff[0]:.3f}, {ci_diff[1]:.3f}] pp")
print(f"  CIs overlap: {ci_fixed[1] > ci_taxonomy[0]}")
print(f"  Diff CI excludes 0: {ci_diff[0] > 0 or ci_diff[1] < 0}")


# ================================================================
# Part 3: Rank-biserial correlation
# ================================================================
print("\n" + "=" * 60)
print("Part 3: Rank-Biserial Correlation for Wilcoxon Tests")
print("=" * 60)

# From Table 1: Wilcoxon results
# MVTec image: U statistic from Mann-Whitney
# The paper uses Wilcoxon signed-rank test on paired differences
# For rank-biserial: r_rb = 1 - 2U/(n*(n+1)/2) for signed-rank
# Or equivalently: r_rb = Z / sqrt(n) for approximate normal

# Compute from per-category AUROC data
for ds_eval_key, ds_name, n_cats in [("MVTec AD", "MVTec", 15), ("VisA", "VisA", 12)]:
    ds = eval_data[ds_eval_key]
    clip_vals = []
    dino_vals = []
    for cat in ds["CLIP only"]["per_category"]:
        clip_vals.append(ds["CLIP only"]["per_category"][cat])
        dino_vals.append(ds["DINOv2 only"]["per_category"][cat])

    clip_arr = np.array(clip_vals)
    dino_arr = np.array(dino_vals)
    diffs = dino_arr - clip_arr

    # Wilcoxon signed-rank test
    stat, p = stats.wilcoxon(diffs, alternative='two-sided')

    # Rank-biserial correlation for Wilcoxon signed-rank
    n = len(diffs)
    # r_rb = 1 - (2 * T_minus) / (n * (n+1) / 2)
    # where T_minus = sum of ranks for negative differences
    # stat from scipy is min(T+, T-)
    total_rank_sum = n * (n + 1) / 2
    # stat is T+ (the test statistic is the sum of positive ranks, or min)
    # For wilcoxon, stat = min(T+, T-) in older scipy, or T+ in newer
    # Let's compute manually
    abs_diffs = np.abs(diffs)
    nonzero = abs_diffs > 0
    abs_diffs_nz = abs_diffs[nonzero]
    diffs_nz = diffs[nonzero]
    ranks = stats.rankdata(abs_diffs_nz)
    T_plus = np.sum(ranks[diffs_nz > 0])
    T_minus = np.sum(ranks[diffs_nz < 0])
    n_nz = len(diffs_nz)
    total_rs = n_nz * (n_nz + 1) / 2

    r_rb = (T_plus - T_minus) / total_rs

    print(f"  {ds_name} (N={n_cats}):")
    print(f"    Wilcoxon stat={stat:.1f}, p={p:.4f}")
    print(f"    T+ = {T_plus:.1f}, T- = {T_minus:.1f}")
    print(f"    Rank-biserial r_rb = {r_rb:.3f}")
    print(f"    (r_rb > 0 means DINOv2 > CLIP)")


# ================================================================
# Part 4: r-value verification for Figure 5
# ================================================================
print("\n" + "=" * 60)
print("Part 4: r-value Verification (Figure 5)")
print("=" * 60)

# Reproduce the exact computation from generate_figures.py
rows = []
for ds_key, alpha_key in [("MVTec AD", "mvtec_image"), ("VisA", "visa_image"), ("BTAD", "btad_image")]:
    ds = eval_data[ds_key]
    categories = list(ds["CLIP only"]["per_category"].keys())
    for cat in categories:
        clip = ds["CLIP only"]["per_category"][cat]
        dino = ds["DINOv2 only"]["per_category"][cat]
        oracle = ds["Oracle"]["per_category"][cat]
        best = max(clip, dino)
        gain = oracle - best
        rows.append({"best_single": best, "fusion_gain": gain})

x = np.array([r["best_single"] for r in rows])
y = np.array([r["fusion_gain"] for r in rows])
slope, intercept, r_val, p_val, _ = stats.linregress(x, y)

print(f"  linregress r = {r_val:.4f}")
print(f"  linregress p = {p_val:.2e}")
print(f"  Pearson r, p = {stats.pearsonr(x, y)}")
print(f"  N = {len(x)}")
rho_val, rho_p = stats.spearmanr(x, y)
print(f"  Spearman rho, p = {rho_val:.4f}, {rho_p:.2e}")

# Also check from fusion_gain_results.json
with open(src_dir / 'fusion_gain_results.json') as f:
    fg_data = json.load(f)
print(f"  fusion_gain_results.json r = {fg_data['experiment_18_fusion_gain_analysis']['part_b']['best_single']['r']:.4f}")


# ================================================================
# Part 5: Robust reversal count from bootstrap data
# ================================================================
print("\n" + "=" * 60)
print("Part 5: Robust Reversal Count")
print("=" * 60)

# From Part A data: check which reversal categories have CIs not crossing 0.5
part_a = rr_data['part_a']
reversal_cats = [c for c in part_a['categories'] if c['is_reversal']]
robust_reversals = [c for c in reversal_cats if not c['crosses_05']]

print(f"  Total categories in bootstrap: {part_a['n_total']}")
print(f"  Reversal categories: {len(reversal_cats)}")
print(f"  Robust reversals (image CI does not cross 0.5): {len(robust_reversals)}")
print(f"  Uncertain reversals (image CI crosses 0.5): {len(reversal_cats) - len(robust_reversals)}")

print(f"\n  Robust reversal categories:")
for c in robust_reversals:
    print(f"    {c['category']} ({c['dataset']}): α*_img={c['orig_alpha_image']:.2f}, α*_pix={c['orig_alpha_pixel']:.2f}, CI=[{c['ci_lo']:.2f}, {c['ci_hi']:.2f}]")

print(f"\n  Uncertain reversal categories:")
uncertain = [c for c in reversal_cats if c['crosses_05']]
for c in uncertain:
    print(f"    {c['category']} ({c['dataset']}): α*_img={c['orig_alpha_image']:.2f}, α*_pix={c['orig_alpha_pixel']:.2f}, CI=[{c['ci_lo']:.2f}, {c['ci_hi']:.2f}]")

# Count per-dataset reversals from the full 30-category data
print(f"\n  Per-dataset reversal counts (from original oracle alphas):")
for ds_key, ds_name, n_total in [('mvtec', 'MVTec', 15), ('visa', 'VisA', 12), ('btad', 'BTAD', 3)]:
    img_alphas = oracle_alphas[f'{ds_key}_image']
    pix_alphas = oracle_alphas[f'{ds_key}_pixel']
    n_rev = 0
    rev_cats = []
    for cat in img_alphas:
        img_a = img_alphas[cat]
        pix_a = pix_alphas[cat]
        img_dino = img_a >= 0.5
        pix_dino = pix_a >= 0.5
        if img_dino != pix_dino:
            n_rev += 1
            rev_cats.append(cat)
    print(f"  {ds_name}: {n_rev}/{n_total} ({n_rev/n_total*100:.0f}%) reversals")
    if rev_cats:
        print(f"    Categories: {rev_cats}")


# Save results
results = {
    "part1_naive_baseline": {
        "n_dino_preferred": p1["n_dino"],
        "n_clip_preferred": p1["n_clip"],
        "n_total": p1["n_total"],
        "naive_accuracy": p1["n_dino"] / p1["n_total"],
        "taxonomy_accuracy": p1["tax_correct"] / p1["n_total"],
        "improvement_categories": p1["tax_correct"] - p1["n_dino"],
    },
    "part2_regret_bootstrap": {
        "mean_regret_fixed": float(np.mean(regrets_fixed)),
        "mean_regret_taxonomy": float(np.mean(regrets_taxonomy)),
        "ci_fixed": [float(ci_fixed[0]), float(ci_fixed[1])],
        "ci_taxonomy": [float(ci_taxonomy[0]), float(ci_taxonomy[1])],
        "ci_diff": [float(ci_diff[0]), float(ci_diff[1])],
    },
    "part4_r_value": {
        "pearson_r": float(r_val),
        "pearson_p": float(p_val),
        "spearman_rho": float(rho_val),
        "spearman_p": float(rho_p),
        "n": int(len(x)),
    },
    "part5_robust_reversals": {
        "n_reversal": len(reversal_cats),
        "n_robust": len(robust_reversals),
        "n_uncertain": len(reversal_cats) - len(robust_reversals),
    },
}

with open(src_dir / 'third_round_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

print(f"\n\nResults saved to third_round_results.json")
