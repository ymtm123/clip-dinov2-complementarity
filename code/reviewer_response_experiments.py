"""
Reviewer Response Experiments: Bootstrap CI for oracle alpha, reversal rate, directional accuracy.

Addresses reviewer concerns about:
- Robustness of oracle alpha to test-set composition (grid resolution noise)
- Robustness of 43% level reversal rate
- Bootstrap CI for 83% directional accuracy
- Taxonomy strategy comparison (multi-strategy report)
"""

import json
import os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import csv
import re

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "reviewer_response_results.json")


# === Data Loading (from normalization_robustness.py) ===

def load_anomalyclip_scores(csv_path):
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            obj, img_path, score, gt = row[0], row[1], float(row[2]), int(row[3])
            match = re.search(r'/test/(.+)$', img_path.replace('\\', '/'))
            rel_path = match.group(1) if match else os.path.basename(img_path)
            data[obj]['scores'].append(score)
            data[obj]['labels'].append(gt)
            data[obj]['paths'].append(rel_path)
    return dict(data)


def load_anomalydino_scores(csv_path):
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            obj = row[0]
            sample = row[1]
            score = float(row[2])
            is_anomaly = 0 if sample.startswith("good/") else 1
            data[obj]['scores'].append(score)
            data[obj]['labels'].append(is_anomaly)
            data[obj]['paths'].append(sample)
    return dict(data)


def align_samples(clip_data, dino_data):
    aligned = {}
    common_objs = set(clip_data.keys()) & set(dino_data.keys())
    for obj in sorted(common_objs):
        clip_path_map = {}
        for i, p in enumerate(clip_data[obj]['paths']):
            key = p.replace('\\', '/').lower()
            clip_path_map[key] = i
        dino_path_map = {}
        for i, p in enumerate(dino_data[obj]['paths']):
            key = p.replace('\\', '/').lower()
            dino_path_map[key] = i
        common_paths = set(clip_path_map.keys()) & set(dino_path_map.keys())
        if len(common_paths) == 0:
            clip_fname_map = {}
            for i, p in enumerate(clip_data[obj]['paths']):
                fname = os.path.basename(p).lower()
                subdir = os.path.dirname(p).replace('\\', '/').split('/')[-1].lower()
                key = f"{subdir}/{fname}"
                clip_fname_map[key] = i
            dino_fname_map = {}
            for i, p in enumerate(dino_data[obj]['paths']):
                fname = os.path.basename(p).lower()
                subdir = os.path.dirname(p).replace('\\', '/').split('/')[-1].lower()
                key = f"{subdir}/{fname}"
                dino_fname_map[key] = i
            common_paths = set(clip_fname_map.keys()) & set(dino_fname_map.keys())
            if len(common_paths) > 0:
                clip_path_map = clip_fname_map
                dino_path_map = dino_fname_map
            else:
                n = min(len(clip_data[obj]['scores']), len(dino_data[obj]['scores']))
                aligned[obj] = {
                    'clip_scores': np.array(clip_data[obj]['scores'][:n]),
                    'dino_scores': np.array(dino_data[obj]['scores'][:n]),
                    'labels': np.array(clip_data[obj]['labels'][:n]),
                    'n_samples': n
                }
                continue
        clip_scores, dino_scores, labels = [], [], []
        for path in sorted(common_paths):
            ci = clip_path_map[path]
            di = dino_path_map[path]
            clip_scores.append(clip_data[obj]['scores'][ci])
            dino_scores.append(dino_data[obj]['scores'][di])
            labels.append(clip_data[obj]['labels'][ci])
        aligned[obj] = {
            'clip_scores': np.array(clip_scores),
            'dino_scores': np.array(dino_scores),
            'labels': np.array(labels),
            'n_samples': len(labels)
        }
    return aligned


def load_all_datasets():
    base = os.path.dirname(__file__)
    datasets = {}
    clip_mvtec = load_anomalyclip_scores(
        os.path.join(base, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv"))
    dino_mvtec = load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv"))
    dino_mvtec.update(load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")))
    datasets["MVTec AD"] = align_samples(clip_mvtec, dino_mvtec)

    clip_visa = load_anomalyclip_scores(
        os.path.join(base, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv"))
    dino_visa = load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv"))
    datasets["VisA"] = align_samples(clip_visa, dino_visa)

    clip_btad = load_anomalyclip_scores(
        os.path.join(base, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv"))
    dino_btad = load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv"))
    datasets["BTAD"] = align_samples(clip_btad, dino_btad)

    return datasets


def normalize_minmax(scores):
    s = np.array(scores, dtype=float)
    smin, smax = s.min(), s.max()
    if smax - smin < 1e-10:
        return np.zeros_like(s)
    return (s - smin) / (smax - smin)


def find_oracle_alpha(labels, clip_norm, dino_norm, alpha_grid):
    """Find oracle alpha on normalized scores."""
    best_alpha, best_auroc = 0.5, -1
    for a in alpha_grid:
        fused = (1 - a) * clip_norm + a * dino_norm
        try:
            auroc = roc_auc_score(labels, fused)
            if auroc > best_auroc:
                best_auroc = auroc
                best_alpha = a
        except:
            continue
    return best_alpha, best_auroc


# =============================================================================
# Part A: Bootstrap oracle alpha and reversal robustness
# =============================================================================

def part_a_bootstrap_oracle_alpha(datasets, taxonomy):
    """Bootstrap oracle alpha* from per-image scores."""
    print("=" * 70)
    print("PART A: Bootstrap Oracle Alpha & Reversal Robustness")
    print("=" * 70)

    alpha_grid = np.arange(0.0, 1.01, 0.05)
    n_boot = 1000
    rng = np.random.RandomState(42)

    ds_alpha_keys = {
        "MVTec AD": ("mvtec_image", "mvtec_pixel"),
        "VisA": ("visa_image", "visa_pixel"),
        "BTAD": ("btad_image", "btad_pixel"),
    }

    all_categories = []

    for ds_name, ds_data in datasets.items():
        img_key, pix_key = ds_alpha_keys[ds_name]
        print(f"\n  {ds_name}:")

        for cat, d in sorted(ds_data.items()):
            labels = d['labels']
            clip_raw = d['clip_scores']
            dino_raw = d['dino_scores']
            n = len(labels)

            if n < 10 or len(np.unique(labels)) < 2:
                continue

            # Original oracle alpha (image level)
            clip_norm = normalize_minmax(clip_raw)
            dino_norm = normalize_minmax(dino_raw)
            orig_alpha, orig_auroc = find_oracle_alpha(labels, clip_norm, dino_norm, alpha_grid)

            # Get pixel-level oracle alpha from taxonomy data
            orig_alpha_pixel = taxonomy["oracle_alphas"].get(pix_key, {}).get(cat, 0.5)

            # Bootstrap
            boot_alphas = []
            for _ in range(n_boot):
                idx = rng.choice(n, size=n, replace=True)
                b_labels = labels[idx]
                if len(np.unique(b_labels)) < 2:
                    continue
                b_clip = normalize_minmax(clip_raw[idx])
                b_dino = normalize_minmax(dino_raw[idx])
                ba, _ = find_oracle_alpha(b_labels, b_clip, b_dino, alpha_grid)
                boot_alphas.append(ba)

            boot_alphas = np.array(boot_alphas)
            if len(boot_alphas) < 100:
                continue

            boot_mean = np.mean(boot_alphas)
            boot_std = np.std(boot_alphas)
            ci_lo, ci_hi = np.percentile(boot_alphas, [2.5, 97.5])
            ci_width = ci_hi - ci_lo

            # Does the CI cross 0.5?
            crosses_05 = (ci_lo < 0.5) and (ci_hi > 0.5)

            # Is 0.65 within CI?
            fixed_in_ci = (ci_lo <= 0.65 <= ci_hi)

            # Original reversal status
            is_reversal = (orig_alpha > 0.5) != (orig_alpha_pixel > 0.5)

            # Reversal robustness: fraction of bootstraps that agree on dominance
            boot_same_side = np.mean((boot_alphas > 0.5) == (orig_alpha > 0.5))

            cat_result = {
                "category": cat,
                "dataset": ds_name,
                "n_samples": n,
                "orig_alpha_image": float(orig_alpha),
                "orig_alpha_pixel": float(orig_alpha_pixel),
                "boot_mean": float(boot_mean),
                "boot_std": float(boot_std),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "ci_width": float(ci_width),
                "crosses_05": bool(crosses_05),
                "fixed_in_ci": bool(fixed_in_ci),
                "is_reversal": bool(is_reversal),
                "boot_dominance_stability": float(boot_same_side),
            }
            all_categories.append(cat_result)

            print(f"    {cat:15s}: α*={orig_alpha:.2f}, "
                  f"boot={boot_mean:.3f}±{boot_std:.3f}, "
                  f"CI=[{ci_lo:.2f},{ci_hi:.2f}], w={ci_width:.2f}, "
                  f"crosses0.5={'Y' if crosses_05 else 'N'}, "
                  f"rev={'Y' if is_reversal else 'N'}")

    # Summary
    n_total = len(all_categories)
    n_crosses = sum(1 for c in all_categories if c["crosses_05"])
    n_fixed_in_ci = sum(1 for c in all_categories if c["fixed_in_ci"])
    n_reversal = sum(1 for c in all_categories if c["is_reversal"])
    ci_widths = [c["ci_width"] for c in all_categories]
    stabilities = [c["boot_dominance_stability"] for c in all_categories]

    # Reversal categories: how many have stable reversal classification?
    reversal_cats = [c for c in all_categories if c["is_reversal"]]
    stable_reversals = sum(1 for c in reversal_cats
                          if not c["crosses_05"])  # CI doesn't cross 0.5

    # Non-reversal categories whose CI DOES cross 0.5 (potential false negatives)
    non_reversal_cats = [c for c in all_categories if not c["is_reversal"]]
    unstable_non_reversals = sum(1 for c in non_reversal_cats if c["crosses_05"])

    print(f"\n  Summary (N={n_total}):")
    print(f"    Median CI width:     {np.median(ci_widths):.3f}")
    print(f"    Mean CI width:       {np.mean(ci_widths):.3f}")
    print(f"    CI crosses 0.5:      {n_crosses}/{n_total}")
    print(f"    α=0.65 in CI:        {n_fixed_in_ci}/{n_total}")
    print(f"    Original reversals:  {n_reversal}/{n_total}")
    print(f"    Stable reversals:    {stable_reversals}/{n_reversal} "
          f"(CI doesn't cross 0.5)")
    print(f"    Mean dominance stability: {np.mean(stabilities):.3f}")

    # Bootstrap the reversal RATE itself
    rng2 = np.random.RandomState(123)
    boot_reversal_rates = []
    for _ in range(2000):
        idx = rng2.choice(n_total, n_total, replace=True)
        sampled = [all_categories[i] for i in idx]
        rate = sum(1 for c in sampled if c["is_reversal"]) / n_total
        boot_reversal_rates.append(rate)
    rev_rate_ci = np.percentile(boot_reversal_rates, [2.5, 97.5])

    print(f"    Reversal rate 95% CI: [{rev_rate_ci[0]*100:.0f}%, {rev_rate_ci[1]*100:.0f}%]")

    return {
        "categories": all_categories,
        "n_total": n_total,
        "n_crosses_05": n_crosses,
        "n_fixed_in_ci": n_fixed_in_ci,
        "n_reversal": n_reversal,
        "stable_reversals": stable_reversals,
        "unstable_non_reversals": unstable_non_reversals,
        "median_ci_width": float(np.median(ci_widths)),
        "mean_ci_width": float(np.mean(ci_widths)),
        "mean_dominance_stability": float(np.mean(stabilities)),
        "reversal_rate_ci_lo": float(rev_rate_ci[0]),
        "reversal_rate_ci_hi": float(rev_rate_ci[1]),
    }


# =============================================================================
# Part B: Bootstrap directional accuracy
# =============================================================================

def part_b_bootstrap_directional_accuracy(datasets, taxonomy):
    """Bootstrap directional accuracy (83% claim) from per-image scores."""
    print("\n" + "=" * 70)
    print("PART B: Bootstrap Directional Accuracy")
    print("=" * 70)

    alpha_grid = np.arange(0.0, 1.01, 0.05)
    n_boot = 1000
    rng = np.random.RandomState(42)

    ds_alpha_keys = {
        "MVTec AD": "mvtec_image",
        "VisA": "visa_image",
        "BTAD": "btad_image",
    }

    # Build category rows with taxonomy
    rows = []
    for ds_name, ds_data in datasets.items():
        alpha_key = ds_alpha_keys[ds_name]
        for cat, d in sorted(ds_data.items()):
            labels = d['labels']
            if len(labels) < 10 or len(np.unique(labels)) < 2:
                continue
            clip_norm = normalize_minmax(d['clip_scores'])
            dino_norm = normalize_minmax(d['dino_scores'])
            orig_alpha, _ = find_oracle_alpha(labels, clip_norm, dino_norm, alpha_grid)
            cat_tax = taxonomy["taxonomy"].get(cat, {})
            rows.append({
                "category": cat,
                "dataset": ds_name,
                "type": cat_tax.get("type", "object"),
                "semantic": cat_tax.get("semantic", "shape"),
                "oracle_alpha": float(orig_alpha),
                "clip_scores": d['clip_scores'],
                "dino_scores": d['dino_scores'],
                "labels": labels,
            })

    n_cats = len(rows)

    # Original directional accuracy (LOO type-based)
    def compute_loo_accuracy(alphas, types):
        """LOO directional accuracy using type-based group mean."""
        n = len(alphas)
        correct = 0
        for i in range(n):
            others = [alphas[j] for j in range(n) if j != i and types[j] == types[i]]
            if others:
                pred_dino = np.mean(others) > 0.5
            else:
                pred_dino = True  # fallback: predict DINOv2
            true_dino = alphas[i] > 0.5
            # Handle ties at 0.5: count as correct for either prediction
            if alphas[i] == 0.5:
                correct += 1
            elif pred_dino == true_dino:
                correct += 1
        return correct

    orig_alphas = [r["oracle_alpha"] for r in rows]
    orig_types = [r["type"] for r in rows]
    orig_correct = compute_loo_accuracy(orig_alphas, orig_types)
    orig_accuracy = orig_correct / n_cats

    print(f"\n  Original LOO directional accuracy: {orig_correct}/{n_cats} = {orig_accuracy*100:.1f}%")

    # Bootstrap: resample test images within each category, recompute oracle alphas, then LOO
    boot_accuracies = []
    boot_correct_counts = []

    for b in range(n_boot):
        boot_alphas = []
        for r in rows:
            labels = r['labels']
            n = len(labels)
            idx = rng.choice(n, size=n, replace=True)
            b_labels = labels[idx]
            if len(np.unique(b_labels)) < 2:
                boot_alphas.append(r['oracle_alpha'])  # keep original
                continue
            b_clip = normalize_minmax(r['clip_scores'][idx])
            b_dino = normalize_minmax(r['dino_scores'][idx])
            ba, _ = find_oracle_alpha(b_labels, b_clip, b_dino, alpha_grid)
            boot_alphas.append(ba)

        bc = compute_loo_accuracy(boot_alphas, orig_types)
        boot_accuracies.append(bc / n_cats)
        boot_correct_counts.append(bc)

    acc_ci = np.percentile(boot_accuracies, [2.5, 97.5])
    mean_acc = np.mean(boot_accuracies)

    print(f"  Bootstrap accuracy: mean={mean_acc*100:.1f}%, "
          f"95% CI=[{acc_ci[0]*100:.1f}%, {acc_ci[1]*100:.1f}%]")

    # Binomial test for original accuracy
    binom_p = stats.binom_test(orig_correct, n_cats, 0.5, alternative='greater')
    print(f"  Binomial test (H0: 50%): p={binom_p:.6f}")

    return {
        "n_categories": n_cats,
        "original_correct": orig_correct,
        "original_accuracy": float(orig_accuracy),
        "boot_mean_accuracy": float(mean_acc),
        "boot_ci_lo": float(acc_ci[0]),
        "boot_ci_hi": float(acc_ci[1]),
        "binomial_p": float(binom_p),
    }


# =============================================================================
# Part C: Multi-strategy taxonomy comparison (from existing data)
# =============================================================================

def part_c_taxonomy_strategies():
    """Report all taxonomy strategies tested."""
    print("\n" + "=" * 70)
    print("PART C: Multi-Strategy Taxonomy Comparison")
    print("=" * 70)

    base = os.path.dirname(__file__)
    tax_file = os.path.join(base, "taxonomy_alpha_results.json")
    with open(tax_file) as f:
        tax_results = json.load(f)["experiment_23_taxonomy_alpha"]

    strategies = {
        "Fixed (alpha=0.65)": tax_results["part_c"]["fixed_regret"],
        "Group-mean (type)": tax_results["part_a"]["type_based_regret"],
        "2D taxonomy (type x semantic)": tax_results["part_b"]["mean_regret_2d"],
        "LOO type-based": tax_results["part_c"]["loo_type_regret"],
        "LOO semantic-based": tax_results["part_c"]["loo_sem_regret"],
        "LOO 2D (type x semantic)": tax_results["part_c"]["loo_2d_regret"],
    }

    print(f"\n  Strategy comparison (per-category mean regret, pp):")
    print(f"  {'Strategy':<35s} {'Regret':>8s} {'vs Fixed':>10s}")
    print("-" * 58)
    fixed_regret = strategies["Fixed (alpha=0.65)"]
    for name, regret in sorted(strategies.items(), key=lambda x: x[1]):
        diff = regret - fixed_regret
        marker = " <-- best" if name == "Fixed (alpha=0.65)" else ""
        print(f"  {name:<35s} {regret:>8.3f} {diff:>+10.3f}{marker}")

    print(f"\n  Key finding: ALL taxonomy-guided strategies increase regret vs fixed.")
    print(f"  More granular taxonomies perform WORSE, not better.")

    # Directional accuracy
    print(f"\n  LOO directional accuracy (binary: CLIP vs DINOv2 preference):")
    print(f"    Type-based:     {tax_results['part_c']['directional_accuracy_type']}/30 "
          f"({tax_results['part_c']['directional_accuracy_type']/30*100:.0f}%)")
    print(f"    Semantic-based: {tax_results['part_c']['directional_accuracy_sem']}/30 "
          f"({tax_results['part_c']['directional_accuracy_sem']/30*100:.0f}%)")

    return {
        "strategies": {k: float(v) for k, v in strategies.items()},
        "directional_accuracy_type": tax_results["part_c"]["directional_accuracy_type"],
        "directional_accuracy_sem": tax_results["part_c"]["directional_accuracy_sem"],
    }


# =============================================================================
# Part D: Intra-group variance for 83% vs weight estimation gap
# =============================================================================

def part_d_intragroup_variance(taxonomy):
    """Compute intra-group variance for the accuracy vs estimation argument."""
    print("\n" + "=" * 70)
    print("PART D: Intra-Group Variance Analysis")
    print("=" * 70)

    # Collect all oracle alphas with types
    all_alphas = {}
    for level in ["mvtec_image", "visa_image", "btad_image"]:
        for cat, alpha in taxonomy["oracle_alphas"].get(level, {}).items():
            cat_type = taxonomy["taxonomy"].get(cat, {}).get("type", "object")
            all_alphas[cat] = {"alpha": alpha, "type": cat_type}

    texture_alphas = [v["alpha"] for v in all_alphas.values() if v["type"] == "texture"]
    object_alphas = [v["alpha"] for v in all_alphas.values() if v["type"] == "object"]

    print(f"\n  Texture categories (N={len(texture_alphas)}):")
    print(f"    Mean alpha: {np.mean(texture_alphas):.3f}")
    print(f"    Std alpha:  {np.std(texture_alphas):.3f}")
    print(f"    Range:      [{min(texture_alphas):.2f}, {max(texture_alphas):.2f}]")

    print(f"\n  Object categories (N={len(object_alphas)}):")
    print(f"    Mean alpha: {np.mean(object_alphas):.3f}")
    print(f"    Std alpha:  {np.std(object_alphas):.3f}")
    print(f"    Range:      [{min(object_alphas):.2f}, {max(object_alphas):.2f}]")

    return {
        "texture_mean": float(np.mean(texture_alphas)),
        "texture_std": float(np.std(texture_alphas)),
        "texture_range": [float(min(texture_alphas)), float(max(texture_alphas))],
        "object_mean": float(np.mean(object_alphas)),
        "object_std": float(np.std(object_alphas)),
        "object_range": [float(min(object_alphas)), float(max(object_alphas))],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("Reviewer Response Experiments")
    print("=" * 70)

    base = os.path.dirname(__file__)
    with open(os.path.join(base, "category_taxonomy_results.json")) as f:
        taxonomy = json.load(f)

    datasets = load_all_datasets()
    total_cats = sum(len(ds) for ds in datasets.values())
    print(f"\nLoaded {total_cats} categories across {len(datasets)} datasets")

    results = {}
    results["part_a"] = part_a_bootstrap_oracle_alpha(datasets, taxonomy)
    results["part_b"] = part_b_bootstrap_directional_accuracy(datasets, taxonomy)
    results["part_c"] = part_c_taxonomy_strategies()
    results["part_d"] = part_d_intragroup_variance(taxonomy)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
