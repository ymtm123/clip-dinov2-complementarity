"""
Experiment 27: Normalization Robustness & Bootstrap Confidence Analysis

Tests whether the paper's key findings are robust to:
1. Choice of score normalization method (min-max, z-score, rank-based)
2. Sampling variability (bootstrap confidence intervals)

Parts:
A. Fusion benefit across normalization methods
B. Optimal alpha stability across normalization methods
C. Bootstrap CIs for key AUROC differences
D. F1-max analysis (threshold-dependent metric)
E. Score distribution characterization (normal vs anomaly separation)
"""

import csv
import json
import os
import re
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score
from collections import defaultdict

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "normalization_robustness_results.json")


# === Data Loading (reuse from score_fusion.py) ===

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

    # MVTec
    clip_mvtec = load_anomalyclip_scores(
        os.path.join(base, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv"))
    dino_mvtec = load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv"))
    dino_mvtec.update(load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")))
    datasets["MVTec AD"] = align_samples(clip_mvtec, dino_mvtec)

    # VisA
    clip_visa = load_anomalyclip_scores(
        os.path.join(base, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv"))
    dino_visa = load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv"))
    datasets["VisA"] = align_samples(clip_visa, dino_visa)

    # BTAD
    clip_btad = load_anomalyclip_scores(
        os.path.join(base, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv"))
    dino_btad = load_anomalydino_scores(
        os.path.join(base, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv"))
    datasets["BTAD"] = align_samples(clip_btad, dino_btad)

    return datasets


# === Normalization Methods ===

def normalize_minmax(scores):
    s = np.array(scores, dtype=float)
    smin, smax = s.min(), s.max()
    if smax - smin < 1e-10:
        return np.zeros_like(s)
    return (s - smin) / (smax - smin)


def normalize_zscore(scores):
    s = np.array(scores, dtype=float)
    mu, std = s.mean(), s.std()
    if std < 1e-10:
        return np.zeros_like(s)
    return (s - mu) / std


def normalize_rank(scores):
    from scipy.stats import rankdata
    s = np.array(scores, dtype=float)
    return rankdata(s) / len(s)


def normalize_quantile(scores):
    """Map to uniform distribution via empirical CDF."""
    from scipy.stats import rankdata
    s = np.array(scores, dtype=float)
    return (rankdata(s) - 0.5) / len(s)


NORM_METHODS = {
    'minmax': normalize_minmax,
    'zscore': normalize_zscore,
    'rank': normalize_rank,
    'quantile': normalize_quantile,
}


# === Core Analysis ===

def compute_auroc(labels, scores):
    labels = np.array(labels)
    if len(np.unique(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, scores) * 100


def compute_f1max(labels, scores):
    """Compute maximum F1 score across all thresholds."""
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        return float('nan')
    # Use sorted unique scores as thresholds
    thresholds = np.unique(scores)
    if len(thresholds) > 500:
        thresholds = np.percentile(scores, np.linspace(0, 100, 500))
    best_f1 = 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1 * 100


def fuse_scores(clip_z, dino_z, alpha):
    return alpha * dino_z + (1 - alpha) * clip_z


def evaluate_strategy(labels, clip_scores, dino_scores, alpha, norm_fn, metric_fn=compute_auroc):
    clip_z = norm_fn(clip_scores)
    dino_z = norm_fn(dino_scores)
    fused = fuse_scores(clip_z, dino_z, alpha)
    return metric_fn(labels, fused)


def find_oracle_alpha(labels, clip_scores, dino_scores, norm_fn, metric_fn=compute_auroc):
    best_alpha, best_score = 0.5, -1
    for a in np.arange(0, 1.01, 0.05):
        s = evaluate_strategy(labels, clip_scores, dino_scores, a, norm_fn, metric_fn)
        if s > best_score:
            best_score = s
            best_alpha = a
    return best_alpha, best_score


def part_a_norm_robustness(datasets):
    """Part A: Fusion benefit consistency across normalization methods."""
    print("=" * 70)
    print("PART A: Fusion Benefit Across Normalization Methods")
    print("=" * 70)

    results = {}
    for norm_name, norm_fn in NORM_METHODS.items():
        print(f"\n  --- {norm_name} normalization ---")
        ds_results = {}
        all_clip_aurocs = []
        all_dino_aurocs = []
        all_equal_aurocs = []
        all_fixed_aurocs = []
        all_oracle_aurocs = []

        for ds_name, ds_data in datasets.items():
            clip_aurocs, dino_aurocs = [], []
            equal_aurocs, fixed_aurocs, oracle_aurocs = [], [], []

            for cat, d in ds_data.items():
                labels = d['labels']
                clip_z = norm_fn(d['clip_scores'])
                dino_z = norm_fn(d['dino_scores'])

                clip_auc = compute_auroc(labels, clip_z)
                dino_auc = compute_auroc(labels, dino_z)
                equal_auc = compute_auroc(labels, fuse_scores(clip_z, dino_z, 0.5))
                fixed_auc = compute_auroc(labels, fuse_scores(clip_z, dino_z, 0.65))

                _, oracle_auc = find_oracle_alpha(labels, d['clip_scores'], d['dino_scores'], norm_fn)

                clip_aurocs.append(clip_auc)
                dino_aurocs.append(dino_auc)
                equal_aurocs.append(equal_auc)
                fixed_aurocs.append(fixed_auc)
                oracle_aurocs.append(oracle_auc)

            ds_results[ds_name] = {
                'clip': np.mean(clip_aurocs),
                'dino': np.mean(dino_aurocs),
                'equal': np.mean(equal_aurocs),
                'fixed': np.mean(fixed_aurocs),
                'oracle': np.mean(oracle_aurocs),
            }
            all_clip_aurocs.extend(clip_aurocs)
            all_dino_aurocs.extend(dino_aurocs)
            all_equal_aurocs.extend(equal_aurocs)
            all_fixed_aurocs.extend(fixed_aurocs)
            all_oracle_aurocs.extend(oracle_aurocs)

            print(f"    {ds_name:10s}: CLIP={np.mean(clip_aurocs):.2f}, "
                  f"DINO={np.mean(dino_aurocs):.2f}, "
                  f"Equal={np.mean(equal_aurocs):.2f}, "
                  f"Fixed={np.mean(fixed_aurocs):.2f}, "
                  f"Oracle={np.mean(oracle_aurocs):.2f}")

        overall = {
            'clip': float(np.mean(all_clip_aurocs)),
            'dino': float(np.mean(all_dino_aurocs)),
            'equal': float(np.mean(all_equal_aurocs)),
            'fixed': float(np.mean(all_fixed_aurocs)),
            'oracle': float(np.mean(all_oracle_aurocs)),
            'fusion_gain_equal': float(np.mean(all_equal_aurocs) - max(np.mean(all_clip_aurocs), np.mean(all_dino_aurocs))),
            'fusion_gain_fixed': float(np.mean(all_fixed_aurocs) - max(np.mean(all_clip_aurocs), np.mean(all_dino_aurocs))),
            'regret_fixed': float(np.mean(all_oracle_aurocs) - np.mean(all_fixed_aurocs)),
        }
        results[norm_name] = overall

        print(f"    Overall: Equal gain={overall['fusion_gain_equal']:+.2f}pp, "
              f"Fixed gain={overall['fusion_gain_fixed']:+.2f}pp, "
              f"Fixed regret={overall['regret_fixed']:.2f}pp")

    # Summary comparison
    print(f"\n  Summary: Fusion benefit is robust across normalization?")
    print(f"  {'Method':>10s}  {'Equal gain':>10s}  {'Fixed gain':>10s}  {'Regret':>8s}")
    print("-" * 45)
    for nm, r in results.items():
        print(f"  {nm:>10s}  {r['fusion_gain_equal']:+.2f}pp     "
              f"{r['fusion_gain_fixed']:+.2f}pp     "
              f"{r['regret_fixed']:.2f}pp")

    all_positive = all(r['fusion_gain_equal'] > 0 for r in results.values())
    print(f"\n  Fusion always positive across all norms: {all_positive}")

    return results


def part_b_optimal_alpha_stability(datasets):
    """Part B: Does optimal alpha change with normalization method?"""
    print("\n" + "=" * 70)
    print("PART B: Optimal Alpha Stability Across Normalization Methods")
    print("=" * 70)

    # Per-category oracle alpha for each norm
    cat_alphas = {}
    for norm_name, norm_fn in NORM_METHODS.items():
        cat_alphas[norm_name] = {}
        for ds_name, ds_data in datasets.items():
            for cat, d in ds_data.items():
                key = f"{ds_name}/{cat}"
                alpha, _ = find_oracle_alpha(
                    d['labels'], d['clip_scores'], d['dino_scores'], norm_fn)
                cat_alphas[norm_name][key] = alpha

    # Global optimal alpha for each norm
    print(f"\n  Global optimal alpha per normalization:")
    global_alphas = {}
    for norm_name, norm_fn in NORM_METHODS.items():
        best_global, best_mean = 0.5, -1
        for a in np.arange(0, 1.01, 0.05):
            aurocs = []
            for ds_name, ds_data in datasets.items():
                for cat, d in ds_data.items():
                    auc = evaluate_strategy(
                        d['labels'], d['clip_scores'], d['dino_scores'], a, norm_fn)
                    aurocs.append(auc)
            m = np.mean(aurocs)
            if m > best_mean:
                best_mean = m
                best_global = a
        global_alphas[norm_name] = best_global
        print(f"    {norm_name:>10s}: alpha*={best_global:.2f}")

    # Pairwise correlation of per-category oracle alphas between norms
    norms = list(NORM_METHODS.keys())
    print(f"\n  Pairwise Spearman correlation of per-category oracle alpha:")
    cats = sorted(cat_alphas[norms[0]].keys())
    correlations = {}
    for i in range(len(norms)):
        for j in range(i+1, len(norms)):
            a1 = [cat_alphas[norms[i]][c] for c in cats]
            a2 = [cat_alphas[norms[j]][c] for c in cats]
            r, p = stats.spearmanr(a1, a2)
            key = f"{norms[i]}_vs_{norms[j]}"
            correlations[key] = {'rho': float(r), 'p': float(p)}
            print(f"    {norms[i]:>10s} vs {norms[j]:<10s}: rho={r:.3f}, p={p:.4f}")

    # Mean absolute difference in oracle alpha across norm methods
    alpha_diffs = []
    for c in cats:
        vals = [cat_alphas[n][c] for n in norms]
        alpha_diffs.append(np.max(vals) - np.min(vals))
    print(f"\n  Oracle alpha range across norms (per category):")
    print(f"    Mean range: {np.mean(alpha_diffs):.3f}")
    print(f"    Median range: {np.median(alpha_diffs):.3f}")
    print(f"    Max range: {np.max(alpha_diffs):.3f}")
    print(f"    Categories with range > 0.2: {sum(1 for d in alpha_diffs if d > 0.2)}/30")

    return {
        'global_alphas': global_alphas,
        'correlations': correlations,
        'mean_alpha_range': float(np.mean(alpha_diffs)),
    }


def part_c_bootstrap_ci(datasets):
    """Part C: Bootstrap CIs for key AUROC differences."""
    print("\n" + "=" * 70)
    print("PART C: Bootstrap Confidence Intervals")
    print("=" * 70)

    n_bootstrap = 2000
    np.random.seed(42)

    norm_fn = normalize_minmax

    # Collect per-category AUROC for each strategy
    strategy_aurocs = {
        'clip': [], 'dino': [], 'equal': [], 'fixed': [], 'oracle': []
    }

    for ds_name, ds_data in datasets.items():
        for cat, d in ds_data.items():
            labels = d['labels']
            clip_z = norm_fn(d['clip_scores'])
            dino_z = norm_fn(d['dino_scores'])

            strategy_aurocs['clip'].append(compute_auroc(labels, clip_z))
            strategy_aurocs['dino'].append(compute_auroc(labels, dino_z))
            strategy_aurocs['equal'].append(compute_auroc(labels, fuse_scores(clip_z, dino_z, 0.5)))
            strategy_aurocs['fixed'].append(compute_auroc(labels, fuse_scores(clip_z, dino_z, 0.65)))
            _, oauc = find_oracle_alpha(labels, d['clip_scores'], d['dino_scores'], norm_fn)
            strategy_aurocs['oracle'].append(oauc)

    n_cats = len(strategy_aurocs['clip'])

    # Bootstrap: resample categories (not images) for population-level inference
    boot_means = {k: [] for k in strategy_aurocs}
    boot_diffs = {
        'fusion_vs_best_single': [],
        'fixed_vs_equal': [],
        'oracle_vs_fixed': [],
        'dino_vs_clip': [],
    }

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_cats, n_cats, replace=True)
        for k in strategy_aurocs:
            vals = np.array(strategy_aurocs[k])[idx]
            boot_means[k].append(np.mean(vals))

        clip_b = np.array(strategy_aurocs['clip'])[idx]
        dino_b = np.array(strategy_aurocs['dino'])[idx]
        equal_b = np.array(strategy_aurocs['equal'])[idx]
        fixed_b = np.array(strategy_aurocs['fixed'])[idx]
        oracle_b = np.array(strategy_aurocs['oracle'])[idx]

        best_single = np.maximum(np.mean(clip_b), np.mean(dino_b))
        boot_diffs['fusion_vs_best_single'].append(np.mean(equal_b) - best_single)
        boot_diffs['fixed_vs_equal'].append(np.mean(fixed_b) - np.mean(equal_b))
        boot_diffs['oracle_vs_fixed'].append(np.mean(oracle_b) - np.mean(fixed_b))
        boot_diffs['dino_vs_clip'].append(np.mean(dino_b) - np.mean(clip_b))

    ci_results = {}
    print(f"\n  Strategy mean AUROC [95% CI] (N={n_cats} categories, {n_bootstrap} bootstraps):")
    for k in ['clip', 'dino', 'equal', 'fixed', 'oracle']:
        lo, hi = np.percentile(boot_means[k], [2.5, 97.5])
        mean = np.mean(strategy_aurocs[k])
        ci_results[k] = {'mean': float(mean), 'ci_lo': float(lo), 'ci_hi': float(hi)}
        print(f"    {k:>8s}: {mean:.2f}% [{lo:.2f}, {hi:.2f}]")

    print(f"\n  Key differences [95% CI]:")
    diff_results = {}
    for k, label in [
        ('fusion_vs_best_single', 'Equal fusion - best single'),
        ('fixed_vs_equal', 'Fixed(0.65) - Equal(0.5)'),
        ('oracle_vs_fixed', 'Oracle - Fixed(0.65)'),
        ('dino_vs_clip', 'DINOv2 - CLIP'),
    ]:
        vals = boot_diffs[k]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        mean = np.mean(vals)
        sig = "***" if lo > 0 or hi < 0 else "n.s."
        diff_results[k] = {
            'mean': float(mean), 'ci_lo': float(lo), 'ci_hi': float(hi),
            'significant': lo > 0 or hi < 0
        }
        print(f"    {label:>30s}: {mean:+.2f}pp [{lo:+.2f}, {hi:+.2f}] {sig}")

    return {
        'strategy_cis': ci_results,
        'diff_cis': diff_results,
    }


def part_d_f1max_analysis(datasets):
    """Part D: Does fusion benefit hold for F1-max (threshold-dependent)?"""
    print("\n" + "=" * 70)
    print("PART D: F1-max Analysis (Threshold-Dependent Metric)")
    print("=" * 70)

    norm_fn = normalize_minmax

    all_results = {}
    all_clip_f1 = []
    all_dino_f1 = []
    all_equal_f1 = []
    all_fixed_f1 = []
    all_oracle_f1 = []

    for ds_name, ds_data in datasets.items():
        ds_clip, ds_dino, ds_equal, ds_fixed, ds_oracle = [], [], [], [], []

        for cat, d in ds_data.items():
            labels = d['labels']
            clip_z = norm_fn(d['clip_scores'])
            dino_z = norm_fn(d['dino_scores'])

            clip_f1 = compute_f1max(labels, clip_z)
            dino_f1 = compute_f1max(labels, dino_z)
            equal_f1 = compute_f1max(labels, fuse_scores(clip_z, dino_z, 0.5))
            fixed_f1 = compute_f1max(labels, fuse_scores(clip_z, dino_z, 0.65))

            # Oracle F1 over alpha
            best_f1, best_a = -1, 0.5
            for a in np.arange(0, 1.01, 0.05):
                f1 = compute_f1max(labels, fuse_scores(clip_z, dino_z, a))
                if f1 > best_f1:
                    best_f1 = f1
                    best_a = a
            oracle_f1 = best_f1

            ds_clip.append(clip_f1)
            ds_dino.append(dino_f1)
            ds_equal.append(equal_f1)
            ds_fixed.append(fixed_f1)
            ds_oracle.append(oracle_f1)

        all_results[ds_name] = {
            'clip': float(np.mean(ds_clip)),
            'dino': float(np.mean(ds_dino)),
            'equal': float(np.mean(ds_equal)),
            'fixed': float(np.mean(ds_fixed)),
            'oracle': float(np.mean(ds_oracle)),
        }

        all_clip_f1.extend(ds_clip)
        all_dino_f1.extend(ds_dino)
        all_equal_f1.extend(ds_equal)
        all_fixed_f1.extend(ds_fixed)
        all_oracle_f1.extend(ds_oracle)

        print(f"\n  {ds_name}:")
        print(f"    CLIP={np.mean(ds_clip):.2f}%, DINO={np.mean(ds_dino):.2f}%, "
              f"Equal={np.mean(ds_equal):.2f}%, Fixed={np.mean(ds_fixed):.2f}%, "
              f"Oracle={np.mean(ds_oracle):.2f}%")

    # Overall
    fusion_gain_eq = np.mean(all_equal_f1) - max(np.mean(all_clip_f1), np.mean(all_dino_f1))
    fusion_gain_fix = np.mean(all_fixed_f1) - max(np.mean(all_clip_f1), np.mean(all_dino_f1))
    regret = np.mean(all_oracle_f1) - np.mean(all_fixed_f1)

    print(f"\n  Overall F1-max:")
    print(f"    CLIP:   {np.mean(all_clip_f1):.2f}%")
    print(f"    DINOv2: {np.mean(all_dino_f1):.2f}%")
    print(f"    Equal:  {np.mean(all_equal_f1):.2f}%  (gain: {fusion_gain_eq:+.2f}pp)")
    print(f"    Fixed:  {np.mean(all_fixed_f1):.2f}%  (gain: {fusion_gain_fix:+.2f}pp)")
    print(f"    Oracle: {np.mean(all_oracle_f1):.2f}%  (Fixed regret: {regret:.2f}pp)")

    # AUROC vs F1 correlation
    norm_fn = normalize_minmax
    aurocs, f1maxs = [], []
    for ds_name, ds_data in datasets.items():
        for cat, d in ds_data.items():
            labels = d['labels']
            clip_z = norm_fn(d['clip_scores'])
            dino_z = norm_fn(d['dino_scores'])
            fused = fuse_scores(clip_z, dino_z, 0.65)
            aurocs.append(compute_auroc(labels, fused))
            f1maxs.append(compute_f1max(labels, fused))
    r, p = stats.pearsonr(aurocs, f1maxs)
    print(f"\n  AUROC vs F1-max correlation (Fixed, per-category): r={r:.3f}, p={p:.6f}")

    return {
        'overall': {
            'clip_f1': float(np.mean(all_clip_f1)),
            'dino_f1': float(np.mean(all_dino_f1)),
            'equal_f1': float(np.mean(all_equal_f1)),
            'fixed_f1': float(np.mean(all_fixed_f1)),
            'oracle_f1': float(np.mean(all_oracle_f1)),
            'fusion_gain_equal': float(fusion_gain_eq),
            'fusion_gain_fixed': float(fusion_gain_fix),
            'regret': float(regret),
        },
        'auroc_f1_correlation': {'r': float(r), 'p': float(p)},
        'per_dataset': all_results,
    }


def part_e_score_distributions(datasets):
    """Part E: Characterize score distributions for normal vs anomaly."""
    print("\n" + "=" * 70)
    print("PART E: Score Distribution Characterization")
    print("=" * 70)

    norm_fn = normalize_minmax
    separation_metrics = []

    for ds_name, ds_data in datasets.items():
        for cat, d in ds_data.items():
            labels = d['labels']
            clip_z = norm_fn(d['clip_scores'])
            dino_z = norm_fn(d['dino_scores'])
            fused = fuse_scores(clip_z, dino_z, 0.65)

            normal_mask = labels == 0
            anomaly_mask = labels == 1

            if sum(anomaly_mask) == 0 or sum(normal_mask) == 0:
                continue

            # Separation metric: Cohen's d between normal and anomaly score distributions
            for model_name, scores in [('clip', clip_z), ('dino', dino_z), ('fused', fused)]:
                normal_scores = scores[normal_mask]
                anomaly_scores = scores[anomaly_mask]
                cohens_d = (np.mean(anomaly_scores) - np.mean(normal_scores)) / \
                    np.sqrt((np.var(normal_scores) + np.var(anomaly_scores)) / 2 + 1e-10)

                separation_metrics.append({
                    'dataset': ds_name,
                    'category': cat,
                    'model': model_name,
                    'cohens_d': cohens_d,
                    'normal_mean': float(np.mean(normal_scores)),
                    'anomaly_mean': float(np.mean(anomaly_scores)),
                    'overlap': float(np.sum(normal_scores >= np.percentile(anomaly_scores, 25)) /
                                    len(normal_scores)),
                })

    # Summarize
    print(f"\n  Score separation (Cohen's d between normal and anomaly):")
    for model in ['clip', 'dino', 'fused']:
        ds = [s['cohens_d'] for s in separation_metrics if s['model'] == model]
        print(f"    {model:>6s}: mean d={np.mean(ds):.2f}, median={np.median(ds):.2f}, "
              f"min={np.min(ds):.2f}, max={np.max(ds):.2f}")

    # Is fusion always better separation?
    fusion_better_count = 0
    total_count = 0
    for cat_group in set(f"{s['dataset']}/{s['category']}" for s in separation_metrics):
        ds, cat = cat_group.split("/", 1)
        fused_d = [s['cohens_d'] for s in separation_metrics
                   if s['dataset'] == ds and s['category'] == cat and s['model'] == 'fused']
        best_single_d = max(
            [s['cohens_d'] for s in separation_metrics
             if s['dataset'] == ds and s['category'] == cat and s['model'] == 'clip'][0],
            [s['cohens_d'] for s in separation_metrics
             if s['dataset'] == ds and s['category'] == cat and s['model'] == 'dino'][0],
        )
        if fused_d and fused_d[0] > best_single_d:
            fusion_better_count += 1
        total_count += 1

    print(f"\n  Fusion has better score separation than best single: "
          f"{fusion_better_count}/{total_count} ({fusion_better_count/total_count*100:.0f}%)")

    # Summarize distribution characteristics
    clip_ds = [s['cohens_d'] for s in separation_metrics if s['model'] == 'clip']
    dino_ds = [s['cohens_d'] for s in separation_metrics if s['model'] == 'dino']
    fused_ds = [s['cohens_d'] for s in separation_metrics if s['model'] == 'fused']

    return {
        'mean_separation': {
            'clip': float(np.mean(clip_ds)),
            'dino': float(np.mean(dino_ds)),
            'fused': float(np.mean(fused_ds)),
        },
        'fusion_better_separation_rate': float(fusion_better_count / total_count),
    }


def main():
    print("Experiment 27: Normalization Robustness & Bootstrap Confidence Analysis")
    print("=" * 70)

    datasets = load_all_datasets()

    # Summary of loaded data
    total_cats = 0
    for ds_name, ds_data in datasets.items():
        n_cats = len(ds_data)
        total_cats += n_cats
        print(f"  {ds_name}: {n_cats} categories loaded")
    print(f"  Total: {total_cats} categories")

    results = {}
    results["part_a"] = part_a_norm_robustness(datasets)
    results["part_b"] = part_b_optimal_alpha_stability(datasets)
    results["part_c"] = part_c_bootstrap_ci(datasets)
    results["part_d"] = part_d_f1max_analysis(datasets)
    results["part_e"] = part_e_score_distributions(datasets)

    # Custom encoder for numpy types
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
        json.dump({"experiment_27_normalization_robustness": results}, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
