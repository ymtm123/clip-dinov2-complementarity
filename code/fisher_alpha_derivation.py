"""
Experiment A: Theoretical Derivation of Optimal Fusion Weight from Fisher Discriminant Ratio

Hypothesis:
    alpha* = d_DINOv2 / (d_CLIP + d_DINOv2)
where d is Cohen's d (signal-to-noise ratio of anomaly detection scores).

Key questions:
1. Does the global mean ratio predict alpha=0.65?
2. Does the per-category ratio correlate with oracle alpha?
3. Can we derive a formula that practitioners can use for new model pairs?
"""

import os
import csv
import numpy as np
from collections import defaultdict
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


# ============================================================
# 1. Load oracle alphas from existing results
# ============================================================

def load_oracle_alphas():
    path = os.path.join(BASE_DIR, "category_taxonomy_results.json")
    with open(path) as f:
        data = json.load(f)
    return data["oracle_alphas"]


# ============================================================
# 2. Load scores (CLIP + DINO for all datasets)
# ============================================================

def load_clip_scores(csv_path):
    """Returns {category: {'scores': [], 'labels': []}}"""
    data = defaultdict(lambda: {"scores": [], "labels": []})
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            cat = row[0].strip()
            data[cat]["scores"].append(float(row[2]))
            data[cat]["labels"].append(int(row[3]))
    return dict(data)


def load_dino_scores(dino_dir):
    """Load from all subdirectories, merging categories."""
    data = defaultdict(lambda: {"scores": [], "labels": []})
    for subdir in sorted(os.listdir(dino_dir)):
        subdir_path = os.path.join(dino_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        csv_path = os.path.join(subdir_path, "measurements_seed=0.csv")
        if not os.path.exists(csv_path):
            continue
        # Skip strong_cats / weak_cats if the main agnostic file has the category
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 3:
                    continue
                cat = row[0].strip()
                sample = row[1].strip()
                score = float(row[2])
                label = 0 if (sample.startswith("good") or sample.startswith("ok/")) else 1
                # Avoid duplicates: only add if this category not already fully loaded
                data[cat]["scores"].append(score)
                data[cat]["labels"].append(label)
    # Deduplicate (categories may appear in multiple subdirs)
    deduped = {}
    for cat in data:
        scores = data[cat]["scores"]
        labels = data[cat]["labels"]
        # Keep unique (score, label) pairs via set
        seen = set()
        unique_s, unique_l = [], []
        for s, l in zip(scores, labels):
            key = (round(s, 6), l)
            if key not in seen:
                seen.add(key)
                unique_s.append(s)
                unique_l.append(l)
        deduped[cat] = {"scores": unique_s, "labels": unique_l}
    return deduped


# ============================================================
# 3. Compute Cohen's d
# ============================================================

def cohens_d(scores, labels):
    """Cohen's d = |mu_anomaly - mu_normal| / pooled_std"""
    s = np.array(scores)
    l = np.array(labels)
    normal = s[l == 0]
    anomaly = s[l == 1]
    if len(normal) < 2 or len(anomaly) < 2:
        return None
    mu_n, mu_a = np.mean(normal), np.mean(anomaly)
    var_n, var_a = np.var(normal, ddof=1), np.var(anomaly, ddof=1)
    pooled = np.sqrt((var_n + var_a) / 2)
    if pooled < 1e-10:
        return None
    return abs(mu_a - mu_n) / pooled


# ============================================================
# 4. Spearman correlation (no scipy needed)
# ============================================================

def rankdata(arr):
    n = len(arr)
    indexed = sorted(range(n), key=lambda i: arr[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[indexed[j + 1]] == arr[indexed[j]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def spearman_r(x, y):
    rx = rankdata(list(x))
    ry = rankdata(list(y))
    n = len(rx)
    mx, my = np.mean(rx), np.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = np.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = np.sqrt(sum((b - my) ** 2 for b in ry))
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return cov / (sx * sy)


def pearson_r(x, y):
    x, y = np.array(x), np.array(y)
    mx, my = np.mean(x), np.mean(y)
    cov = np.sum((x - mx) * (y - my))
    sx = np.sqrt(np.sum((x - mx) ** 2))
    sy = np.sqrt(np.sum((y - my) ** 2))
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return cov / (sx * sy)


# Simple t-test for correlation significance
def corr_pvalue(r, n):
    """Two-tailed p-value for Pearson r with n samples."""
    if abs(r) >= 1.0 or n <= 2:
        return 0.0
    import math
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # Approximate p from t distribution using normal for large n
    # For small n, use a better approximation
    df = n - 2
    # Use the incomplete beta function approximation
    x = df / (df + t * t)
    # Simple approximation for p-value
    if df >= 30:
        import math
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    else:
        # More precise: use Welch-Satterthwaite or t-table approximation
        # For our purposes, use the normal approximation with correction
        a = abs(t) * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
        p = 2 * (1 - 0.5 * (1 + math.erf(a / math.sqrt(2))))
    return p


# ============================================================
# 5. Main analysis
# ============================================================

def main():
    print("=" * 78)
    print("  EXPERIMENT A: Fisher Discriminant Ratio -> Fusion Weight Derivation")
    print("=" * 78)

    oracle_alphas = load_oracle_alphas()

    CLIP_PATHS = {
        "MVTec": os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv"),
        "VisA":  os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_visa",  "anomalyclip_scores.csv"),
        "BTAD":  os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_btad",  "anomalyclip_scores.csv"),
    }
    DINO_DIRS = {
        "MVTec": os.path.join(BASE_DIR, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448"),
        "VisA":  os.path.join(BASE_DIR, "AnomalyDINO", "results_VisA",  "dinov2_vitl14_448"),
        "BTAD":  os.path.join(BASE_DIR, "AnomalyDINO", "results_BTAD",  "dinov2_vitl14_448"),
    }
    ORACLE_KEYS = {"MVTec": "mvtec_image", "VisA": "visa_image", "BTAD": "btad_image"}

    # Collect all results
    all_d_clip, all_d_dino = [], []
    all_pred_alpha, all_oracle_alpha = [], []
    all_rows = []

    for ds in ["MVTec", "VisA", "BTAD"]:
        clip_data = load_clip_scores(CLIP_PATHS[ds])
        dino_data = load_dino_scores(DINO_DIRS[ds])
        oa = oracle_alphas[ORACLE_KEYS[ds]]

        print(f"\n--- {ds} ---")
        print(f"  {'Category':<16} {'d_CLIP':>8} {'d_DINO':>8} {'d_ratio':>8} "
              f"{'a_pred':>8} {'a_oracle':>8} {'|err|':>8}")
        print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        # Normalize category names for matching
        clip_cats = {c.lower(): c for c in clip_data}
        dino_cats = {c.lower(): c for c in dino_data}
        oa_cats = {c.lower(): c for c in oa}

        for cat_lower in sorted(set(clip_cats.keys()) & set(dino_cats.keys()) & set(oa_cats.keys())):
            cc = clip_cats[cat_lower]
            dc = dino_cats[cat_lower]
            oc = oa_cats[cat_lower]

            d_c = cohens_d(clip_data[cc]["scores"], clip_data[cc]["labels"])
            d_d = cohens_d(dino_data[dc]["scores"], dino_data[dc]["labels"])

            if d_c is None or d_d is None:
                continue

            alpha_pred = d_d / (d_c + d_d) if (d_c + d_d) > 0 else 0.5
            alpha_oracle = oa[oc]
            err = abs(alpha_pred - alpha_oracle)
            ratio = d_d / d_c if d_c > 0 else float('inf')

            print(f"  {cc:<16} {d_c:8.3f} {d_d:8.3f} {ratio:8.2f} "
                  f"{alpha_pred:8.3f} {alpha_oracle:8.3f} {err:8.3f}")

            all_d_clip.append(d_c)
            all_d_dino.append(d_d)
            all_pred_alpha.append(alpha_pred)
            all_oracle_alpha.append(alpha_oracle)
            all_rows.append({
                "dataset": ds, "category": cc,
                "d_clip": round(d_c, 4), "d_dino": round(d_d, 4),
                "ratio": round(ratio, 4),
                "alpha_pred": round(alpha_pred, 4),
                "alpha_oracle": alpha_oracle
            })

    # ============================================================
    # Global Summary
    # ============================================================
    print(f"\n{'=' * 78}")
    print("  GLOBAL SUMMARY")
    print(f"{'=' * 78}")

    d_clips = np.array(all_d_clip)
    d_dinos = np.array(all_d_dino)
    pred = np.array(all_pred_alpha)
    oracle = np.array(all_oracle_alpha)
    n = len(pred)

    mean_dc = np.mean(d_clips)
    mean_dd = np.mean(d_dinos)
    global_ratio = mean_dd / mean_dc
    derived_alpha = mean_dd / (mean_dc + mean_dd)

    print(f"\n  N categories:             {n}")
    print(f"  Mean Cohen's d (CLIP):    {mean_dc:.4f}")
    print(f"  Mean Cohen's d (DINOv2):  {mean_dd:.4f}")
    print(f"  Mean ratio (DINO/CLIP):   {global_ratio:.4f}")

    print(f"\n  *** DERIVED alpha = d_DINO / (d_CLIP + d_DINO) = {derived_alpha:.4f} ***")
    print(f"  *** Empirical optimal alpha (from paper)       = 0.6500          ***")
    print(f"  *** Difference                                 = {abs(derived_alpha - 0.65):.4f}          ***")

    r_pearson = pearson_r(pred, oracle)
    r_spearman = spearman_r(pred, oracle)
    mae = np.mean(np.abs(pred - oracle))
    rmse = np.sqrt(np.mean((pred - oracle) ** 2))
    p_val = corr_pvalue(r_pearson, n)

    print(f"\n  Per-category prediction quality:")
    print(f"    Pearson  r  = {r_pearson:.4f}  (p = {p_val:.4f})")
    print(f"    Spearman rho = {r_spearman:.4f}")
    print(f"    MAE          = {mae:.4f}")
    print(f"    RMSE         = {rmse:.4f}")

    # Compare with fixed alpha regret
    print(f"\n  Mean alpha_predicted:   {np.mean(pred):.4f}")
    print(f"  Median alpha_predicted: {np.median(pred):.4f}")
    print(f"  Mean alpha_oracle:      {np.mean(oracle):.4f}")
    print(f"  Median alpha_oracle:    {np.median(oracle):.4f}")

    # Additional: does d_ratio explain oracle alpha better than taxonomy?
    ratios = d_dinos / np.maximum(d_clips, 1e-10)
    r_ratio_oracle = pearson_r(ratios, oracle)
    print(f"\n  Correlation(d_ratio, alpha_oracle):")
    print(f"    Pearson r = {r_ratio_oracle:.4f}")

    # Breakdown by DINOv2 vs CLIP dominance
    dino_better = d_dinos > d_clips
    clip_better = ~dino_better
    print(f"\n  Categories where d_DINO > d_CLIP: {np.sum(dino_better)}/{n}")
    print(f"  Categories where d_CLIP > d_DINO: {np.sum(clip_better)}/{n}")
    print(f"  Mean oracle alpha (DINO-dominant):  {np.mean(oracle[dino_better]):.4f}")
    print(f"  Mean oracle alpha (CLIP-dominant):  {np.mean(oracle[clip_better]):.4f}" if np.sum(clip_better) > 0 else "  (none)")

    # Save
    output = {
        "global": {
            "n": n,
            "mean_d_clip": round(float(mean_dc), 4),
            "mean_d_dino": round(float(mean_dd), 4),
            "ratio_dino_clip": round(float(global_ratio), 4),
            "derived_alpha": round(float(derived_alpha), 4),
            "empirical_alpha": 0.65,
            "pearson_r": round(float(r_pearson), 4),
            "spearman_rho": round(float(r_spearman), 4),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
        },
        "per_category": all_rows
    }
    out_path = os.path.join(PROJECT_DIR, "results", "fisher_alpha_derivation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
