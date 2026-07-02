"""
Alpha Discovery Cycle 2: Cross-Model Pseudo-Labeling

Cycle 1 failure analysis:
  - All methods scored BELOW fixed alpha=0.65
  - Pseudo-AUROC methods collapsed to alpha=0.5 (degenerate)
  - Root cause: evaluating each model independently loses relative information

New hypotheses:
  A: Cross-model pseudo-labeling: Use Model A's top-k predictions as
     pseudo-anomalies, evaluate Model B on those pseudo-labels, and vice versa.
     This measures RELATIVE discriminability.

  B: Raw score analysis: Don't normalize before pseudo-AUROC computation.
     Normalization destroys absolute discriminability information.

  C: Expected anomaly rate prior: In manufacturing AD, anomaly rate ~5-30%.
     Use this prior to calibrate pseudo-labels.

  D: Score rank cross-evaluation: Use one model's ranking to evaluate the
     other model's AUROC-like metric.

  E: Concordance-based: Use samples where BOTH models agree (high/low)
     as reliable pseudo-labels, then measure per-model performance on
     disagreement samples only.
"""

import os
import sys
import numpy as np
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fusion import (
    load_anomalyclip_scores,
    load_anomalydino_scores,
    align_samples,
    normalize_scores,
    compute_auroc,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_btad_dino_scores(csv_path):
    data = defaultdict(lambda: {'scores': [], 'labels': [], 'paths': []})
    import csv as csv_mod
    with open(csv_path, 'r') as f:
        reader = csv_mod.reader(f)
        next(reader)
        for row in reader:
            obj, sample, score = row[0], row[1], float(row[2])
            is_anomaly = 0 if sample.startswith("ok/") else 1
            data[obj]['scores'].append(score)
            data[obj]['labels'].append(is_anomaly)
            data[obj]['paths'].append(sample)
    return dict(data)


# ============================================================================
# Hypothesis A: Cross-model pseudo-labeling
# ============================================================================

def cross_model_pseudo_auroc(clip_scores, dino_scores, top_pct=0.2):
    """
    Use CLIP's top-k as pseudo-anomalies, compute DINO's AUROC on those.
    Use DINO's top-k as pseudo-anomalies, compute CLIP's AUROC on those.

    If DINO achieves higher pseudo-AUROC when evaluated against CLIP's labels,
    it means DINO also detects what CLIP detects = DINO is good.
    If CLIP achieves lower pseudo-AUROC against DINO's labels,
    it means CLIP misses what DINO catches = DINO is better.
    """
    n = len(clip_scores)
    k = max(1, int(n * top_pct))

    # CLIP's top-k as pseudo-anomalies
    clip_order = np.argsort(clip_scores)[::-1]
    pseudo_labels_clip = np.zeros(n, dtype=int)
    pseudo_labels_clip[clip_order[:k]] = 1

    # DINO's top-k as pseudo-anomalies
    dino_order = np.argsort(dino_scores)[::-1]
    pseudo_labels_dino = np.zeros(n, dtype=int)
    pseudo_labels_dino[dino_order[:k]] = 1

    from sklearn.metrics import roc_auc_score

    # DINO evaluated on CLIP's pseudo-labels
    # (how well does DINO detect what CLIP says is anomalous?)
    dino_on_clip_labels = roc_auc_score(pseudo_labels_clip, dino_scores)

    # CLIP evaluated on DINO's pseudo-labels
    # (how well does CLIP detect what DINO says is anomalous?)
    clip_on_dino_labels = roc_auc_score(pseudo_labels_dino, clip_scores)

    return clip_on_dino_labels, dino_on_clip_labels


def method_cross_pseudo_auroc(d, top_pct=0.2):
    """
    Cross-model pseudo-AUROC → inverse error weighting.

    alpha = f(how well each model catches what the OTHER model flags)

    Intuition: If DINO catches CLIP's anomalies (high DINO-on-CLIP AUROC)
    AND CLIP misses DINO's anomalies (low CLIP-on-DINO AUROC),
    then DINO is strictly better → alpha should be high.
    """
    clip_on_dino, dino_on_clip = cross_model_pseudo_auroc(
        d['clip_scores'], d['dino_scores'], top_pct
    )

    # Self-evaluation: each model's AUROC on its own pseudo-labels (always ~1.0)
    # Cross-evaluation tells us about relative quality
    err_c = max(1 - clip_on_dino, 1e-6)  # CLIP fails to catch DINO's targets
    err_d = max(1 - dino_on_clip, 1e-6)  # DINO fails to catch CLIP's targets

    # Lower cross-error = model also catches the other's anomalies = better model
    # So weight INVERSELY to cross-error
    alpha = (1 / err_d) / (1 / err_c + 1 / err_d)
    return np.clip(alpha, 0.05, 0.95), clip_on_dino, dino_on_clip


def method_cross_pseudo_reversed(d, top_pct=0.2):
    """
    Reversed logic: if DINO catches CLIP's anomalies well (high dino_on_clip),
    DINO is reliable. Weight DINO more.

    alpha = dino_on_clip / (clip_on_dino + dino_on_clip)
    """
    clip_on_dino, dino_on_clip = cross_model_pseudo_auroc(
        d['clip_scores'], d['dino_scores'], top_pct
    )
    total = clip_on_dino + dino_on_clip
    if total < 1e-10:
        return 0.5
    alpha = dino_on_clip / total
    return np.clip(alpha, 0.05, 0.95)


# ============================================================================
# Hypothesis B: Raw score pseudo-AUROC
# ============================================================================

def method_raw_gmm_pseudo_auroc(d):
    """
    Apply GMM on RAW (unnormalized) scores.
    The absolute scale differences between models may carry information.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import roc_auc_score

    results = {}
    for name, scores in [('clip', d['clip_scores']), ('dino', d['dino_scores'])]:
        X = scores.reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
            gmm.fit(X)
            means = gmm.means_.flatten()
            anomaly_comp = np.argmax(means)
            probs = gmm.predict_proba(X)[:, anomaly_comp]
            pseudo_labels = (probs > 0.5).astype(int)
            if len(np.unique(pseudo_labels)) < 2:
                results[name] = 0.5
            else:
                results[name] = roc_auc_score(pseudo_labels, scores)
        except Exception:
            results[name] = 0.5

    err_c = max(1 - results['clip'], 1e-6)
    err_d = max(1 - results['dino'], 1e-6)
    alpha = (1 / err_d) / (1 / err_c + 1 / err_d)
    return np.clip(alpha, 0.05, 0.95)


# ============================================================================
# Hypothesis C: Expected anomaly rate prior
# ============================================================================

def method_anomaly_rate_prior(d, expected_rate=0.15):
    """
    Assume approximately expected_rate fraction of test images are anomalous.
    Use this to set threshold and compute pseudo-AUROC.
    """
    from sklearn.metrics import roc_auc_score
    n = len(d['clip_scores'])
    k = max(1, int(n * expected_rate))

    results = {}
    for name, scores in [('clip', d['clip_scores']), ('dino', d['dino_scores'])]:
        order = np.argsort(scores)[::-1]
        pseudo_labels = np.zeros(n, dtype=int)
        pseudo_labels[order[:k]] = 1

        # Pseudo-AUROC: but this is trivially high since we defined labels from scores
        # Instead, measure how "separated" the top-k are from the rest
        top_scores = scores[order[:k]]
        rest_scores = scores[order[k:]]

        # Cohen's d as proxy for discriminability
        if np.std(top_scores) + np.std(rest_scores) < 1e-10:
            results[name] = 0
        else:
            pooled = np.sqrt((np.var(top_scores) + np.var(rest_scores)) / 2)
            results[name] = abs(np.mean(top_scores) - np.mean(rest_scores)) / (pooled + 1e-10)

    total = results['clip'] + results['dino']
    if total < 1e-10:
        return 0.5
    alpha = results['dino'] / total
    return np.clip(alpha, 0.05, 0.95)


# ============================================================================
# Hypothesis D: Score rank cross-evaluation
# ============================================================================

def method_rank_cross_auroc(d, top_pct=0.2):
    """
    Use rank-transformed scores for cross-evaluation.
    Rank is more robust to scale differences.
    """
    from scipy.stats import rankdata
    clip_rank = rankdata(d['clip_scores']) / len(d['clip_scores'])
    dino_rank = rankdata(d['dino_scores']) / len(d['dino_scores'])

    n = len(clip_rank)
    k = max(1, int(n * top_pct))

    clip_order = np.argsort(clip_rank)[::-1]
    dino_order = np.argsort(dino_rank)[::-1]

    pseudo_clip = np.zeros(n, dtype=int)
    pseudo_clip[clip_order[:k]] = 1
    pseudo_dino = np.zeros(n, dtype=int)
    pseudo_dino[dino_order[:k]] = 1

    from sklearn.metrics import roc_auc_score
    dino_on_clip = roc_auc_score(pseudo_clip, dino_rank)
    clip_on_dino = roc_auc_score(pseudo_dino, clip_rank)

    err_c = max(1 - clip_on_dino, 1e-6)
    err_d = max(1 - dino_on_clip, 1e-6)
    alpha = (1 / err_d) / (1 / err_c + 1 / err_d)
    return np.clip(alpha, 0.05, 0.95)


# ============================================================================
# Hypothesis E: Concordance-based
# ============================================================================

def method_concordance(d, agree_pct=0.3):
    """
    Find samples where both models agree (both in top-k OR both in bottom-k).
    These are reliable pseudo-labels.
    Then evaluate each model's score separation on the DISAGREEMENT set.
    The model with better separation on disagreements is more informative.
    """
    clip_norm = normalize_scores(d['clip_scores'], 'minmax')
    dino_norm = normalize_scores(d['dino_scores'], 'minmax')
    n = len(clip_norm)
    k = max(1, int(n * agree_pct))

    clip_top = set(np.argsort(clip_norm)[-k:])
    clip_bot = set(np.argsort(clip_norm)[:k])
    dino_top = set(np.argsort(dino_norm)[-k:])
    dino_bot = set(np.argsort(dino_norm)[:k])

    # Agree-high: both models say anomalous
    agree_high = clip_top & dino_top
    # Agree-low: both models say normal
    agree_low = clip_bot & dino_bot
    # Disagreement set: everything else
    agree_set = agree_high | agree_low
    disagree_idx = [i for i in range(n) if i not in agree_set]

    if len(agree_high) < 2 or len(agree_low) < 2 or len(disagree_idx) < 2:
        return 0.5

    # Use agree_high/agree_low as reliable pseudo-labels for the agree set
    # Then check: on the DISAGREE set, which model's scores are more extreme?
    clip_disagree = clip_norm[disagree_idx]
    dino_disagree = dino_norm[disagree_idx]

    # Model with higher variance on disagreement set is more opinionated
    clip_var = np.std(clip_disagree)
    dino_var = np.std(dino_disagree)

    total = clip_var + dino_var
    if total < 1e-10:
        return 0.5
    alpha = dino_var / total
    return np.clip(alpha, 0.05, 0.95)


# ============================================================================
# Hypothesis F: Mutual cross-AUROC (new idea from analysis)
# ============================================================================

def method_mutual_auroc(d, top_pcts=[0.1, 0.15, 0.2, 0.25, 0.3]):
    """
    Average cross-AUROC across multiple top-k thresholds for robustness.
    """
    clip_total, dino_total = 0, 0
    for pct in top_pcts:
        clip_on_dino, dino_on_clip = cross_model_pseudo_auroc(
            d['clip_scores'], d['dino_scores'], pct
        )
        clip_total += clip_on_dino
        dino_total += dino_on_clip

    clip_avg = clip_total / len(top_pcts)
    dino_avg = dino_total / len(top_pcts)

    err_c = max(1 - clip_avg, 1e-6)
    err_d = max(1 - dino_avg, 1e-6)
    alpha = (1 / err_d) / (1 / err_c + 1 / err_d)
    return np.clip(alpha, 0.05, 0.95)


def method_mutual_auroc_direct(d, top_pcts=[0.1, 0.15, 0.2, 0.25, 0.3]):
    """
    Direct version: alpha = avg(dino_on_clip) / (avg(clip_on_dino) + avg(dino_on_clip))
    """
    clip_total, dino_total = 0, 0
    for pct in top_pcts:
        clip_on_dino, dino_on_clip = cross_model_pseudo_auroc(
            d['clip_scores'], d['dino_scores'], pct
        )
        clip_total += clip_on_dino
        dino_total += dino_on_clip

    clip_avg = clip_total / len(top_pcts)
    dino_avg = dino_total / len(top_pcts)

    total = clip_avg + dino_avg
    if total < 1e-10:
        return 0.5
    alpha = dino_avg / total
    return np.clip(alpha, 0.05, 0.95)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(datasets):
    alpha_steps = [i / 100.0 for i in range(0, 101, 1)]

    all_categories = []
    for ds_name, aligned in datasets.items():
        for obj in sorted(aligned.keys()):
            d = aligned[obj]
            if len(np.unique(d['labels'])) < 2:
                continue
            all_categories.append((ds_name, obj, d))

    # Define methods
    methods = {}

    # Baselines
    methods['Fixed-0.65'] = lambda d: 0.65
    methods['Fixed-0.50'] = lambda d: 0.50

    # Hypothesis A: Cross-model pseudo-labeling
    for pct in [0.10, 0.15, 0.20, 0.25, 0.30]:
        pct_str = f"{int(pct*100)}"
        methods[f'CrossPseudo-{pct_str}%'] = lambda d, p=pct: method_cross_pseudo_auroc(d, p)[0]
        methods[f'CrossReversed-{pct_str}%'] = lambda d, p=pct: method_cross_pseudo_reversed(d, p)

    # Hypothesis B: Raw scores
    methods['RawGMM'] = method_raw_gmm_pseudo_auroc

    # Hypothesis C: Anomaly rate prior
    for rate in [0.10, 0.15, 0.20, 0.25, 0.30]:
        r_str = f"{int(rate*100)}"
        methods[f'AnomalyRate-{r_str}%'] = lambda d, r=rate: method_anomaly_rate_prior(d, r)

    # Hypothesis D: Rank cross-evaluation
    for pct in [0.10, 0.15, 0.20, 0.25]:
        pct_str = f"{int(pct*100)}"
        methods[f'RankCross-{pct_str}%'] = lambda d, p=pct: method_rank_cross_auroc(d, p)

    # Hypothesis E: Concordance
    for pct in [0.20, 0.30, 0.40]:
        pct_str = f"{int(pct*100)}"
        methods[f'Concordance-{pct_str}%'] = lambda d, p=pct: method_concordance(d, p)

    # Hypothesis F: Mutual cross-AUROC
    methods['MutualAUROC-inv'] = method_mutual_auroc
    methods['MutualAUROC-direct'] = method_mutual_auroc_direct

    print(f"\n{'=' * 110}")
    print(f"  CYCLE 2: {len(methods)} methods x {len(all_categories)} categories")
    print(f"{'=' * 110}")

    method_results = {}

    for method_name, method_fn in methods.items():
        aurocs = []
        alphas_pred = []
        alphas_oracle = []
        per_cat = []

        for ds_name, obj, d in all_categories:
            labels = d['labels']
            clip_norm = normalize_scores(d['clip_scores'], 'minmax')
            dino_norm = normalize_scores(d['dino_scores'], 'minmax')

            try:
                result = method_fn(d)
                a_pred = float(np.clip(result, 0.0, 1.0))
            except Exception:
                a_pred = 0.65

            fused = (1 - a_pred) * clip_norm + a_pred * dino_norm
            auroc = compute_auroc(labels, fused)

            # Oracle
            best_auroc, best_alpha = 0, 0.5
            for a in alpha_steps:
                f = (1 - a) * clip_norm + a * dino_norm
                au = compute_auroc(labels, f)
                if au > best_auroc:
                    best_auroc = au
                    best_alpha = a

            aurocs.append(auroc)
            alphas_pred.append(a_pred)
            alphas_oracle.append(best_alpha)
            per_cat.append({
                'ds': ds_name, 'cat': obj,
                'a_pred': round(a_pred, 3), 'a_oracle': best_alpha,
                'auroc': round(auroc, 2), 'auroc_oracle': round(best_auroc, 2)
            })

        mean_auroc = np.mean(aurocs)
        pred_arr, orc_arr = np.array(alphas_pred), np.array(alphas_oracle)

        if len(set(alphas_pred)) > 1:
            mx, my = np.mean(pred_arr), np.mean(orc_arr)
            cov = np.sum((pred_arr - mx) * (orc_arr - my))
            sx = np.sqrt(np.sum((pred_arr - mx) ** 2))
            sy = np.sqrt(np.sum((orc_arr - my) ** 2))
            pearson_r = cov / (sx * sy) if sx > 1e-10 and sy > 1e-10 else 0
        else:
            pearson_r = 0

        mae = np.mean(np.abs(pred_arr - orc_arr))
        regret = np.mean(np.array([p['auroc_oracle'] for p in per_cat]) -
                         np.array([p['auroc'] for p in per_cat]))

        method_results[method_name] = {
            'mean_auroc': round(mean_auroc, 3),
            'pearson_r': round(pearson_r, 3),
            'mae_alpha': round(mae, 3),
            'regret': round(regret, 3),
            'per_category': per_cat,
        }

    # Print summary
    fixed_auroc = method_results['Fixed-0.65']['mean_auroc']
    sorted_methods = sorted(method_results.items(), key=lambda x: -x[1]['mean_auroc'])

    print(f"\n  {'Method':<28} {'AUROC':>8} {'r':>8} {'MAE(a)':>8} {'Regret':>8} {'vs0.65':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, r in sorted_methods:
        diff = r['mean_auroc'] - fixed_auroc
        marker = " ***" if diff > 0.05 else (" **" if diff > 0.01 else (" *" if diff > 0 else ""))
        print(f"  {name:<28} {r['mean_auroc']:>8.3f} {r['pearson_r']:>8.3f} "
              f"{r['mae_alpha']:>8.3f} {r['regret']:>8.3f} {diff:>+8.3f}{marker}")

    # Show top 3 non-fixed methods per-category
    print(f"\n{'=' * 110}")
    print(f"  TOP 3 NON-FIXED METHODS: Per-Category Detail")
    print(f"{'=' * 110}")

    non_fixed = [(n, r) for n, r in sorted_methods if not n.startswith('Fixed')][:3]
    for method_name, r in non_fixed:
        print(f"\n  --- {method_name} (AUROC: {r['mean_auroc']:.3f}, r={r['pearson_r']:.3f}) ---")
        print(f"  {'Category':<14} {'a_pred':>7} {'a_orc':>6} {'|da|':>6} "
              f"{'AUROC':>7} {'Orc':>7} {'diff':>7}")
        for pc in sorted(r['per_category'], key=lambda x: x['cat']):
            da = abs(pc['a_pred'] - pc['a_oracle'])
            diff = pc['auroc'] - pc['auroc_oracle']
            print(f"  {pc['cat']:<14} {pc['a_pred']:>7.3f} {pc['a_oracle']:>6.2f} {da:>6.3f} "
                  f"{pc['auroc']:>7.1f} {pc['auroc_oracle']:>7.1f} {diff:>+7.2f}")

    # Save
    output = {name: {k: v for k, v in r.items() if k != 'per_category'}
              for name, r in sorted_methods}
    out_path = os.path.join(os.path.dirname(BASE_DIR), "results", "alpha_discovery_cycle2.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return method_results


def main():
    print("=" * 110)
    print("  ALPHA DISCOVERY CYCLE 2: Cross-Model Approaches")
    print("=" * 110)

    datasets = {}

    # MVTec
    clip_mvtec = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv")
    dino_mvtec_weak = os.path.join(BASE_DIR, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                    "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv")
    dino_mvtec_strong = os.path.join(BASE_DIR, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                      "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")
    if os.path.exists(clip_mvtec):
        clip_data = load_anomalyclip_scores(clip_mvtec)
        dino_data = {}
        for dp in [dino_mvtec_weak, dino_mvtec_strong]:
            if os.path.exists(dp):
                dino_data.update(load_anomalydino_scores(dp))
        if dino_data:
            print("\n--- Aligning MVTec ---")
            datasets['MVTec'] = align_samples(clip_data, dino_data)

    # VisA
    clip_visa = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv")
    dino_visa = os.path.join(BASE_DIR, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv")
    if os.path.exists(clip_visa) and os.path.exists(dino_visa):
        clip_data = load_anomalyclip_scores(clip_visa)
        dino_data = load_anomalydino_scores(dino_visa)
        print("\n--- Aligning VisA ---")
        datasets['VisA'] = align_samples(clip_data, dino_data)

    # BTAD
    clip_btad = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv")
    dino_btad = os.path.join(BASE_DIR, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv")
    if os.path.exists(clip_btad) and os.path.exists(dino_btad):
        clip_data = load_anomalyclip_scores(clip_btad)
        dino_data = load_btad_dino_scores(dino_btad)
        print("\n--- Aligning BTAD ---")
        datasets['BTAD'] = align_samples(clip_data, dino_data)
    else:
        print(f"[WARN] BTAD scores not found -- run would cover only 27 categories:\n  clip={clip_btad}\n  dino={dino_btad}")

    evaluate(datasets)


if __name__ == "__main__":
    main()
