"""
Phase 3: Comprehensive Evaluation Across 3 Datasets
=====================================================
Runs all fusion methods (baselines + AdaFuse) on MVTec, VisA, BTAD
and creates unified evaluation tables for paper.

Evaluation Protocol:
- OGRR: Oracle Gap Reduction Rate
- Cross-dataset Stability
- Per-category Regret
"""

import csv
import json
import os
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from collections import defaultdict


# ============================================================================
# Data Loading (from existing modules)
# ============================================================================

def load_anomalyclip_scores(csv_path):
    """Load AnomalyCLIP scores from CSV."""
    scores = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row['Object']
            score = float(row['AnomalyScore'])
            gt = int(row['GroundTruth'])
            ip = row.get('ImagePath', '').replace('\\', '/')
            key = '/'.join(ip.split('/')[-2:]).lower()
            scores[obj].append({'score': score, 'gt': gt, 'sample': key})
    return dict(scores)


def load_anomalydino_scores(csv_path):
    """Load AnomalyDINO scores from CSV."""
    scores = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row['Object']
            sample = row['Sample']
            score = float(row['Anomaly_Score'])
            anomaly_type = sample.split('/')[0]
            gt = 0 if anomaly_type in ('good', 'ok') else 1
            scores[obj].append({'score': score, 'gt': gt, 'sample': sample})
    return dict(scores)


def align_scores(clip_data, dino_data):
    """Align CLIP and DINO scores by sample key (subdir/filename) within each category."""
    aligned = {}
    common = set(clip_data.keys()) & set(dino_data.keys())
    for cat in sorted(common):
        cmap = {x['sample']: x for x in clip_data[cat]}
        dmap = {'/'.join(x['sample'].replace('\\', '/').split('/')[-2:]).lower(): x
                for x in dino_data[cat]}
        keys = sorted(set(cmap) & set(dmap))
        if not keys:
            raise RuntimeError(f"no matching samples for {cat}")
        aligned[cat] = {
            'clip_scores': np.array([cmap[k]['score'] for k in keys]),
            'dino_scores': np.array([dmap[k]['score'] for k in keys]),
            'labels': np.array([cmap[k]['gt'] for k in keys]),
        }
    return aligned


def normalize_scores(scores, method='minmax'):
    arr = np.array(scores)
    if method == 'minmax':
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    elif method == 'rank':
        from scipy.stats import rankdata
        return rankdata(arr) / len(arr)
    return arr


def compute_auroc(labels, scores):
    if len(np.unique(labels)) < 2:
        return 0.0
    return roc_auc_score(labels, scores) * 100


# ============================================================================
# Signal Functions
# ============================================================================

def sig_variance(scores): return np.std(scores)
def sig_iqr(scores):
    q75, q25 = np.percentile(scores, [75, 25])
    return q75 - q25
def sig_skewness(scores): return abs(scipy_stats.skew(scores))
def sig_topk_gap(scores, k_ratio=0.2):
    s = np.sort(scores)[::-1]
    k = max(1, int(len(scores) * k_ratio))
    return s[:k].mean() - s[k:].mean()
def sig_bimodality(scores):
    if len(scores) < 4: return 0.0
    sk = scipy_stats.skew(scores)
    ku = scipy_stats.kurtosis(scores, fisher=False)
    return (sk ** 2 + 1) / ku if ku > 1e-10 else 0.0
def sig_gmm_sep(scores):
    if len(scores) < 10: return 0.0
    X = scores.reshape(-1, 1)
    try:
        gmm = GaussianMixture(2, random_state=42, max_iter=200).fit(X)
        mu = gmm.means_.flatten()
        s = np.sqrt(gmm.covariances_.flatten())
        return np.sqrt(2) * abs(mu[0]-mu[1]) / np.sqrt(s[0]**2 + s[1]**2) if s[0]**2+s[1]**2 > 1e-10 else 0.0
    except: return 0.0

ALL_SIGNALS = {
    'variance': sig_variance, 'iqr': sig_iqr, 'skewness': sig_skewness,
    'topk_gap': sig_topk_gap, 'bimodality': sig_bimodality, 'gmm_sep': sig_gmm_sep,
}


# ============================================================================
# Fusion Methods
# ============================================================================

def eval_fixed_fusion(aligned, alpha):
    """Fixed alpha fusion."""
    aurocs = {}
    for obj, d in aligned.items():
        if len(np.unique(d['labels'])) < 2: continue
        cn = normalize_scores(d['clip_scores'], 'minmax')
        dn = normalize_scores(d['dino_scores'], 'minmax')
        aurocs[obj] = compute_auroc(d['labels'], (1-alpha)*cn + alpha*dn)
    return aurocs


def eval_oracle(aligned):
    """Per-category oracle."""
    aurocs = {}
    alphas = {}
    for obj, d in aligned.items():
        if len(np.unique(d['labels'])) < 2: continue
        cn = normalize_scores(d['clip_scores'], 'minmax')
        dn = normalize_scores(d['dino_scores'], 'minmax')
        best_auroc, best_alpha = 0, 0.5
        for a_int in range(21):
            a = a_int / 20.0
            auroc = compute_auroc(d['labels'], (1-a)*cn + a*dn)
            if auroc > best_auroc:
                best_auroc = auroc
                best_alpha = a
        aurocs[obj] = best_auroc
        alphas[obj] = best_alpha
    return aurocs, alphas


def eval_paramfree(aligned, base_alpha=0.65):
    """Parameter-free adaptive fusion using all signals."""
    aurocs = {}
    for obj, d in aligned.items():
        if len(np.unique(d['labels'])) < 2: continue
        cn = normalize_scores(d['clip_scores'], 'minmax')
        dn = normalize_scores(d['dino_scores'], 'minmax')
        ratios = []
        for fn in ALL_SIGNALS.values():
            sc, sd = fn(cn), fn(dn)
            ratios.append(sd / (sc + sd) if sc + sd > 1e-10 else 0.5)
        alpha = base_alpha + 0.5 * (np.mean(ratios) - 0.5)
        alpha = np.clip(alpha, 0.1, 0.9)
        aurocs[obj] = compute_auroc(d['labels'], (1-alpha)*cn + alpha*dn)
    return aurocs


def eval_hybrid(aligned, base_alpha=0.65):
    """Hybrid category + image level adaptive fusion."""
    aurocs = {}
    for obj, d in aligned.items():
        if len(np.unique(d['labels'])) < 2: continue
        cn = normalize_scores(d['clip_scores'], 'minmax')
        dn = normalize_scores(d['dino_scores'], 'minmax')
        # Category-level signal
        sig_c, sig_d = np.std(cn), np.std(dn)
        cat_alpha = base_alpha + 0.5 * (sig_d/(sig_c+sig_d+1e-10) - 0.5)
        # Image-level
        clip_conf = np.abs(cn - 0.5)
        dino_conf = np.abs(dn - 0.5)
        img_alpha = dino_conf / (clip_conf + dino_conf + 1e-10)
        # Blend
        alpha_per_image = np.clip(0.7 * cat_alpha + 0.3 * img_alpha, 0.1, 0.9)
        fused = (1 - alpha_per_image) * cn + alpha_per_image * dn
        aurocs[obj] = compute_auroc(d['labels'], fused)
    return aurocs


def eval_ks_test(aligned, base_alpha=0.65):
    """KS-test based distribution selection."""
    aurocs = {}
    for obj, d in aligned.items():
        if len(np.unique(d['labels'])) < 2: continue
        cn = normalize_scores(d['clip_scores'], 'minmax')
        dn = normalize_scores(d['dino_scores'], 'minmax')
        def ks_sep(s):
            med = np.median(s)
            lo, hi = s[s<=med], s[s>med]
            return scipy_stats.ks_2samp(lo, hi)[0] if len(lo)>=2 and len(hi)>=2 else 0.0
        ks_c, ks_d = ks_sep(cn), ks_sep(dn)
        alpha = base_alpha + 0.3 * (ks_d/(ks_c+ks_d+1e-10) - 0.5)
        alpha = np.clip(alpha, 0.1, 0.9)
        aurocs[obj] = compute_auroc(d['labels'], (1-alpha)*cn + alpha*dn)
    return aurocs


def eval_max_fusion(aligned):
    """Max fusion: take maximum of normalized scores."""
    aurocs = {}
    for obj, d in aligned.items():
        if len(np.unique(d['labels'])) < 2: continue
        cn = normalize_scores(d['clip_scores'], 'minmax')
        dn = normalize_scores(d['dino_scores'], 'minmax')
        aurocs[obj] = compute_auroc(d['labels'], np.maximum(cn, dn))
    return aurocs


# ============================================================================
# Evaluation Protocol
# ============================================================================

def compute_ogrr(method_aurocs, oracle_aurocs, clip_aurocs, dino_aurocs):
    """Oracle Gap Reduction Rate."""
    ogrrs = []
    for cat in method_aurocs:
        best_single = max(clip_aurocs.get(cat, 0), dino_aurocs.get(cat, 0))
        oracle_val = oracle_aurocs.get(cat, best_single)
        method_val = method_aurocs[cat]
        gap = oracle_val - best_single
        if gap > 0.01:
            ogrr = (method_val - best_single) / gap
        elif method_val >= oracle_val - 0.01:
            ogrr = 1.0
        else:
            ogrr = 0.0
        ogrrs.append(ogrr)
    return np.mean(ogrrs) if ogrrs else 0.0


def compute_regret(method_aurocs, oracle_aurocs):
    """Mean per-category regret (Oracle - Method)."""
    regrets = [oracle_aurocs[cat] - method_aurocs[cat]
               for cat in method_aurocs if cat in oracle_aurocs]
    return np.mean(regrets) if regrets else 0.0


# ============================================================================
# Main
# ============================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load all datasets
    datasets = {}

    # MVTec
    clip_mvtec = os.path.join(base_dir, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv")
    dino_mvtec_w = os.path.join(base_dir, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                 "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv")
    dino_mvtec_s = os.path.join(base_dir, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                                 "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")
    clip_d = load_anomalyclip_scores(clip_mvtec)
    dino_d = {}
    for p in [dino_mvtec_w, dino_mvtec_s]:
        if os.path.exists(p):
            dino_d.update(load_anomalydino_scores(p))
    datasets['MVTec AD'] = align_scores(clip_d, dino_d)
    print(f"MVTec AD: {len(datasets['MVTec AD'])} categories")

    # VisA
    clip_visa = os.path.join(base_dir, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv")
    dino_visa = os.path.join(base_dir, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv")
    clip_d = load_anomalyclip_scores(clip_visa)
    dino_d = load_anomalydino_scores(dino_visa)
    datasets['VisA'] = align_scores(clip_d, dino_d)
    print(f"VisA: {len(datasets['VisA'])} categories")

    # BTAD
    clip_btad = os.path.join(base_dir, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv")
    dino_btad = os.path.join(base_dir, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                              "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv")
    clip_d = load_anomalyclip_scores(clip_btad)
    dino_d = load_anomalydino_scores(dino_btad)
    datasets['BTAD'] = align_scores(clip_d, dino_d)
    print(f"BTAD: {len(datasets['BTAD'])} categories")

    # ============================================
    # Run all methods on all datasets
    # ============================================
    all_results = {}

    methods = {
        'CLIP only':       lambda a: eval_fixed_fusion(a, 0.0),
        'DINOv2 only':     lambda a: eval_fixed_fusion(a, 1.0),
        'Equal (a=0.5)':   lambda a: eval_fixed_fusion(a, 0.5),
        'Fixed (a=0.65)':  lambda a: eval_fixed_fusion(a, 0.65),
        'DINO-heavy (a=0.7)': lambda a: eval_fixed_fusion(a, 0.7),
        'Max fusion':      eval_max_fusion,
        'AdaFuse-PF':      eval_paramfree,
        'AdaFuse-Hybrid':  eval_hybrid,
        'AdaFuse-KS':      eval_ks_test,
    }

    for ds_name, aligned in datasets.items():
        print(f"\n{'='*80}")
        print(f"  {ds_name}")
        print(f"{'='*80}")

        ds_results = {}
        oracle_aurocs, oracle_alphas = eval_oracle(aligned)
        clip_aurocs = eval_fixed_fusion(aligned, 0.0)
        dino_aurocs = eval_fixed_fusion(aligned, 1.0)

        for method_name, method_fn in methods.items():
            aurocs = method_fn(aligned)
            mean_auroc = np.mean(list(aurocs.values()))
            ogrr = compute_ogrr(aurocs, oracle_aurocs, clip_aurocs, dino_aurocs)
            regret = compute_regret(aurocs, oracle_aurocs)

            ds_results[method_name] = {
                'mean_auroc': round(float(mean_auroc), 2),
                'ogrr': round(float(ogrr), 3),
                'regret': round(float(regret), 2),
                'per_category': {k: round(float(v), 2) for k, v in aurocs.items()},
            }

        # Oracle
        oracle_mean = np.mean(list(oracle_aurocs.values()))
        ds_results['Oracle'] = {
            'mean_auroc': round(float(oracle_mean), 2),
            'ogrr': 1.0,
            'regret': 0.0,
            'per_category': {k: round(float(v), 2) for k, v in oracle_aurocs.items()},
        }

        # Print summary
        print(f"\n  {'Method':<25} {'Mean AUROC':>10} {'OGRR':>8} {'Regret':>8}")
        print(f"  {'-'*55}")
        for name in list(methods.keys()) + ['Oracle']:
            r = ds_results[name]
            marker = '**' if 'AdaFuse' in name else '  '
            print(f"  {marker}{name:<23} {r['mean_auroc']:>9.2f}% {r['ogrr']:>7.3f} {r['regret']:>+7.2f}%")

        all_results[ds_name] = ds_results

    # ============================================
    # Cross-Dataset Summary Table (for paper)
    # ============================================
    print(f"\n\n{'='*80}")
    print(f"  CROSS-DATASET SUMMARY (Image-Level AUROC)")
    print(f"{'='*80}")

    method_order = ['CLIP only', 'DINOv2 only', 'Equal (a=0.5)', 'Fixed (a=0.65)',
                    'DINO-heavy (a=0.7)', 'Max fusion', 'AdaFuse-PF', 'AdaFuse-Hybrid',
                    'AdaFuse-KS', 'Oracle']

    header = f"  {'Method':<25}"
    for ds in datasets:
        header += f" {ds:>12}"
    header += f" {'Avg':>8}"
    print(header)
    print(f"  {'-'*(25 + 12*len(datasets) + 10)}")

    for method in method_order:
        row = f"  {method:<25}"
        vals = []
        for ds in datasets:
            v = all_results[ds].get(method, {}).get('mean_auroc', 0)
            row += f" {v:>11.2f}%"
            vals.append(v)
        avg = np.mean(vals)
        row += f" {avg:>7.2f}%"
        print(row)

    # ============================================
    # OGRR Summary
    # ============================================
    print(f"\n\n{'='*80}")
    print(f"  ORACLE GAP REDUCTION RATE (OGRR)")
    print(f"{'='*80}")

    header = f"  {'Method':<25}"
    for ds in datasets:
        header += f" {ds:>12}"
    header += f" {'Avg':>8}"
    print(header)
    print(f"  {'-'*(25 + 12*len(datasets) + 10)}")

    for method in method_order:
        if method in ['CLIP only', 'DINOv2 only', 'Oracle']:
            continue
        row = f"  {method:<25}"
        vals = []
        for ds in datasets:
            v = all_results[ds].get(method, {}).get('ogrr', 0)
            row += f" {v:>11.3f}"
            vals.append(v)
        avg = np.mean(vals)
        row += f" {avg:>7.3f}"
        print(row)

    # ============================================
    # Cross-Dataset Stability
    # ============================================
    print(f"\n\n{'='*80}")
    print(f"  CROSS-DATASET STABILITY (Std across datasets)")
    print(f"{'='*80}")

    for method in method_order:
        if method in ['CLIP only', 'DINOv2 only', 'Oracle']:
            continue
        vals = [all_results[ds].get(method, {}).get('mean_auroc', 0) for ds in datasets]
        std = np.std(vals)
        mean = np.mean(vals)
        cv = std / mean * 100 if mean > 0 else 0
        print(f"  {method:<25} Mean={mean:.2f}%, Std={std:.2f}%, CV={cv:.1f}%")

    # ============================================
    # Save
    # ============================================
    output_path = os.path.join(base_dir, "comprehensive_eval_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
