"""Deployment-legal (calibration-set) score normalization."""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fusion import load_anomalyclip_scores, load_anomalydino_scores, align_samples, compute_auroc
from fine_grained_alpha_sweep import load_btad_dino_scores

BASE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = [round(i / 100.0, 2) for i in range(0, 101)]
N_SPLITS = 20
CALIB_FRAC = 0.5
MIN_CALIB = 5
RNG = np.random.RandomState(42)


def load_30():
    AD = os.path.join(BASE, "AnomalyDINO")
    clip = {ds: load_anomalyclip_scores(os.path.join(BASE, "AnomalyCLIP", "results", f"scores_{k}", "anomalyclip_scores.csv"))
            for ds, k in [("MVTec", "mvtec"), ("VisA", "visa"), ("BTAD", "btad")]}
    dino = {}
    dm = {}
    dm.update(load_anomalydino_scores(os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv")))
    dm.update(load_anomalydino_scores(os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")))
    dino["MVTec"] = dm
    dino["VisA"] = load_anomalydino_scores(os.path.join(AD, "results_VisA", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv"))
    dino["BTAD"] = load_btad_dino_scores(os.path.join(AD, "results_BTAD", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv"))
    table = {}
    for ds in ["MVTec", "VisA", "BTAD"]:
        for cat, d in align_samples(clip[ds], dino[ds]).items():
            if len(np.unique(d["labels"])) >= 2:
                table[cat] = (ds, np.asarray(d["clip_scores"], float), np.asarray(d["dino_scores"], float), np.asarray(d["labels"]))
    return table


# --- param estimators return (loc, scale) so normalize = (x-loc)/scale ---
def p_minmax(s):
    lo, hi = s.min(), s.max(); return lo, (hi - lo if hi - lo > 1e-9 else 1.0)
def p_zscore(s):
    mu, sd = s.mean(), s.std(); return mu, (sd if sd > 1e-9 else 1.0)
def p_robust_minmax(s):
    lo, hi = np.percentile(s, 1), np.percentile(s, 99); return lo, (hi - lo if hi - lo > 1e-9 else 1.0)


PROTOCOLS = {
    "transductive_minmax":  ("eval", p_minmax),       # paper baseline
    "transductive_zscore":  ("eval", p_zscore),
    "calib_zscore":         ("calib", p_zscore),      # deployment-legal
    "calib_robust_minmax":  ("calib", p_robust_minmax),
}


def per_cat_curves(clip, dino, y, calib_n=None):
    """Return {protocol: {alpha: auroc}} on the eval set, given one normal split.
    calib_n: indices of calibration normals (fit calib params on these); eval = rest."""
    normals = np.where(y == 0)[0]
    anoms = np.where(y == 1)[0]
    if calib_n is None:  # use all normals as calibration, eval = full set
        calib = normals; eval_idx = np.arange(len(y))
    else:
        eval_norm = np.array([i for i in normals if i not in set(calib_n)])
        eval_idx = np.concatenate([eval_norm, anoms])
        calib = np.array(calib_n)
    yi = y[eval_idx]
    if len(np.unique(yi)) < 2:
        return None
    out = {}
    for name, (src, est) in PROTOCOLS.items():
        fit_idx = eval_idx if src == "eval" else calib
        cl_loc, cl_sc = est(clip[fit_idx]); dn_loc, dn_sc = est(dino[fit_idx])
        cn = (clip[eval_idx] - cl_loc) / cl_sc
        dn = (dino[eval_idx] - dn_loc) / dn_sc
        out[name] = {a: compute_auroc(yi, (1 - a) * cn + a * dn) for a in ALPHAS}
    return out


def aggregate(table, calib_frac=CALIB_FRAC, n_splits=N_SPLITS):
    """Average per-category alpha curves over random normal splits, per protocol."""
    cat_curves = {p: {} for p in PROTOCOLS}
    for cat, (ds, clip, dino, y) in table.items():
        normals = np.where(y == 0)[0]
        k = max(MIN_CALIB, int(round(len(normals) * calib_frac)))
        k = min(k, len(normals) - 2)  # leave >=2 eval normals
        acc = {p: [] for p in PROTOCOLS}
        for _ in range(n_splits):
            calib_n = RNG.choice(normals, size=k, replace=False)
            res = per_cat_curves(clip, dino, y, calib_n)
            if res is None:
                continue
            for p in PROTOCOLS:
                acc[p].append([res[p][a] for a in ALPHAS])
        for p in PROTOCOLS:
            cat_curves[p][cat] = dict(zip(ALPHAS, np.mean(acc[p], axis=0)))
    return cat_curves


def summarize(cat_curves):
    """Per protocol: mean strategy AUROCs, global alpha, regret, fusion gain."""
    res = {}
    for p, curves in cat_curves.items():
        cats = list(curves)
        glob_a = max(ALPHAS, key=lambda a: np.mean([curves[c][a] for c in cats]))
        clip = np.mean([curves[c][0.0] for c in cats])
        dino = np.mean([curves[c][1.0] for c in cats])
        equal = np.mean([curves[c][0.5] for c in cats])
        fixed = np.mean([curves[c][glob_a] for c in cats])
        f065 = np.mean([curves[c][0.65] for c in cats])
        oracle = np.mean([max(curves[c].values()) for c in cats])
        best_single = np.mean([max(curves[c][0.0], curves[c][1.0]) for c in cats])
        regret = np.mean([max(curves[c].values()) - curves[c][glob_a] for c in cats])
        res[p] = dict(global_alpha=glob_a, clip_only=round(clip, 2), dino_only=round(dino, 2),
                      equal=round(equal, 2), fixed_globalalpha=round(fixed, 2), fixed_065=round(f065, 2),
                      oracle=round(oracle, 2), best_single=round(best_single, 2),
                      fusion_gain_vs_best=round(fixed - best_single, 3), regret_pp=round(regret, 3))
    return res


def main():
    table = load_30()
    print(f"Loaded {len(table)} categories.")
    cc = aggregate(table)
    summ = summarize(cc)

    # E3: calibration-size sweep (deployment-legal calib_zscore)
    sizes = [5, 10, 20, 40]
    size_sweep = []
    for k in sizes:
        cat_curves = {p: {} for p in PROTOCOLS}
        for cat, (ds, clip, dino, y) in table.items():
            normals = np.where(y == 0)[0]
            kk = min(k, len(normals) - 2)
            if kk < 2:
                continue
            acc = []
            for _ in range(N_SPLITS):
                calib_n = RNG.choice(normals, size=kk, replace=False)
                r = per_cat_curves(clip, dino, y, calib_n)
                if r:
                    acc.append([r["calib_zscore"][a] for a in ALPHAS])
            cat_curves["calib_zscore"][cat] = dict(zip(ALPHAS, np.mean(acc, axis=0)))
        s = summarize({"calib_zscore": cat_curves["calib_zscore"]})["calib_zscore"]
        size_sweep.append(dict(calib_size=k, global_alpha=s["global_alpha"],
                               regret_pp=s["regret_pp"], fusion_gain=s["fusion_gain_vs_best"]))

    out = {"experiment": "practical (calibration-set) normalization",
           "n_categories": len(table), "n_splits": N_SPLITS, "calib_frac": CALIB_FRAC,
           "protocols": summ, "calib_size_sweep": size_sweep}
    with open(os.path.join(os.path.dirname(BASE), "results", "practical_normalization.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 100)
    print("  MAIN FIXED-FUSION RESULTS UNDER TRANSDUCTIVE vs DEPLOYMENT-LEGAL NORMALIZATION")
    print("=" * 100)
    print(f"  {'protocol':<22} {'a*':>4} {'CLIP':>6} {'DINO':>6} {'best1':>6} {'Equal':>6} {'Fixed.65':>8} {'Oracle':>7} {'gain':>6} {'regret':>7}")
    for p in PROTOCOLS:
        s = summ[p]
        legal = "  (deployment-legal)" if p.startswith("calib") else ""
        print(f"  {p:<22} {s['global_alpha']:>4} {s['clip_only']:>6} {s['dino_only']:>6} {s['best_single']:>6} "
              f"{s['equal']:>6} {s['fixed_065']:>8} {s['oracle']:>7} {s['fusion_gain_vs_best']:>+6} {s['regret_pp']:>7}{legal}")
    print(f"\n  Calibration-size sweep (calib_zscore): " +
          ", ".join(f"K={r['calib_size']}: a*={r['global_alpha']}, regret={r['regret_pp']}pp" for r in size_sweep))
    print(f"\n  Saved: results/practical_normalization.json")


if __name__ == "__main__":
    main()
