"""Leave-one-dataset-out (LODO) fusion-weight transfer."""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fusion import (
    load_anomalyclip_scores,
    load_anomalydino_scores,
    align_samples,
    normalize_scores,
    compute_auroc,
)
from fine_grained_alpha_sweep import load_btad_dino_scores

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHAS = [round(i / 100.0, 2) for i in range(0, 101)]  # 0.00 .. 1.00 step 0.01


def load_all_datasets():
    """Return {ds_name: aligned_data} for MVTec, VisA, BTAD."""
    datasets = {}

    # MVTec (weak + strong category splits)
    clip_mvtec = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv")
    dino_mvtec = [
        os.path.join(BASE_DIR, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv"),
        os.path.join(BASE_DIR, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv"),
    ]
    if os.path.exists(clip_mvtec):
        clip_data = load_anomalyclip_scores(clip_mvtec)
        dino_data = {}
        for dp in dino_mvtec:
            if os.path.exists(dp):
                dino_data.update(load_anomalydino_scores(dp))
        if dino_data:
            datasets['MVTec'] = align_samples(clip_data, dino_data)

    # VisA
    clip_visa = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv")
    dino_visa = os.path.join(BASE_DIR, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                             "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv")
    if os.path.exists(clip_visa) and os.path.exists(dino_visa):
        datasets['VisA'] = align_samples(load_anomalyclip_scores(clip_visa),
                                         load_anomalydino_scores(dino_visa))

    # BTAD
    clip_btad = os.path.join(BASE_DIR, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv")
    dino_btad = os.path.join(BASE_DIR, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                             "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv")
    if os.path.exists(clip_btad) and os.path.exists(dino_btad):
        datasets['BTAD'] = align_samples(load_anomalyclip_scores(clip_btad),
                                         load_btad_dino_scores(dino_btad))
    else:
        raise RuntimeError(f"BTAD not found -- would run on 27 categories.\n  {dino_btad}")
    return datasets


def build_curves(datasets):
    """Return curves[cat] = {alpha: auroc}, and cat2ds[cat] = dataset name."""
    curves, cat2ds = {}, {}
    for ds, aligned in datasets.items():
        for cat in sorted(aligned.keys()):
            d = aligned[cat]
            labels = d['labels']
            if len(np.unique(labels)) < 2:
                continue
            clip_n = normalize_scores(d['clip_scores'], 'minmax')
            dino_n = normalize_scores(d['dino_scores'], 'minmax')
            curves[cat] = {a: compute_auroc(labels, (1 - a) * clip_n + a * dino_n) for a in ALPHAS}
            cat2ds[cat] = ds
    return curves, cat2ds


def global_alpha(curves, cats):
    """Alpha maximizing mean AUROC over the given category set (paper's selection rule)."""
    best_a, best_m = None, -1.0
    for a in ALPHAS:
        m = np.mean([curves[c][a] for c in cats])
        if m > best_m:
            best_m, best_a = m, a
    return best_a, best_m


def oracle(curves, cat):
    a = max(ALPHAS, key=lambda x: curves[cat][x])
    return a, curves[cat][a]


def main():
    print("=" * 78)
    print("  LEAVE-ONE-DATASET-OUT ALPHA TRANSFER")
    print("=" * 78)

    datasets = load_all_datasets()
    curves, cat2ds = build_curves(datasets)
    all_cats = list(curves.keys())
    ds_names = list(datasets.keys())
    by_ds = {ds: [c for c in all_cats if cat2ds[c] == ds] for ds in ds_names}
    n = len(all_cats)
    print(f"\n  Loaded {n} categories: " + ", ".join(f"{ds}={len(by_ds[ds])}" for ds in ds_names))

    # Reference: global alpha on all 30, and per-category oracle
    g30_a, _ = global_alpha(curves, all_cats)
    oracle_auroc = {c: oracle(curves, c)[1] for c in all_cats}
    fixed_regret = {c: oracle_auroc[c] - curves[c][0.65] for c in all_cats}
    g30_regret = {c: oracle_auroc[c] - curves[c][g30_a] for c in all_cats}

    # ---- LODO folds ----
    folds = {}
    transfer_alpha_per_cat = {}
    transfer_regret = {}
    for held in ds_names:
        train_cats = [c for c in all_cats if cat2ds[c] != held]
        a_T, _ = global_alpha(curves, train_cats)
        held_cats = by_ds[held]
        treg = {c: oracle_auroc[c] - curves[c][a_T] for c in held_cats}
        tauroc = np.mean([curves[c][a_T] for c in held_cats])
        oauroc = np.mean([oracle_auroc[c] for c in held_cats])
        freg = np.mean([fixed_regret[c] for c in held_cats])
        folds[held] = {
            "transfer_alpha": a_T,
            "n_held": len(held_cats),
            "held_transfer_auroc": round(float(tauroc), 3),
            "held_oracle_auroc": round(float(oauroc), 3),
            "held_transfer_regret_pp": round(float(np.mean(list(treg.values()))), 3),
            "held_fixed065_regret_pp": round(float(freg), 3),
        }
        for c in held_cats:
            transfer_alpha_per_cat[c] = a_T
            transfer_regret[c] = treg[c]

    # ---- pooled comparison over all 30 categories ----
    tr = np.array([transfer_regret[c] for c in all_cats])
    fr = np.array([fixed_regret[c] for c in all_cats])
    gr = np.array([g30_regret[c] for c in all_cats])
    # paired Wilcoxon: transfer vs in-sample fixed-0.65
    try:
        w_stat, w_p = stats.wilcoxon(tr, fr)
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')

    print(f"\n  Global alpha on all 30 categories: {g30_a}")
    print(f"\n  {'Held-out':<8} {'alpha_T':>8} {'n':>3} {'transferAUROC':>14} {'oracleAUROC':>12} "
          f"{'transferRegret':>15} {'fixed0.65Regret':>16}")
    for held in ds_names:
        f = folds[held]
        print(f"  {held:<8} {f['transfer_alpha']:>8} {f['n_held']:>3} {f['held_transfer_auroc']:>14} "
              f"{f['held_oracle_auroc']:>12} {f['held_transfer_regret_pp']:>15} {f['held_fixed065_regret_pp']:>16}")

    print(f"\n  POOLED over all 30 categories (each held out exactly once):")
    print(f"    LODO transfer regret (mean):     {tr.mean():.3f} pp   (max {tr.max():.3f})")
    print(f"    In-sample fixed-0.65 regret:     {fr.mean():.3f} pp   (max {fr.max():.3f})")
    print(f"    In-sample global-{g30_a} regret:    {gr.mean():.3f} pp")
    print(f"    transfer - fixed (mean diff):    {(tr - fr).mean():+.3f} pp")
    print(f"    Wilcoxon transfer vs fixed-0.65: W={w_stat:.1f}, p={w_p:.3f} "
          f"({'indistinguishable' if w_p > 0.05 else 'different'})")
    print(f"    transferred alphas per fold:     " +
          ", ".join(f"{held}->{folds[held]['transfer_alpha']}" for held in ds_names))

    out = {
        "experiment": "leave-one-dataset-out alpha transfer",
        "n_categories": n,
        "alpha_grid_step": 0.01,
        "normalization": "per-category min-max",
        "global_alpha_all30": g30_a,
        "folds": folds,
        "pooled": {
            "lodo_transfer_regret_mean_pp": round(float(tr.mean()), 3),
            "lodo_transfer_regret_max_pp": round(float(tr.max()), 3),
            "insample_fixed065_regret_mean_pp": round(float(fr.mean()), 3),
            "insample_fixed065_regret_max_pp": round(float(fr.max()), 3),
            "insample_global_alpha_regret_mean_pp": round(float(gr.mean()), 3),
            "transfer_minus_fixed_mean_pp": round(float((tr - fr).mean()), 3),
            "wilcoxon_transfer_vs_fixed_W": None if np.isnan(w_stat) else round(float(w_stat), 2),
            "wilcoxon_transfer_vs_fixed_p": None if np.isnan(w_p) else round(float(w_p), 4),
        },
        "transferred_alpha_per_fold": {held: folds[held]["transfer_alpha"] for held in ds_names},
    }
    out_path = os.path.join(os.path.dirname(BASE_DIR), "results", "leave_one_dataset_out_alpha.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
