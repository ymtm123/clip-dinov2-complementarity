"""Seed-averaged fusion metrics over the 10 AnomalyDINO reference-image seeds."""
import os, sys, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fusion import (load_anomalyclip_scores, load_anomalydino_scores, align_samples)
from fine_grained_alpha_sweep import load_btad_dino_scores
from leave_one_dataset_out_alpha import build_curves, global_alpha, oracle, ALPHAS, BASE_DIR

AD = os.path.join(BASE_DIR, "AnomalyDINO")
CLIP = os.path.join(BASE_DIR, "AnomalyCLIP", "results")


def load_all_datasets(seed):
    """Same loaders/paths as leave_one_dataset_out_alpha.load_all_datasets, parameterized by seed."""
    d = {}
    dino_mvtec = [
        os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_weak_cats", f"measurements_seed={seed}.csv"),
        os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_strong_cats", f"measurements_seed={seed}.csv"),
    ]
    clip_mvtec = load_anomalyclip_scores(os.path.join(CLIP, "scores_mvtec", "anomalyclip_scores.csv"))
    dino_mv = {}
    for dp in dino_mvtec:
        dino_mv.update(load_anomalydino_scores(dp))
    d['MVTec'] = align_samples(clip_mvtec, dino_mv)
    d['VisA'] = align_samples(
        load_anomalyclip_scores(os.path.join(CLIP, "scores_visa", "anomalyclip_scores.csv")),
        load_anomalydino_scores(os.path.join(AD, "results_VisA", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_all", f"measurements_seed={seed}.csv")))
    d['BTAD'] = align_samples(
        load_anomalyclip_scores(os.path.join(CLIP, "scores_btad", "anomalyclip_scores.csv")),
        load_btad_dino_scores(os.path.join(AD, "results_BTAD", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_btad_all", f"measurements_seed={seed}.csv")))
    return d


def per_seed_metrics(seed):
    ds = load_all_datasets(seed)
    curves, cat2ds = build_curves(ds)
    cats = list(curves.keys())
    names = list(ds.keys())
    by = {n: [c for c in cats if cat2ds[c] == n] for n in names}
    orc = {c: oracle(curves, c)[1] for c in cats}
    fixed_reg = {c: orc[c] - curves[c][0.65] for c in cats}
    pooled = {a: float(np.mean([orc[c] - curves[c][a] for c in cats])) for a in ALPHAS}
    pmin_a = min(ALPHAS, key=lambda a: pooled[a]); pmin = pooled[pmin_a]
    band45 = [a for a in ALPHAS if pooled[a] <= pmin + 0.45]
    band20 = [a for a in ALPHAS if pooled[a] <= pmin + 0.20]
    optd = {}
    for n in names:
        pc = {a: float(np.mean([orc[c] - curves[c][a] for c in by[n]])) for a in ALPHAS}
        optd[n] = min(ALPHAS, key=lambda a: pc[a])
    treg = {}; falpha = {}
    for held in names:
        aT, _ = global_alpha(curves, [c for c in cats if cat2ds[c] != held]); falpha[held] = aT
        for c in by[held]:
            treg[c] = orc[c] - curves[c][aT]
    tr = np.array([treg[c] for c in cats]); fr = np.array([fixed_reg[c] for c in cats])
    g_a, _ = global_alpha(curves, cats)
    return dict(n=len(cats), pooled_opt_alpha=pmin_a, pooled_min_regret=pmin,
               regret_050=pooled[0.50], regret_065=pooled[0.65], regret_070=pooled[0.70],
               band45_lo=min(band45), band45_hi=max(band45), band20_lo=min(band20), band20_hi=max(band20),
               opt_MVTec=optd.get('MVTec'), opt_VisA=optd.get('VisA'), opt_BTAD=optd.get('BTAD'),
               lodo_transfer_regret=float(tr.mean()), lodo_transfer_regret_max=float(tr.max()),
               fixed065_regret=float(fr.mean()), transfer_minus_fixed=float((tr - fr).mean()),
               global_alpha=g_a, fold_alpha_MVTec=falpha.get('MVTec'),
               fold_alpha_VisA=falpha.get('VisA'), fold_alpha_BTAD=falpha.get('BTAD'))


def ci95(xs):
    a = np.array(xs, float); m = a.mean()
    if len(a) < 2:
        return float(m), float(m), float(m)
    se = a.std(ddof=1) / np.sqrt(len(a)); h = stats.t.ppf(0.975, len(a) - 1) * se
    return float(m), float(m - h), float(m + h)


def main():
    seeds = list(range(10))
    rows = [per_seed_metrics(s) for s in seeds]
    agg = {}
    for k in [k for k in rows[0] if k != 'n']:
        vals = [r[k] for r in rows]
        m, lo, hi = ci95(vals)
        agg[k] = dict(mean=round(m, 4), ci95=[round(lo, 4), round(hi, 4)],
                      seed0=round(float(rows[0][k]), 4), per_seed=[round(float(v), 4) for v in vals])
    out = dict(experiment="seed-averaged LODO + regret-vs-alpha",
               seeds=seeds, n_categories=rows[0]['n'],
               method="per-seed metric, then mean +/- t(9) 95% CI; per-category min-max norm",
               aggregate=agg)
    outp = os.path.join(os.path.dirname(BASE_DIR), "results", "seed_avg_fusion_metrics.json")
    json.dump(out, open(outp, 'w'), indent=2)

    def show(k, lbl, pp=True):
        a = agg[k]; u = ' pp' if pp else ''
        print(f"  {lbl:<40} {a['mean']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]{u}   (seed0 {a['seed0']})")
    print("=" * 80); print("  SEED-AVERAGED FUSION METRICS (10 seeds) - LODO + regret-vs-alpha"); print("=" * 80)
    show('lodo_transfer_regret', 'LODO transfer regret')
    show('fixed065_regret', 'In-sample fixed-0.65 regret')
    show('transfer_minus_fixed', 'transfer - fixed')
    show('pooled_min_regret', 'pooled min regret')
    show('regret_065', 'regret at alpha=0.65')
    show('band45_lo', 'band (<=0.45pp) lower edge', False); show('band45_hi', 'band (<=0.45pp) upper edge', False)
    show('band20_lo', 'band (<=0.20pp) lower edge', False); show('band20_hi', 'band (<=0.20pp) upper edge', False)
    show('opt_MVTec', 'per-dataset optimum MVTec', False); show('opt_VisA', 'per-dataset optimum VisA', False); show('opt_BTAD', 'per-dataset optimum BTAD', False)
    print(f"\n  Saved: {outp}")


if __name__ == "__main__":
    main()
