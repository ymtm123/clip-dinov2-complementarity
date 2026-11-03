"""Seed-averaged deployment-legal normalization over the 10 reference-image seeds."""
import os, sys, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import practical_normalization as pn
from score_fusion import load_anomalyclip_scores, load_anomalydino_scores, align_samples
from fine_grained_alpha_sweep import load_btad_dino_scores

AD = os.path.join(pn.BASE, "AnomalyDINO")


def load_30_seed(seed):
    clip = {ds: load_anomalyclip_scores(os.path.join(pn.BASE, "AnomalyCLIP", "results", f"scores_{k}", "anomalyclip_scores.csv"))
            for ds, k in [("MVTec", "mvtec"), ("VisA", "visa"), ("BTAD", "btad")]}
    dm = {}
    dm.update(load_anomalydino_scores(os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_weak_cats", f"measurements_seed={seed}.csv")))
    dm.update(load_anomalydino_scores(os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_strong_cats", f"measurements_seed={seed}.csv")))
    dino = {"MVTec": dm,
            "VisA": load_anomalydino_scores(os.path.join(AD, "results_VisA", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_all", f"measurements_seed={seed}.csv")),
            "BTAD": load_btad_dino_scores(os.path.join(AD, "results_BTAD", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_btad_all", f"measurements_seed={seed}.csv"))}
    table = {}
    for ds in ["MVTec", "VisA", "BTAD"]:
        for cat, d in align_samples(clip[ds], dino[ds]).items():
            if len(np.unique(d["labels"])) >= 2:
                table[cat] = (ds, np.asarray(d["clip_scores"], float), np.asarray(d["dino_scores"], float), np.asarray(d["labels"]))
    return table


def ci95(xs):
    a = np.array(xs, float); m = a.mean()
    if len(a) < 2:
        return float(m), float(m), float(m)
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(m), float(m - h), float(m + h)


def main():
    fields = ["regret_pp", "fusion_gain_vs_best", "fixed_065", "global_alpha", "oracle", "best_single"]
    acc = {p: {f: [] for f in fields} for p in pn.PROTOCOLS}
    for s in range(10):
        summ = pn.summarize(pn.aggregate(load_30_seed(s)))
        for p in pn.PROTOCOLS:
            for f in fields:
                acc[p][f].append(summ[p][f])
        print(f"  seed {s}: calib_zscore regret={summ['calib_zscore']['regret_pp']:.3f}  transductive_minmax regret={summ['transductive_minmax']['regret_pp']:.3f}")
    out = {"experiment": "seed-averaged deployment-legal normalization", "seeds": list(range(10)),
           "n_splits": pn.N_SPLITS, "method": "per-seed protocol summary, then mean +/- t(9) 95% CI", "protocols": {}}
    for p in pn.PROTOCOLS:
        d = {}
        for f in fields:
            m, lo, hi = ci95(acc[p][f])
            d[f] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "seed0": round(float(acc[p][f][0]), 4)}
        out["protocols"][p] = d
    json.dump(out, open(os.path.join(os.path.dirname(pn.BASE), "results", "seed_avg_practical_norm.json"), "w"), indent=2)
    print("\n" + "=" * 86)
    print("  SEED-AVERAGED DEPLOYMENT-LEGAL NORMALIZATION (mean [95% CI]; seed0 in parens)")
    print("=" * 86)
    for p in pn.PROTOCOLS:
        r = out["protocols"][p]["regret_pp"]; g = out["protocols"][p]["fusion_gain_vs_best"]; a = out["protocols"][p]["global_alpha"]
        legal = " (deployment-legal)" if p.startswith("calib") else ""
        print(f"  {p:<22} a*={a['mean']:.2f}  regret={r['mean']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  gain={g['mean']:+.3f} (seed0 regret {r['seed0']}){legal}")
    print("\n  Saved: results/seed_avg_practical_norm.json")


if __name__ == "__main__":
    main()
