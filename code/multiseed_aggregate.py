"""Aggregate the multi-seed AnomalyDINO runs ."""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fusion import (load_anomalyclip_scores, load_anomalydino_scores,
                          align_samples, normalize_scores, compute_auroc)
from fine_grained_alpha_sweep import load_btad_dino_scores

BASE = os.path.dirname(os.path.abspath(__file__))
AD = os.path.join(BASE, "AnomalyDINO")
ALPHAS = [round(i / 100.0, 2) for i in range(0, 101)]
SEEDS = list(range(0, 10))

CLIP = {
    "MVTec": os.path.join(BASE, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv"),
    "VisA": os.path.join(BASE, "AnomalyCLIP", "results", "scores_visa", "anomalyclip_scores.csv"),
    "BTAD": os.path.join(BASE, "AnomalyCLIP", "results", "scores_btad", "anomalyclip_scores.csv"),
}


def dino_csv(ds, tag, seed):
    sub = {"MVTec": "results_MVTec", "VisA": "results_VisA", "BTAD": "results_BTAD"}[ds]
    return os.path.join(AD, sub, "dinov2_vitl14_448", f"1-shot_preprocess={tag}", f"measurements_seed={seed}.csv")


def load_dino_seed(ds, seed):
    if ds == "MVTec":
        d = {}
        d.update(load_anomalydino_scores(dino_csv("MVTec", "agnostic_weak_cats", seed)))
        d.update(load_anomalydino_scores(dino_csv("MVTec", "agnostic_strong_cats", seed)))
        return d
    if ds == "VisA":
        return load_anomalydino_scores(dino_csv("VisA", "agnostic_all", seed))
    return load_btad_dino_scores(dino_csv("BTAD", "agnostic_btad_all", seed))


def per_cat_metrics(aligned):
    """For each category: dino-only AUROC, clip-only AUROC, oracle alpha+auroc, fused@0.65."""
    out = {}
    for cat, d in aligned.items():
        y = d["labels"]
        if len(np.unique(y)) < 2:
            continue
        cn = normalize_scores(d["clip_scores"], "minmax")
        dn = normalize_scores(d["dino_scores"], "minmax")
        aurocs = {a: compute_auroc(y, (1 - a) * cn + a * dn) for a in ALPHAS}
        oa = max(ALPHAS, key=lambda x: aurocs[x])
        out[cat] = dict(clip=aurocs[0.0], dino=aurocs[1.0], fixed065=aurocs[0.65],
                        oracle_alpha=oa, oracle_auroc=aurocs[oa])
    return out


def ci95(vals):
    v = np.array(vals, dtype=float)
    m, sd = v.mean(), v.std(ddof=1)
    half = 1.96 * sd / np.sqrt(len(v))
    return float(m), float(sd), float(m - half), float(m + half)


def main():
    clip_data = {ds: load_anomalyclip_scores(p) for ds, p in CLIP.items()}

    # per (seed, cat) metrics
    by_seed = {}   # seed -> {cat: metrics}
    cat_ds = {}
    for s in SEEDS:
        merged = {}
        for ds in ["MVTec", "VisA", "BTAD"]:
            aligned = align_samples(clip_data[ds], load_dino_seed(ds, s))
            m = per_cat_metrics(aligned)
            for c in m:
                cat_ds[c] = ds
            merged.update(m)
        by_seed[s] = merged
    cats = sorted(by_seed[0].keys())
    print(f"Seeds {SEEDS[0]}-{SEEDS[-1]} x {len(cats)} categories aligned.")

    # ---------- E1: per-category DINO-only AUROC mean +/- SD ----------
    e1 = {}
    for c in cats:
        dvals = [by_seed[s][c]["dino"] for s in SEEDS]
        fvals = [by_seed[s][c]["fixed065"] for s in SEEDS]
        m, sd, lo, hi = ci95(dvals)
        fm, fsd, _, _ = ci95(fvals)
        e1[c] = dict(dataset=cat_ds[c], dino_mean=round(m, 2), dino_sd=round(sd, 2),
                     dino_ci=[round(lo, 2), round(hi, 2)], seed0_dino=round(by_seed[0][c]["dino"], 2),
                     fixed065_mean=round(fm, 2), fixed065_sd=round(fsd, 2))
    dino_sds = [e1[c]["dino_sd"] for c in cats]
    fused_sds = [e1[c]["fixed065_sd"] for c in cats]
    print(f"\n[E1] DINOv2-only per-cat AUROC SD across seeds: median {np.median(dino_sds):.2f}, "
          f"max {np.max(dino_sds):.2f} ({cats[int(np.argmax(dino_sds))]}); "
          f">2pp SD in {sum(1 for x in dino_sds if x>2)}/{len(cats)} cats")
    # variance-averaging preview (E4): fused SD vs dino SD
    print(f"     fixed-0.65 fusion per-cat AUROC SD: median {np.median(fused_sds):.2f} "
          f"(vs DINO median {np.median(dino_sds):.2f}) -> fusion {'reduces' if np.median(fused_sds)<np.median(dino_sds) else 'does not reduce'} seed variance")

    # ---------- E2: borderline flip audit ----------
    a0 = {c: by_seed[0][c]["oracle_alpha"] for c in cats}
    borderline = [c for c in cats if abs(a0[c] - 0.5) <= 0.15]
    flips = {}
    for c in cats:
        prefs = [(by_seed[s][c]["oracle_alpha"] > 0.5) for s in SEEDS]  # True=DINO
        n_dino = sum(prefs)
        flips[c] = dict(seed0_alpha=round(a0[c], 2), n_seeds_dino=int(n_dino),
                        n_seeds_clip=int(len(SEEDS) - n_dino),
                        flips=int(min(n_dino, len(SEEDS) - n_dino)),
                        alpha_sd=round(float(np.std([by_seed[s][c]["oracle_alpha"] for s in SEEDS], ddof=1)), 3))
    flip_border = {c: flips[c] for c in borderline}
    nonborder_flips = sum(flips[c]["flips"] for c in cats if c not in borderline)
    print(f"\n[E2] borderline (|alpha*-0.5|<=0.15) seed-0 cats: {borderline}")
    print(f"     flips among borderline: " + ", ".join(f"{c}={flips[c]['flips']}/10" for c in borderline))
    print(f"     total preferred-model flips in NON-borderline cats: {nonborder_flips} "
          f"(of {len(cats)-len(borderline)} cats)")

    # ---------- E3: seed-averaged headline numbers ----------
    dino_pref = [sum(1 for c in cats if by_seed[s][c]["oracle_alpha"] > 0.5) for s in SEEDS]
    penalty = [float(np.mean([abs(by_seed[s][c]["dino"] - by_seed[s][c]["clip"]) for c in cats])) for s in SEEDS]
    regret = [float(np.mean([by_seed[s][c]["oracle_auroc"] - by_seed[s][c]["fixed065"] for c in cats])) for s in SEEDS]
    def fmt(vals):
        m, sd, lo, hi = ci95(vals); return dict(mean=round(m, 3), sd=round(sd, 3), ci95=[round(lo, 3), round(hi, 3)],
                                                seed0=round(vals[0], 3), range=[round(min(vals), 3), round(max(vals), 3)])
    e3 = {"dino_preferred_count_of_30": fmt([float(x) for x in dino_pref]),
          "wrong_model_penalty_pp": fmt(penalty),
          "fixed065_regret_pp": fmt(regret)}
    print(f"\n[E3] seed-averaged headline numbers (mean +/- 95% CI over 10 seeds):")
    print(f"     DINOv2-preferred categories: {e3['dino_preferred_count_of_30']['mean']:.1f}/30 "
          f"(range {e3['dino_preferred_count_of_30']['range']}, seed0 {e3['dino_preferred_count_of_30']['seed0']:.0f})")
    print(f"     wrong-model penalty: {e3['wrong_model_penalty_pp']['mean']:.2f} pp "
          f"CI{e3['wrong_model_penalty_pp']['ci95']} (seed0 {e3['wrong_model_penalty_pp']['seed0']:.2f})")
    print(f"     fixed-0.65 regret:   {e3['fixed065_regret_pp']['mean']:.3f} pp "
          f"CI{e3['fixed065_regret_pp']['ci95']} (seed0 {e3['fixed065_regret_pp']['seed0']:.3f})")

    out = {"experiment": "multi-seed aggregate",
           "n_seeds": len(SEEDS), "n_categories": len(cats),
           "E1_per_category": e1,
           "E1_summary": {"dino_sd_median": round(float(np.median(dino_sds)), 3),
                          "dino_sd_max": round(float(np.max(dino_sds)), 3),
                          "fused_sd_median": round(float(np.median(fused_sds)), 3)},
           "E2_borderline": flip_border, "E2_nonborderline_total_flips": int(nonborder_flips),
           "E2_all_flips": flips, "E3_headline": e3}
    with open(os.path.join(os.path.dirname(BASE), "results", "multiseed_aggregate.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/multiseed_aggregate.json")


if __name__ == "__main__":
    main()
