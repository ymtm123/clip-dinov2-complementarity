"""Statistical power / sensitivity of the fusion-gain effect vs N (secondary illustration)."""
import os
import sys
import csv
import io
import json
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from seed_avg_fusion_metrics import load_all_datasets
from leave_one_dataset_out_alpha import build_curves, ALPHAS
from score_fusion import load_anomalyclip_scores, normalize_scores, compute_auroc

AD = os.path.join(BASE, "AnomalyDINO")
CL = os.path.join(BASE, "AnomalyCLIP", "results")
RES = os.path.join(os.path.dirname(BASE), "results")
FIXED_A = 0.65

LOCO_CLIP = os.path.join(CL, "loco_split", "loco_clip_scores.csv")
LOCO_DINO = os.path.join(AD, "results_MVTEC_LOCO", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_loco", "measurements_seed=0.csv")
DAGM_CLIP = os.path.join(CL, "scores_dagm", "anomalyclip_scores.csv")
DAGM_DINO = os.path.join(AD, "results_DAGM", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_dagm_all", "measurements_seed=0.csv")


def _gains(clip, dino, y):
    cl = normalize_scores(clip); dn = normalize_scores(dino)
    curve = {a: compute_auroc(y, (1 - a) * cl + a * dn) for a in ALPHAS}
    best = max(curve[0.0], curve[1.0]); oa = max(ALPHAS, key=lambda a: curve[a])
    return dict(equal_gain=curve[0.5] - best, fixed065_gain=curve[FIXED_A] - best,
                oracle_gain=curve[oa] - best, oracle_alpha=oa)


def core_rows():
    curves, cat2ds = build_curves(load_all_datasets(0))
    rows = []
    for c in curves:
        best = max(curves[c][0.0], curves[c][1.0]); oa = max(ALPHAS, key=lambda a: curves[c][a])
        rows.append(dict(category=c, dataset=cat2ds[c], equal_gain=curves[c][0.5] - best,
                         fixed065_gain=curves[c][FIXED_A] - best, oracle_gain=curves[c][oa] - best, oracle_alpha=oa))
    return rows


def loco_structural_rows():
    if not (os.path.exists(LOCO_CLIP) and os.path.exists(LOCO_DINO)):
        return []
    clip = {}
    with io.open(LOCO_CLIP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            at = r["AnomType"]
            if at not in ("good", "structural"):
                continue
            obj = r["Object"]; stem = os.path.splitext(os.path.basename(r["Sample"]))[0]
            clip.setdefault(obj, {})[("good" if at == "good" else "structural", stem)] = (float(r["Score"]), 0 if at == "good" else 1)
    dino = {}
    with io.open(LOCO_DINO, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            samp = r["Sample"].replace("\\", "/"); sub = samp.split("/")[0]
            if sub not in ("good", "structural_anomalies"):
                continue
            obj = r["Object"]; stem = os.path.splitext(os.path.basename(samp))[0]
            dino.setdefault(obj, {})[("good" if sub == "good" else "structural", stem)] = float(r["Anomaly_Score"])
    rows = []
    for obj in set(clip) & set(dino):
        common = sorted(set(clip[obj]) & set(dino[obj]))
        if len(common) < 10:
            continue
        cl = np.array([clip[obj][k][0] for k in common]); y = np.array([clip[obj][k][1] for k in common])
        dn = np.array([dino[obj][k] for k in common])
        if len(np.unique(y)) < 2:
            continue
        g = _gains(cl, dn, y); g.update(category=obj, dataset="MVTec-LOCO (structural)")
        rows.append(g)
    return rows


def dagm_rows():
    if not (os.path.exists(DAGM_CLIP) and os.path.exists(DAGM_DINO)):
        return []
    from score_fusion import load_anomalydino_scores, align_samples
    aligned = align_samples(load_anomalyclip_scores(DAGM_CLIP), load_anomalydino_scores(DAGM_DINO))
    rows = []
    for cat, a in aligned.items():
        y = np.asarray(a["labels"])
        if len(np.unique(y)) < 2:
            continue
        g = _gains(np.asarray(a["clip_scores"], float), np.asarray(a["dino_scores"], float), y)
        g.update(category=cat, dataset="DAGM")
        rows.append(g)
    return rows


def ci95(xs):
    a = np.asarray(xs, float); m = a.mean()
    if len(a) < 2:
        return float(m), float(m), float(m)
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(m), float(m - h), float(m + h)


def t_power(mean, sd, n, alpha=0.05):
    if sd <= 0 or n < 2:
        return 1.0 if mean != 0 else 0.0
    ncp = mean / sd * np.sqrt(n); crit = stats.t.ppf(1 - alpha / 2, n - 1)
    return float(1 - stats.nct.cdf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp))


def min_detectable_effect(sd, n, target=0.80, alpha=0.05):
    if n < 2 or sd <= 0:
        return None
    lo, hi = 0.0, 10.0 * sd
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t_power(mid, sd, n, alpha) < target else (lo, mid)
    return 0.5 * (lo + hi)


def effect(rows, key):
    g = np.array([r[key] for r in rows], float)
    m, lo, hi = ci95(g); sd = float(g.std(ddof=1)) if len(g) > 1 else 0.0
    mde = min_detectable_effect(sd, len(rows))
    return dict(pooled_mean=round(m, 3), ci95=[round(lo, 3), round(hi, 3)], sd=round(sd, 3),
                achieved_power=round(t_power(m, sd, len(rows)), 3),
                min_detectable_effect_80pct=(round(mde, 3) if mde is not None else None),
                per_dataset={ds: dict(n=sum(1 for r in rows if r["dataset"] == ds),
                                      mean=round(float(np.mean([r[key] for r in rows if r["dataset"] == ds])), 3))
                             for ds in sorted(set(r["dataset"] for r in rows))})


def main():
    core = core_rows(); loco = loco_structural_rows(); dagm = dagm_rows()
    print(f"core={len(core)}  DAGM={len(dagm)}  LOCO-structural={len(loco)} (LOCO reported separately, not pooled)")

    # power-vs-N pool: SAME traditional-AD regime only (core + DAGM); LOCO is heterogeneity, kept out.
    pool_stages = [("N30_core", core)]
    if dagm:
        pool_stages.append((f"N{len(core)+len(dagm)}_core+DAGM", core + dagm))

    out = dict(
        experiment="secondary pooled-t power illustration (primary = cluster_bootstrap_headline_ci.json)",
        lead_inference="results/cluster_bootstrap_headline_ci.json (cluster-robust bootstrap)",
        primary_effect="fixed alpha=0.65 fusion gain (AUROC@0.65 - best single), seed-0, image-level",
        ruling=("Pooled-t treats categories as independent, ignoring the dataset clustering, so this is a "
                "SECONDARY illustration; the cluster-robust bootstrap is the real inference. oracle_gain>=0 "
                "by construction + post-hoc alpha = near-tautological (caveated note only). LOCO-structural "
                "is a separate heterogeneity/scope finding, NOT pooled into power-vs-N."),
        power_vs_N={}, supplementary={}, loco_structural_heterogeneity={})

    for name, rows in pool_stages:
        out["power_vs_N"][name] = dict(N=len(rows), n_datasets=len(set(r["dataset"] for r in rows)),
                                       fixed065_gain=effect(rows, "fixed065_gain"))
        e = out["power_vs_N"][name]["fixed065_gain"]
        print(f"  [PRIMARY fixed-0.65] {name:<14} N={len(rows):<3}  gain={e['pooled_mean']:+.3f} "
              f"[{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]pp  power={e['achieved_power']:.2f}  MDE@80%={e['min_detectable_effect_80pct']}pp")
        print(f"        per-dataset: {e['per_dataset']}")

    # supplementary (caveated): oracle + equal on the full pool, clearly labelled
    full = core + dagm
    out["supplementary"] = dict(
        note="oracle_gain is >=0 by construction (alpha in {0,1} in grid) and uses a post-hoc alpha -> "
             "near-tautological; equal_gain is the alpha=0.5 fixed variant. Not the headline.",
        oracle_gain=effect(full, "oracle_gain"), equal_gain=effect(full, "equal_gain"))

    if loco:
        out["loco_structural_heterogeneity"] = dict(
            note="MVTec-LOCO STRUCTURAL: fixed alpha does NOT transfer to this regime (supports the "
                 "understanding-optimization gap). Reported as SCOPE/heterogeneity, not pooled into power-vs-N.",
            n=len(loco), fixed065_gain_mean=round(float(np.mean([r["fixed065_gain"] for r in loco])), 3),
            equal_gain_mean=round(float(np.mean([r["equal_gain"] for r in loco])), 3),
            oracle_gain_mean=round(float(np.mean([r["oracle_gain"] for r in loco])), 3),
            per_category={r["category"]: dict(fixed065=round(r["fixed065_gain"], 2), oracle_alpha=r["oracle_alpha"]) for r in loco})
        h = out["loco_structural_heterogeneity"]
        print(f"  [SCOPE LOCO-structural, NOT pooled] n={h['n']}  fixed-0.65 gain mean={h['fixed065_gain_mean']:+.2f}pp "
              f"(equal {h['equal_gain_mean']:+.2f}, oracle {h['oracle_gain_mean']:+.2f})")

    json.dump(out, open(os.path.join(RES, "power_sensitivity_analysis.json"), "w"), indent=2)
    print(f"\n  Primary inference = results/cluster_bootstrap_headline_ci.json.")
    print(f"  Saved: results/power_sensitivity_analysis.json")


if __name__ == "__main__":
    main()
