"""T3 §4e : evaluate the reference-selection strategies across all 30 categories."""
import os
import sys
import json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from score_fusion import (load_anomalyclip_scores, load_anomalydino_scores, align_samples,
                          normalize_scores, compute_auroc)
from fine_grained_alpha_sweep import load_btad_dino_scores

AD = os.path.join(BASE, "AnomalyDINO")
CL = os.path.join(BASE, "AnomalyCLIP", "results")
ALPHAS = [round(i / 100.0, 2) for i in range(0, 101)]
FIXED_A = 0.65
STRATS = ["sorted", "medoid", "farthest"]

DINO_DIRS = {
    "MVTec AD": [os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", f"1-shot_preprocess=agnostic_{s}")
                 for s in ("weak_cats", "strong_cats")],
    "VisA": [os.path.join(AD, "results_VisA", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_all")],
    "BTAD": [os.path.join(AD, "results_BTAD", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_btad_all")],
}
CLIP_CSV = {"MVTec AD": os.path.join(CL, "scores_mvtec", "anomalyclip_scores.csv"),
            "VisA": os.path.join(CL, "scores_visa", "anomalyclip_scores.csv"),
            "BTAD": os.path.join(CL, "scores_btad", "anomalyclip_scores.csv")}


def load_dino_strategy(ds, strat):
    loader = load_btad_dino_scores if ds == "BTAD" else load_anomalydino_scores
    d = {}
    for dpath in DINO_DIRS[ds]:
        f = os.path.join(dpath, f"measurements_strategy={strat}.csv")
        if os.path.exists(f):
            d.update(loader(f))
    return d


def per_cat(ds, strat, clip_cache):
    dino = load_dino_strategy(ds, strat)
    if not dino:
        return {}
    aligned = align_samples(clip_cache[ds], dino)
    rows = {}
    for cat, a in aligned.items():
        y = np.asarray(a["labels"])
        if len(np.unique(y)) < 2:
            continue
        cl = normalize_scores(a["clip_scores"]); dn = normalize_scores(a["dino_scores"])
        curve = {al: compute_auroc(y, (1 - al) * cl + al * dn) for al in ALPHAS}
        oa = max(ALPHAS, key=lambda al: curve[al])
        rows[cat] = dict(dataset=ds, dino_only=curve[1.0], clip_only=curve[0.0],
                         fused_065=curve[FIXED_A], oracle=curve[oa], oracle_alpha=oa)
    return rows


def main():
    clip_cache = {ds: load_anomalyclip_scores(CLIP_CSV[ds]) for ds in DINO_DIRS}
    available = []
    for s in STRATS:
        if any(os.path.exists(os.path.join(dp, f"measurements_strategy={s}.csv"))
               for ds in DINO_DIRS for dp in DINO_DIRS[ds]):
            available.append(s)
    if "sorted" not in available:
        print("No 'sorted' control CSVs found; run run_refstrategy_categories.py first."); return

    tables = {}
    for s in available:
        rows = {}
        for ds in DINO_DIRS:
            rows.update(per_cat(ds, s, clip_cache))
        tables[s] = rows
        print(f"  strategy {s}: {len(rows)} categories")

    cats = sorted(set.intersection(*[set(tables[s]) for s in available]))
    ctrl = tables["sorted"]
    out = {"experiment": "reference-selection strategies (seed-0 representative)",
           "strategies": available, "n_categories": len(cats), "fixed_alpha": FIXED_A, "by_strategy": {}, "vs_sorted": {}}
    for s in available:
        t = tables[s]
        out["by_strategy"][s] = dict(
            mean_dino_only=round(float(np.mean([t[c]["dino_only"] for c in cats])), 2),
            mean_fused_065=round(float(np.mean([t[c]["fused_065"] for c in cats])), 2),
            mean_oracle=round(float(np.mean([t[c]["oracle"] for c in cats])), 2),
            mean_oracle_alpha=round(float(np.mean([t[c]["oracle_alpha"] for c in cats])), 3))
    for s in available:
        if s == "sorted":
            continue
        t = tables[s]
        d_dino = [t[c]["dino_only"] - ctrl[c]["dino_only"] for c in cats]
        d_f065 = [t[c]["fused_065"] - ctrl[c]["fused_065"] for c in cats]
        dom_flips = sum(1 for c in cats if (t[c]["oracle_alpha"] > 0.5) != (ctrl[c]["oracle_alpha"] > 0.5))
        out["vs_sorted"][s] = dict(
            mean_abs_delta_dino=round(float(np.mean(np.abs(d_dino))), 3),
            max_abs_delta_dino=round(float(np.max(np.abs(d_dino))), 2),
            mean_abs_delta_fused065=round(float(np.mean(np.abs(d_f065))), 3),
            mean_delta_fused065=round(float(np.mean(d_f065)), 3),
            n_dominance_flips=dom_flips)
    out["per_category"] = {s: {c: {k: round(v, 3) if isinstance(v, float) else v
                                   for k, v in tables[s][c].items()} for c in cats} for s in available}
    json.dump(out, open(os.path.join(os.path.dirname(BASE), "results", "refstrategy_eval.json"), "w"), indent=2)

    print("\n" + "=" * 80)
    print("  REFERENCE-SELECTION STRATEGIES (seed-0 representative, 30 cats)")
    print("=" * 80)
    print(f"  {'strategy':<10}{'DINO-only':>10}{'fused.65':>10}{'oracle':>9}{'oracle a*':>10}")
    for s in available:
        b = out["by_strategy"][s]
        print(f"  {s:<10}{b['mean_dino_only']:>10}{b['mean_fused_065']:>10}{b['mean_oracle']:>9}{b['mean_oracle_alpha']:>10}")
    for s in available:
        if s == "sorted":
            continue
        v = out["vs_sorted"][s]
        print(f"  vs sorted [{s}]: mean|ΔDINO|={v['mean_abs_delta_dino']}pp (max {v['max_abs_delta_dino']}), "
              f"mean|Δfused.65|={v['mean_abs_delta_fused065']}pp, dominance flips={v['n_dominance_flips']}/{len(cats)}")
    print("\n  Saved: results/refstrategy_eval.json")


if __name__ == "__main__":
    main()
